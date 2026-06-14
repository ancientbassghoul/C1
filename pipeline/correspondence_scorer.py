"""
pipeline/correspondence_scorer.py – Quality scoring for manual correspondences.

Replaces SuperPoint descriptor matching with CLIP visual-embedding cosine
similarity.  For each correspondence, a square crop centred on each
manually-picked point is embedded with the CLIP vision encoder.  Cosine
similarity between all frame-pair embeddings is averaged to produce the
per-correspondence appearance score.

The CLIP model is loaded once, used, then torn down (del + empty_cache) so
that subsequent pipeline steps can load MASt3R without OOM.

Public API (unchanged — --show-scores viewer uses this directly):
    score_correspondences(frames, json_path) -> dict[int, float | None]
"""

from __future__ import annotations

import gc
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from pipeline.frame import Frame
import config

logger = logging.getLogger(__name__)

CROP_HALF_SIZE = 48   # half-width of the patch crop in pixels


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity ∈ [-1, 1] clamped to [0, 1]."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def _crop_patch(img_bgr: np.ndarray, cx: float, cy: float, half: int) -> Optional[Image.Image]:
    """Crop a (2*half) × (2*half) patch centred on (cx, cy) from a BGR image.

    Returns None if the patch would be entirely outside the image.
    """
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(round(cx)) - half)
    y1 = max(0, int(round(cy)) - half)
    x2 = min(w, int(round(cx)) + half)
    y2 = min(h, int(round(cy)) + half)
    if x2 <= x1 or y2 <= y1:
        return None
    patch_bgr = img_bgr[y1:y2, x1:x2]
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(patch_rgb)


def score_correspondences(
    frames    : list[Frame],
    json_path : Path,
) -> dict[int, Optional[float]]:
    """
    Compute CLIP-based appearance scores for all correspondences in *json_path*
    and write the results back into the file.

    Returns {id: score} where score is None when no patch could be extracted
    for at least one frame in a pair.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        logger.warning("No correspondence file at %s — nothing to score.", json_path)
        return {}

    data = json.loads(json_path.read_text())
    correspondences = data.get("correspondences", [])
    if not correspondences:
        logger.info("No correspondences to score.")
        return {}

    by_stem = {f.stem: f for f in frames}

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    import torch
    from transformers import CLIPModel, CLIPProcessor

    logger.info("Loading CLIP model for correspondence scoring …")
    clip_model = CLIPModel.from_pretrained(
        config.CLIP_MODEL,
        cache_dir=config.MODEL_CACHE_DIR,
    )
    clip_proc  = CLIPProcessor.from_pretrained(
        config.CLIP_MODEL,
        cache_dir=config.MODEL_CACHE_DIR,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device).eval()

    def _embed(pil_img: Image.Image) -> np.ndarray:
        inputs = clip_proc(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    # ── Score each correspondence ─────────────────────────────────────────────
    scores: dict[int, Optional[float]] = {}

    for entry in correspondences:
        corr_id = int(entry["id"])
        points  = entry.get("points", {})   # {stem: [x, y]} or {stem: [[x,y],[x,y]]}

        # For pair-type entries (wheel_axis, roof_edge), use the first point only
        embeds_per_frame: dict[str, Optional[np.ndarray]] = {}
        for stem, pts_val in points.items():
            f = by_stem.get(stem)
            if f is None or f.undistorted is None:
                embeds_per_frame[stem] = None
                continue

            # Support both single [x,y] and pair [[x,y],[x,y]]
            if isinstance(pts_val[0], (list, tuple)):
                px, py = float(pts_val[0][0]), float(pts_val[0][1])
            else:
                px, py = float(pts_val[0]), float(pts_val[1])

            patch = _crop_patch(f.undistorted, px, py, CROP_HALF_SIZE)
            if patch is None:
                embeds_per_frame[stem] = None
                logger.debug("  corr %d / %s: patch out of bounds at (%.0f, %.0f)",
                             corr_id, stem[-8:], px, py)
            else:
                embeds_per_frame[stem] = _embed(patch)

        # Pairwise cosine similarities
        stems     = list(points.keys())
        sims:      list[float] = []
        pair_info: list[dict]  = []

        for sa, sb in combinations(stems, 2):
            ea = embeds_per_frame.get(sa)
            eb = embeds_per_frame.get(sb)
            if ea is None or eb is None:
                pair_info.append({"frames": [sa, sb], "similarity": None})
                continue
            sim = _cosine(ea, eb)
            sims.append(sim)
            pair_info.append({"frames": [sa, sb], "similarity": round(sim, 4)})

        overall = float(np.mean(sims)) if sims else None
        scores[corr_id] = overall

        level = ("good" if overall and overall > 0.6
                 else "ok"   if overall and overall > 0.35
                 else "poor")
        logger.info(
            "  Correspondence %2d: appearance=%-5s  score=%s  (%d pairs)",
            corr_id,
            level,
            f"{overall:.3f}" if overall is not None else "N/A",
            len(sims),
        )

        entry["scores"] = {
            "appearance": round(overall, 4) if overall is not None else None,
            "pairs":      pair_info,
        }

    # Write scores back into the JSON
    json_path.write_text(json.dumps(data, indent=2))
    logger.info("Scores written to %s", json_path)

    # ── Tear down CLIP to free VRAM ───────────────────────────────────────────
    del clip_model, clip_proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("CLIP model released.")

    return scores
