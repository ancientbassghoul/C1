"""
pipeline/correspondence_scorer.py – Quality scoring for manual correspondences.

Two complementary scores per correspondence
────────────────────────────────────────────
APPEARANCE (0–1)
    For each pair of frames in a correspondence, extract the SuperPoint
    descriptor of the nearest detected keypoint to the manually-picked
    location, then compute cosine similarity between the two descriptors.
    Average over all frame pairs → overall appearance score.

    High score = patches look like the same feature to SuperPoint.
    Low score  = patches look different, possibly a blunder — or just a
                 large viewpoint change (oblique drone → expected).

    When no SuperPoint keypoint falls within SCORE_SNAP_RADIUS pixels of
    a manually-picked point the score for that pair is marked None (shown
    as grey in the viewer).

Scores are written back into the same JSON that the manual picker uses,
under a "scores" key on each correspondence entry, so they persist between
runs and are immediately visible in the viewer without re-scoring.

Public API
──────────
    score_correspondences(frames, json_path) -> dict[int, float | None]
        Compute / refresh scores for all correspondences in json_path.
        Returns {correspondence_id: score}.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.frame import Frame
import config

logger = logging.getLogger(__name__)

# Max pixel distance from a manually-picked point to the nearest SuperPoint
# keypoint for the descriptor to be considered valid.
SCORE_SNAP_RADIUS = 25.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity ∈ [-1, 1] clamped to [0, 1]."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def score_correspondences(
    frames    : list[Frame],
    json_path : Path,
) -> dict[int, Optional[float]]:
    """
    Compute appearance scores for all correspondences in *json_path* and
    write the results back into the file.

    Returns {id: score} where score is None when descriptors could not be
    extracted (no SuperPoint keypoint near the picked location in ≥1 frame).
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

    # ── Extract SuperPoint keypoints + descriptors for every relevant frame ───
    import pipeline.feature_matcher as fm
    import torch
    from lightglue.utils import rbd

    fm._load_models()
    device      = fm._device
    device_type = device.type

    # Gather which frames actually appear in the correspondences
    needed_stems: set[str] = set()
    for entry in correspondences:
        needed_stems.update(entry.get("points", {}).keys())

    needed_frames = [f for f in frames if f.stem in needed_stems]
    logger.info("Scoring: running SuperPoint on %d frame(s)…", len(needed_frames))

    # frame → (kps (N,2) float32, descs (N,256) float32)
    frame_features: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for f in needed_frames:
        if f.undistorted is None:
            continue
        t = fm._to_tensor(f.undistorted, device)
        with torch.autocast(device_type=device_type,
                            enabled=(device_type == "cuda")):
            with torch.no_grad():
                feats = fm._extractor.extract(t)
        feats_rb = rbd(feats)
        kps   = feats_rb["keypoints"].cpu().numpy()    # (N, 2)
        descs = feats_rb["descriptors"].cpu().numpy()  # (N, D)
        frame_features[f.stem] = (kps, descs)
        logger.debug("  %s: %d keypoints", f.stem[-12:], len(kps))

    # ── Score each correspondence ─────────────────────────────────────────────
    scores: dict[int, Optional[float]] = {}

    for entry in correspondences:
        corr_id = int(entry["id"])
        points  = entry.get("points", {})   # {stem: [x, y]}

        # Resolve descriptors at each picked location
        descs_per_frame: dict[str, Optional[np.ndarray]] = {}
        for stem, (px, py) in points.items():
            if stem not in frame_features:
                descs_per_frame[stem] = None
                continue
            kps, descs = frame_features[stem]
            if len(kps) == 0:
                descs_per_frame[stem] = None
                continue
            dists   = np.linalg.norm(kps - [px, py], axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] > SCORE_SNAP_RADIUS:
                descs_per_frame[stem] = None
                logger.debug(
                    "  corr %d / %s: no keypoint within %.0fpx "
                    "(nearest=%.1fpx)",
                    corr_id, stem[-8:], SCORE_SNAP_RADIUS, dists[nearest]
                )
            else:
                descs_per_frame[stem] = descs[nearest]

        # Pairwise cosine similarity across all frame pairs
        stems   = list(points.keys())
        sims    : list[float] = []
        pair_info: list[dict] = []

        for sa, sb in combinations(stems, 2):
            da, db = descs_per_frame.get(sa), descs_per_frame.get(sb)
            if da is None or db is None:
                pair_info.append({"frames": [sa, sb], "similarity": None})
                continue
            sim = _cosine(da, db)
            sims.append(sim)
            pair_info.append({"frames": [sa, sb], "similarity": round(sim, 4)})

        overall = float(np.mean(sims)) if sims else None
        scores[corr_id] = overall

        level = "good" if overall and overall > 0.6 else ("ok" if overall and overall > 0.35 else "poor")
        logger.info("  Correspondence %2d: appearance=%-5s  score=%s  (%d pairs)",
                    corr_id,
                    level,
                    f"{overall:.3f}" if overall is not None else "N/A",
                    len(sims))

        entry["scores"] = {
            "appearance" : round(overall, 4) if overall is not None else None,
            "pairs"      : pair_info,
        }

    # Write scores back into the JSON
    json_path.write_text(json.dumps(data, indent=2))
    logger.info("Scores written to %s", json_path)
    return scores
