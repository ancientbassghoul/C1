"""
pipeline/detect_anchor.py – Open-world anchor object discovery via Qwen VL + CLIP.

Four phases:
  1. Per-frame Qwen VL object discovery (generalized labels, bboxes).
  2. Label consolidation + cross-frame coverage count.
  3. Anchor selection — highest coverage, Qwen reasoning to pick best 3D anchor.
  4. CLIP cosine similarity weighting per frame.

VRAM note: Qwen and MASt3R both require significant VRAM.  This module
explicitly tears down Qwen and CLIP before returning so MASt3R can load
without OOM on 16–24 GB pods.

Qwen bbox coordinates: Qwen2.5-VL returns bboxes in a [0, 1000] normalized
window relative to its internal processing resolution (with possible
aspect-ratio letterboxing).  We use the processor's image_processor metadata
to map back to actual pixel coordinates.
"""

from __future__ import annotations

import gc
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnchorResult:
    label: str                                    # normalized anchor label
    bboxes:    dict = field(default_factory=dict) # frame.stem → [y1,x1,y2,x2] pixel
    centroids: dict = field(default_factory=dict) # frame.stem → (u, v) pixel
    weights:   dict = field(default_factory=dict) # frame.stem → CLIP cosine similarity


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Per-frame Qwen VL discovery
# ─────────────────────────────────────────────────────────────────────────────

_DISCOVERY_PROMPT = (
    "List every discrete physical object you can see in this image. "
    "For each object provide a bounding box and a label. "
    "Use the most generalized category term possible — e.g. 'vehicle', 'tree', 'building'. "
    "Only specialize if MULTIPLE instances of the same category appear in this image "
    "(e.g. 'vehicle-truck' vs 'vehicle-private', then by color if still ambiguous). "
    "Return ONLY a JSON array, no other text. "
    'Format: [{"label": "...", "bbox": [ymin, xmin, ymax, xmax]}, ...]  '
    "where bbox values are integers in [0, 1000] normalized to the image dimensions."
)


def _qwen_discover_frame(model, processor, frame_bgr: np.ndarray) -> list[dict]:
    """Run Qwen VL on one frame; return list of {label, bbox} dicts."""
    import torch
    import cv2
    from PIL import Image

    # Convert BGR → RGB PIL
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text": _DISCOVERY_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt").to(
        model.device
    )

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=512)

    generated = out_ids[0][inputs["input_ids"].shape[1]:]
    response  = processor.decode(generated, skip_special_tokens=True).strip()

    # Extract JSON array from response
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        objects = json.loads(clean)
        if not isinstance(objects, list):
            objects = []
    except json.JSONDecodeError:
        logger.warning("Qwen JSON parse failed; raw response: %s", response[:200])
        objects = []

    return objects


def _scale_bbox_to_pixels(
    bbox_norm: list[int],
    img_h: int,
    img_w: int,
) -> list[int]:
    """Map Qwen's [0,1000] normalized bbox to actual pixel coordinates.

    Qwen2.5-VL normalizes bboxes to a 1000×1000 window (with aspect-ratio
    letterboxing).  We recover pixel coords by scaling by img_h/1000 and
    img_w/1000 independently — this matches the model's coordinate convention.
    """
    y1, x1, y2, x2 = bbox_norm
    py1 = int(round(y1 * img_h / 1000.0))
    px1 = int(round(x1 * img_w / 1000.0))
    py2 = int(round(y2 * img_h / 1000.0))
    px2 = int(round(x2 * img_w / 1000.0))
    # Clamp to image bounds
    py1 = max(0, min(py1, img_h - 1))
    px1 = max(0, min(px1, img_w - 1))
    py2 = max(0, min(py2, img_h - 1))
    px2 = max(0, min(px2, img_w - 1))
    return [py1, px1, py2, px2]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Label consolidation
# ─────────────────────────────────────────────────────────────────────────────

def _consolidate_labels(
    per_frame_objects: dict[str, list[dict]],
) -> dict[str, list[Optional[dict]]]:
    """
    Group per-frame object detections by normalized label.

    Returns {label: [detection_or_None_per_frame_stem]} where None means the
    label was not seen in that frame.  Sorted descending by frame coverage.
    """
    stems = list(per_frame_objects.keys())
    label_detections: dict[str, dict[str, dict]] = defaultdict(dict)

    for stem, objects in per_frame_objects.items():
        for obj in objects:
            lbl = obj["label"].strip().lower()
            # Keep the highest-area detection per label per frame
            if lbl not in label_detections or _bbox_area(obj["bbox"]) > _bbox_area(
                label_detections[lbl].get(stem, {}).get("bbox", [0, 0, 0, 0])
            ):
                label_detections[lbl][stem] = obj

    # Convert to sorted list of (label, coverage, per_stem_det)
    result = {}
    for lbl, stem_map in label_detections.items():
        coverage = len(stem_map)
        result[lbl] = {"coverage": coverage, "by_stem": stem_map}

    return dict(sorted(result.items(), key=lambda x: -x[1]["coverage"]))


def _bbox_area(bbox: list) -> float:
    if len(bbox) < 4:
        return 0.0
    y1, x1, y2, x2 = bbox
    return max(0, (y2 - y1) * (x2 - x1))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Anchor selection
# ─────────────────────────────────────────────────────────────────────────────

_SELECTION_PROMPT_TMPL = (
    "I am calibrating drone cameras and need a single fixed 3D reference point "
    "that appears across multiple frames. "
    "Here are the candidate objects found across my frames: {candidates}. "
    "Which ONE of these would be the most reliable fixed 3D reference point for camera calibration? "
    "Exclude featureless surfaces such as sky, ground, road, or crops — those are not 3D landmarks. "
    "Reply with ONLY the exact label string from the list, nothing else."
)


def _qwen_select_anchor(model, processor, candidates: list[str]) -> str:
    """Ask Qwen to pick the best anchor label from a candidate list."""
    import torch

    if len(candidates) == 1:
        return candidates[0]

    prompt = _SELECTION_PROMPT_TMPL.format(candidates=", ".join(candidates))
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=32)

    generated = out_ids[0][inputs["input_ids"].shape[1]:]
    answer    = processor.decode(generated, skip_special_tokens=True).strip().lower()

    # Match answer to closest candidate
    for cand in candidates:
        if cand.lower() in answer or answer in cand.lower():
            return cand

    logger.warning("Qwen anchor selection unclear (%r); using highest-coverage candidate", answer)
    return candidates[0]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — CLIP weighting
# ─────────────────────────────────────────────────────────────────────────────

def _compute_clip_weights(
    clip_model,
    clip_processor,
    frames: list,
    label: str,
    bboxes_pixel: dict[str, list[int]],
) -> dict[str, float]:
    """
    Compute CLIP cosine similarity between each frame's anchor crop and a
    descriptive text prompt.

    Uses `f"a crisp, clear aerial photograph of a {label}"` rather than raw
    label text to place the text embedding in a richer manifold region,
    improving discrimination between sharp and blurry frames.
    """
    import torch
    import cv2
    from PIL import Image

    text_prompt = f"a crisp, clear aerial photograph of a {label}"
    weights: dict[str, float] = {}

    # Encode text once
    text_inputs = clip_processor(
        text=[text_prompt], return_tensors="pt", padding=True
    ).to(clip_model.device)
    with torch.no_grad():
        text_feats = clip_model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    frame_map = {f.stem: f for f in frames}

    for stem, bbox in bboxes_pixel.items():
        f = frame_map.get(stem)
        if f is None or f.undistorted is None:
            weights[stem] = 0.0
            continue

        y1, x1, y2, x2 = bbox
        if y2 <= y1 or x2 <= x1:
            weights[stem] = 0.0
            continue

        crop_bgr = f.undistorted[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)

        img_inputs = clip_processor(images=pil_crop, return_tensors="pt").to(
            clip_model.device
        )
        with torch.no_grad():
            img_feats = clip_model.get_image_features(**img_inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        similarity = float((img_feats @ text_feats.T).squeeze())
        weights[stem] = max(0.0, similarity)
        logger.info("  [%s] CLIP weight for '%s': %.3f", stem[-12:], label, similarity)

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def detect_anchor(frames: list) -> AnchorResult:
    """
    Discover the best shared anchor object across all frames using Qwen VL
    and compute per-frame CLIP confidence weights.

    Qwen and CLIP are torn down before returning to free VRAM for MASt3R.
    """
    # ── Load Qwen ─────────────────────────────────────────────────────────────
    logger.info("Loading Qwen VL model from %s …", config.QWEN_MODEL)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.QWEN_MODEL,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=config.MODEL_CACHE_DIR,
    )
    qwen_model.eval()
    qwen_proc = AutoProcessor.from_pretrained(
        config.QWEN_MODEL,
        cache_dir=config.MODEL_CACHE_DIR,
    )

    # ── Phase 1: Per-frame discovery ──────────────────────────────────────────
    logger.info("Phase 1: per-frame Qwen object discovery (%d frames) …", len(frames))
    per_frame_objects: dict[str, list[dict]] = {}

    for frame in frames:
        if frame.undistorted is None:
            logger.warning("  [%s] no undistorted image — skipping", frame.stem)
            per_frame_objects[frame.stem] = []
            continue

        logger.info("  Querying Qwen for frame %s …", frame.stem[-12:])
        raw_objects = _qwen_discover_frame(qwen_model, qwen_proc, frame.undistorted)

        h, w = frame.undistorted.shape[:2]
        scaled = []
        for obj in raw_objects:
            bbox_norm = obj.get("bbox", [])
            if len(bbox_norm) != 4:
                continue
            bbox_px = _scale_bbox_to_pixels(bbox_norm, h, w)
            scaled.append({"label": obj.get("label", "unknown"), "bbox": bbox_px})

        per_frame_objects[frame.stem] = scaled
        logger.info("    → %d objects found", len(scaled))

    # ── Phase 2: Label consolidation ─────────────────────────────────────────
    logger.info("Phase 2: label consolidation …")
    consolidated = _consolidate_labels(per_frame_objects)

    for lbl, info in consolidated.items():
        logger.info("  '%s': coverage %d/%d frames", lbl, info["coverage"], len(frames))

    # ── Phase 3: Anchor selection ─────────────────────────────────────────────
    logger.info("Phase 3: anchor selection …")

    # Prefer labels that appear in all frames; fall back to best coverage
    max_coverage = max((v["coverage"] for v in consolidated.values()), default=0)
    candidates   = [lbl for lbl, v in consolidated.items() if v["coverage"] == max_coverage]

    if not candidates:
        logger.error("No objects found across any frames — returning empty AnchorResult")
        _teardown_models(qwen_model, qwen_proc)
        return AnchorResult(label="")

    logger.info("  Candidate labels with max coverage (%d): %s", max_coverage, candidates)

    if len(candidates) > 1:
        chosen_label = _qwen_select_anchor(qwen_model, qwen_proc, candidates)
    else:
        chosen_label = candidates[0]

    logger.info("  Chosen anchor label: '%s'", chosen_label)

    anchor_info = consolidated[chosen_label]
    bboxes_pixel: dict[str, list[int]] = {}
    centroids:    dict[str, tuple]     = {}

    for stem, det in anchor_info["by_stem"].items():
        bbox = det["bbox"]
        bboxes_pixel[stem] = bbox
        cy = (bbox[0] + bbox[2]) / 2.0
        cx = (bbox[1] + bbox[3]) / 2.0
        centroids[stem] = (cx, cy)   # (u, v) pixel

    # ── Tear down Qwen before loading CLIP ────────────────────────────────────
    _teardown_models(qwen_model, qwen_proc)

    # ── Phase 4: CLIP weighting ───────────────────────────────────────────────
    logger.info("Phase 4: CLIP cosine similarity weighting …")
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(
        config.CLIP_MODEL,
        cache_dir=config.MODEL_CACHE_DIR,
    )
    clip_proc  = CLIPProcessor.from_pretrained(
        config.CLIP_MODEL,
        cache_dir=config.MODEL_CACHE_DIR,
    )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device).eval()

    weights = _compute_clip_weights(clip_model, clip_proc, frames, chosen_label, bboxes_pixel)

    _teardown_models(clip_model, clip_proc)

    result = AnchorResult(
        label=chosen_label,
        bboxes=bboxes_pixel,
        centroids=centroids,
        weights=weights,
    )

    n_above = sum(1 for w in weights.values() if w >= config.CLIP_ANCHOR_THRESHOLD)
    logger.info(
        "Anchor detection complete: label='%s'  frames=%d  above_threshold=%d",
        chosen_label, len(bboxes_pixel), n_above,
    )
    return result


def _teardown_models(*models) -> None:
    """Explicitly delete models and flush CUDA cache to free VRAM."""
    import torch

    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model VRAM released.")
