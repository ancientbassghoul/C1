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
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_anchor_result(result: AnchorResult, path: str) -> None:
    """Serialize AnchorResult to JSON for --use-saved-qwen reuse."""
    import json
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {
        "label": result.label,
        "bboxes": result.bboxes,
        "centroids": {k: list(v) for k, v in result.centroids.items()},
        "weights": result.weights,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_anchor_result(path: str) -> AnchorResult:
    """Deserialize AnchorResult from JSON saved by save_anchor_result()."""
    import json
    with open(path) as f:
        data = json.load(f)
    return AnchorResult(
        label=data["label"],
        bboxes=data["bboxes"],
        centroids={k: tuple(v) for k, v in data["centroids"].items()},
        weights=data["weights"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Per-frame Qwen VL discovery
# ─────────────────────────────────────────────────────────────────────────────

_CROSSHAIR_PROMPT = (
    "This is a drone FPV video frame with a HUD overlay. "
    "Find the white crosshair or aiming reticle (a + or ⊕ symbol) overlaid on the image. "
    'Return ONLY a single JSON object: {"bbox": [xmin, ymin, xmax, ymax]} '
    "where values are absolute pixel coordinates (x=column, y=row, origin top-left). "
    'If no crosshair is visible, return {"bbox": null}.'
)

_DISCOVERY_PROMPT = (
    "List every discrete, bounded physical object you can see in this image. "
    "Only label man-made objects and structures: vehicles, buildings, towers, poles, "
    "fences, and other distinct man-made entities. "
    "Do NOT label scene-background areas — do not label sky, ground, field, road, crops, "
    "soil, terrain, farmland, dirt, grass, trees, vegetation, or any other natural or "
    "amorphous scene region. "
    "For each object provide a bounding box and a label. "
    "Use the most generalized category term possible — e.g. 'vehicle', 'building'. "
    "Only specialize if MULTIPLE instances of the same category appear "
    "(e.g. 'vehicle-truck' vs 'vehicle-private', then by color if still ambiguous). "
    "Return ONLY a JSON array, no other text. "
    'Format: [{"label": "...", "bbox": [xmin, ymin, xmax, ymax]}, ...]  '
    "where bbox values are absolute pixel coordinates (x=column, y=row, origin top-left)."
)


def _qwen_find_crosshair(model, processor, frame_bgr: np.ndarray):
    """Locate the HUD crosshair in one frame via a short targeted Qwen query.

    Returns (xmin, ymin, xmax, ymax) in processed-image pixel coords, or None.
    """
    import torch
    import cv2
    from PIL import Image

    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text",  "text": _CROSSHAIR_PROMPT},
    ]}]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=64)

    generated = out_ids[0][inputs["input_ids"].shape[1]:]
    response  = processor.decode(generated, skip_special_tokens=True).strip()

    try:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        data  = json.loads(clean)
        bbox  = data.get("bbox") if isinstance(data, dict) else None
        if bbox and len(bbox) == 4:
            return tuple(int(v) for v in bbox)
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.debug("Crosshair parse failed; raw: %s", response[:100])
    return None


def _bbox_iou_norm(a, b) -> float:
    """IoU of two [x1,y1,x2,y2] bboxes in any consistent coordinate space."""
    iy1 = max(a[0], b[0]); ix1 = max(a[1], b[1])
    iy2 = min(a[2], b[2]); ix2 = min(a[3], b[3])
    inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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
        out_ids = model.generate(**inputs, max_new_tokens=1024)

    generated = out_ids[0][inputs["input_ids"].shape[1]:]
    response  = processor.decode(generated, skip_special_tokens=True).strip()

    # Extract JSON array from response
    objects = []
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        objects = json.loads(clean)
        if not isinstance(objects, list):
            objects = []
    except json.JSONDecodeError:
        # Attempt partial recovery for truncated JSON arrays
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        last_brace = clean.rfind("}")
        if last_brace != -1:
            try:
                open_br = clean.index("[")
                objects = json.loads(clean[open_br : last_brace + 1] + "]")
                logger.warning(
                    "Qwen JSON truncated; recovered %d object(s) via partial parse",
                    len(objects),
                )
            except (json.JSONDecodeError, ValueError):
                pass
        if not isinstance(objects, list) or not objects:
            logger.warning("Qwen JSON parse failed; raw response: %s", response[:200])
            objects = []

    return objects


def _proc_dims(img_h: int, img_w: int, factor: int = 28) -> tuple[int, int]:
    """Height/width of Qwen's internally smart_resize'd image (nearest multiple of `factor`)."""
    h_proc = max(factor, round(img_h / factor) * factor)
    w_proc = max(factor, round(img_w / factor) * factor)
    return h_proc, w_proc


def _scale_bbox_to_pixels(
    bbox_proc: list[int],
    img_h: int,
    img_w: int,
) -> list[int]:
    """Map Qwen's [xmin, ymin, xmax, ymax] pixel bbox to original image pixel coordinates.

    Qwen2.5-VL returns absolute pixel coordinates in the space of its internally
    smart_resize'd image (rounds H and W to multiples of 28; e.g. 1280×720 →
    1288×728). Scale back to original dimensions.  Returns [py1, px1, py2, px2]
    for downstream consistency with the rest of the pipeline.
    """
    h_proc, w_proc = _proc_dims(img_h, img_w)
    x1, y1, x2, y2 = bbox_proc
    px1 = max(0, min(int(round(x1 * img_w / w_proc)), img_w - 1))
    py1 = max(0, min(int(round(y1 * img_h / h_proc)), img_h - 1))
    px2 = max(0, min(int(round(x2 * img_w / w_proc)), img_w - 1))
    py2 = max(0, min(int(round(y2 * img_h / h_proc)), img_h - 1))
    return [py1, px1, py2, px2]


# ─────────────────────────────────────────────────────────────────────────────
# Background label deny-list
# ─────────────────────────────────────────────────────────────────────────────

_BACKGROUND_LABELS = {
    "field", "sky", "ground", "road", "crop", "crops", "soil", "terrain",
    "dirt", "grass", "pavement", "asphalt", "earth", "land", "farmland",
    "water", "sand", "mud", "gravel", "vegetation", "landscape", "horizon",
    "path", "track", "lane",
}


def _is_background_label(label: str) -> bool:
    """Return True if label matches a known scene-background category."""
    words = set(re.split(r"[-_ ]+", label.lower()))
    return bool(words & _BACKGROUND_LABELS)


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
    "Strongly prefer vehicles, buildings, and man-made structures. "
    "Exclude featureless surfaces such as sky, ground, road, or crops — those are not 3D landmarks. "
    "Also exclude repetitive natural features such as tree rows, hedgerows, or orchards — "
    "only choose a tree if it is a single, isolated, highly distinctive landmark "
    "and no man-made object is available. "
    "Reply with ONLY the exact label string from the list, nothing else."
)

_TARGETED_PROMPT_TMPL = (
    "I need to find a {label} in this aerial drone image. "
    "If you can see a {label}, return its bounding box as a JSON object: "
    '{{"bbox": [xmin, ymin, xmax, ymax]}} where values are absolute pixel coordinates '
    "(x=column, y=row, origin top-left). "
    'If no {label} is visible, return {{"bbox": null}}.'
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


def _qwen_find_object(model, processor, frame_bgr: np.ndarray, label: str):
    """Targeted query: find a specific object in one frame.

    Returns (xmin, ymin, xmax, ymax) in processed-image pixel coords, or None.
    Used as a second-pass to fill in frames where the anchor wasn't found
    during open-world discovery.
    """
    import torch
    import cv2
    from PIL import Image

    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    prompt  = _TARGETED_PROMPT_TMPL.format(label=label)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text",  "text": prompt},
    ]}]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=64)

    generated = out_ids[0][inputs["input_ids"].shape[1]:]
    response  = processor.decode(generated, skip_special_tokens=True).strip()

    try:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        data  = json.loads(clean)
        bbox  = data.get("bbox") if isinstance(data, dict) else None
        if bbox and len(bbox) == 4 and bbox[0] is not None:
            return tuple(int(v) for v in bbox)
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.debug("Targeted find parse failed for '%s'; raw: %s", label, response[:100])
    return None


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
        if not isinstance(text_feats, torch.Tensor):
            # transformers >= 4.50 returns an output object; pooler_output is the
            # already-projected CLIP embedding — use it directly, no extra projection.
            text_feats = text_feats.pooler_output
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
            if not isinstance(img_feats, torch.Tensor):
                img_feats = img_feats.pooler_output
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

        # Step A: find crosshair first so we can exclude it from anchor candidates
        logger.info("  Finding crosshair in %s …", frame.stem[-12:])
        ch_bbox = _qwen_find_crosshair(qwen_model, qwen_proc, frame.undistorted)
        frame.crosshair_bbox_px = ch_bbox
        if ch_bbox is not None:
            logger.info("    → crosshair at %s (proc-image pixels)", ch_bbox)
        else:
            logger.info("    → no crosshair detected")

        # Step B: standard per-frame object discovery
        logger.info("  Querying Qwen for frame %s …", frame.stem[-12:])
        raw_objects = _qwen_discover_frame(qwen_model, qwen_proc, frame.undistorted)
        logger.info("    → raw detections: %s",
                    [(o.get("label"), o.get("bbox")) for o in raw_objects])

        # Filter out any object whose bbox overlaps the detected crosshair.
        # Both ch_bbox and discovery bboxes are in the same processed-pixel space,
        # so IoU is computed correctly regardless of axis labelling.
        if ch_bbox is not None:
            keep, drop = [], []
            for obj in raw_objects:
                iou = _bbox_iou_norm(obj.get("bbox", [0, 0, 0, 0]), list(ch_bbox))
                (keep if iou <= config.CROSSHAIR_IOU_EXCLUDE else drop).append((obj, iou))
            if drop:
                for obj, iou in drop:
                    logger.info("    → EXCLUDED '%s' bbox=%s  IoU=%.3f > threshold %.2f",
                                obj.get("label"), obj.get("bbox"), iou,
                                config.CROSSHAIR_IOU_EXCLUDE)
            raw_objects = [obj for obj, _ in keep]

        h, w = frame.undistorted.shape[:2]
        scaled = []
        for obj in raw_objects:
            bbox_norm = obj.get("bbox", [])
            if len(bbox_norm) != 4:
                continue
            lbl     = obj.get("label", "unknown")
            bbox_px = _scale_bbox_to_pixels(bbox_norm, h, w)
            area_frac = (
                (bbox_px[2] - bbox_px[0]) * (bbox_px[3] - bbox_px[1])
            ) / (h * w)
            logger.info("    → '%s'  raw=%s → px=%s  area=%.1f%%",
                        lbl, bbox_norm, bbox_px, area_frac * 100)
            if area_frac > config.MAX_ANCHOR_BBOX_FRACTION:
                logger.info(
                    "    Skipping '%s' — bbox covers %.0f%% of image (too large)",
                    lbl, area_frac * 100,
                )
                continue
            scaled.append({"label": lbl, "bbox": bbox_px})

        per_frame_objects[frame.stem] = scaled
        logger.info("    → %d object(s) accepted", len(scaled))

    # ── Phase 2: Label consolidation ─────────────────────────────────────────
    logger.info("Phase 2: label consolidation …")
    consolidated = _consolidate_labels(per_frame_objects)

    # Filter out known scene-background labels before coverage ranking
    bg_filtered = [lbl for lbl in consolidated if _is_background_label(lbl)]
    if bg_filtered:
        logger.info("  Filtering background labels: %s", bg_filtered)
        consolidated = {lbl: v for lbl, v in consolidated.items()
                        if not _is_background_label(lbl)}

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

    # ── Second-pass: targeted query for frames that missed the anchor ─────────
    missed_frames = [f for f in frames
                     if f.stem not in bboxes_pixel and f.undistorted is not None]
    if missed_frames:
        logger.info(
            "Phase 3b: targeted second-pass for %d frame(s) missing '%s' …",
            len(missed_frames), chosen_label,
        )
        for frame in missed_frames:
            h, w = frame.undistorted.shape[:2]
            bbox_norm = _qwen_find_object(qwen_model, qwen_proc, frame.undistorted, chosen_label)
            logger.info("    [%s] second-pass raw bbox: %s", frame.stem[-12:], bbox_norm)
            if bbox_norm is not None:
                bbox_px = _scale_bbox_to_pixels(list(bbox_norm), h, w)
                area_frac = (
                    (bbox_px[2] - bbox_px[0]) * (bbox_px[3] - bbox_px[1])
                ) / (h * w)
                if area_frac <= config.MAX_ANCHOR_BBOX_FRACTION:
                    bboxes_pixel[frame.stem] = bbox_px
                    cy = (bbox_px[0] + bbox_px[2]) / 2.0
                    cx = (bbox_px[1] + bbox_px[3]) / 2.0
                    centroids[frame.stem] = (cx, cy)
                    logger.info("    [%s] found '%s' on second pass", frame.stem[-12:], chosen_label)
                else:
                    logger.debug(
                        "    [%s] second-pass bbox too large (%.0f%%) — skipped",
                        frame.stem[-12:], area_frac * 100,
                    )
            else:
                logger.info("    [%s] '%s' not found on second pass", frame.stem[-12:], chosen_label)

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
    save_anchor_result(result, config.ANCHOR_CACHE_FILE)
    logger.info("Saved Qwen/CLIP anchor cache → %s", config.ANCHOR_CACHE_FILE)
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
