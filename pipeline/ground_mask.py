"""
pipeline/ground_mask.py – Per-frame ground segmentation via GroundedSAM.

Strategy
────────
HUD MASK  (pure geometry, no ML)
    Each rectangle in config.HUD_REGIONS is defined in raw-frame pixel
    coordinates.  Its four corners are pushed through cv2.fisheye.undistortPoints
    — the same transform used by undistort.py — to produce a quadrilateral in
    undistorted-image space, which is then filled.  The result is identical for
    every frame (same camera model) so it is computed once and cached.
    No GroundingDINO or SAM involved.  Zero variance, pixel-accurate.

POSITIVE PASS  (GroundingDINO + SAM, on undistorted frame)
    Detects pixels that ARE ground (soil, dirt, track…).
    Skipped gracefully when GROUND_INCLUDE_PROMPT = "".

EXCLUSION PASS  (GroundingDINO + SAM, on undistorted frame)
    Detects pixels that are NOT ground (sky, vegetation, van…).
    Skipped gracefully when GROUND_EXCLUDE_PROMPT = "".

COMBINE
    ground = positive ∩ ¬(exclusion ∪ hud) ∩ ¬black_border
    Fallback when positive is empty: ¬(exclusion ∪ hud) ∩ ¬black_border

Public API
──────────
    get_ground_masks(frames, save_debug) → dict[Frame, np.ndarray]
    mask_to_row_bounds(mask)             → (row_min, row_max)

Models are lazy-loaded and cached for the process lifetime.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

import config
from pipeline.frame import Frame

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model state  (GroundingDINO + SAM for ground/sky/veg passes)
# ─────────────────────────────────────────────────────────────────────────────
_sam_predictor : Optional[object] = None
_gdino_model   : Optional[object] = None


def _load_models() -> None:
    global _sam_predictor, _gdino_model
    if _sam_predictor is not None:
        return

    repo = Path(config.GROUNDED_SAM_DIR)

    # ── SAM ───────────────────────────────────────────────────────────────────
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError(
            "segment_anything not importable.  "
            "Make sure Grounded-Segment-Anything is installed in this venv."
        )

    sam_ckpt = repo / "sam_vit_h_4b8939.pth"
    if not sam_ckpt.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam    = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt))
    sam.to(device)
    _sam_predictor = SamPredictor(sam)
    logger.info("Ground mask: SAM vit_h loaded on %s", device)

    # ── GroundingDINO ─────────────────────────────────────────────────────────
    try:
        import groundingdino
        from groundingdino.util.inference import load_model
    except ImportError:
        raise ImportError("groundingdino not importable.  See README Setup.")

    gdino_cfg  = os.path.join(os.path.dirname(groundingdino.__file__),
                               "config", "GroundingDINO_SwinT_OGC.py")
    gdino_ckpt = "groundingdino_swint_ogc.pth"

    if not os.path.exists(gdino_ckpt):
        logger.info("Downloading GroundingDINO weights (~700 MB)…")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/IDEA-Research/GroundingDINO/"
            "releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            gdino_ckpt,
        )

    _gdino_model = load_model(gdino_cfg, gdino_ckpt)
    logger.info("Ground mask: GroundingDINO loaded")


# ─────────────────────────────────────────────────────────────────────────────
# HUD mask  (pure geometry — no ML, computed once and cached)
# ─────────────────────────────────────────────────────────────────────────────
_hud_mask_cache: Optional[np.ndarray] = None


def _make_hud_mask(h: int, w: int) -> np.ndarray:
    """
    Build a boolean (H, W) HUD exclusion mask from config.HUD_REGIONS.

    Each entry in HUD_REGIONS is (y1, y2, x1, x2) in raw-frame pixels.
    The four corners are undistorted via cv2.fisheye.undistortPoints (same
    camera model as undistort.py) and filled as a convex polygon.

    All frames share one camera, so the mask is identical — computed once
    and cached for the process lifetime.
    """
    global _hud_mask_cache
    if _hud_mask_cache is not None:
        return _hud_mask_cache

    from pipeline.undistort import build_K, build_D, build_K_new
    K     = build_K()
    D     = build_D()
    K_new = build_K_new(K)

    canvas = np.zeros((h, w), dtype=np.uint8)

    for (y1, y2, x1, x2) in config.HUD_REGIONS:
        corners = np.array([
            [[float(x1), float(y1)]],
            [[float(x2), float(y1)]],
            [[float(x2), float(y2)]],
            [[float(x1), float(y2)]],
        ], dtype=np.float64)

        undist = cv2.fisheye.undistortPoints(corners, K, D, P=K_new)
        poly   = undist.reshape(-1, 1, 2).astype(np.int32)
        cv2.fillConvexPoly(canvas, poly, 1)

    _hud_mask_cache = canvas.astype(bool)
    logger.info("HUD mask: %d region(s), %.1f%% of undistorted frame",
                len(config.HUD_REGIONS), _hud_mask_cache.mean() * 100)
    return _hud_mask_cache


# ─────────────────────────────────────────────────────────────────────────────
# GroundingDINO + SAM helper
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_for_gdino(image_bgr: np.ndarray):
    """Return (image_rgb_np, image_tensor) ready for GroundingDINO."""
    import groundingdino.datasets.transforms as T
    from PIL import Image as PILImage

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil       = PILImage.fromarray(image_rgb)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(pil, None)
    return image_rgb, image_tensor


def _segment_prompt(
    image_bgr   : np.ndarray,
    image_rgb   : np.ndarray,
    image_tensor,
    prompt      : str,
    box_thresh  : float,
    text_thresh : float,
) -> np.ndarray:
    """
    Run GroundingDINO + SAM for *prompt*.  Returns a boolean (H, W) mask.
    Returns all-False immediately if prompt is empty — no model call made.
    """
    h, w = image_bgr.shape[:2]

    if not prompt.strip():
        return np.zeros((h, w), dtype=bool)

    from groundingdino.util.inference import predict

    mask = np.zeros((h, w), dtype=bool)

    with torch.no_grad():
        boxes_cx, scores, _ = predict(
            model          = _gdino_model,
            image          = image_tensor,
            caption        = prompt,
            box_threshold  = box_thresh,
            text_threshold = text_thresh,
        )

    if len(boxes_cx) == 0:
        return mask

    boxes_xyxy = boxes_cx.clone()
    boxes_xyxy[:, 0] = (boxes_cx[:, 0] - boxes_cx[:, 2] / 2) * w
    boxes_xyxy[:, 1] = (boxes_cx[:, 1] - boxes_cx[:, 3] / 2) * h
    boxes_xyxy[:, 2] = (boxes_cx[:, 0] + boxes_cx[:, 2] / 2) * w
    boxes_xyxy[:, 3] = (boxes_cx[:, 1] + boxes_cx[:, 3] / 2) * h
    boxes_xyxy = boxes_xyxy.cpu().numpy()

    _sam_predictor.set_image(image_rgb)

    for box in boxes_xyxy:
        sam_masks, sam_scores, _ = _sam_predictor.predict(
            box=box, multimask_output=True,
        )
        best = sam_masks[int(np.argmax(sam_scores))].astype(bool)
        mask |= best

    logger.debug("    '%s…': %d detections → %.1f%% of frame",
                 prompt[:35], len(boxes_cx), mask.mean() * 100)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Debug image helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_hud_mask_preview(frames: list[Frame]) -> None:
    """
    Save a side-by-side debug PNG per frame showing the geometric HUD mask.
    Output: {OUTPUT_DIR}/debug/hud_masks/<stem>_hud.png
    Requires frames to already be undistorted (uses frame.undistorted as base).
    """
    import pathlib
    out_dir = pathlib.Path(config.OUTPUT_DIR) / "debug" / "hud_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        img = (frame.undistorted.copy()
               if frame.undistorted is not None
               else np.zeros((config.IMAGE_H, config.IMAGE_W, 3), np.uint8))
        h, w = img.shape[:2]
        hud  = _make_hud_mask(h, w)

        overlay = img.copy().astype(np.float32)
        overlay[hud] = overlay[hud] * 0.35 + np.array([255, 0, 200], np.float32) * 0.65
        overlay = overlay.astype(np.uint8)

        def _label(i, txt):
            cv2.rectangle(i, (0, 0), (i.shape[1], 22), (0, 0, 0), cv2.FILLED)
            cv2.putText(i, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (220, 220, 220), 1, cv2.LINE_AA)

        _label(img,     f"UNDISTORTED  {frame.stem[-16:]}")
        _label(overlay, f"HUD MASK ({hud.mean()*100:.1f}%)  "
                        f"{len(config.HUD_REGIONS)} region(s) from config.HUD_REGIONS")

        canvas   = np.hstack([img, overlay])
        out_path = out_dir / f"{frame.stem}_hud.png"
        cv2.imwrite(str(out_path), canvas)
        logger.info("HUD preview: %s", out_path)


def _save_mask_debug(
    frame     : Frame,
    pos_mask  : np.ndarray,
    excl_mask : np.ndarray,
    hud_mask  : np.ndarray,
    final_mask: np.ndarray,
) -> None:
    """
    Save a four-panel debug PNG to {OUTPUT_DIR}/debug/masks/<stem>_masks.png:
        BLUE    – positive mask  (what IS ground)
        RED     – exclusion mask (sky / veg / van)
        MAGENTA – HUD mask       (geometric, from config.HUD_REGIONS)
        GREEN   – final mask     (what feature matching will use)
    """
    import pathlib
    out_dir = pathlib.Path(config.OUTPUT_DIR) / "debug" / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = pos_mask.shape
    base = (frame.undistorted.copy()
            if frame.undistorted is not None
            else np.zeros((h, w, 3), np.uint8))

    def _overlay(img, mask, colour):
        out = img.copy().astype(np.float32)
        out[mask] = out[mask] * 0.45 + np.array(colour, np.float32) * 0.55
        return out.astype(np.uint8)

    def _label(img, text):
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (220, 220, 220), 1, cv2.LINE_AA)

    p_pos  = _overlay(base, pos_mask,   (255,  80,  80))
    p_excl = _overlay(base, excl_mask,  ( 80,  80, 255))
    p_hud  = _overlay(base, hud_mask,   (200,   0, 200))
    p_fin  = _overlay(base, final_mask, ( 80, 220,  80))

    _label(p_pos,  f"POSITIVE  ({pos_mask.mean()*100:.1f}%)  "
                   f"{config.GROUND_INCLUDE_PROMPT[:45] or '(disabled)'}")
    _label(p_excl, f"EXCLUSION ({excl_mask.mean()*100:.1f}%)  "
                   f"{config.GROUND_EXCLUDE_PROMPT[:45] or '(disabled)'}")
    _label(p_hud,  f"HUD MASK  ({hud_mask.mean()*100:.1f}%)  "
                   f"{len(config.HUD_REGIONS)} region(s)")
    _label(p_fin,  f"FINAL     ({final_mask.mean()*100:.1f}%)")

    canvas   = np.hstack([p_pos, p_excl, p_hud, p_fin])
    out_path = out_dir / f"{frame.stem}_masks.png"
    cv2.imwrite(str(out_path), canvas)
    logger.info("  Mask debug image: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_ground_masks(
    frames    : list[Frame],
    save_debug: bool = False,
) -> dict[Frame, np.ndarray]:
    """
    Compute and return ground masks for all frames.

    Each mask is boolean (H, W) in undistorted-image space:
        True  = ground pixel  → eligible for feature matching
        False = non-ground    → excluded

    HUD mask is computed geometrically from config.HUD_REGIONS — no ML needed
    for that step.  GroundingDINO + SAM are only called for the positive and
    exclusion passes; either pass is skipped when its prompt is empty ("").

    If save_debug=True, saves a four-panel PNG per frame to
    {OUTPUT_DIR}/debug/masks/.
    """
    needs_models = (config.GROUND_INCLUDE_PROMPT.strip() or
                    config.GROUND_EXCLUDE_PROMPT.strip())
    if needs_models:
        _load_models()

    masks: dict[Frame, np.ndarray] = {}

    for i, frame in enumerate(frames):
        if frame.undistorted is None:
            logger.warning("[%s] No undistorted image — skipping", frame.stem[-12:])
            masks[frame] = np.ones((config.IMAGE_H, config.IMAGE_W), dtype=bool)
            continue

        logger.info("Ground mask %d/%d: %s", i + 1, len(frames), frame.stem[-12:])
        image_bgr = frame.undistorted
        h, w      = image_bgr.shape[:2]

        # ── Geometric HUD mask (always computed, no ML) ───────────────────────
        hud_mask = _make_hud_mask(h, w)

        # ── GroundingDINO + SAM passes (skipped when prompt is empty) ─────────
        if needs_models:
            image_rgb, image_tensor = _preprocess_for_gdino(image_bgr)
        else:
            image_rgb = image_tensor = None   # unused

        pos_mask = _segment_prompt(
            image_bgr, image_rgb, image_tensor,
            config.GROUND_INCLUDE_PROMPT,
            config.GROUND_INCLUDE_BOX_THRESHOLD,
            config.GROUND_INCLUDE_TEXT_THRESHOLD,
        ) if needs_models or True else np.zeros((h, w), dtype=bool)

        excl_mask = _segment_prompt(
            image_bgr, image_rgb, image_tensor,
            config.GROUND_EXCLUDE_PROMPT,
            config.GROUND_EXCLUDE_BOX_THRESHOLD,
            config.GROUND_EXCLUDE_TEXT_THRESHOLD,
        ) if needs_models or True else np.zeros((h, w), dtype=bool)

        # ── Black border (undistortion artefact) ──────────────────────────────
        black_mask = image_bgr.max(axis=2) < 8

        # ── Combine ───────────────────────────────────────────────────────────
        combined_excl = excl_mask | hud_mask

        if pos_mask.any():
            final = pos_mask & ~combined_excl & ~black_mask
        else:
            final = ~combined_excl & ~black_mask
            if config.GROUND_INCLUDE_PROMPT.strip():
                logger.info("  Positive pass empty — exclusion-only fallback")

        masks[frame] = final
        logger.info("  → pos=%.1f%%  excl=%.1f%%  hud=%.1f%%  black=%.1f%%  final=%.1f%%",
                    pos_mask.mean()  * 100,
                    excl_mask.mean() * 100,
                    hud_mask.mean()  * 100,
                    black_mask.mean()* 100,
                    final.mean()     * 100)

        if save_debug:
            _save_mask_debug(frame, pos_mask, excl_mask, hud_mask, final)

    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_row_bounds(mask: np.ndarray) -> tuple[int, int]:
    """
    Return (row_min, row_max) spanning the True region of *mask*.
    Used to define the grid extent for _grid_subsample.
    """
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return 0, mask.shape[0] - 1
    return int(rows[0]), int(rows[-1])
