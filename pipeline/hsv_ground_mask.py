"""
pipeline/hsv_ground_mask.py – HSV-based ground segmentation (experimental).

An alternative to GroundingDINO + SAM for the exclusion pass.
Uses deterministic HSV colour thresholding to identify sky and vegetation,
then combines with the geometric HUD mask and (optionally) the van bbox.

No neural network inference — runs in milliseconds per frame.
Tune HSV_SKY_RANGES and HSV_VEG_RANGES in config.py, then iterate with
    python raycast.py --frames_dir ... --preview-hsv-masks

Public API
──────────
    get_hsv_ground_masks(frames, van_detections, save_debug)
        → dict[Frame, np.ndarray]   (True = ground pixel)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from pipeline.frame import Frame
from pipeline.ground_mask import _make_hud_mask   # reuse geometric HUD mask

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hsv_union(hsv: np.ndarray, ranges: list) -> np.ndarray:
    """
    OR-combine multiple HSV inRange calls.
    Each range is ((H_lo, S_lo, V_lo), (H_hi, S_hi, V_hi)).
    """
    h, w  = hsv.shape[:2]
    mask  = np.zeros((h, w), dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv,
                            np.array(lo, dtype=np.uint8),
                            np.array(hi, dtype=np.uint8))
    return mask


def _morph_clean(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Close small holes then open small noise specks."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m,    cv2.MORPH_OPEN,  k)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame exclusion mask
# ─────────────────────────────────────────────────────────────────────────────

def compute_hsv_exclusion(
    image_bgr  : np.ndarray,
    van_bbox   : Optional[tuple] = None,   # (x1, y1, x2, y2) or None
) -> dict[str, np.ndarray]:
    """
    Compute individual exclusion layers and return them as a dict of bool masks.

    Keys: 'sky', 'veg', 'van', 'hud', 'black', 'exclusion', 'ground'
    All masks are (H, W) bool in undistorted-image space.
    """
    h, w  = image_bgr.shape[:2]
    k     = config.HSV_MORPH_KERNEL

    hsv   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # ── Sky ───────────────────────────────────────────────────────────────────
    sky_raw  = _hsv_union(hsv, config.HSV_SKY_RANGES)
    sky_mask = _morph_clean(sky_raw, k).astype(bool)

    # ── Vegetation ────────────────────────────────────────────────────────────
    veg_raw  = _hsv_union(hsv, config.HSV_VEG_RANGES)
    veg_mask = _morph_clean(veg_raw, k).astype(bool)

    # ── Van bbox ──────────────────────────────────────────────────────────────
    van_mask = np.zeros((h, w), dtype=bool)
    if van_bbox is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in van_bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        van_mask[y1:y2, x1:x2] = True

    # ── HUD (geometric, from config.HUD_REGIONS) ──────────────────────────────
    hud_mask = _make_hud_mask(h, w)

    # ── Black border (undistortion artefact) ──────────────────────────────────
    black_mask = image_bgr.max(axis=2) < 8

    # ── Combine ───────────────────────────────────────────────────────────────
    excl   = sky_mask | veg_mask | van_mask | hud_mask | black_mask
    ground = ~excl

    return {
        "sky"      : sky_mask,
        "veg"      : veg_mask,
        "van"      : van_mask,
        "hud"      : hud_mask,
        "black"    : black_mask,
        "exclusion": excl,
        "ground"   : ground,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Debug image
# ─────────────────────────────────────────────────────────────────────────────

def _save_hsv_debug(
    frame   : Frame,
    layers  : dict[str, np.ndarray],
) -> None:
    """
    Save a multi-panel debug PNG to {OUTPUT_DIR}/debug/hsv_masks/<stem>_hsv.png.

    Panels (left → right):
        SKY (blue) | VEG (green) | VAN+HUD (yellow) | EXCLUSION (red) | GROUND (teal)
    """
    out_dir = Path(config.OUTPUT_DIR) / "debug" / "hsv_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = (frame.undistorted.copy()
            if frame.undistorted is not None
            else np.zeros((config.IMAGE_H, config.IMAGE_W, 3), np.uint8))

    def _overlay(mask, colour):
        out = base.copy().astype(np.float32)
        out[mask] = out[mask] * 0.40 + np.array(colour, np.float32) * 0.60
        return out.astype(np.uint8)

    def _label(img, text):
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (220, 220, 220), 1, cv2.LINE_AA)

    sky    = layers["sky"]
    veg    = layers["veg"]
    van    = layers["van"]
    hud    = layers["hud"]
    excl   = layers["exclusion"]
    ground = layers["ground"]

    p_sky    = _overlay(sky,            ( 30,  30, 220))  # red tint  (BGR)
    p_veg    = _overlay(veg,            ( 30, 180,  30))  # green
    p_vanh   = _overlay(van | hud,      (  0, 200, 200))  # yellow
    p_excl   = _overlay(excl,           ( 60,  60, 200))  # orange-red
    p_ground = _overlay(ground,         (180, 180,  30))  # teal

    pct = lambda m: m.mean() * 100

    _label(p_sky,    f"SKY      ({pct(sky):.1f}%)")
    _label(p_veg,    f"VEG      ({pct(veg):.1f}%)")
    _label(p_vanh,   f"VAN+HUD  ({pct(van | hud):.1f}%)")
    _label(p_excl,   f"EXCLUSION ({pct(excl):.1f}%)")
    _label(p_ground, f"GROUND   ({pct(ground):.1f}%)")

    canvas   = np.hstack([p_sky, p_veg, p_vanh, p_excl, p_ground])
    out_path = out_dir / f"{frame.stem}_hsv.png"
    cv2.imwrite(str(out_path), canvas)
    logger.info("  HSV debug: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_hsv_ground_masks(
    frames         : list[Frame],
    van_detections : dict = None,   # Frame → (x1,y1,x2,y2); pass {} to skip
    save_debug     : bool = False,
) -> dict[Frame, np.ndarray]:
    """
    Compute HSV-based ground masks for all frames.

    Returns {frame: bool mask (H, W)} — True = ground pixel.

    Parameters
    ----------
    frames          : undistorted Frame objects
    van_detections  : optional dict from VanDetector.detect_all(); pass None
                      to skip van exclusion (useful during preview tuning)
    save_debug      : if True, saves 5-panel PNG per frame to
                      {OUTPUT_DIR}/debug/hsv_masks/
    """
    if van_detections is None:
        van_detections = {}

    masks: dict[Frame, np.ndarray] = {}

    for i, frame in enumerate(frames):
        if frame.undistorted is None:
            logger.warning("[%s] No undistorted image — skipping", frame.stem[-12:])
            masks[frame] = np.ones((config.IMAGE_H, config.IMAGE_W), dtype=bool)
            continue

        logger.info("HSV ground mask %d/%d: %s", i + 1, len(frames), frame.stem[-12:])
        bbox   = van_detections.get(frame)
        layers = compute_hsv_exclusion(frame.undistorted, van_bbox=bbox)

        masks[frame] = layers["ground"]
        logger.info(
            "  → sky=%.1f%%  veg=%.1f%%  van=%.1f%%  hud=%.1f%%  ground=%.1f%%",
            layers["sky"].mean()   * 100,
            layers["veg"].mean()   * 100,
            layers["van"].mean()   * 100,
            layers["hud"].mean()   * 100,
            layers["ground"].mean()* 100,
        )

        if save_debug:
            _save_hsv_debug(frame, layers)

    return masks
