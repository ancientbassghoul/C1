"""
pipeline/undistort.py – Fisheye lens undistortion.

The drone camera shows strong barrel/fisheye distortion (see the curved
vegetation boundary in overhead shots).  We use OpenCV's fisheye model,
which fits wide-angle lenses better than the standard Brown-Conrady model.

Fisheye projection model (equidistant):
    r_d = f * θ * (1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)
where θ is the angle from the optical axis.

After undistortion every frame is a standard rectilinear (pinhole)
projection that can be used directly for ray-casting geometry.

Tuning workflow
───────────────
1.  Run:  python raycast.py --preview-undistort --frames_dir ./frames
    This shows the undistorted result for every frame.
2.  Adjust FOCAL_LENGTH, FISHEYE_K1, FISHEYE_K2 in config.py until
    straight real-world lines (tyre tracks, vegetation boundary) look
    straight in the output.
3.  Adjust UNDISTORT_SCALE (0.5–1.0) to trade off FOV vs. black borders.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from pipeline.frame import Frame
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Build camera matrices from config
# ─────────────────────────────────────────────────────────────────────────────

def build_K() -> np.ndarray:
    """3 × 3 camera intrinsic matrix estimated from config parameters."""
    f = config.FOCAL_LENGTH
    return np.array([
        [f,   0.0, config.CX],
        [0.0, f,   config.CY],
        [0.0, 0.0, 1.0       ],
    ], dtype=np.float64)


def build_D() -> np.ndarray:
    """4-element fisheye distortion vector [k1, k2, k3, k4]."""
    return np.array([
        [config.FISHEYE_K1],
        [config.FISHEYE_K2],
        [config.FISHEYE_K3],
        [config.FISHEYE_K4],
    ], dtype=np.float64)


def build_K_new(K: np.ndarray, scale: float = None) -> np.ndarray:
    """
    New camera matrix for the undistorted (rectilinear) output image.

    Scaling < 1.0 zooms out, trading pixel density for a larger visible
    FOV with no black-border cropping.  The principal point is kept at
    the image centre.
    """
    if scale is None:
        scale = config.UNDISTORT_SCALE
    K_new = K.copy()
    K_new[0, 0] *= scale   # fx
    K_new[1, 1] *= scale   # fy
    # Keep cx, cy at image centre (already set by build_K).
    return K_new


# ─────────────────────────────────────────────────────────────────────────────
# Undistortion
# ─────────────────────────────────────────────────────────────────────────────

def undistort_image(
    img: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    K_new: np.ndarray,
) -> np.ndarray:
    """
    Apply fisheye undistortion to *img* and return the corrected image.

    Uses cv2.fisheye.undistortImage which maps each output pixel through
    the inverse of the fisheye projection model.
    """
    h, w = img.shape[:2]
    # Pre-compute the undistortion + rectification map (faster for batch).
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D,
        np.eye(3),   # no rectification rotation
        K_new,
        (w, h),
        cv2.CV_16SC2,
    )
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def undistort_point(
    px: float,
    py: float,
    K: np.ndarray,
    D: np.ndarray,
    K_new: np.ndarray,
) -> tuple[float, float]:
    """
    Undistort a single 2-D pixel coordinate.

    Useful for converting a user's click in the *raw* image to the
    corresponding point in the undistorted image (and vice versa).
    """
    pts = np.array([[[px, py]]], dtype=np.float64)
    undist = cv2.fisheye.undistortPoints(pts, K, D, P=K_new)
    return float(undist[0, 0, 0]), float(undist[0, 0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def undistort_frame(
    frame: Frame,
    K: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
    K_new: Optional[np.ndarray] = None,
) -> None:
    """
    Undistort *frame.raw* and store the result in *frame.undistorted*.
    Also sets *frame.K_undist* to the new camera matrix for later use in
    ray-casting.
    """
    if K is None:     K     = build_K()
    if D is None:     D     = build_D()
    if K_new is None: K_new = build_K_new(K)

    frame.undistorted = undistort_image(frame.raw, K, D, K_new)
    frame.K_undist    = K_new.copy()
    logger.debug("[%s] undistorted OK", frame.stem)


def undistort_all(frames: list[Frame]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Undistort every frame in *frames* using a single shared K / D / K_new.

    HUD masking is applied to frame.raw BEFORE undistortion so that
    cv2.remap's bilinear interpolation never samples HUD pixels into
    scene content.  OCR and bracket-roll detection must already have run
    on frame.raw before this function is called.

    Returns (K, D, K_new) so callers can reuse them for point operations.
    """
    K     = build_K()
    D     = build_D()
    K_new = build_K_new(K)

    logger.info(
        "Undistorting %d frame(s)  [f=%.0f  k1=%.3f  scale=%.2f]",
        len(frames), config.FOCAL_LENGTH, config.FISHEYE_K1, config.UNDISTORT_SCALE,
    )
    for i, frame in enumerate(frames):
        logger.info("  Undistorting %d/%d: %s", i + 1, len(frames), frame.stem)
        _apply_hud_mask_raw(frame)       # mask raw FIRST
        undistort_frame(frame, K, D, K_new)

    return K, D, K_new


def _apply_hud_mask_raw(frame: Frame) -> None:
    """Paint HUD overlay regions solid black on frame.raw (in-place).

    HUD_REGIONS are defined in raw/distorted pixel coordinates.  Masking
    before undistortion ensures the remap interpolation never blends HUD
    content into adjacent scene pixels.
    """
    if frame.raw is None:
        return
    for (y1, y2, x1, x2) in config.HUD_REGIONS:
        frame.raw[y1:y2, x1:x2] = 0


def suppress_center_overlays(frames: list[Frame]) -> None:
    """Gaussian-blur bracket and crosshair HUD overlays in frame.undistorted.

    Called after estimate_poses (brackets known) and detect_anchor (crosshair
    known), and before MASt3R, so the feature matcher never sees HUD lines.
    """
    K, D, K_new = build_K(), build_D(), build_K_new(build_K())
    ksize = config.OVERLAY_BLUR_KERNEL
    pad   = config.OVERLAY_BLUR_PAD

    for frame in frames:
        if frame.undistorted is None:
            continue
        img  = frame.undistorted
        h, w = img.shape[:2]

        # ── Brackets ─────────────────────────────────────────────────────────
        if frame.bracket_rects_raw:
            rects_to_blur = list(frame.bracket_rects_raw)
            if len(frame.bracket_rects_raw) == 1:
                # Mirror the single detected bracket through the raw-image centre
                # so the undetected opposite bracket is also blurred.
                ry1, ry2, rx1, rx2 = frame.bracket_rects_raw[0]
                raw_h, raw_w = frame.raw.shape[:2]
                rcx = (rx1 + rx2) / 2.0
                rcy = (ry1 + ry2) / 2.0
                hw  = (rx2 - rx1) / 2.0
                hh  = (ry2 - ry1) / 2.0
                mx  = 2 * (raw_w // 2) - rcx
                my  = 2 * (raw_h // 2) - rcy
                rects_to_blur.append((
                    int(my - hh), int(my + hh),
                    int(mx - hw), int(mx + hw),
                ))
                logger.debug("[%s] bracket symmetry: added mirror rect (%.0f,%.0f)→(%.0f,%.0f)",
                             frame.stem, mx - hw, my - hh, mx + hw, my + hh)
            for (ry1, ry2, rx1, rx2) in rects_to_blur:
                # Map raw-space corners to undistorted space, take axis-aligned bbox
                corners = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]
                pts_u   = [undistort_point(x, y, K, D, K_new) for x, y in corners]
                ux = [p[0] for p in pts_u]
                uy = [p[1] for p in pts_u]
                y1c = max(0, int(min(uy)) - pad)
                y2c = min(h, int(max(uy)) + pad)
                x1c = max(0, int(min(ux)) - pad)
                x2c = min(w, int(max(ux)) + pad)
                if y2c > y1c and x2c > x1c:
                    roi = img[y1c:y2c, x1c:x2c]
                    img[y1c:y2c, x1c:x2c] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        else:
            # Fallback: blur entire center window (brackets not detected)
            cx, cy = w // 2, h // 2
            margin = config.BRACKET_SEARCH_MARGIN
            y1c = max(0, cy - margin); y2c = min(h, cy + margin)
            x1c = max(0, cx - margin); x2c = min(w, cx + margin)
            roi = img[y1c:y2c, x1c:x2c]
            img[y1c:y2c, x1c:x2c] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            logger.debug("[%s] bracket fallback: blurred full ±%dpx center window",
                         frame.stem, margin)

        # ── Crosshair (Qwen-detected) ─────────────────────────────────────────
        if frame.crosshair_bbox_px is not None:
            x1p, y1p, x2p, y2p = frame.crosshair_bbox_px  # [xmin,ymin,xmax,ymax] proc pixels
            h_proc = max(28, round(h / 28) * 28)
            w_proc = max(28, round(w / 28) * 28)
            y1c = max(0, int(y1p * h / h_proc) - pad)
            y2c = min(h, int(y2p * h / h_proc) + pad)
            x1c = max(0, int(x1p * w / w_proc) - pad)
            x2c = min(w, int(x2p * w / w_proc) + pad)
            if y2c > y1c and x2c > x1c:
                roi = img[y1c:y2c, x1c:x2c]
                img[y1c:y2c, x1c:x2c] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            logger.debug("[%s] crosshair blurred at undist px [%d:%d, %d:%d]",
                         frame.stem, y1c, y2c, x1c, x2c)

        # ── Fixed center reticle (always present, ~17×17 px "+" symbol) ───────
        # Blur unconditionally — idempotent if Qwen already covered this region.
        half = 8  # 8+1+8 = 17 px
        cy, cx = h // 2, w // 2
        y1c = max(0, cy - half - pad)
        y2c = min(h, cy + half + pad)
        x1c = max(0, cx - half - pad)
        x2c = min(w, cx + half + pad)
        roi = img[y1c:y2c, x1c:x2c]
        img[y1c:y2c, x1c:x2c] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
