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
        undistort_frame(frame, K, D, K_new)
        _apply_hud_mask(frame)

    return K, D, K_new


def _apply_hud_mask(frame: Frame) -> None:
    """Paint HUD overlay regions solid black on frame.undistorted (in-place).

    MASt3R's transformer assigns zero confidence to zero-entropy pixels,
    automatically excluding HUD text/graphics from feature matching.
    """
    if frame.undistorted is None:
        return
    for (y1, y2, x1, x2) in config.HUD_REGIONS:
        frame.undistorted[y1:y2, x1:x2] = 0
