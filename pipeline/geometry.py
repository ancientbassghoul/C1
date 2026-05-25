"""
pipeline/geometry.py – Core ray-casting mathematics.

This module implements the three-step re-projection pipeline:

  1. unproject_pixel   – pixel → 3-D ray in world (ENU) frame
  2. intersect_ground  – ray → world point on the Z=0 ground plane
  3. project_to_frame  – world point → pixel in another frame

All operations work on the UNDISTORTED image and the K_undist intrinsics.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Pixel → World ray
# ─────────────────────────────────────────────────────────────────────────────

def unproject_pixel(
    px: float,
    py: float,
    K: np.ndarray,
    R: np.ndarray,
    position_enu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2-D pixel in an undistorted image into a 3-D ray in world frame.

    Parameters
    ──────────
    px, py        : pixel coordinates (column, row) in the undistorted image.
    K             : 3×3 undistorted camera intrinsic matrix.
    R             : 3×3 rotation  R_cam_from_world for this frame.
    position_enu  : (3,) camera position in ENU world frame (metres).

    Returns
    ───────
    origin    : (3,) ray origin  = camera position in world frame.
    direction : (3,) unit ray direction in world frame.

    Math
    ────
    Normalised camera coordinates:  p_n = K_inv @ [px, py, 1]ᵀ
    Direction in camera frame:       d_cam = p_n  (already a direction)
    Direction in world frame:        d_world = R.T @ d_cam
                                             = R_world_from_cam @ d_cam
    """
    K_inv = np.linalg.inv(K)
    p_hom = np.array([px, py, 1.0], dtype=np.float64)

    # Direction in camera frame
    d_cam = K_inv @ p_hom

    # Rotate to world frame  (R.T = R_world_from_cam because R is orthonormal)
    d_world = R.T @ d_cam
    d_world /= np.linalg.norm(d_world)

    return position_enu.copy(), d_world


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Ray → Ground-plane intersection
# ─────────────────────────────────────────────────────────────────────────────

def intersect_ground_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    z: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Intersect a 3-D ray with the horizontal ground plane  Z = *z*.

    Parameters
    ──────────
    origin    : (3,) ray start point in world frame.
    direction : (3,) unit ray direction in world frame.
    z         : height of the ground plane in ENU metres (config.GROUND_Z_M).

    Returns
    ───────
    (3,) world point on the ground plane, or None if:
      - the ray is parallel to the plane (|direction.Z| < ε), or
      - the intersection is *behind* the camera (t < 0).

    Math
    ────
    Parametric ray:    P(t) = origin + t · direction
    Ground plane:      P.Z  = z
    Solving:           t    = (z – origin.Z) / direction.Z
    """
    dz = direction[2]
    if abs(dz) < 1e-6:
        logger.debug("Ray parallel to ground plane – no intersection.")
        return None

    t = (z - origin[2]) / dz
    if t < 0:
        logger.debug("Ground intersection behind camera (t=%.3f).", t)
        return None

    return origin + t * direction


# ─────────────────────────────────────────────────────────────────────────────
# 3.  World point → Pixel in target frame
# ─────────────────────────────────────────────────────────────────────────────

def project_to_frame(
    world_pt: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    position_enu: np.ndarray,
    image_shape: tuple[int, int],
) -> Optional[tuple[float, float]]:
    """
    Project a 3-D world point into a camera's image plane.

    Parameters
    ──────────
    world_pt      : (3,) point in ENU world frame.
    K             : 3×3 undistorted intrinsic matrix of the target frame.
    R             : 3×3 R_cam_from_world for the target frame.
    position_enu  : (3,) camera position in ENU for the target frame.
    image_shape   : (height, width) of the undistorted image.

    Returns
    ───────
    (px, py) pixel coordinates, or None if the point is behind the camera
    or projects outside the image bounds.

    Math
    ────
    p_cam = R @ (world_pt – position_enu)   → camera-frame 3-D point
    Perspective divide:  p_n = p_cam[:2] / p_cam[2]
    Pixel coords:        [px, py, 1]ᵀ = K @ [p_n[0], p_n[1], 1]ᵀ
    """
    p_cam = R @ (world_pt - position_enu)

    # Point must be in front of the camera
    if p_cam[2] <= 0:
        return None

    # Perspective projection
    p_norm = p_cam[:2] / p_cam[2]

    # Apply intrinsics
    px = K[0, 0] * p_norm[0] + K[0, 2]
    py = K[1, 1] * p_norm[1] + K[1, 2]

    # Bounds check
    h, w = image_shape
    if not (0 <= px < w and 0 <= py < h):
        return None

    return float(px), float(py)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full pipeline for one pick
# ─────────────────────────────────────────────────────────────────────────────

def reproject_pick(
    px: float,
    py: float,
    source_frame,            # pipeline.frame.Frame
    target_frames: list,     # list[pipeline.frame.Frame]
) -> dict:
    """
    Full re-projection pipeline for a single pixel pick.

    Given a pixel (px, py) in *source_frame*'s undistorted image, compute
    the corresponding pixel in every *target_frame* that has a valid pose.

    Returns a dict mapping Frame → (px, py) for every successful projection.
    """
    if not source_frame.ready:
        logger.error("[%s] Source frame not ready for ray-casting.", source_frame.stem)
        return {}

    # Step 1 – Unproject to world ray
    origin, direction = unproject_pixel(
        px, py,
        source_frame.K_undist,
        source_frame.R,
        source_frame.position_enu,
    )
    logger.info(
        "Ray from [%s] pixel (%.1f, %.1f): origin=[%.2f, %.2f, %.2f]  "
        "dir=[%.4f, %.4f, %.4f]",
        source_frame.stem, px, py,
        *origin, *direction,
    )

    # Step 2 – Intersect with ground plane.
    world_pt = intersect_ground_plane(origin, direction, z=config.GROUND_Z_M)
    if world_pt is None:
        logger.warning("Ray did not intersect the ground plane.")
        return {}

    logger.info("Ground intersection: [%.2f E, %.2f N, %.2f U] m", *world_pt)

    # Step 3 – Project into each target frame
    results: dict = {}
    for tf in target_frames:
        if tf is source_frame or not tf.ready:
            continue
        proj = project_to_frame(
            world_pt,
            tf.K_undist,
            tf.R,
            tf.position_enu,
            tf.undistorted.shape[:2],
        )
        if proj is not None:
            results[tf] = proj
            logger.info("  → [%s]: pixel (%.1f, %.1f)", tf.stem, *proj)
        else:
            logger.info("  → [%s]: point not visible.", tf.stem)

    return results
