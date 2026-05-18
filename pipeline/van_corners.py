"""
pipeline/van_corners.py – Van corner geometry for the orientation solver.

Provides the 3-D corner layout of the van in van-local coordinates,
visibility determination per camera frame, and matching of visible projected
corners to observed bounding-box extremes.

Van-local coordinate frame
──────────────────────────
  Origin : centre of the van base (ground contact centre)
  X      : forward  (along van heading)
  Y      : left     (90° CCW from heading when viewed from above)
  Z      : up

The 8 corners are labelled by face combination:
  FL / FR / RL / RR = Front-Left / Front-Right / Rear-Left / Rear-Right
  B / T             = Bottom (Z=0) / Top (Z=VAN_HEIGHT_M)

Coordinate signs in van-local:
  Front-Left-Bottom  → [+L/2, +W/2, 0]
  Front-Right-Bottom → [+L/2, -W/2, 0]
  Rear-Left-Bottom   → [-L/2, +W/2, 0]
  Rear-Right-Bottom  → [-L/2, -W/2, 0]
  (Top corners same but Z = VAN_HEIGHT_M)
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

import config
from pipeline.frame import Frame
from pipeline.pose  import build_rotation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Corner table (van-local)
# ─────────────────────────────────────────────────────────────────────────────

def _make_corners() -> np.ndarray:
    """Return (8, 3) array of van corners in van-local frame."""
    L, W, H = config.VAN_LENGTH_M / 2, config.VAN_WIDTH_M / 2, config.VAN_HEIGHT_M
    return np.array([
        [ L,  W, 0],   # 0  Front-Left-Bottom
        [ L, -W, 0],   # 1  Front-Right-Bottom
        [-L,  W, 0],   # 2  Rear-Left-Bottom
        [-L, -W, 0],   # 3  Rear-Right-Bottom
        [ L,  W, H],   # 4  Front-Left-Top
        [ L, -W, H],   # 5  Front-Right-Top
        [-L,  W, H],   # 6  Rear-Left-Top
        [-L, -W, H],   # 7  Rear-Right-Top
    ], dtype=np.float64)

CORNERS_LOCAL = _make_corners()   # (8, 3)

# Human-readable labels for logging
CORNER_NAMES = [
    "FL-Bot", "FR-Bot", "RL-Bot", "RR-Bot",
    "FL-Top", "FR-Top", "RL-Top", "RR-Top",
]


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transforms
# ─────────────────────────────────────────────────────────────────────────────

def van_rotation(van_heading_deg: float) -> np.ndarray:
    """
    3×3 rotation matrix from van-local to ENU world frame.

    Van-local X (forward) maps to ENU [sin(H), cos(H), 0].
    Van-local Y (left)    maps to ENU [-cos(H), sin(H), 0].
    Van-local Z (up)      maps to ENU [0, 0, 1].
    """
    H = math.radians(van_heading_deg)
    s, c = math.sin(H), math.cos(H)
    return np.array([
        [ s, -c, 0.0],   # ENU East   components of local X, Y, Z
        [ c,  s, 0.0],   # ENU North
        [0.0, 0.0, 1.0], # ENU Up
    ], dtype=np.float64)


def corners_in_world(van_east: float, van_north: float,
                     van_heading_deg: float, van_z: float) -> np.ndarray:
    """
    Transform all 8 corners from van-local to ENU world frame.

    Returns (8, 3) array of world-frame corner positions.
    """
    R = van_rotation(van_heading_deg)
    origin = np.array([van_east, van_north, van_z])
    return (R @ CORNERS_LOCAL.T).T + origin   # (8, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Visibility
# ─────────────────────────────────────────────────────────────────────────────

def visible_corner_mask(van_to_cam_enu: np.ndarray,
                        van_heading_deg: float) -> np.ndarray:
    """
    Boolean mask of which corners are visible from the camera.

    A corner is visible when the dot product of the van→camera vector with
    the corner's outward direction (sign of its local coordinates) is > 0.
    This is exact for a convex box viewed from outside.

    Args:
        van_to_cam_enu : (3,) vector from van centre to camera, in ENU.
        van_heading_deg: van heading in degrees.

    Returns:
        (8,) bool array, True = corner potentially visible.
    """
    R = van_rotation(van_heading_deg)
    # Rotate van→camera from ENU to van-local frame
    van_to_cam_local = R.T @ van_to_cam_enu   # (3,)

    # Sign of each corner's local coordinates = outward direction
    corner_signs = np.sign(CORNERS_LOCAL)       # (8, 3)
    # For Z=0 bottom corners, Z sign = 0; treat as visible if camera is above
    # (handled by the dot product with full 3-D vector)
    dots = corner_signs @ van_to_cam_local      # (8,)
    return dots > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Project corners → image
# ─────────────────────────────────────────────────────────────────────────────

def project_corners(corners_world: np.ndarray,
                    K: np.ndarray,
                    R_cam_from_world: np.ndarray,
                    pos_enu: np.ndarray,
                    image_shape: tuple[int, int]) -> dict[int, tuple[float, float]]:
    """
    Project world-frame corners into the image.

    Returns a dict mapping corner_index → (u, v) for corners that are
    in front of the camera and within the image bounds.
    """
    h, w = image_shape
    result: dict[int, tuple[float, float]] = {}
    for idx, corner in enumerate(corners_world):
        p_cam = R_cam_from_world @ (corner - pos_enu)
        if p_cam[2] <= 0:
            continue
        u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
        v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
        if 0 <= u < w and 0 <= v < h:
            result[idx] = (float(u), float(v))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Match projected corners → bbox extremes
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_bbox_corner(
    px: float, py: float,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return the bbox corner (x1,y1 / x1,y2 / x2,y1 / x2,y2) nearest to (px, py)."""
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    return min(corners, key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)


def get_van_observations(
    frame: Frame,
    bbox: tuple[float, float, float, float],
    van_east_init: float,
    van_north_init: float,
    van_heading_init: float,
    van_z: float,
) -> list[tuple[np.ndarray, tuple[float, float]]]:
    """
    Compute van corner reprojection observations for one frame.

    Uses the *initial* van pose and camera pose (pitch=0°) to determine which
    corners are visible, projects them, then matches each to the nearest bbox
    corner.  Returns a list of (corner_local_3d, pixel_obs) pairs that the
    orientation solver will drive to zero reprojection error.

    Args:
        frame            : Frame with valid position_enu, K_undist, heading_deg,
                           camera_roll_deg, and undistorted image.
        bbox             : GroundingDINO bounding box (x1, y1, x2, y2).
        van_east_init    : Initial van ENU east position (metres).
        van_north_init   : Initial van ENU north position (metres).
        van_heading_init : Initial van heading (degrees).
        van_z            : Fixed van base Z in ENU (metres).

    Returns:
        List of (corner_local_3d, pixel_obs) where corner_local_3d is shape (3,)
        and pixel_obs is (u, v).  May be empty if no corners project cleanly.
    """
    if frame.undistorted is None or frame.position_enu is None:
        return []

    # Initial camera rotation (pitch=0 seed)
    R_cam = build_rotation(frame.heading_deg, 0.0, frame.camera_roll_deg)

    # Van→camera vector in ENU (for visibility test)
    van_centre_enu = np.array([van_east_init, van_north_init, van_z])
    van_to_cam = frame.position_enu - van_centre_enu

    # Visibility mask
    mask = visible_corner_mask(van_to_cam, van_heading_init)

    # World positions of all corners (initial van pose)
    corners_world = corners_in_world(van_east_init, van_north_init,
                                     van_heading_init, van_z)

    # Project visible corners into the image
    h, w = frame.undistorted.shape[:2]
    projected = project_corners(
        corners_world[mask],           # only visible subset
        frame.K_undist,
        R_cam,
        frame.position_enu,
        (h, w),
    )
    # Re-index back to global corner indices
    visible_indices = np.where(mask)[0]
    global_projected: dict[int, tuple[float, float]] = {}
    for local_idx, corner_pix in projected.items():
        global_idx = int(visible_indices[local_idx])
        global_projected[global_idx] = corner_pix

    if not global_projected:
        logger.debug("[%s] No van corners projected into image.", frame.stem)
        return []

    # Match each projected corner to the nearest bbox corner (full 2-D obs)
    observations: list[tuple[np.ndarray, tuple[float, float]]] = []
    seen_bbox_corners: set[tuple[float, float]] = set()

    for corner_idx, (pu, pv) in global_projected.items():
        obs_pixel = _nearest_bbox_corner(pu, pv, bbox)
        # Skip if this bbox corner is already claimed by a closer projected corner
        if obs_pixel in seen_bbox_corners:
            continue
        seen_bbox_corners.add(obs_pixel)
        observations.append((CORNERS_LOCAL[corner_idx].copy(), obs_pixel))
        logger.debug(
            "[%s] Corner %s proj=(%.0f,%.0f) → bbox obs=(%.0f,%.0f)",
            frame.stem, CORNER_NAMES[corner_idx], pu, pv, *obs_pixel,
        )

    logger.info("[%s] %d van corner observation(s).", frame.stem, len(observations))
    return observations


# ─────────────────────────────────────────────────────────────────────────────
# Van position initialisation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_van_position(frame: Frame,
                           bbox: tuple[float, float, float, float],
                           van_z: float) -> tuple[float, float]:
    """
    Estimate the van's ENU (east, north) position from one frame's bbox.

    Raycasts the centre-bottom pixel of the bbox to the ground plane at
    van_z to get an initial van position estimate.  Accuracy depends on
    pitch=0° seed — good enough to seed the solver.

    Returns (van_east, van_north).
    """
    x1, y1, x2, y2 = bbox
    u = (x1 + x2) / 2.0   # horizontal centre of bbox
    v = y2                  # bottom edge ≈ ground contact

    K_inv = np.linalg.inv(frame.K_undist)
    R_cam = build_rotation(frame.heading_deg, 0.0, frame.camera_roll_deg)

    d_cam   = K_inv @ np.array([u, v, 1.0])
    d_world = R_cam.T @ d_cam
    d_world /= np.linalg.norm(d_world)

    dz = d_world[2]
    if abs(dz) < 1e-6:
        logger.warning("Van position estimate: ray parallel to ground — using camera position.")
        return float(frame.position_enu[0]), float(frame.position_enu[1])

    t = (van_z - frame.position_enu[2]) / dz
    if t <= 0:
        logger.warning("Van position estimate: intersection behind camera — using camera position.")
        return float(frame.position_enu[0]), float(frame.position_enu[1])

    P = frame.position_enu + t * d_world
    logger.info(
        "[%s] Van position estimate from bbox: east=%.1f m  north=%.1f m",
        frame.stem, P[0], P[1],
    )
    return float(P[0]), float(P[1])
