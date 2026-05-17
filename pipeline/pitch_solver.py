"""
pipeline/pitch_solver.py – Ground-scatter pitch calibration for a multi-camera rig.

Given : camera positions, yaw, roll, intrinsics, and matched 2-D pixel
        coordinates for shared ground features (z = ground_z).
Find  : the pitch angle of each camera that minimises the scatter of
        ray-ground intersections across cameras observing the same feature.

Rotation convention (ENU world frame, OpenCV camera frame)
──────────────────────────────────────────────────────────
  yaw   : degrees, 0 = North (+Y), 90 = East (+X), clockwise positive
  pitch : degrees, 0 = horizontal, negative = looking DOWN
  roll  : degrees, positive = roll right

  make_rotation(yaw, pitch, roll) returns the 3×3 matrix R_world_from_cam
  such that:  d_world = R @ d_cam

  At pitch=0 the camera looks horizontally in the heading direction.
  At pitch=-90° the camera looks straight down (-Z in ENU).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ─────────────────────────────────────────────────────────────────────────────
# Rotation matrix  (ENU world + OpenCV camera convention)
# ─────────────────────────────────────────────────────────────────────────────

def make_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build the camera-to-world rotation matrix R_world_from_cam.

    Matches build_rotation() in pipeline/pose.py exactly so that the
    optimizer and the ray-caster use identical geometry.

    Returns R_world_from_cam such that:  d_world = R @ d_cam
    (build_rotation returns the transpose: R_cam_from_world)
    """
    import math as _math
    yaw   = _math.radians(yaw_deg)
    pitch = _math.radians(pitch_deg)
    roll  = _math.radians(roll_deg)

    fwd = np.array([
        _math.sin(yaw) * _math.cos(pitch),
        _math.cos(yaw) * _math.cos(pitch),
        _math.sin(pitch),
    ])

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(fwd[2]) > 0.99:
        ref   = np.array([_math.sin(yaw), _math.cos(yaw), 0.0])
        right = np.cross(fwd, ref)
    else:
        right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)

    down = np.cross(fwd, right)
    down /= np.linalg.norm(down)

    # Roll: rotate right and down around the optical axis (fwd).
    cr, sr   = _math.cos(roll), _math.sin(roll)
    right_r  =  cr * right + sr * down
    down_r   = -sr * right + cr * down

    R_world_from_cam = np.column_stack([right_r, down_r, fwd])
    return R_world_from_cam


# ─────────────────────────────────────────────────────────────────────────────
# Ray – ground plane intersection
# ─────────────────────────────────────────────────────────────────────────────

def ray_plane_intersect(
    origin:    np.ndarray,   # (3,) camera position in world (ENU metres)
    direction: np.ndarray,   # (3,) ray direction in world (need not be unit)
    z:         float = 0.0,  # ground plane height in ENU metres
) -> np.ndarray | None:
    """
    Intersect ray  P(t) = origin + t * direction  with plane  z = const.

    Returns the 3-D intersection point, or None if:
      - the ray is parallel to the plane  (|dz| < 1e-9)
      - the intersection is behind the camera  (t ≤ 0)
    """
    dz = float(direction[2])
    if abs(dz) < 1e-9:
        return None
    t = (z - float(origin[2])) / dz
    if t <= 0:
        return None
    return origin + t * direction


# ─────────────────────────────────────────────────────────────────────────────
# Residual function
# ─────────────────────────────────────────────────────────────────────────────

# Penalty applied (metres, per axis) when a ray fails to hit the ground.
_MISS_PENALTY = 1000.0


def compute_residuals(
    pitches:  np.ndarray,      # (N,)  current pitch guess (degrees)
    cameras:  list[dict],      # N dicts – see optimize_pitches() docstring
    features: list[dict],      # M dicts {cam_idx: (u, v), ...}
    z_ground: float = 0.0,
) -> np.ndarray:
    """
    Flat residual vector for scipy.optimize.least_squares.

    For each feature j observed by cameras S_j (|S_j| ≥ 2):
      ┌─ For each i ∈ S_j
      │   d_cam   = K_i⁻¹ · [u, v, 1]ᵀ
      │   d_world = R(yaw_i, pitch_i, roll_i) · d_cam
      │   W_ij    = ray_plane_intersect(C_i, d_world, z)
      └─ μ_j      = mean of all valid W_ij

      Residual for camera i:  [W_ij_x − μ_j_x,  W_ij_y − μ_j_y]
      (If ray misses the plane a penalty of ±1000 m is used.)

    The returned vector has FIXED length = 2 × Σ_j |S_j|, which is
    required by least_squares (constant across calls).
    """
    K_invs = [np.linalg.inv(cam['K']) for cam in cameras]
    residuals: list[float] = []

    for feat in features:
        cam_indices = sorted(i for i in feat if isinstance(i, int))
        if len(cam_indices) < 2:
            continue

        # ── Shoot rays → ground hits ──────────────────────────────────────────
        hits: list[np.ndarray | None] = []
        for i in cam_indices:
            cam   = cameras[i]
            u, v  = feat[i]
            d_cam = K_invs[i] @ np.array([u, v, 1.0])

            R       = make_rotation(cam['yaw'], float(pitches[i]), cam['roll'])
            d_world = R @ d_cam

            hits.append(ray_plane_intersect(cam['pos'], d_world, z=z_ground))

        # ── Centroid of valid hits ────────────────────────────────────────────
        valid = [h for h in hits if h is not None]
        if len(valid) >= 2:
            mu = np.mean(valid, axis=0)
        elif len(valid) == 1:
            mu = valid[0]
        else:
            mu = np.zeros(3)

        # ── Per-camera residuals (fixed size) ─────────────────────────────────
        for h in hits:
            if h is not None:
                residuals.append(float(h[0] - mu[0]))   # x (East)
                residuals.append(float(h[1] - mu[1]))   # y (North)
            else:
                residuals.append(_MISS_PENALTY)
                residuals.append(_MISS_PENALTY)

    return np.array(residuals, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation function
# ─────────────────────────────────────────────────────────────────────────────

def optimize_pitches(
    cameras:         list[dict],
    features:        list[dict],
    initial_pitches: list[float] | None = None,
    z_ground:        float = 0.0,
    pitch_min:       float = -89.0,
    pitch_max:       float =  15.0,
    verbose:         bool  = True,
) -> tuple[np.ndarray, object]:
    """
    Find the pitch of each camera that minimises ground-scatter.

    Parameters
    ──────────
    cameras : list of N dicts, each with:
        'pos'  : (3,) np.ndarray  — camera position [East, North, Up] metres
        'yaw'  : float            — heading in degrees (0 = N, 90 = E, CW)
        'roll' : float            — roll in degrees (positive = roll right)
        'K'    : (3, 3) np.ndarray — camera intrinsic matrix

    features : list of M dicts, each mapping
        int (camera index) → (u, v) pixel in that camera's image
        Features seen by < 2 cameras are silently ignored.

    initial_pitches : (N,) starting pitches in degrees.
        Defaults to 0° for every camera if None.

    z_ground  : height of the ground plane in world ENU metres.
    pitch_min / pitch_max : search bounds in degrees.

    Returns
    ───────
    optimized_pitches : (N,) np.ndarray — refined pitch per camera (degrees)
    result            : scipy OptimizeResult (cost, nfev, message, …)
    """
    N = len(cameras)

    if initial_pitches is None:
        x0 = np.zeros(N)
    else:
        x0 = np.clip(initial_pitches, pitch_min, pitch_max).astype(float)

    lo = np.full(N, pitch_min)
    hi = np.full(N, pitch_max)

    result = least_squares(
        compute_residuals,
        x0,
        args    = (cameras, features, z_ground),
        bounds  = (lo, hi),
        method  = 'trf',        # Trust Region Reflective – handles bounds well
        loss    = 'cauchy',     # Robust: down-weights wrong correspondences
        f_scale = 5.0,          # residuals > 5 m get suppressed (wrong matches)
        ftol    = 1e-5,
        xtol    = 1e-5,
        gtol    = 1e-5,
        max_nfev= 2000,
        verbose = 2 if verbose else 0,
    )

    return result.x, result
