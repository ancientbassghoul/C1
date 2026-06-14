"""
pipeline/orientation_solver.py – Full bundle-adjustment with pyceres.

Optimises jointly:
  Per-camera  : 6-DoF pose [rx, ry, rz, tx, ty, tz] (axis-angle + translation)
  Per-3D-point: [x, y, z] in MASt3R's world frame  (FREE — not fixed)
  P_anchor    : [x, y, z] single shared 3D anchor point

Residual types:
  MASt3RReprojCost  — dense 3D point reprojection (from MASt3R observations)
  AnchorRayCost     — CLIP-weighted centroid ray to shared P_anchor
  ManualReprojCost  — higher-weight reprojection for hand-picked correspondences
  VanMetricCost     — known-distance constraint for wheel_axis / roof_edge pairs
  AxisPairCost      — wheel_axis / roof_edge: distance + cross-frame reprojection

After Ceres the coordinate frame is still MASt3R's arbitrary frame.
Call align_to_telemetry_sim3() to map to ENU via Umeyama Sim(3).
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.spatial.transform import Rotation

import config
from pipeline.pose import build_rotation

logger = logging.getLogger(__name__)

_MISS_PENALTY = 1000.0
_FD_EPS       = 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aa_to_R(r: np.ndarray) -> np.ndarray:
    """Axis-angle 3-vector → 3×3 rotation matrix (scipy)."""
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(r).as_matrix()


def _project(r3: np.ndarray, t3: np.ndarray, P: np.ndarray, K: np.ndarray):
    """Project world point P using camera [r,t] and intrinsic K.

    Returns projected (u, v) or None if the point is behind the camera.
    """
    R     = _aa_to_R(r3)
    p_cam = R @ P + t3
    if float(p_cam[2]) <= 1e-6:
        return None
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    return np.array([u, v])


def _w2c_from_params(cam6: np.ndarray) -> np.ndarray:
    """Convert 6-vector [rx,ry,rz,tx,ty,tz] → 4×4 world-to-camera matrix."""
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = _aa_to_R(cam6[:3])
    M[:3,  3] = cam6[3:6]
    return M


def _params_from_w2c(W2C: np.ndarray) -> np.ndarray:
    """Convert 4×4 world-to-camera matrix → 6-vector [rx,ry,rz,tx,ty,tz]."""
    r = Rotation.from_matrix(W2C[:3, :3]).as_rotvec()
    t = W2C[:3, 3]
    return np.concatenate([r, t]).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Cost functors (plain Python callables — wrapped by _NumericDiff)
# ─────────────────────────────────────────────────────────────────────────────

class MASt3RReprojCost:
    """
    Reprojection error for one (camera, 3D point, observed pixel) triple.

    Call: (cam6[6], point[3]) → residual[2]
    """
    def __init__(self, pixel_uv: np.ndarray, K: np.ndarray, confidence: float):
        self._obs  = np.asarray(pixel_uv, dtype=np.float64)
        self._K    = K.astype(np.float64)
        self._conf = float(confidence)

    def __call__(self, cam6: np.ndarray, point3: np.ndarray) -> np.ndarray:
        proj = _project(cam6[:3], cam6[3:], point3, self._K)
        if proj is None:
            return np.array([_MISS_PENALTY, _MISS_PENALTY]) * self._conf
        return (proj - self._obs) * self._conf


class AnchorRayCost:
    """
    CLIP-weighted reprojection of the shared P_anchor into one frame.

    Call: (cam6[6], p_anchor[3]) → residual[2]  (scaled by CLIP weight)
    """
    def __init__(self, pixel_uv: np.ndarray, K: np.ndarray, clip_weight: float):
        self._obs    = np.asarray(pixel_uv, dtype=np.float64)
        self._K      = K.astype(np.float64)
        self._weight = float(clip_weight)

    def __call__(self, cam6: np.ndarray, p_anchor: np.ndarray) -> np.ndarray:
        proj = _project(cam6[:3], cam6[3:], p_anchor, self._K)
        if proj is None:
            return np.array([_MISS_PENALTY, _MISS_PENALTY]) * self._weight
        return (proj - self._obs) * self._weight


class ManualReprojCost(MASt3RReprojCost):
    """Reprojection cost for hand-picked points (higher confidence weight)."""
    def __init__(self, pixel_uv: np.ndarray, K: np.ndarray):
        super().__init__(pixel_uv, K, confidence=config.MANUAL_CORRESPONDENCE_WEIGHT)


class VanMetricCost:
    """
    Distance constraint between two 3D points (wheel_axis / roof_edge pairs).

    Call: (point_a[3], point_b[3]) → residual[1]
    """
    def __init__(self, distance_m: float):
        self._dist = float(distance_m)

    def __call__(self, point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
        return np.array([float(np.linalg.norm(point_b - point_a)) - self._dist])


# ─────────────────────────────────────────────────────────────────────────────
# Ceres problem builder
# ─────────────────────────────────────────────────────────────────────────────

def ceres_solve(
    frames,
    mast3r_observations,
    anchor_rays,
    cam_poses_init,
    points_3d_init: np.ndarray,
    p_anchor_init: np.ndarray,
    manual_features=None,
):
    """
    Build and solve the pyceres full-BA problem.

    Parameters
    ----------
    frames               : list[Frame]
    mast3r_observations  : list[Observation] from mast3r_matcher
    anchor_rays          : list[(frame_stem, pixel_uv, clip_weight)]
    cam_poses_init       : dict[str, np.ndarray(4,4)] — MASt3R initial W2C matrices
    points_3d_init       : np.ndarray(N, 3) — MASt3R initial 3D points
    p_anchor_init        : np.ndarray(3,) — initial anchor 3D position
    manual_features      : list of manual correspondence feature dicts (optional)

    Returns
    -------
    cam_params_out : dict[str, np.ndarray(6,)] — solved camera params per stem
    points_3d_out  : np.ndarray(N, 3) — refined 3D points
    p_anchor_out   : np.ndarray(3,) — refined anchor position
    report         : str — Ceres BriefReport
    """
    try:
        import pyceres
    except ImportError:
        raise ImportError("\npyceres not installed — run SETUP_SECOND.bat\n")

    # ── Generic numeric-diff wrapper ──────────────────────────────────────────
    class _NumericDiff(pyceres.CostFunction):
        def __init__(self, callable_cost, num_residuals, block_sizes):
            super().__init__()
            self._cost        = callable_cost
            self._block_sizes = block_sizes
            self.set_num_residuals(num_residuals)
            self.set_parameter_block_sizes(block_sizes)

        def Evaluate(self, parameters, residuals, jacobians):
            res = self._cost(*parameters)
            residuals[:] = res
            if jacobians is not None:
                for bi, bs in enumerate(self._block_sizes):
                    if jacobians[bi] is None:
                        continue
                    jac = jacobians[bi]
                    for k in range(bs):
                        p_plus  = [p.copy() for p in parameters]
                        p_minus = [p.copy() for p in parameters]
                        p_plus[bi][k]  += _FD_EPS
                        p_minus[bi][k] -= _FD_EPS
                        jac[k::bs] = (
                            self._cost(*p_plus) - self._cost(*p_minus)
                        ) / (2.0 * _FD_EPS)
            return True

    # ── Initialise parameter blocks ───────────────────────────────────────────
    frame_map  = {f.stem: f for f in frames}
    cam_params = {}   # stem → np.ndarray(6,)

    for f in frames:
        W2C = cam_poses_init.get(f.stem)
        if W2C is not None:
            cam_params[f.stem] = _params_from_w2c(W2C)
        else:
            logger.warning("[%s] no MASt3R pose — using identity", f.stem)
            cam_params[f.stem] = np.zeros(6, dtype=np.float64)

    n_pts       = len(points_3d_init)
    point_params = [
        np.asarray(points_3d_init[k], dtype=np.float64).copy()
        for k in range(n_pts)
    ]
    p_anchor = np.asarray(p_anchor_init, dtype=np.float64).copy()

    logger.info(
        "Ceres Full-BA: %d cameras, %d 3D points, %d observations, "
        "%d anchor rays, P_anchor=%s",
        len(frames), n_pts, len(mast3r_observations), len(anchor_rays),
        np.round(p_anchor, 2),
    )

    problem    = pyceres.Problem()
    reproj_loss = pyceres.HuberLoss(1.5)   # 1.5 px inlier threshold
    anchor_loss = pyceres.HuberLoss(3.0)
    metric_loss = pyceres.HuberLoss(0.05)  # 5 cm van geometry

    frames_in_problem: set = set()

    # ── MASt3R reprojection residuals ─────────────────────────────────────────
    for obs in mast3r_observations:
        stem = obs.frame_stem
        f    = frame_map.get(stem)
        if f is None or f.K_undist is None:
            continue
        kidx = obs.point_idx
        if kidx >= n_pts:
            continue
        functor = MASt3RReprojCost(obs.pixel_uv, f.K_undist, obs.confidence)
        cost    = _NumericDiff(functor, num_residuals=2, block_sizes=[6, 3])
        problem.add_residual_block(
            cost, reproj_loss,
            [cam_params[stem], point_params[kidx]],
        )
        frames_in_problem.add(stem)

    # ── Anchor ray residuals ──────────────────────────────────────────────────
    for stem, pixel_uv, clip_weight in anchor_rays:
        if clip_weight < config.CLIP_ANCHOR_MIN_WEIGHT:
            continue
        f = frame_map.get(stem)
        if f is None or f.K_undist is None:
            continue
        functor = AnchorRayCost(pixel_uv, f.K_undist, clip_weight)
        cost    = _NumericDiff(functor, num_residuals=2, block_sizes=[6, 3])
        problem.add_residual_block(
            cost, anchor_loss,
            [cam_params[stem], p_anchor],
        )
        frames_in_problem.add(stem)

    # ── Manual feature residuals ──────────────────────────────────────────────
    if manual_features:
        # manual_point_params: (feature_idx, 'A'/'B') → np.ndarray(3,)
        # shared across frames that observe the same feature
        manual_point_params: dict[tuple, np.ndarray] = {}

        def _get_manual_point(feat_idx: int, end: str) -> np.ndarray:
            key = (feat_idx, end)
            if key not in manual_point_params:
                manual_point_params[key] = np.zeros(3, dtype=np.float64)
            return manual_point_params[key]

        # We need to initialise manual point params from MASt3R's point cloud
        # or from reprojection if possible.  For simplicity, seed with zeros
        # and let Ceres solve from there.  The reprojection residuals pull them
        # toward the correct world position.

        for feat_idx, feat in enumerate(manual_features):
            feat_type  = feat.get("type", "ground")
            is_pair    = feat.get("is_pair", False)
            distance_m = float(feat.get("distance_m", 1.0)) if is_pair else None
            points_per_frame = feat.get("points", {})

            for stem, pts_list in points_per_frame.items():
                f = frame_map.get(stem)
                if f is None or f.K_undist is None:
                    continue

                if is_pair and len(pts_list) >= 2:
                    pix_a = np.asarray(pts_list[0], dtype=np.float64)
                    pix_b = np.asarray(pts_list[1], dtype=np.float64)
                    P_a   = _get_manual_point(feat_idx, "A")
                    P_b   = _get_manual_point(feat_idx, "B")

                    cost_a = _NumericDiff(
                        ManualReprojCost(pix_a, f.K_undist),
                        num_residuals=2, block_sizes=[6, 3],
                    )
                    cost_b = _NumericDiff(
                        ManualReprojCost(pix_b, f.K_undist),
                        num_residuals=2, block_sizes=[6, 3],
                    )
                    problem.add_residual_block(cost_a, None, [cam_params[stem], P_a])
                    problem.add_residual_block(cost_b, None, [cam_params[stem], P_b])
                    frames_in_problem.add(stem)

                elif not is_pair and len(pts_list) >= 1:
                    pix = np.asarray(pts_list[0], dtype=np.float64)
                    P   = _get_manual_point(feat_idx, "A")
                    cost = _NumericDiff(
                        ManualReprojCost(pix, f.K_undist),
                        num_residuals=2, block_sizes=[6, 3],
                    )
                    problem.add_residual_block(cost, None, [cam_params[stem], P])
                    frames_in_problem.add(stem)

            # Van metric constraint (once per feature, not per frame)
            if is_pair and distance_m is not None:
                P_a = _get_manual_point(feat_idx, "A")
                P_b = _get_manual_point(feat_idx, "B")
                cost = _NumericDiff(
                    VanMetricCost(distance_m),
                    num_residuals=1, block_sizes=[3, 3],
                )
                problem.add_residual_block(cost, metric_loss, [P_a, P_b])

        logger.info("Added %d manual feature constraints.", len(manual_features))

    n_unconstrained = len(set(f.stem for f in frames) - frames_in_problem)
    if n_unconstrained:
        logger.warning(
            "%d frame(s) have no residuals — their poses will remain at MASt3R seeds.",
            n_unconstrained,
        )

    # ── Bounds on camera rotations (small window around MASt3R seeds) ─────────
    for stem, p in cam_params.items():
        # Rotation: ±SOLVER_PITCH_OFFSET degrees around seed (applied to each component)
        rot_range = math.radians(config.SOLVER_PITCH_OFFSET)
        for i in range(3):
            problem.set_parameter_lower_bound(p, i, float(p[i]) - rot_range)
            problem.set_parameter_upper_bound(p, i, float(p[i]) + rot_range)
        # Translation: ±SOLVER_POSITION_RANGE_H around seed
        for i, rng in enumerate([
            config.SOLVER_POSITION_RANGE_H,
            config.SOLVER_POSITION_RANGE_N,
            config.SOLVER_POSITION_RANGE_V,
        ]):
            problem.set_parameter_lower_bound(p, 3 + i, float(p[3 + i]) - rng)
            problem.set_parameter_upper_bound(p, 3 + i, float(p[3 + i]) + rng)

    # ── Solver options ────────────────────────────────────────────────────────
    options = pyceres.SolverOptions()
    options.linear_solver_type          = pyceres.LinearSolverType.SPARSE_SCHUR
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations          = config.SOLVER_MAX_ITERATIONS
    options.function_tolerance          = 1e-6
    options.gradient_tolerance          = 1e-8
    options.parameter_tolerance         = 1e-8

    logger.info("Starting Ceres full-BA solve …")
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)

    report = summary.BriefReport()
    logger.info("Ceres solve complete: %s", report)

    # ── Collect results ───────────────────────────────────────────────────────
    points_3d_out = np.stack(point_params, axis=0)

    return cam_params, points_3d_out, p_anchor, report


# ─────────────────────────────────────────────────────────────────────────────
# Sim(3) ENU alignment
# ─────────────────────────────────────────────────────────────────────────────

def _umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate Similarity(3) transform s, R, t such that dst ≈ s * R @ src + t.

    Uses the Umeyama algorithm (zero-mean SVD).
    Returns (scale, R_3x3, t_3).
    """
    n = src.shape[0]
    assert n == dst.shape[0] and n >= 3

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = float(np.mean(np.sum(src_c ** 2, axis=1)))
    if var_src < 1e-12:
        return 1.0, np.eye(3), mu_dst - mu_src

    H   = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(H)

    det_sign = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, det_sign])

    R   = U @ D @ Vt
    s   = float(np.sum(S * np.diag(D))) / var_src
    t   = mu_dst - s * R @ mu_src

    return s, R, t


def align_to_telemetry_sim3(
    frames,
    cam_params_solved: dict[str, np.ndarray],
    anchor_result=None,
) -> None:
    """
    Map solved camera poses from MASt3R's coordinate frame to GPS/ENU.

    Control points for the Sim(3) fit: only frames with CLIP anchor weight
    w >= CLIP_ANCHOR_THRESHOLD (high-confidence, non-blurry).

    If no anchor_result is available, falls back to all frames.

    After this function, each frame's position_enu, heading_deg,
    gimbal_pitch_deg, camera_roll_deg, and R are updated in-place.
    """
    frame_map = {f.stem: f for f in frames}

    # ── Identify control points ───────────────────────────────────────────────
    if anchor_result is not None:
        ctrl_stems = [
            stem for stem, w in anchor_result.weights.items()
            if w >= config.CLIP_ANCHOR_THRESHOLD
        ]
    else:
        ctrl_stems = list(frame_map.keys())

    if len(ctrl_stems) < 3:
        logger.warning(
            "Only %d high-confidence frames for Sim(3); using all frames as fallback.",
            len(ctrl_stems),
        )
        ctrl_stems = list(frame_map.keys())

    # Camera centre in MASt3R frame: C = -R.T @ t
    def _camera_centre(cam6: np.ndarray) -> np.ndarray:
        R = _aa_to_R(cam6[:3])
        t = cam6[3:6]
        return -R.T @ t

    src_pts, dst_pts, ctrl_frames = [], [], []
    for stem in ctrl_stems:
        f   = frame_map.get(stem)
        p6  = cam_params_solved.get(stem)
        if f is None or p6 is None or f.position_enu is None:
            continue
        src_pts.append(_camera_centre(p6))
        dst_pts.append(f.position_enu.copy())
        ctrl_frames.append(stem)

    if len(ctrl_frames) < 3:
        logger.error(
            "Sim(3): fewer than 3 valid control points (%d) — skipping ENU alignment.",
            len(ctrl_frames),
        )
        return

    src_arr = np.stack(src_pts, axis=0)
    dst_arr = np.stack(dst_pts, axis=0)

    scale, R_sim, t_sim = _umeyama_sim3(src_arr, dst_arr)
    logger.info(
        "Sim(3) alignment: scale=%.4f  using %d/%d control frames",
        scale, len(ctrl_frames), len(frames),
    )

    # ── Apply transform to all cameras ────────────────────────────────────────
    for f in frames:
        p6 = cam_params_solved.get(f.stem)
        if p6 is None:
            continue

        # Camera centre in ENU
        C_mast3r = _camera_centre(p6)
        C_enu    = scale * R_sim @ C_mast3r + t_sim
        f.position_enu = C_enu

        # Rotation: R_enu_wc = R_mast3r_wc @ R_sim^T  (change of world frame)
        R_mast3r_wc = _aa_to_R(p6[:3])
        R_enu_wc    = R_mast3r_wc @ R_sim.T

        # Decompose R_enu_wc into heading / pitch / roll and update frame
        _apply_rotation_to_frame(f, R_enu_wc)

    logger.info("ENU alignment applied to %d frames.", len(frames))


def _apply_rotation_to_frame(frame, R_wc: np.ndarray) -> None:
    """
    Decompose a world-to-camera rotation matrix (in ENU frame) into
    heading_deg, gimbal_pitch_deg, camera_roll_deg and store into the frame.

    ENU → OpenCV camera: R_wc maps ENU world → OpenCV cam frame
    We parameterise as:  R_wc = R_roll @ R_pitch @ R_yaw
    where yaw is compass heading (Z-up in ENU, but note ENU has Z=Up).

    Uses scipy Rotation to extract intrinsic ZYX Euler angles.
    """
    # scipy intrinsic ZYX: first Z (yaw), then Y (pitch), then X (roll)
    # but we need to account for the ENU → camera axis mapping.
    # Instead, we store R directly and decompose as yaw/pitch/roll matching
    # the convention in pipeline/pose.py build_rotation().
    #
    # build_rotation(heading, pitch, roll) computes:
    #   R = R_x(roll) @ R_y(pitch) @ R_z(heading)   [in the ENU tangent frame]
    # We extract by: r = Rotation.from_matrix(R_wc); ZYX angles.
    try:
        rot = Rotation.from_matrix(R_wc)
        # Extrinsic ZYX = intrinsic XYZ applied in reverse
        angles_zyx = rot.as_euler("ZYX", degrees=True)
        heading_deg = float(angles_zyx[0])
        pitch_deg   = float(angles_zyx[1])
        roll_deg    = float(angles_zyx[2])
    except Exception:
        logger.warning("[%s] rotation decomposition failed", frame.stem[-12:])
        return

    frame.heading_deg      = heading_deg % 360.0
    frame.gimbal_pitch_deg = pitch_deg
    frame.camera_roll_deg  = roll_deg
    frame.R                = R_wc.copy()
    frame.ready            = True
