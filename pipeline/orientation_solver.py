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
    if n_pts == 0:
        logger.error(
            "Ceres solve aborted: zero 3D points (all frames below "
            "MAST3R_CONF_THRESHOLD=%.1f). Returning MASt3R seed cameras.",
            config.MAST3R_CONF_THRESHOLD,
        )
        return cam_params, np.empty((0, 3), dtype=np.float64), \
               np.asarray(p_anchor_init, dtype=np.float64).copy(), \
               "Aborted: zero 3D points"

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
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """
    Map solved camera poses from MASt3R's coordinate frame to GPS/ENU.

    Control points for the Sim(3) fit: only frames with CLIP anchor weight
    w >= CLIP_ANCHOR_THRESHOLD (high-confidence, non-blurry).

    If no anchor_result is available, falls back to all frames.

    After this function, each frame's position_enu, heading_deg,
    gimbal_pitch_deg, camera_roll_deg, and R are updated in-place.

    Returns
    -------
    (scale, R_sim, t_sim) — the fitted Sim(3) transform (MASt3R frame → ENU),
    so callers can apply the identical transform to other MASt3R-frame
    artifacts (e.g. the dense point cloud, for the Blender mesh export).
    None if alignment was skipped (fewer than 3 valid control points).
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
        return None

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
    return scale, R_sim, t_sim


# ─────────────────────────────────────────────────────────────────────────────
# Ground-plane leveling
# ─────────────────────────────────────────────────────────────────────────────

def _ransac_plane(
    pts: np.ndarray,
    thresh: float,
    iters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RANSAC-fit the dominant plane in a point cloud.

    Returns (normal, centroid, inlier_mask):
      normal      : (3,) unit plane normal (refit to the best consensus set).
      centroid    : (3,) mean of the inlier points (a point on the plane).
      inlier_mask : (N,) bool mask of inliers within `thresh` of the plane.
    """
    n = len(pts)
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0

    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = pts[idx]
        nrm = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:                      # degenerate (collinear) sample
            continue
        nrm = nrm / nn
        dist = np.abs((pts - p1) @ nrm)
        mask = dist < thresh
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_count < 3:
        # No consensus — fall back to a total-least-squares fit over everything.
        best_mask = np.ones(n, dtype=bool)

    inliers = pts[best_mask]
    centroid = inliers.mean(axis=0)
    # Refit: smallest singular vector of the mean-centred inliers is the normal.
    _, _, Vt = np.linalg.svd(inliers - centroid)
    normal = Vt[-1]
    normal = normal / np.linalg.norm(normal)
    return normal, centroid, best_mask


def _rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix R such that R @ a == b (a, b unit vectors)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        # Parallel or antiparallel.
        if c > 0:
            return np.eye(3)
        # Antiparallel: 180° about any axis orthogonal to a.
        ortho = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def level_ground_plane(
    frames,
    points_3d_solved: np.ndarray,
    sim3: tuple[float, np.ndarray, np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Correct the residual scene tilt left by the camera-centre Sim(3) fit.

    The telemetry Sim(3) fits camera centres only; on a near-constant-altitude
    arc those are nearly coplanar, so its vertical axis is poorly constrained
    and the ground comes out tilted. Here we RANSAC-fit the dominant ground
    plane to the Ceres-refined sparse points (mapped into ENU via `sim3`),
    then apply a corrective rigid transform that rotates the plane normal to
    +Z and shifts it to config.GROUND_Z_M.

    Keeps the GPS-derived scale and horizontal alignment: the correction is a
    rotation about a horizontal axis (heading preserved) plus a Z shift.

    Mutates each frame's R / position_enu in place and returns the COMPOSED
    Sim(3) (level ∘ sim3) so downstream artifacts (debug .glb, MASt3R-ENU
    cameras) stay consistent.  On any guard failure the scene is left untouched
    and the original `sim3` is returned.
    """
    scale, R_sim, t_sim = sim3

    pts = np.asarray(points_3d_solved, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        logger.warning("Ground leveling skipped — too few solved points (%d).", len(pts))
        return sim3

    # Map MASt3R-frame points into the current ENU frame.
    pts_enu = scale * (pts @ R_sim.T) + t_sim

    rng = np.random.default_rng(0)   # deterministic
    normal, centroid, inlier_mask = _ransac_plane(
        pts_enu,
        thresh=float(config.GROUND_RANSAC_THRESH_M),
        iters=int(config.GROUND_RANSAC_ITERS),
        rng=rng,
    )

    inlier_frac = float(inlier_mask.mean())
    if inlier_frac < float(config.GROUND_RANSAC_MIN_INLIER_FRAC):
        logger.warning(
            "Ground leveling skipped — no dominant plane (inlier frac %.2f < %.2f).",
            inlier_frac, config.GROUND_RANSAC_MIN_INLIER_FRAC,
        )
        return sim3

    # Orient the normal "up" so the corrective rotation is the minimal one.
    if normal[2] < 0:
        normal = -normal

    tilt_deg = math.degrees(math.acos(np.clip(normal[2], -1.0, 1.0)))
    if tilt_deg > float(config.GROUND_LEVEL_MAX_TILT_DEG):
        logger.warning(
            "Ground leveling skipped — correction tilt %.1f° exceeds max %.1f° "
            "(RANSAC likely locked onto the van or a spike fan, not the ground).",
            tilt_deg, config.GROUND_LEVEL_MAX_TILT_DEG,
        )
        return sim3

    # Rotation taking the ground normal to +Z, pivoting about the inlier
    # centroid, then a Z shift so the plane sits at GROUND_Z_M.
    z_axis  = np.array([0.0, 0.0, 1.0])
    R_level = _rotation_aligning(normal, z_axis)
    z_shift = np.array([0.0, 0.0, centroid[2] - float(config.GROUND_Z_M)])
    b       = centroid - R_level @ centroid - z_shift

    logger.info(
        "Ground leveling: %d/%d inliers (%.0f%%), tilt corrected %.2f°, "
        "ground z %.2f → %.2f m.",
        int(inlier_mask.sum()), len(pts_enu), 100.0 * inlier_frac,
        tilt_deg, float(centroid[2]), float(config.GROUND_Z_M),
    )

    # Apply level ∘ (existing ENU pose) to every frame.
    for f in frames:
        if f.position_enu is None or f.R is None:
            continue
        f.position_enu = R_level @ f.position_enu + b
        _apply_rotation_to_frame(f, f.R @ R_level.T)

    # Composed Sim(3): x ↦ R_level @ (s·R_sim·x + t_sim) + b
    return scale, R_level @ R_sim, R_level @ t_sim + b


def decompose_R_wc(R_wc: np.ndarray) -> tuple[float, float, float] | None:
    """
    Decompose a world-to-camera rotation matrix (in ENU frame) into
    (heading_deg, pitch_deg, roll_deg).

    This is the EXACT analytic inverse of pipeline/pose.py build_rotation() —
    NOT a generic Euler-angle decomposition. build_rotation() is not a plain
    3-matrix Euler product (Rx@Ry@Rz or similar): it computes `fwd` from
    yaw/pitch via explicit ENU trig, derives `right = fwd × world_up`, then
    applies roll as a Rodrigues rotation of `right` *around the fwd axis*
    (roll about the camera's own boresight, not a fixed world axis). A
    generic scipy `as_euler(...)` decomposition assumes rotations about fixed
    axes and does NOT correctly invert this — verified numerically: feeding
    its output back into build_rotation() reproduced a completely different
    matrix (max abs diff ~1.3, not a rounding error). This function instead
    reverses build_rotation()'s actual construction step by step.

    R_world_from_cam's columns are [right | down | fwd] (see build_rotation),
    and R_wc = R_world_from_cam.T, so:
      row 0 of R_wc = right
      row 2 of R_wc = fwd

    Returns None on numerical failure.
    """
    try:
        fwd   = R_wc[2, :].copy()
        right = R_wc[0, :].copy()
        fwd  /= np.linalg.norm(fwd)

        pitch_deg   = math.degrees(math.asin(np.clip(fwd[2], -1.0, 1.0)))
        heading_deg = math.degrees(math.atan2(fwd[0], fwd[1]))

        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fwd, world_up)) > 0.999:
            yaw_rad = math.radians(heading_deg)
            right0  = np.array([math.cos(yaw_rad), -math.sin(yaw_rad), 0.0])
        else:
            right0  = np.cross(fwd, world_up)
            right0 /= np.linalg.norm(right0)

        roll_deg = math.degrees(math.atan2(
            float(np.dot(np.cross(right0, right), fwd)),
            float(np.dot(right, right0)),
        ))
    except Exception:
        return None

    return heading_deg % 360.0, pitch_deg, roll_deg


def _apply_rotation_to_frame(frame, R_wc: np.ndarray) -> None:
    """Decompose R_wc and store heading/pitch/roll/R into the frame."""
    result = decompose_R_wc(R_wc)
    if result is None:
        logger.warning("[%s] rotation decomposition failed", frame.stem[-12:])
        return
    frame.heading_deg      = result[0]
    frame.gimbal_pitch_deg = result[1]
    frame.camera_roll_deg  = result[2]
    frame.R                = R_wc.copy()
    # frame.ready is a computed property (pipeline/frame.py) derived from
    # undistorted/R/position_enu/K_undist — no setter exists, none needed:
    # it already evaluates True once those fields are populated.
