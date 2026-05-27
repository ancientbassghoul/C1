"""
pipeline/orientation_solver.py – Camera orientation calibration using pyceres.

Jointly optimises:
  Per-camera  : pitch, yaw_offset, roll_offset, dx, dy, dz   (6 × N parameters)
  Van heading : van_heading_deg                               (1 parameter)

Two types of constraints:
  1. Ground scatter (pairwise): for each matched ground feature seen in two
     frames, the ray-ground intersections from both cameras must agree.
     Residual: [ΔEast, ΔNorth] of the two intersection points.

  2. Van plane features (manual): wheel centres at known Z constrain wheelbase
     distance and axis direction; roof marks constrain roof-plane scatter;
     roof-edge pairs constrain lateral width.  Provided via the manual
     correspondence JSON (type: wheel_axis, roof, roof_edge).

pyceres cost functions are subclasses of pyceres.CostFunction that implement
evaluate(parameters, residuals, jacobians).  Jacobians are computed via central
finite differences inside a generic _NumericDiff wrapper so that the cost
functors stay as plain Python callables.

Install pyceres:
  venv\\Scripts\\pip install pyceres
"""

from __future__ import annotations

import logging
import math

import numpy as np

import config
from pipeline.pose import build_rotation

logger = logging.getLogger(__name__)

_MISS_PENALTY = 1000.0
_FD_EPS       = 1e-5   # central-difference step size


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers (plain Python / numpy — no pyceres dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _ray_ground(pos, d_world, z):
    dz = float(d_world[2])
    if abs(dz) < 1e-9:
        return None
    t = (z - float(pos[2])) / dz
    if t <= 0:
        return None
    return pos + t * d_world


def _unproject(u, v, K_inv, yaw, pitch, roll, pos, z_ground):
    d_cam   = K_inv @ np.array([u, v, 1.0])
    R       = build_rotation(yaw, pitch, roll)
    d_world = R.T @ d_cam
    norm    = np.linalg.norm(d_world)
    if norm < 1e-12:
        return None
    d_world = d_world / norm
    return _ray_ground(pos, d_world, z_ground)


# ─────────────────────────────────────────────────────────────────────────────
# Plain-callable cost functors
# ─────────────────────────────────────────────────────────────────────────────

def _k_inv_from_f(f, cx, cy):
    """Analytical K_inv for a camera with focal length f and principal point cx,cy."""
    return np.array([
        [1.0/f,   0.0, -cx/f],
        [  0.0, 1.0/f, -cy/f],
        [  0.0,   0.0,   1.0],
    ])


class GroundScatterCost:
    """
    Pairwise ground-scatter cost functor.

    Call signature: (params_i, params_j) or
                    (params_i, params_j, focal_params)
    params_* = [pitch, yaw_offset, roll_offset, dx, dy, dz]
    focal_params = [focal_length]  (optional shared block)
    Residuals = [P_i.east - P_j.east,  P_i.north - P_j.north]
    """
    def __init__(self, pixel_i, pixel_j, pos_i, pos_j,
                 K_i, K_j, heading_i, heading_j, roll_i, roll_j,
                 z_ground=None):
        self._ui, self._vi = pixel_i
        self._uj, self._vj = pixel_j
        self._pos_i  = pos_i.copy()
        self._pos_j  = pos_j.copy()
        # Store fixed K_inv for when focal length is not free
        self._Ki_inv_fixed = np.linalg.inv(K_i)
        self._Kj_inv_fixed = np.linalg.inv(K_j)
        # Store cx/cy for analytical K_inv recomputation when f is free
        self._cx_i = float(K_i[0, 2]);  self._cy_i = float(K_i[1, 2])
        self._cx_j = float(K_j[0, 2]);  self._cy_j = float(K_j[1, 2])
        self._hdg_i  = heading_i
        self._hdg_j  = heading_j
        self._roll_i = roll_i
        self._roll_j = roll_j
        self._z      = z_ground if z_ground is not None else config.GROUND_Z_M

    def __call__(self, params_i, params_j, focal_params=None):
        # params layout: [pitch, yaw_off, roll_off, dx, dy, dz]
        pos_i = self._pos_i + np.array([params_i[3], params_i[4], params_i[5]])
        pos_j = self._pos_j + np.array([params_j[3], params_j[4], params_j[5]])
        if focal_params is not None:
            f      = float(focal_params[0])
            Ki_inv = _k_inv_from_f(f, self._cx_i, self._cy_i)
            Kj_inv = _k_inv_from_f(f, self._cx_j, self._cy_j)
        else:
            Ki_inv = self._Ki_inv_fixed
            Kj_inv = self._Kj_inv_fixed
        Pi = _unproject(self._ui, self._vi, Ki_inv,
                        self._hdg_i + params_i[1], params_i[0],
                        self._roll_i + params_i[2], pos_i, self._z)
        Pj = _unproject(self._uj, self._vj, Kj_inv,
                        self._hdg_j + params_j[1], params_j[0],
                        self._roll_j + params_j[2], pos_j, self._z)
        if Pi is None or Pj is None:
            return np.array([_MISS_PENALTY, _MISS_PENALTY])
        return np.array([Pi[0] - Pj[0], Pi[1] - Pj[1]])




# ─────────────────────────────────────────────────────────────────────────────
# Van feature constraints (plane scatter + wheel distance)
# ─────────────────────────────────────────────────────────────────────────────

class PlaneScatterCost(GroundScatterCost):
    """
    Pairwise scatter cost for points on an arbitrary horizontal plane Z=z_plane.

    Identical to GroundScatterCost — inherits everything, just passes a
    different z_ground.  Used for van roof (z=VAN_HEIGHT_M) and wheel
    centres (z=WHEEL_RADIUS_M).
    """
    # No override needed — GroundScatterCost already accepts z_ground.


class WheelDistanceCost:
    """
    Metric distance constraint between two wheel centres in the same frame.

    Given two pixels in the same frame that are wheel centres at height
    z=WHEEL_RADIUS_M, their back-projected world positions must be exactly
    *distance_m* apart (track width or wheelbase).

    Call signature: (cam_params,) -> np.ndarray shape (1,)
    """
    def __init__(self, pixel_a, pixel_b, pos, K,
                 heading_deg, roll_deg, z_wheel, distance_m):
        self._ua, self._va = pixel_a
        self._ub, self._vb = pixel_b
        self._pos          = pos.copy()
        self._K_inv        = np.linalg.inv(K)
        self._cx           = float(K[0, 2])
        self._cy           = float(K[1, 2])
        self._hdg          = heading_deg
        self._roll         = roll_deg
        self._z            = z_wheel
        self._dist         = distance_m

    def __call__(self, cam_params, focal_params=None):
        pos = self._pos + np.array([cam_params[3], cam_params[4], cam_params[5]])
        if focal_params is not None:
            f      = float(focal_params[0])
            K_inv  = np.array([[1/f, 0, -self._cx/f],
                                [0, 1/f, -self._cy/f],
                                [0,   0,           1]])
        else:
            K_inv = self._K_inv

        Pa = _unproject(self._ua, self._va, K_inv,
                        self._hdg + cam_params[1], cam_params[0],
                        self._roll + cam_params[2], pos, self._z)
        Pb = _unproject(self._ub, self._vb, K_inv,
                        self._hdg + cam_params[1], cam_params[0],
                        self._roll + cam_params[2], pos, self._z)
        if Pa is None or Pb is None:
            return np.array([_MISS_PENALTY])
        dist = float(np.linalg.norm(Pa - Pb))
        return np.array([dist - self._dist])


class AxisPairCost:
    """
    Within-frame constraint for a pair of points on the van at known height.

    Residuals (3):
      [0]  distance_error = |P_b - P_a| - distance_m          (metres)
      [1]  direction_x   = cross(diff_unit, axis_unit).x * dist (metres)
      [2]  direction_y   = cross(diff_unit, axis_unit).y * dist (metres)

    Call: (cam_params[6], van_params[3], [focal_params[1]]) -> (3,)
    """
    def __init__(self, pixel_a, pixel_b, pos, K,
                 heading_deg, roll_deg, z_plane, distance_m, axis="forward"):
        self._ua, self._va = pixel_a
        self._ub, self._vb = pixel_b
        self._pos   = pos.copy()
        self._K_inv = np.linalg.inv(K)
        self._cx    = float(K[0, 2])
        self._cy    = float(K[1, 2])
        self._hdg   = heading_deg
        self._roll  = roll_deg
        self._z     = z_plane
        self._dist  = distance_m
        self._axis  = axis   # "forward" or "lateral"

    def __call__(self, cam_params, van_params, focal_params=None):
        pos   = self._pos + cam_params[3:6]
        K_inv = (_k_inv_from_f(float(focal_params[0]), self._cx, self._cy)
                 if focal_params is not None else self._K_inv)

        Pa = _unproject(self._ua, self._va, K_inv,
                        self._hdg + cam_params[1], cam_params[0],
                        self._roll + cam_params[2], pos, self._z)
        Pb = _unproject(self._ub, self._vb, K_inv,
                        self._hdg + cam_params[1], cam_params[0],
                        self._roll + cam_params[2], pos, self._z)

        if Pa is None or Pb is None:
            return np.array([_MISS_PENALTY] * 3)

        diff2d  = (Pb - Pa)[:2]
        horiz   = float(np.linalg.norm(diff2d))
        dist_res = horiz - self._dist

        if horiz < 1e-6:
            return np.array([dist_res, _MISS_PENALTY, _MISS_PENALTY])

        diff_unit = diff2d / horiz

        # Van heading in ENU: forward = (sin hdg, cos hdg)
        van_hdg_rad = math.radians(float(van_params[2]))
        if self._axis == "forward":
            axis_unit = np.array([math.sin(van_hdg_rad), math.cos(van_hdg_rad)])
        else:   # lateral — perpendicular to forward
            axis_unit = np.array([math.cos(van_hdg_rad), -math.sin(van_hdg_rad)])

        # 2D cross product (scalar): measures misalignment
        cross = diff_unit[0]*axis_unit[1] - diff_unit[1]*axis_unit[0]
        # Scale to metres so residuals are comparable
        dir_res = cross * self._dist

        return np.array([dist_res, dir_res, 0.0])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def solve(frames, pairwise_features, pitch_seeds=None,
          van_plane_features=None,
          cam_params_init=None,
          van_params_init=None,
          calibrated_frames=None,
          delta_rotation_bound=None,
          delta_position_bound=None):
    """
    Build and solve the pyceres orientation problem.

    Args
    ----
    frames              : list of Frame objects (all eligible frames).
    pairwise_features   : list of (frame_i, pixel_i, frame_j, pixel_j).
    pitch_seeds         : optional dict Frame -> float (degrees).  When provided,
                          each camera's pitch search window is centred on the seed
                          (± SOLVER_PITCH_OFFSET) rather than the full physical
                          range.  Supplied by GeoCalib estimates from pose.py.
    cam_params_init     : optional dict Frame -> np.ndarray([p, yo, ro, dx, dy, dz]).
                          When provided, initialises cam_params from these values
                          instead of from GeoCalib seeds.  Used for stage-2 solves
                          so that LightGlue refinement starts from the stage-1 result.
    delta_rotation_bound: if set, clamp each rotation offset ± this many degrees
                          AROUND its initialised value (stage-2 tight bounds).
    delta_position_bound: if set, clamp each position offset ± this many metres
                          AROUND its initialised value (stage-2 tight bounds).

    Returns
    -------
    cam_results : dict  Frame -> np.ndarray([pitch, yaw_off, roll_off, dx, dy, dz])
    van_pose    : np.ndarray([0, 0, van_heading_deg])  — heading solved from axis pairs
    report      : Ceres BriefReport string
    focal_params: np.ndarray([focal_length]) (undistorted px)
    """
    try:
        import pyceres
    except ImportError:
        raise ImportError(
            "\npyceres not installed. Run:\n"
            "  venv\\Scripts\\pip install pyceres\n"
        )

    # ── Generic numeric-diff wrapper ──────────────────────────────────────────
    # Defined here so it can subclass pyceres.CostFunction after import.
    class _NumericDiff(pyceres.CostFunction):
        """
        Wraps a plain Python callable as a pyceres.CostFunction.
        Jacobians are computed via central finite differences.

        callable : (*param_blocks) -> np.ndarray of shape (num_residuals,)
        """
        def __init__(self, callable_cost, num_residuals, block_sizes):
            super().__init__()
            self._cost        = callable_cost
            self._block_sizes = block_sizes
            self.set_num_residuals(num_residuals)
            self.set_parameter_block_sizes(block_sizes)

        def Evaluate(self, parameters, residuals, jacobians):
            # parameters is a list of numpy arrays (one per block)
            res = self._cost(*parameters)
            residuals[:] = res

            if jacobians is not None:
                for bi, bs in enumerate(self._block_sizes):
                    if jacobians[bi] is None:
                        continue
                    jac = jacobians[bi]          # flat (num_residuals * bs,), row-major
                    for k in range(bs):
                        p_plus  = [p.copy() for p in parameters]
                        p_minus = [p.copy() for p in parameters]
                        p_plus[bi][k]  += _FD_EPS
                        p_minus[bi][k] -= _FD_EPS
                        jac[k::bs] = (
                            self._cost(*p_plus) - self._cost(*p_minus)
                        ) / (2.0 * _FD_EPS)

            return True

    # ── Problem setup ─────────────────────────────────────────────────────────
    logger.info(
        "Building Ceres problem: %d frames, %d ground pairs.",
        len(frames), len(pairwise_features),
    )

    problem    = pyceres.Problem()
    # Initialise camera parameters.
    # Layout: [pitch, yaw_off, roll_off, dx, dy, dz]
    # If cam_params_init is given (stage-2), start from those values.
    # Otherwise start from GeoCalib pitch seeds with zero offsets.
    cam_params = {}
    for f in frames:
        if cam_params_init and f in cam_params_init:
            cam_params[f] = cam_params_init[f].copy()
        else:
            seed_pitch = pitch_seeds.get(f, 0.0) if pitch_seeds else 0.0
            cam_params[f] = np.array([seed_pitch, 0.0, 0.0, 0.0, 0.0, 0.0])
    # van_params: [van_east, van_north, van_heading_deg]
    # east/north are unconstrained placeholders; heading is solved by AxisPairCost.
    van_params = (np.array(van_params_init, dtype=np.float64)
                  if van_params_init is not None
                  else np.zeros(3, dtype=np.float64))

    ground_loss = pyceres.CauchyLoss(5.0)

    # Track which frames have at least one residual so we only set bounds on
    # those — Ceres rejects set_parameter_*_bound on unknown parameter blocks.
    frames_in_problem: set = set()

    # Optional shared focal length parameter
    # Solver works in undistorted image space where
    # f_undist = FOCAL_LENGTH * UNDISTORT_SCALE (see undistort.py build_K_new).
    # After solving, convert back: FOCAL_LENGTH = f_undist / UNDISTORT_SCALE.
    estimate_f    = config.ESTIMATE_FOCAL_LENGTH
    f_undist_seed = config.FOCAL_LENGTH * config.UNDISTORT_SCALE
    focal_params  = np.array([f_undist_seed], dtype=np.float64)

    # Ground scatter residuals
    for fi, pixel_i, fj, pixel_j in pairwise_features:
        functor = GroundScatterCost(
            pixel_i, pixel_j,
            fi.position_enu, fj.position_enu,
            fi.K_undist,     fj.K_undist,
            fi.heading_deg,  fj.heading_deg,
            fi.camera_roll_deg, fj.camera_roll_deg,
            z_ground=config.GROUND_Z_M,
        )
        if estimate_f:
            cost = _NumericDiff(functor, num_residuals=2, block_sizes=[6, 6, 1])
            problem.add_residual_block(cost, ground_loss,
                                     [cam_params[fi], cam_params[fj], focal_params])
        else:
            cost = _NumericDiff(functor, num_residuals=2, block_sizes=[6, 6])
            problem.add_residual_block(cost, ground_loss,
                                     [cam_params[fi], cam_params[fj]])
        frames_in_problem.add(fi)
        frames_in_problem.add(fj)

    # Van plane feature residuals (roof scatter + wheel distance)
    plane_loss = pyceres.HuberLoss(2.0)
    wheel_loss = pyceres.HuberLoss(0.10)
    frame_by_stem = {f.stem: f for f in frames}
    if van_plane_features:
        for feat in van_plane_features:
            feat_type  = feat.get("type", "ground")
            z_plane    = float(feat.get("z_plane", config.GROUND_Z_M))
            is_pair    = feat.get("is_pair", False)
            distance_m = float(feat.get("distance_m", 1.0)) if is_pair else None
            axis       = feat.get("axis", "forward")
            points     = feat.get("points", {})

            if is_pair:
                # Each frame contributes [[xa,ya],[xb,yb]]
                # Per-frame: AxisPairCost (distance + direction vs van heading)
                # Cross-frame: PlaneScatterCost on A-points and B-points separately
                pair_frames = []
                for stem, pts in points.items():
                    f = frame_by_stem.get(stem)
                    if f is None or len(pts) < 2:
                        continue
                    pix_a = tuple(pts[0]);  pix_b = tuple(pts[1])
                    pair_frames.append((f, pix_a, pix_b))
                    functor = AxisPairCost(
                        pix_a, pix_b, f.position_enu, f.K_undist,
                        f.heading_deg, f.camera_roll_deg,
                        z_plane, distance_m, axis,
                    )
                    if estimate_f:
                        cost = _NumericDiff(functor, num_residuals=3,
                                           block_sizes=[6, 3, 1])
                        problem.add_residual_block(cost, plane_loss,
                                                 [cam_params[f], van_params, focal_params])
                    else:
                        cost = _NumericDiff(functor, num_residuals=3,
                                           block_sizes=[6, 3])
                        problem.add_residual_block(cost, plane_loss,
                                                 [cam_params[f], van_params])
                    frames_in_problem.add(f)
                # Cross-frame scatter: same physical endpoints seen in multiple frames
                for i in range(len(pair_frames)):
                    for j in range(i + 1, len(pair_frames)):
                        fi, pa_i, pb_i = pair_frames[i]
                        fj, pa_j, pb_j = pair_frames[j]
                        for pix_i, pix_j in [(pa_i, pa_j), (pb_i, pb_j)]:
                            functor = PlaneScatterCost(
                                pix_i, pix_j,
                                fi.position_enu, fj.position_enu,
                                fi.K_undist, fj.K_undist,
                                fi.heading_deg, fj.heading_deg,
                                fi.camera_roll_deg, fj.camera_roll_deg,
                                z_ground=z_plane,
                            )
                            if estimate_f:
                                cost = _NumericDiff(functor, num_residuals=2,
                                                   block_sizes=[6, 6, 1])
                                problem.add_residual_block(cost, plane_loss,
                                                         [cam_params[fi], cam_params[fj],
                                                          focal_params])
                            else:
                                cost = _NumericDiff(functor, num_residuals=2,
                                                   block_sizes=[6, 6])
                                problem.add_residual_block(cost, plane_loss,
                                                         [cam_params[fi], cam_params[fj]])
                            frames_in_problem.add(fi)
                            frames_in_problem.add(fj)
            else:
                # Single-point per frame — plain plane scatter
                feat_frames = [(frame_by_stem[s], tuple(pt))
                               for s, pt in points.items() if s in frame_by_stem]
                for i in range(len(feat_frames)):
                    for j in range(i + 1, len(feat_frames)):
                        fi, pix_i = feat_frames[i]
                        fj, pix_j = feat_frames[j]
                        functor = PlaneScatterCost(
                            pix_i, pix_j,
                            fi.position_enu, fj.position_enu,
                            fi.K_undist, fj.K_undist,
                            fi.heading_deg, fj.heading_deg,
                            fi.camera_roll_deg, fj.camera_roll_deg,
                            z_ground=z_plane,
                        )
                        if estimate_f:
                            cost = _NumericDiff(functor, num_residuals=2,
                                               block_sizes=[6, 6, 1])
                            problem.add_residual_block(cost, plane_loss,
                                                     [cam_params[fi], cam_params[fj],
                                                      focal_params])
                        else:
                            cost = _NumericDiff(functor, num_residuals=2,
                                               block_sizes=[6, 6])
                            problem.add_residual_block(cost, plane_loss,
                                                     [cam_params[fi], cam_params[fj]])
                        frames_in_problem.add(fi)
                        frames_in_problem.add(fj)
        logger.info("Added %d van plane feature constraint(s).", len(van_plane_features))

    n_unconstrained = len(set(frames) - frames_in_problem)
    if n_unconstrained:
        logger.warning(
            "%d frame(s) have no residuals in this solve — their camera params "
            "will remain at initial values (no bounds set).", n_unconstrained,
        )

    # Camera bounds — only for frames that appear in at least one residual block.
    # Calibrated frames (had manual correspondences in stage-1) get tight delta
    # bounds so LightGlue can only make small adjustments.
    # New frames (GeoCalib seeds only) get the normal wide bounds so they can
    # actually move to where the geometry pulls them.
    for f in frames:
        if f not in frames_in_problem:
            continue
        p          = cam_params[f]
        is_cal     = (calibrated_frames is None) or (f in calibrated_frames)

        if is_cal and delta_rotation_bound is not None:
            # Tight window AROUND the stage-1 value
            dr = delta_rotation_bound
            problem.set_parameter_lower_bound(p, 0, float(p[0]) - dr)
            problem.set_parameter_upper_bound(p, 0, float(p[0]) + dr)
            problem.set_parameter_lower_bound(p, 1, float(p[1]) - dr)
            problem.set_parameter_upper_bound(p, 1, float(p[1]) + dr)
            problem.set_parameter_lower_bound(p, 2, float(p[2]) - dr)
            problem.set_parameter_upper_bound(p, 2, float(p[2]) + dr)
        else:
            # Normal wide bounds centred on GeoCalib seed (or stage-1 value
            # for new frames initialised from cam_params_init)
            seed_pitch = float(p[0])
            p_lo = max(seed_pitch - config.SOLVER_PITCH_OFFSET, config.SOLVER_PITCH_FLOOR)
            p_hi = min(seed_pitch + config.SOLVER_PITCH_OFFSET, config.SOLVER_PITCH_CEILING)
            problem.set_parameter_lower_bound(p, 0, p_lo)
            problem.set_parameter_upper_bound(p, 0, p_hi)
            problem.set_parameter_lower_bound(p, 1, -config.SOLVER_YAW_OFFSET_RANGE)
            problem.set_parameter_upper_bound(p, 1,  config.SOLVER_YAW_OFFSET_RANGE)
            problem.set_parameter_lower_bound(p, 2, -config.SOLVER_ROLL_OFFSET_RANGE)
            problem.set_parameter_upper_bound(p, 2,  config.SOLVER_ROLL_OFFSET_RANGE)
            logger.debug("[%s] pitch window: [%.1f°, %.1f°]  seed=%.1f°",
                         f.stem[-12:], p_lo, p_hi, seed_pitch)

        if is_cal and delta_position_bound is not None:
            dp = delta_position_bound
            for idx in (3, 4, 5):
                problem.set_parameter_lower_bound(p, idx, float(p[idx]) - dp)
                problem.set_parameter_upper_bound(p, idx, float(p[idx]) + dp)
        else:
            problem.set_parameter_lower_bound(p, 3, -config.SOLVER_POSITION_RANGE_H)
            problem.set_parameter_upper_bound(p, 3,  config.SOLVER_POSITION_RANGE_H)
            problem.set_parameter_lower_bound(p, 4, -config.SOLVER_POSITION_RANGE_H)
            problem.set_parameter_upper_bound(p, 4,  config.SOLVER_POSITION_RANGE_H)
            problem.set_parameter_lower_bound(p, 5, -config.SOLVER_POSITION_RANGE_V)
            problem.set_parameter_upper_bound(p, 5,  config.SOLVER_POSITION_RANGE_V)

    # Focal length bounds
    if estimate_f:
        f_lo = f_undist_seed * (1.0 - config.FOCAL_LENGTH_RANGE)
        f_hi = f_undist_seed * (1.0 + config.FOCAL_LENGTH_RANGE)
        problem.set_parameter_lower_bound(focal_params, 0, f_lo)
        problem.set_parameter_upper_bound(focal_params, 0, f_hi)
        logger.info(
            "Focal length: seed=%.1fpx (undist)  bounds=[%.1f, %.1f]  "
            "(raw FOCAL_LENGTH seed=%.1fpx)",
            f_undist_seed, f_lo, f_hi, config.FOCAL_LENGTH)

    # Solver options
    options = pyceres.SolverOptions()
    options.linear_solver_type         = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations         = config.SOLVER_MAX_ITERATIONS
    options.function_tolerance         = 1e-6
    options.gradient_tolerance         = 1e-8
    options.parameter_tolerance        = 1e-8

    logger.info("Starting Ceres orientation solve…")
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)

    report = summary.BriefReport()
    logger.info("Ceres solve complete: %s", report)

    if estimate_f:
        f_undist_solved = float(focal_params[0])
        f_raw_solved    = f_undist_solved / config.UNDISTORT_SCALE
        f_delta_pct     = 100.0 * (f_raw_solved - config.FOCAL_LENGTH) / config.FOCAL_LENGTH
        logger.info(
            "Focal length solved: %.2f px (undist)  →  %.2f px (raw)\n"
            "  seed was %.2f px (raw)  delta %+.2f px / %+.1f%%\n"
            "  → Update FOCAL_LENGTH = %.2f in config.py",
            f_undist_solved, f_raw_solved,
            config.FOCAL_LENGTH,
            f_raw_solved - config.FOCAL_LENGTH, f_delta_pct,
            f_raw_solved,
        )

    return cam_params, van_params, report, focal_params
