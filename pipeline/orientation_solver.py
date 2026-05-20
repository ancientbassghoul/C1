"""
pipeline/orientation_solver.py – Camera orientation calibration using pyceres.

Jointly optimises:
  Per-camera  : pitch, yaw_offset, roll_offset        (3 × N parameters)
  Van pose    : east, north, heading                  (3 parameters, Z fixed)

Two types of constraints:
  1. Ground scatter (pairwise): for each matched ground feature seen in two
     frames, the ray-ground intersections from both cameras must agree.
     Residual: [ΔEast, ΔNorth] of the two intersection points.

  2. Van corner reprojection: for each visible van corner in a selected frame,
     the 3-D corner (known local coordinates + solved van pose) must project
     to the observed bounding-box pixel.
     Residual: [Δu, Δv] in image space, scaled by VAN_CORNER_WEIGHT.

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


def _van_corner_world(corner_local, van_east, van_north, van_heading_deg, van_z):
    H = math.radians(van_heading_deg)
    s, c = math.sin(H), math.cos(H)
    cx, cy, cz = corner_local
    return np.array([
        van_east  + cx * s - cy * c,
        van_north + cx * c + cy * s,
        van_z     + cz,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Plain-callable cost functors
# ─────────────────────────────────────────────────────────────────────────────

class GroundScatterCost:
    """
    Pairwise ground-scatter cost functor.

    Call signature: (params_i, params_j) -> np.ndarray shape (2,)
    params_* = [pitch, yaw_offset, roll_offset]
    Residuals = [P_i.east - P_j.east,  P_i.north - P_j.north]
    """
    def __init__(self, pixel_i, pixel_j, pos_i, pos_j,
                 K_i, K_j, heading_i, heading_j, roll_i, roll_j,
                 z_ground=0.0):
        self._ui, self._vi = pixel_i
        self._uj, self._vj = pixel_j
        self._pos_i  = pos_i.copy()   # GPS seed — params[3:6] adds correction
        self._pos_j  = pos_j.copy()
        self._Ki_inv = np.linalg.inv(K_i)
        self._Kj_inv = np.linalg.inv(K_j)
        self._hdg_i  = heading_i
        self._hdg_j  = heading_j
        self._roll_i = roll_i
        self._roll_j = roll_j
        self._z      = z_ground

    def __call__(self, params_i, params_j):
        # params layout: [pitch, yaw_off, roll_off, dx, dy, dz]
        pos_i = self._pos_i + np.array([params_i[3], params_i[4], params_i[5]])
        pos_j = self._pos_j + np.array([params_j[3], params_j[4], params_j[5]])
        Pi = _unproject(self._ui, self._vi, self._Ki_inv,
                        self._hdg_i + params_i[1], params_i[0],
                        self._roll_i + params_i[2], pos_i, self._z)
        Pj = _unproject(self._uj, self._vj, self._Kj_inv,
                        self._hdg_j + params_j[1], params_j[0],
                        self._roll_j + params_j[2], pos_j, self._z)
        if Pi is None or Pj is None:
            return np.array([_MISS_PENALTY, _MISS_PENALTY])
        return np.array([Pi[0] - Pj[0], Pi[1] - Pj[1]])


class VanCornerCost:
    """
    Van corner reprojection cost functor.

    Call signature: (camera_params, van_params) -> np.ndarray shape (2,)
    camera_params = [pitch, yaw_offset, roll_offset]
    van_params    = [van_east, van_north, van_heading_deg]
    Residuals     = VAN_CORNER_WEIGHT * [u_pred - u_obs, v_pred - v_obs]
    """
    def __init__(self, pixel_obs, corner_local, K, pos_enu,
                 heading_deg, roll_deg, van_z):
        self._u_obs, self._v_obs = pixel_obs
        self._corner_local = corner_local.copy()
        self._K      = K.copy()
        self._pos    = pos_enu.copy()
        self._hdg    = heading_deg
        self._roll   = roll_deg
        self._van_z  = van_z
        self._weight = config.VAN_CORNER_WEIGHT

    def __call__(self, camera_params, van_params):
        # params layout: [pitch, yaw_off, roll_off, dx, dy, dz]
        yaw   = self._hdg  + camera_params[1]
        pitch = camera_params[0]
        roll  = self._roll + camera_params[2]
        R_cam = build_rotation(yaw, pitch, roll)
        pos   = self._pos + np.array([camera_params[3], camera_params[4], camera_params[5]])

        corner_world = _van_corner_world(
            self._corner_local,
            float(van_params[0]), float(van_params[1]), float(van_params[2]),
            self._van_z,
        )
        p_cam = R_cam @ (corner_world - pos)
        if p_cam[2] <= 1e-6:
            return np.array([_MISS_PENALTY * self._weight,
                             _MISS_PENALTY * self._weight])

        u_pred = self._K[0, 0] * p_cam[0] / p_cam[2] + self._K[0, 2]
        v_pred = self._K[1, 1] * p_cam[1] / p_cam[2] + self._K[1, 2]
        return np.array([(u_pred - self._u_obs) * self._weight,
                         (v_pred - self._v_obs) * self._weight])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def solve(frames, pairwise_features, van_observations, van_pose_init,
          pitch_seeds=None):
    """
    Build and solve the pyceres orientation problem.

    Args
    ----
    frames            : list of Frame objects (all eligible frames).
    pairwise_features : list of (frame_i, pixel_i, frame_j, pixel_j).
    van_observations  : list of (frame, corner_local_3d, pixel_obs).
    van_pose_init     : [van_east, van_north, van_heading_deg].
    pitch_seeds       : optional dict Frame -> float (degrees).  When provided,
                        each camera's pitch search window is centred on the seed
                        (± SOLVER_PITCH_OFFSET) rather than the full physical
                        range.  Supplied by GeoCalib estimates from pose.py.

    Returns
    -------
    cam_results : dict  Frame -> np.ndarray([pitch, yaw_off, roll_off])
    van_pose    : np.ndarray([van_east, van_north, van_heading_deg])
    report      : Ceres BriefReport string
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
        "Building Ceres problem: %d frames, %d ground pairs, %d van observations.",
        len(frames), len(pairwise_features), len(van_observations),
    )

    problem    = pyceres.Problem()
    # Initialise camera parameters from GeoCalib seeds where available
    # Layout: [pitch, yaw_off, roll_off, dx, dy, dz]
    cam_params = {}
    for f in frames:
        seed_pitch = pitch_seeds.get(f, 0.0) if pitch_seeds else 0.0
        cam_params[f] = np.array([seed_pitch, 0.0, 0.0, 0.0, 0.0, 0.0])
    van_params = np.array(van_pose_init, dtype=np.float64)

    ground_loss = pyceres.CauchyLoss(5.0)
    van_loss    = pyceres.HuberLoss(3.0)

    # Ground scatter residuals
    for fi, pixel_i, fj, pixel_j in pairwise_features:
        functor = GroundScatterCost(
            pixel_i, pixel_j,
            fi.position_enu, fj.position_enu,
            fi.K_undist,     fj.K_undist,
            fi.heading_deg,  fj.heading_deg,
            fi.camera_roll_deg, fj.camera_roll_deg,
        )
        cost = _NumericDiff(functor, num_residuals=2, block_sizes=[6, 6])
        problem.add_residual_block(cost, ground_loss,
                                 [cam_params[fi], cam_params[fj]])

    # Van corner residuals
    for frame, corner_local, pixel_obs in van_observations:
        functor = VanCornerCost(
            pixel_obs, corner_local,
            frame.K_undist, frame.position_enu,
            frame.heading_deg, frame.camera_roll_deg,
            van_z=config.VAN_Z_M,
        )
        cost = _NumericDiff(functor, num_residuals=2, block_sizes=[6, 3])
        problem.add_residual_block(cost, van_loss,
                                 [cam_params[frame], van_params])

    # Camera bounds — pitch window centred on GeoCalib seed, clamped to floor/ceiling
    for f in frames:
        p          = cam_params[f]
        seed_pitch = float(p[0])   # already set from GeoCalib
        p_lo = max(seed_pitch - config.SOLVER_PITCH_OFFSET, config.SOLVER_PITCH_FLOOR)
        p_hi = min(seed_pitch + config.SOLVER_PITCH_OFFSET, config.SOLVER_PITCH_CEILING)
        problem.set_parameter_lower_bound(p, 0, p_lo)
        problem.set_parameter_upper_bound(p, 0, p_hi)
        problem.set_parameter_lower_bound(p, 1, -config.SOLVER_YAW_OFFSET_RANGE)
        problem.set_parameter_upper_bound(p, 1,  config.SOLVER_YAW_OFFSET_RANGE)
        problem.set_parameter_lower_bound(p, 2, -config.SOLVER_ROLL_OFFSET_RANGE)
        problem.set_parameter_upper_bound(p, 2,  config.SOLVER_ROLL_OFFSET_RANGE)
        # Position correction bounds (dx, dy horizontal; dz vertical)
        problem.set_parameter_lower_bound(p, 3, -config.SOLVER_POSITION_RANGE_H)
        problem.set_parameter_upper_bound(p, 3,  config.SOLVER_POSITION_RANGE_H)
        problem.set_parameter_lower_bound(p, 4, -config.SOLVER_POSITION_RANGE_H)
        problem.set_parameter_upper_bound(p, 4,  config.SOLVER_POSITION_RANGE_H)
        problem.set_parameter_lower_bound(p, 5, -config.SOLVER_POSITION_RANGE_V)
        problem.set_parameter_upper_bound(p, 5,  config.SOLVER_POSITION_RANGE_V)
        logger.debug("[%s] pitch window: [%.1f°, %.1f°]  seed=%.1f°",
                     f.stem[-12:], p_lo, p_hi, seed_pitch)

    # Van heading bounds (east/north are free)
    h_lo = config.VAN_HEADING_PRIOR_DEG - config.VAN_HEADING_RANGE_DEG
    h_hi = config.VAN_HEADING_PRIOR_DEG + config.VAN_HEADING_RANGE_DEG
    problem.set_parameter_lower_bound(van_params, 2, h_lo)
    problem.set_parameter_upper_bound(van_params, 2, h_hi)

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

    return cam_params, van_params, report
