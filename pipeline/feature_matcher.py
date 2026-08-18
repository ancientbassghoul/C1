"""
pipeline/feature_matcher.py – MASt3R-SfM orchestration + Ceres full-BA.

Replaces LightGlue / SuperPoint / GroundedSAM with:
  1. MASt3R-SfM complete-graph run  → camera poses + dense 3D point cloud
  2. Qwen VL + CLIP anchor detection → CLIP-weighted anchor rays
  3. Single Ceres full-BA            → jointly refines cameras + 3D points
  4. Umeyama Sim(3)                  → maps MASt3R frame to GPS/ENU

Manual correspondences (--manual-fm-json) are loaded and added as
higher-weight reprojection residuals alongside MASt3R observations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.frame import Frame
import config

logger = logging.getLogger(__name__)


@dataclass
class RefineResult:
    """
    Bundle of intermediate solve products from refine_pitches(), returned so
    callers can optionally build the debug .glb export (--export-mesh)
    without re-running MASt3R/Ceres. None of this is needed for the normal
    pipeline path — raycast.py's default flow ignores the return value.
    """
    mast3r_result: object               # MASt3RResult — carries .dense when keep_dense=True
    points_3d_solved: np.ndarray        # Ceres-refined sparse points, MASt3R frame (pre-Sim3)
    sim3: tuple[float, np.ndarray, np.ndarray] | None  # (scale, R_sim, t_sim), MASt3R frame → ENU (Ceres-leveled)
    mast3r_cameras_enu: dict = None     # MASt3R seed cameras in ENU (before Ceres) — same schema as solved_cameras.json
    mast3r_sim3: tuple[float, np.ndarray, np.ndarray] | None = None  # MASt3R-native composed Sim(3) (its own ground leveled)


def surface_points_enu(
    refine_result, source: str, which: str = "ceres",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Build the ENU ground-point cloud(s) for surface-aware raycast from a solve.

    Returns (sparse_enu, dense_enu) — each (N,3) in ENU metres or None — selected
    by two orthogonal axes:

      source ("sparse" | "dense" | "both") — which cloud type to build.
      which  ("ceres" | "native")          — which solve frame to map into ENU:
        - "ceres":  sim3 = refine_result.sim3 (composed, Ceres-leveled), sparse =
                    points_3d_solved (Ceres-refined). Lines up with solved_cameras.json.
        - "native": sim3 = refine_result.mast3r_sim3 (MASt3R's own ground leveled),
                    sparse = mast3r_result.points_3d (raw). Lines up with
                    solved_camera_mast3r.json.

    dense = all valid MASt3R dense pointmap samples (same points either way),
    transformed by the chosen sim3 — needs robust filtering at GroundSurface
    build time. Returns (None, None) if the chosen sim3 is unavailable.
    """
    from pipeline.mast3r_matcher import _apply_sim3

    if refine_result is None:
        return None, None

    if which == "native":
        sim3 = refine_result.mast3r_sim3
        sparse_src = getattr(refine_result.mast3r_result, "points_3d", None)
    else:
        sim3 = refine_result.sim3
        sparse_src = refine_result.points_3d_solved

    if sim3 is None:
        return None, None
    scale, R_sim, t_sim = sim3

    sparse = dense = None
    if source in ("sparse", "both") and sparse_src is not None:
        pts = np.asarray(sparse_src)
        if len(pts):
            sparse = _apply_sim3(pts, scale, R_sim, t_sim)

    if source in ("dense", "both"):
        chunks = []
        dense_dict = getattr(refine_result.mast3r_result, "dense", {}) or {}
        for _stem, (pts3d_hw3, _img, valid) in dense_dict.items():
            v = pts3d_hw3[valid]
            if len(v):
                chunks.append(np.asarray(v, dtype=np.float64))
        if chunks:
            dense = _apply_sim3(np.concatenate(chunks, axis=0), scale, R_sim, t_sim)

    return sparse, dense


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_manual_features(frames: list[Frame]) -> list[dict]:
    """
    Load all correspondences from MANUAL_CORRESPONDENCES_FILE.

    Returns a list of feature dicts in the format expected by
    orientation_solver.ceres_solve(manual_features=...).
    """
    path = Path(config.MANUAL_CORRESPONDENCES_FILE)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load manual correspondences: %s", exc)
        return []

    stem_set = {f.stem for f in frames}
    features = []
    for entry in data.get("correspondences", []):
        pts = entry.get("points", {})
        filtered_pts = {s: p for s, p in pts.items() if s in stem_set}
        if len(filtered_pts) < 1:
            continue
        feat = dict(entry)
        feat["points"] = filtered_pts
        features.append(feat)

    logger.info("Loaded %d manual feature(s) from %s", len(features), path)
    return features


def _build_mast3r_cameras_enu(frames, w2c_poses: dict, sim3) -> dict:
    """
    Apply the Ceres-derived Sim(3) to MASt3R's initial W2C matrices and
    return a camera dict in the same schema as solved_cameras.json.

    Using the same Sim(3) as the Ceres-solved export keeps both files in the
    same ENU frame so positions are directly comparable.
    """
    if sim3 is None:
        return {}

    from pipeline.orientation_solver import decompose_R_wc

    scale, R_sim, t_sim = sim3
    cameras = {}
    for f in frames:
        w2c = w2c_poses.get(f.stem)
        if w2c is None or f.K_undist is None:
            continue
        R34 = w2c[:3, :3]
        t3  = w2c[:3, 3]

        C_enu    = scale * R_sim @ (-R34.T @ t3) + t_sim
        R_enu_wc = R34 @ R_sim.T
        angles   = decompose_R_wc(R_enu_wc)
        if angles is None:
            logger.warning("[%s] MASt3R camera rotation decomposition failed", f.stem[-12:])
            continue
        heading_deg, pitch_deg, roll_deg = angles

        cameras[f.stem] = {
            "position_enu":     C_enu.tolist(),
            "heading_deg":      heading_deg,
            "gimbal_pitch_deg": pitch_deg,
            "camera_roll_deg":  roll_deg,
            "K_undist":         f.K_undist.tolist(),
        }
    return cameras


def _build_anchor_rays(anchor_result) -> list[tuple]:
    """
    Convert AnchorResult into a list of (frame_stem, pixel_uv, clip_weight)
    tuples for orientation_solver.ceres_solve().
    """
    if anchor_result is None:
        return []

    rays = []
    for stem, centroid in anchor_result.centroids.items():
        w = anchor_result.weights.get(stem, 0.0)
        u, v = centroid
        rays.append((stem, np.array([float(u), float(v)]), float(w)))

    logger.info(
        "Anchor rays: %d frames  (label='%s'  %.0f%% above threshold %.2f)",
        len(rays), anchor_result.label,
        100.0 * sum(1 for *_, w in rays if w >= config.CLIP_ANCHOR_THRESHOLD) / max(1, len(rays)),
        config.CLIP_ANCHOR_THRESHOLD,
    )
    return rays


def _init_p_anchor(
    mast3r_result,
    anchor_result,
    cam_poses_init: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Initialize P_anchor by projecting MASt3R 3D points into frames and
    finding the centroid of points that land inside the anchor bbox in ≥2 frames.

    Falls back to the origin if no suitable points are found.
    """
    if (anchor_result is None
            or not anchor_result.bboxes
            or len(mast3r_result.points_3d) == 0):
        logger.warning("Cannot initialize P_anchor — returning world origin.")
        return np.zeros(3, dtype=np.float64)

    from scipy.spatial.transform import Rotation as _R

    def _aa_to_R(r3):
        angle = float(np.linalg.norm(r3))
        if angle < 1e-12:
            return np.eye(3)
        return _R.from_rotvec(r3).as_matrix()

    pts = mast3r_result.points_3d   # (N, 3)
    votes = np.zeros(len(pts), dtype=int)

    for stem, bbox in anchor_result.bboxes.items():
        W2C = cam_poses_init.get(stem)
        if W2C is None:
            continue
        # We stored W2C as 4×4; extract R and t from it, or use cam6 format
        # (orientation_solver uses axis-angle; cam_poses_init are 4×4 matrices)
        R34 = W2C[:3, :3]
        t3  = W2C[:3, 3]

        # Project all 3D points into this frame
        p_cam = (R34 @ pts.T + t3[:, None]).T   # (N, 3)
        in_front = p_cam[:, 2] > 1e-3

        # Need a K matrix — use a simple default from config
        fx = config.FOCAL_LENGTH * config.UNDISTORT_SCALE
        fy = fx
        cx = config.CX
        cy = config.CY

        u_proj = fx * p_cam[:, 0] / np.maximum(p_cam[:, 2], 1e-6) + cx
        v_proj = fy * p_cam[:, 1] / np.maximum(p_cam[:, 2], 1e-6) + cy

        y1, x1, y2, x2 = bbox
        in_bbox = (
            in_front
            & (u_proj >= x1) & (u_proj <= x2)
            & (v_proj >= y1) & (v_proj <= y2)
        )
        votes[in_bbox] += 1

    candidate_mask = votes >= 2
    if candidate_mask.sum() == 0:
        logger.warning(
            "P_anchor init: no 3D points visible in ≥2 anchor bboxes. "
            "Using mean of single-frame candidates."
        )
        candidate_mask = votes >= 1
        if candidate_mask.sum() == 0:
            return np.zeros(3, dtype=np.float64)

    p_anchor = pts[candidate_mask].mean(axis=0)
    logger.info(
        "P_anchor initialized from centroid of %d candidate 3D points: %s",
        candidate_mask.sum(), np.round(p_anchor, 2),
    )
    return p_anchor


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def refine_pitches(
    frames: list[Frame],
    anchor_result=None,
    use_manual_features: bool = False,
    matcher_only: bool = False,
    save_mast3r_images: str | None = None,
    keep_dense: bool = False,
    export_raw_glb_dir: str | None = None,
) -> RefineResult | None:
    """
    Refine camera poses using MASt3R-SfM + Ceres full bundle adjustment.

    Steps
    -----
    1. Run MASt3R-SfM complete-graph on all eligible frames.
    2. Build CLIP-weighted anchor rays from anchor_result.
    3. Initialize P_anchor from MASt3R point cloud projected into anchor bboxes.
    4. (Optional) Load manual correspondences as additional Ceres residuals.
    5. Single Ceres full-BA solve (MASt3RReprojCost + AnchorRayCost + manual).
    6. Umeyama Sim(3): map solved MASt3R frame → GPS/ENU.

    Parameters
    ----------
    frames              : all loaded frames.
    anchor_result       : AnchorResult from detect_anchor() (Qwen + CLIP).
                          If None, anchor rays are omitted.
    use_manual_features : load and inject MANUAL_CORRESPONDENCES_FILE.
    matcher_only        : run MASt3R only, save auto_matches.json, then return.
    keep_dense          : passed through to run_complete_graph — retain dense
                          per-view data for the Blender mesh exporter.
    export_raw_glb_dir  : passed through to run_complete_graph — write
                          MASt3R's raw reconstruction as a debug .glb.

    Returns
    -------
    A RefineResult bundling mast3r_result / points_3d_solved / sim3, for
    optional debug export (--export-mesh). None on any early abort (too few
    frames, MASt3R failure, or --run-matcher-only).
    """
    # ── Eligible frames ───────────────────────────────────────────────────────
    ready = [
        f for f in frames
        if f.undistorted is not None
        and f.K_undist   is not None
    ]
    if len(ready) < 2:
        logger.warning("refine_pitches: fewer than 2 undistorted frames — aborting.")
        return None

    # ── Step 1: MASt3R-SfM ───────────────────────────────────────────────────
    from pipeline.mast3r_matcher import run_complete_graph, save_auto_matches

    logger.info("Running MASt3R-SfM on %d frames …", len(ready))
    mast3r_result = run_complete_graph(
        ready,
        save_mast3r_images=save_mast3r_images,
        keep_dense=keep_dense,
        export_raw_glb_dir=export_raw_glb_dir,
    )

    if len(mast3r_result.camera_poses) == 0:
        logger.error("MASt3R returned no camera poses — aborting refinement.")
        return None

    if matcher_only:
        save_auto_matches(mast3r_result, config.AUTO_MATCHES_FILE)
        logger.info("--run-matcher-only: saved auto_matches.json, stopping before Ceres.")
        return None

    # ── Step 2: Anchor rays ───────────────────────────────────────────────────
    anchor_rays = _build_anchor_rays(anchor_result)

    # ── Step 3: Initialize P_anchor ──────────────────────────────────────────
    p_anchor_init = _init_p_anchor(
        mast3r_result, anchor_result, mast3r_result.camera_poses
    )

    # ── Step 4: Manual features ───────────────────────────────────────────────
    manual_features = _load_manual_features(ready) if use_manual_features else None

    # ── Step 5: Ceres full-BA ─────────────────────────────────────────────────
    from pipeline.orientation_solver import ceres_solve, align_to_telemetry_sim3

    logger.info("Starting Ceres full-BA …")
    cam_params_solved, points_3d_solved, p_anchor_solved, report = ceres_solve(
        ready,
        mast3r_observations=mast3r_result.observations,
        anchor_rays=anchor_rays,
        cam_poses_init=mast3r_result.camera_poses,
        points_3d_init=mast3r_result.points_3d,
        p_anchor_init=p_anchor_init,
        manual_features=manual_features,
    )
    logger.info("Ceres BA: %s", report)
    logger.info("P_anchor solved: %s", np.round(p_anchor_solved, 3))

    # ── Step 6: Sim(3) alignment to GPS/ENU ──────────────────────────────────
    logger.info("Aligning solved poses to GPS/ENU via Umeyama Sim(3) …")
    sim3 = align_to_telemetry_sim3(ready, cam_params_solved, anchor_result=anchor_result)

    # ── Step 6b: Level the ground plane (corrects the tilt the near-coplanar
    # camera-centre Sim(3) leaves underconstrained) ──────────────────────────
    if config.LEVEL_GROUND_PLANE and sim3 is not None:
        from pipeline.orientation_solver import level_ground_plane
        sim3 = level_ground_plane(ready, points_3d_solved, sim3)

    # Build MASt3R-before-Ceres cameras in ENU using a MASt3R-NATIVE Sim(3)
    # (its own telemetry fit + its own ground leveled to Z=0) — NOT the Ceres
    # composed sim3 above. This keeps solved_camera_mast3r.json loadable in the
    # flat-Z=0 UI on MASt3R's own terms, for a clean before/after-Ceres compare.
    from pipeline.orientation_solver import mast3r_native_leveled_sim3
    mast3r_sim3 = mast3r_native_leveled_sim3(
        ready, mast3r_result.camera_poses, mast3r_result.points_3d,
        anchor_result=anchor_result,
    )
    mast3r_cams_enu = _build_mast3r_cameras_enu(ready, mast3r_result.camera_poses, mast3r_sim3)
    if mast3r_cams_enu:
        logger.info("MASt3R seed cameras converted to ENU (%d frames).", len(mast3r_cams_enu))

    logger.info("refine_pitches complete.")
    return RefineResult(
        mast3r_result=mast3r_result,
        points_3d_solved=points_3d_solved,
        sim3=sim3,
        mast3r_cameras_enu=mast3r_cams_enu,
        mast3r_sim3=mast3r_sim3,
    )
