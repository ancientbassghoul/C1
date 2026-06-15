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
from pathlib import Path

import numpy as np

from pipeline.frame import Frame
from pipeline.pose import build_rotation
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_pose_overrides(frames: list[Frame]) -> int:
    """Apply config.CAMERA_POSE_OVERRIDES to matching frames in-place."""
    overrides = config.CAMERA_POSE_OVERRIDES
    if not overrides:
        return 0
    updated = 0
    for f in frames:
        frame_num = f.stem[-5:]
        if frame_num not in overrides:
            continue
        ov = overrides[frame_num]
        f.position_enu     = np.array([ov['x'], ov['y'], ov['z']])
        f.heading_deg      = ov['heading']
        f.gimbal_pitch_deg = ov['pitch']
        f.camera_roll_deg  = ov['roll']
        f.R = build_rotation(f.heading_deg, f.gimbal_pitch_deg, f.camera_roll_deg)
        logger.info(
            "[%s] Pose from config override: pos=(%.2f, %.2f, %.2f)m  "
            "hdg=%.1f°  pitch=%.1f°  roll=%.1f°",
            f.stem, ov['x'], ov['y'], ov['z'],
            ov['heading'], ov['pitch'], ov['roll'],
        )
        updated += 1
    return updated


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
    cameras_init_from_config: bool = False,
    use_manual_features: bool = False,
    matcher_only: bool = False,
    save_mast3r_images: str | None = None,
) -> None:
    """
    Refine camera poses using MASt3R-SfM + Ceres full bundle adjustment.

    Steps
    -----
    1. (Optional) Apply config.CAMERA_POSE_OVERRIDES.
    2. Run MASt3R-SfM complete-graph on all eligible frames.
    3. Build CLIP-weighted anchor rays from anchor_result.
    4. Initialize P_anchor from MASt3R point cloud projected into anchor bboxes.
    5. (Optional) Load manual correspondences as additional Ceres residuals.
    6. Single Ceres full-BA solve (MASt3RReprojCost + AnchorRayCost + manual).
    7. Umeyama Sim(3): map solved MASt3R frame → GPS/ENU.

    Parameters
    ----------
    frames              : all loaded frames.
    anchor_result       : AnchorResult from detect_anchor() (Qwen + CLIP).
                          If None, anchor rays are omitted.
    cameras_init_from_config : apply CAMERA_POSE_OVERRIDES before solving.
    use_manual_features : load and inject MANUAL_CORRESPONDENCES_FILE.
    matcher_only        : run MASt3R only, save auto_matches.json, then return.
    """
    # ── Apply manual pose overrides ───────────────────────────────────────────
    if cameras_init_from_config:
        n = _apply_pose_overrides(frames)
        logger.info("Applied config pose overrides to %d frame(s).", n)

    # ── Eligible frames ───────────────────────────────────────────────────────
    ready = [
        f for f in frames
        if f.undistorted is not None
        and f.K_undist   is not None
    ]
    if len(ready) < 2:
        logger.warning("refine_pitches: fewer than 2 undistorted frames — aborting.")
        return

    # ── Step 1: MASt3R-SfM ───────────────────────────────────────────────────
    from pipeline.mast3r_matcher import run_complete_graph, save_auto_matches

    logger.info("Running MASt3R-SfM on %d frames …", len(ready))
    mast3r_result = run_complete_graph(ready, save_mast3r_images=save_mast3r_images)

    if len(mast3r_result.camera_poses) == 0:
        logger.error("MASt3R returned no camera poses — aborting refinement.")
        return

    if matcher_only:
        save_auto_matches(mast3r_result, config.AUTO_MATCHES_FILE)
        logger.info("--run-matcher-only: saved auto_matches.json, stopping before Ceres.")
        return

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
    align_to_telemetry_sim3(ready, cam_params_solved, anchor_result=anchor_result)

    logger.info("refine_pitches complete.")
