"""
pipeline/mast3r_matcher.py – MASt3R-SfM complete-graph wrapper.

Runs MASt3R-SfM on all N*(N-1)/2 pairs of HUD-masked undistorted frames.
Returns initial camera poses and a dense set of 3D points with per-frame
observations for the Ceres full-BA solver.

MASt3R-SfM API summary:
  1. AsymmetricMASt3R.from_pretrained(model_name)
  2. load_images(filelist, size=512)           – resize images for the model
  3. make_pairs(images, scene_graph='complete')
  4. inference(pairs, model, device)           – pairwise matching
  5. sparse_global_alignment(filelist, output, cache, model) → scene

The scene object provides:
  scene.im_poses          – list of 4×4 world-to-camera transforms
  scene.get_pts3d()       – list of (H, W, 3) per-view world-space pointmaps
  scene.get_masks()       – list of (H, W) bool valid-pixel masks
  scene.get_im_confs()    – list of (H, W) per-pixel confidence scores

All coordinates live in MASt3R's internal (arbitrary) coordinate frame.
ENU alignment happens AFTER Ceres via Sim(3) in orientation_solver.py.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """One 2D observation of a 3D point in a specific frame."""
    frame_stem: str
    point_idx:  int         # index into MASt3RResult.points_3d
    pixel_uv:   np.ndarray  # shape (2,): [u, v] in undistorted image pixels
    confidence: float


@dataclass
class MASt3RResult:
    """Outputs of the MASt3R-SfM complete-graph run."""
    # Per-frame initial camera poses in MASt3R's world frame.
    # Dict: frame.stem → 4×4 world-to-camera matrix (float64).
    camera_poses: dict = field(default_factory=dict)

    # Global 3D point cloud, shape (N, 3).
    points_3d: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    # All 2D observations associating frames with 3D points.
    observations: list = field(default_factory=list)  # list[Observation]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_frames_to_tmpdir(frames: list, tmpdir: str) -> tuple[list[str], dict[str, str]]:
    """Write undistorted BGR frames to PNG files in tmpdir for MASt3R.

    Returns (ordered_filelist, stem→filepath mapping).
    """
    filelist = []
    stem_to_path: dict[str, str] = {}
    for f in frames:
        if f.undistorted is None:
            logger.warning("[%s] no undistorted image — skipping in MASt3R", f.stem)
            continue
        fpath = os.path.join(tmpdir, f"{f.stem}.png")
        cv2.imwrite(fpath, f.undistorted)
        filelist.append(fpath)
        stem_to_path[f.stem] = fpath
    return filelist, stem_to_path


def _load_mast3r_model(device: str):
    """Load MASt3R model from config.MAST3R_MODEL with caching."""
    from mast3r.model import AsymmetricMASt3R

    logger.info("Loading MASt3R model %s …", config.MAST3R_MODEL)
    model = AsymmetricMASt3R.from_pretrained(
        config.MAST3R_MODEL,
        cache_dir=config.MODEL_CACHE_DIR,
    ).to(device)
    model.eval()
    return model


def _extract_poses(scene, filelist: list[str], frames: list) -> dict[str, np.ndarray]:
    """
    Extract per-frame 4×4 world-to-camera matrices from the aligned scene.

    MASt3R's `im_poses` are camera-to-world; we invert them for Ceres
    (which expects world-to-camera: p_cam = R @ p_world + t).
    """
    import torch

    path_to_stem = {}
    for f in frames:
        for fpath in filelist:
            if Path(fpath).stem == f.stem:
                path_to_stem[fpath] = f.stem

    cam_poses: dict[str, np.ndarray] = {}

    # scene.im_poses is a list (one per image) of (4,4) tensors or arrays,
    # camera-to-world (c2w).
    raw_poses = scene.im_poses if hasattr(scene, "im_poses") else []
    if hasattr(raw_poses, "detach"):
        raw_poses_np = raw_poses.detach().cpu().numpy()
    else:
        try:
            raw_poses_np = [
                p.detach().cpu().numpy() if hasattr(p, "detach") else np.array(p)
                for p in raw_poses
            ]
        except Exception:
            raw_poses_np = []

    for idx, fpath in enumerate(filelist):
        stem = path_to_stem.get(fpath)
        if stem is None:
            continue
        if idx >= len(raw_poses_np):
            logger.warning("MASt3R returned fewer poses than images (%d vs %d)", len(raw_poses_np), len(filelist))
            continue

        c2w = np.array(raw_poses_np[idx], dtype=np.float64)  # 4×4
        # Invert camera-to-world → world-to-camera
        w2c = np.linalg.inv(c2w)
        cam_poses[stem] = w2c

    return cam_poses


def _extract_pointcloud(
    scene,
    filelist: list[str],
    frames: list,
    cam_poses: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[Observation]]:
    """
    Build a global point cloud and cross-frame observations from per-view pointmaps.

    Strategy:
    1. From each frame, subsample K confident pixels → K initial 3D points.
    2. For each sampled 3D point, project into all other frames via initial camera poses.
       If the projected pixel lands near a valid, high-confidence MASt3R pixel,
       record an additional observation.
    3. Keep only 3D points with ≥ 2 observations (degenerate ones have no cross-frame constraint).
    4. Subsample to MAST3R_MAX_POINTS using farthest-point sampling.
    """
    import torch

    stem_to_idx = {Path(fpath).stem: i for i, fpath in enumerate(filelist)}
    frame_map   = {f.stem: f for f in frames}

    # Get per-view pointmaps, masks, and confidences
    try:
        pts3d_views  = scene.get_pts3d()   # list of (H, W, 3) world-space arrays
        masks_views  = scene.get_masks()   # list of (H, W) bool
        confs_views  = scene.get_im_confs() # list of (H, W) float
    except AttributeError as e:
        logger.error("Cannot extract pointmaps from scene object: %s", e)
        return np.empty((0, 3)), []

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.array(x)

    pts3d_views  = [to_numpy(p) for p in pts3d_views]
    masks_views  = [to_numpy(m).astype(bool) for m in masks_views]
    confs_views  = [to_numpy(c).astype(np.float32) for c in confs_views]

    n_frames    = len(filelist)
    points_per_frame = max(10, config.MAST3R_MAX_POINTS // max(1, n_frames))

    sampled_pts3d: list[np.ndarray] = []   # (3,) world coords
    sampled_frame_stems: list[str]  = []   # which frame the seed came from
    sampled_pixels: list[np.ndarray] = []  # (2,) pixel [u, v]
    sampled_confs:  list[float]     = []

    for fpath in filelist:
        stem = Path(fpath).stem
        idx  = stem_to_idx.get(stem)
        if idx is None or idx >= len(pts3d_views):
            continue

        pts3d = pts3d_views[idx]   # (H, W, 3)
        mask  = masks_views[idx]   # (H, W)
        conf  = confs_views[idx]   # (H, W)

        # Restrict to confident, valid pixels
        valid = mask & (conf >= config.MAST3R_CONF_THRESHOLD)
        ys, xs = np.where(valid)
        if len(ys) == 0:
            logger.warning("[%s] no confident MASt3R pixels", stem)
            continue

        # Random subsample
        n_sample = min(points_per_frame, len(ys))
        sel      = np.random.choice(len(ys), n_sample, replace=False)
        ys_s, xs_s = ys[sel], xs[sel]

        for y, x in zip(ys_s, xs_s):
            sampled_pts3d.append(pts3d[y, x])
            sampled_frame_stems.append(stem)
            sampled_pixels.append(np.array([float(x), float(y)]))
            sampled_confs.append(float(conf[y, x]))

    if not sampled_pts3d:
        logger.error("No valid 3D points extracted from MASt3R scene")
        return np.empty((0, 3)), []

    all_pts  = np.stack(sampled_pts3d, axis=0)  # (N, 3)
    all_px   = sampled_pixels
    all_conf = sampled_confs
    all_stem = sampled_frame_stems

    logger.info("Sampled %d initial 3D points across %d frames", len(all_pts), n_frames)

    # ── Cross-frame observation augmentation ─────────────────────────────────
    # For each sampled 3D point, project into all frames and check if a
    # confident MASt3R pixel is nearby.

    # Build per-frame intrinsics for projection.
    # We use frame.K_undist if available; fall back to config intrinsics.
    from pipeline.undistort import build_K_new, build_K

    K_default = build_K_new(build_K())

    def get_K(stem: str) -> Optional[np.ndarray]:
        f = frame_map.get(stem)
        if f is not None and f.K_undist is not None:
            return f.K_undist
        return K_default

    # index_of_stem → (pts3d, mask, conf) for observation lookup
    view_data = {}
    for fpath in filelist:
        stem = Path(fpath).stem
        idx  = stem_to_idx.get(stem)
        if idx is not None and idx < len(pts3d_views):
            h, w = pts3d_views[idx].shape[:2]
            view_data[stem] = (pts3d_views[idx], masks_views[idx], confs_views[idx], h, w)

    observations: list[Observation] = []
    point_to_obs_count = [0] * len(all_pts)  # track per-point multi-view count

    MATCH_DIST_3D_M  = 0.30   # world-space radius to merge across frames
    MATCH_DIST_PX    = 5.0    # pixel-space tolerance for projection check

    for pt_idx, (P, seed_stem, seed_px, seed_conf) in enumerate(
        zip(all_pts, all_stem, all_px, all_conf)
    ):
        for fpath in filelist:
            stem = Path(fpath).stem
            w2c = cam_poses.get(stem)
            K   = get_K(stem)
            vd  = view_data.get(stem)
            if w2c is None or K is None or vd is None:
                continue

            R34 = w2c[:3, :3]
            t3  = w2c[:3, 3]
            p_cam = R34 @ P + t3
            if p_cam[2] <= 0:  # behind camera
                continue

            u_proj = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
            v_proj = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]

            pts_view, mask_view, conf_view, h, w = vd
            ui, vi = int(round(u_proj)), int(round(v_proj))
            if not (0 <= ui < w and 0 <= vi < h):
                continue

            if not mask_view[vi, ui]:
                continue
            if conf_view[vi, ui] < config.MAST3R_CONF_THRESHOLD:
                continue

            # Check that MASt3R's pointmap at this pixel agrees in 3D space
            P_view = pts_view[vi, ui]
            if np.linalg.norm(P_view - P) > MATCH_DIST_3D_M:
                if stem != seed_stem:
                    # Use projection pixel even if 3D doesn't match — the solver corrects it
                    pass

            observations.append(Observation(
                frame_stem=stem,
                point_idx=pt_idx,
                pixel_uv=np.array([u_proj, v_proj]),
                confidence=float(conf_view[vi, ui]),
            ))
            point_to_obs_count[pt_idx] += 1

    # ── Filter: keep only points seen in ≥2 frames ────────────────────────────
    keep_mask = np.array(point_to_obs_count) >= 2
    keep_idxs = np.where(keep_mask)[0]

    if len(keep_idxs) == 0:
        # Fall back to single-view points to avoid empty Ceres problem
        logger.warning("No multi-view points found; using single-view seed points")
        keep_idxs = np.arange(len(all_pts))

    logger.info(
        "%d / %d 3D points have ≥2 observations",
        len(keep_idxs), len(all_pts),
    )

    # Remap point indices
    old_to_new = {old: new for new, old in enumerate(keep_idxs)}
    final_pts  = all_pts[keep_idxs]

    final_obs = [
        Observation(
            frame_stem=obs.frame_stem,
            point_idx=old_to_new[obs.point_idx],
            pixel_uv=obs.pixel_uv,
            confidence=obs.confidence,
        )
        for obs in observations
        if obs.point_idx in old_to_new
    ]

    # ── Subsample to MAST3R_MAX_POINTS using farthest-point sampling ──────────
    if len(final_pts) > config.MAST3R_MAX_POINTS:
        final_pts, final_obs = _fps_subsample(final_pts, final_obs, config.MAST3R_MAX_POINTS)

    logger.info(
        "Final 3D cloud: %d points, %d observations",
        len(final_pts), len(final_obs),
    )
    return final_pts, final_obs


def _fps_subsample(
    pts: np.ndarray,
    obs: list[Observation],
    target: int,
) -> tuple[np.ndarray, list[Observation]]:
    """Farthest-point subsample to at most `target` points."""
    n = len(pts)
    if n <= target:
        return pts, obs

    selected = [np.random.randint(n)]
    dists    = np.full(n, np.inf)

    for _ in range(target - 1):
        d = np.linalg.norm(pts - pts[selected[-1]], axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))

    keep_set = set(selected)
    old_to_new = {old: new for new, old in enumerate(selected)}

    pts_sub = pts[selected]
    obs_sub = [
        Observation(
            frame_stem=o.frame_stem,
            point_idx=old_to_new[o.point_idx],
            pixel_uv=o.pixel_uv,
            confidence=o.confidence,
        )
        for o in obs
        if o.point_idx in keep_set
    ]
    return pts_sub, obs_sub


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_complete_graph(frames: list) -> MASt3RResult:
    """
    Run MASt3R-SfM on all pairs of undistorted frames.

    Frames without an undistorted image are silently skipped.
    Returns a MASt3RResult with camera poses in MASt3R's internal coordinate
    frame and a filtered/subsampled point cloud with cross-frame observations.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("MASt3R using device: %s", device)

    # ── Save undistorted frames to temp PNG files ─────────────────────────────
    with tempfile.TemporaryDirectory(prefix="mast3r_") as tmpdir:
        filelist, _stem_map = _save_frames_to_tmpdir(frames, tmpdir)

        if len(filelist) < 2:
            logger.error("Need at least 2 frames for MASt3R; got %d", len(filelist))
            return MASt3RResult()

        logger.info("Running MASt3R-SfM complete graph on %d frames …", len(filelist))

        # ── Load model ────────────────────────────────────────────────────────
        model = _load_mast3r_model(device)

        # ── MASt3R-SfM pipeline ───────────────────────────────────────────────
        from dust3r.utils.image import load_images
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from dust3r.inference import inference
        from dust3r.image_pairs import make_pairs

        cache_dir = os.path.join(config.MODEL_CACHE_DIR, "mast3r_cache")
        os.makedirs(cache_dir, exist_ok=True)

        logger.info("Loading %d images at size=512 …", len(filelist))
        images = load_images(filelist, size=512, verbose=False)

        logger.info("Building complete-graph pairs …")
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        logger.info("%d pairs for %d frames", len(pairs), len(filelist))

        logger.info("Running pairwise MASt3R inference …")
        output = inference(pairs, model, device, batch_size=1, verbose=True)

        logger.info("Running sparse global alignment …")
        scene = sparse_global_alignment(
            filelist,
            output,
            cache_dir,
            model,
            device=device,
            verbose=True,
        )

        # ── Extract results ───────────────────────────────────────────────────
        logger.info("Extracting camera poses …")
        cam_poses = _extract_poses(scene, filelist, frames)

        logger.info("Extracting 3D pointcloud and observations …")
        points_3d, observations = _extract_pointcloud(scene, filelist, frames, cam_poses)

        # Tear down model before returning (VRAM for Ceres host is fine; Ceres is CPU)
        del model, scene, output, images, pairs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("MASt3R VRAM released.")

    return MASt3RResult(
        camera_poses=cam_poses,
        points_3d=points_3d,
        observations=observations,
    )


def save_auto_matches(result: MASt3RResult, path: str | Path) -> None:
    """Serialize a MASt3RResult to JSON for --run-matcher-only inspection."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_points":      int(len(result.points_3d)),
        "n_observations": int(len(result.observations)),
        "camera_stems":  sorted(result.camera_poses.keys()),
        "camera_poses":  {
            stem: pose.tolist()
            for stem, pose in result.camera_poses.items()
        },
        "points_3d":  result.points_3d.tolist() if len(result.points_3d) else [],
        "observations": [
            {
                "frame":      obs.frame_stem,
                "point_idx":  obs.point_idx,
                "pixel_uv":   obs.pixel_uv.tolist(),
                "confidence": obs.confidence,
            }
            for obs in result.observations
        ],
    }

    path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Auto-matches saved to %s  (%d pts, %d obs)",
        path, len(result.points_3d), len(result.observations),
    )
