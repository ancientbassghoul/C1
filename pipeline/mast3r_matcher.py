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

sparse_global_alignment() returns a mast3r.cloud_opt.sparse_ga.SparseGA
instance — NOT a dust3r dense global_aligner — so it has a different API:
  scene.get_im_poses()                       – (N,4,4) cam2world tensor
  scene.get_dense_pts3d(clean_depth, subsample)
                                              – (pts3d, depthmaps, confs), one
                                                flattened (H*W,3) / (H,W) entry
                                                per image, in world coordinates
  scene.imgs[i]                              – RGB image at the exact
                                                resolution pts3d/confs use
  scene.get_masks()                          – no real per-pixel mask exists;
                                                always returns slice(None) —
                                                confidence is the only filter
                                                (see SparseGA.show(): masks =
                                                [c > 1 for c in confs] — the
                                                confidence scale is NOT 0-1
                                                normalized)

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

    # Dense per-view data for the Blender mesh exporter — only populated when
    # run_complete_graph(..., keep_dense=True). Dict: frame.stem → (pts3d,
    # img, valid), where pts3d is (H,W,3) world-space, img is (H,W,3) uint8
    # RGB, valid is (H,W) bool. Same resolution MASt3R itself worked at, NOT
    # the full undistorted resolution.
    dense: dict = field(default_factory=dict)


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


def _resolve_mast3r_pth(model_id: str, cache_dir: str) -> str:
    """Return a local .pth path for *model_id*, downloading from NAVER CDN if needed.

    model_id may be:
      - an absolute/relative file path  → returned as-is
      - a HF-style identifier           → e.g. "naver/MASt3R_ViTLarge_BaseDecoder_512_capdepth"

    HuggingFace Hub loading is deliberately avoided: the HF route requires either
    network auth (401 on RunPod) or a locally-cached HF model directory, and it
    triggers a MASt3R-internal bug where img_size arrives as an int rather than a
    tuple.  The NAVER CDN .pth files load through load_model() which is not affected.
    """
    import urllib.request

    # Already a file path?
    if os.path.isfile(model_id):
        return model_id

    # Derive .pth filename from the last component of the HF repo id.
    model_name = model_id.split("/")[-1]  # e.g. "MASt3R_ViTLarge_BaseDecoder_512_capdepth"
    pth_name = model_name if model_name.endswith(".pth") else f"{model_name}.pth"
    pth_path = os.path.join(cache_dir, pth_name)

    if os.path.isfile(pth_path):
        logger.info("Using cached MASt3R weights: %s", pth_path)
        return pth_path

    # Download from NAVER CDN — no authentication required.
    cdn_url = f"https://download.europe.naverlabs.com/ComputerVision/MASt3R/{pth_name}"
    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Downloading MASt3R weights from %s → %s", cdn_url, pth_path)

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            if pct % 10 == 0:
                logger.info("  … %d%%", pct)

    urllib.request.urlretrieve(cdn_url, pth_path, reporthook=_progress)
    logger.info("Download complete: %s", pth_path)
    return pth_path


def _load_mast3r_model(device: str):
    """Load MASt3R model, preferring a local .pth file over the HuggingFace Hub path."""
    import sys
    if '/workspace/mast3r' not in sys.path:
        sys.path.insert(0, '/workspace/mast3r')
    from mast3r.model import load_model as mast3r_load_model

    pth_path = _resolve_mast3r_pth(config.MAST3R_MODEL, config.MODEL_CACHE_DIR)
    logger.info("Loading MASt3R model from %s …", pth_path)
    model = mast3r_load_model(pth_path, device=device, verbose=False)
    model.eval()
    return model


def _extract_poses(scene, filelist: list[str], frames: list) -> dict[str, np.ndarray]:
    """
    Extract per-frame 4×4 world-to-camera matrices from the aligned scene.

    scene.get_im_poses() returns camera-to-world; we invert them for Ceres
    (which expects world-to-camera: p_cam = R @ p_world + t).
    """
    import torch

    path_to_stem = {}
    for f in frames:
        for fpath in filelist:
            if Path(fpath).stem == f.stem:
                path_to_stem[fpath] = f.stem

    cam_poses: dict[str, np.ndarray] = {}

    # scene is a mast3r.cloud_opt.sparse_ga.SparseGA — get_im_poses() returns
    # a single (N,4,4) cam2world tensor (there is no .im_poses attribute).
    raw_poses = scene.get_im_poses()
    raw_poses_np = raw_poses.detach().cpu().numpy() if hasattr(raw_poses, "detach") else np.asarray(raw_poses)

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


def _adaptive_conf_mask(conf_hw: np.ndarray) -> np.ndarray:
    """Per-frame adaptive confidence mask, robust to scene-wide confidence
    scale shifts (e.g. the gray HUD fill depressing MASt3R's dense per-pixel
    confidence across the whole frame).

    Keeps pixels above the larger of:
      - config.MAST3R_CONF_FLOOR — an absolute floor just above MASt3R's ~1.0
        "no-information" baseline, dropping genuine garbage regardless of band, and
      - the config.MAST3R_CONF_PERCENTILE-th percentile of the frame's finite
        confidences, so a uniformly low-confidence frame still yields its most
        reliable pixels instead of being wiped out entirely by a fixed cutoff.

    Never returns an all-False mask when any finite confidence exists: if the
    percentile/floor gate clips everything, fall back to the top-N highest pixels.
    """
    finite = np.isfinite(conf_hw)
    if not finite.any():
        return np.zeros_like(conf_hw, dtype=bool)

    vals = conf_hw[finite]
    pct  = float(np.percentile(vals, config.MAST3R_CONF_PERCENTILE))
    thr  = max(float(config.MAST3R_CONF_FLOOR), pct)
    mask = finite & (conf_hw >= thr)

    if not mask.any():
        # Degenerate fallback: keep the top-N highest-confidence pixels so a
        # frame is never dropped wholesale.
        top_n   = min(256, vals.size)
        cutoff  = float(np.partition(vals, vals.size - top_n)[vals.size - top_n])
        mask    = finite & (conf_hw >= cutoff)
    return mask


def _extract_pointcloud(
    scene,
    filelist: list[str],
    frames: list,
    cam_poses: dict[str, np.ndarray],
    keep_dense: bool = False,
) -> tuple[np.ndarray, list[Observation], dict]:
    """
    Build a global point cloud and cross-frame observations from per-view pointmaps.

    Strategy:
    1. From each frame, subsample K confident pixels → K initial 3D points.
    2. For each sampled 3D point, project into all other frames via initial camera poses.
       If the projected pixel lands near a valid, high-confidence MASt3R pixel,
       record an additional observation.
    3. Keep only 3D points with ≥ 2 observations (degenerate ones have no cross-frame constraint).
    4. Subsample to MAST3R_MAX_POINTS using farthest-point sampling.

    If keep_dense, also returns a third value: dict[stem] → (pts3d (H,W,3),
    img (H,W,3) uint8 RGB, valid (H,W) bool) — the full per-view dense data,
    reusing the same get_dense_pts3d() call already made below rather than
    querying the scene a second time. Used by the Blender mesh exporter.
    """
    import torch

    stem_to_idx = {Path(fpath).stem: i for i, fpath in enumerate(filelist)}
    frame_map   = {f.stem: f for f in frames}

    # Get per-view pointmaps and confidences.
    # scene is a mast3r.cloud_opt.sparse_ga.SparseGA — it has no get_pts3d()/
    # get_im_confs() (those belong to dust3r's dense global_aligner classes).
    # The dense per-pixel API here is get_dense_pts3d(), which returns flattened
    # (H*W, 3) / (H, W) entries already in world coordinates. get_masks() exists
    # but is a no-op (always slice(None)) — confidence is the only real filter.
    try:
        pts3d_list, _depthmaps_list, confs_list = scene.get_dense_pts3d(clean_depth=True, subsample=8)
    except AttributeError as e:
        logger.error("Cannot extract dense pointmaps from scene object: %s", e)
        return np.empty((0, 3)), [], {}

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.array(x)

    pts3d_views: list[np.ndarray] = []
    masks_views: list[np.ndarray] = []
    confs_views: list[np.ndarray] = []
    dense_out: dict[str, tuple] = {}
    for i in range(len(filelist)):
        img_native = to_numpy(scene.imgs[i])   # (H, W, 3) float in [0, 1]
        h, w = img_native.shape[:2]
        pts3d_hw3 = to_numpy(pts3d_list[i]).reshape(h, w, 3)
        conf_hw   = to_numpy(confs_list[i]).reshape(h, w).astype(np.float32)
        pts3d_views.append(pts3d_hw3)
        confs_views.append(conf_hw)
        masks_views.append(np.ones((h, w), dtype=bool))

        # Per-frame confidence diagnostics — makes a scene-wide confidence scale
        # shift (e.g. from HUD fill changes) self-diagnosing instead of silently
        # producing "no confident MASt3R pixels".
        adaptive_valid = _adaptive_conf_mask(conf_hw)
        fin = conf_hw[np.isfinite(conf_hw)]
        if fin.size:
            logger.info(
                "[%s] conf min/med/p60/p90/max = %.2f/%.2f/%.2f/%.2f/%.2f  kept=%d/%d",
                Path(filelist[i]).stem,
                float(fin.min()), float(np.median(fin)),
                float(np.percentile(fin, 60)), float(np.percentile(fin, 90)),
                float(fin.max()), int(adaptive_valid.sum()), int(conf_hw.size),
            )

        if keep_dense:
            stem = Path(filelist[i]).stem
            img_u8 = np.clip(img_native * 255.0, 0, 255).astype(np.uint8)
            dense_out[stem] = (pts3d_hw3, img_u8, adaptive_valid)

    n_frames    = len(filelist)

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

        # Restrict to confident, valid pixels (per-frame adaptive gate)
        valid = mask & _adaptive_conf_mask(conf)

        # HUD erosion guard: exclude pixels in (approximately scaled) HUD regions
        # + 20-pixel buffer to prevent boundary-noise points from slipping through.
        H_m, W_m = pts3d.shape[:2]
        for (y1r, y2r, x1r, x2r) in config.HUD_REGIONS:
            y1m = int(y1r * H_m / config.IMAGE_H); y2m = int(y2r * H_m / config.IMAGE_H)
            x1m = int(x1r * W_m / config.IMAGE_W); x2m = int(x2r * W_m / config.IMAGE_W)
            buf_y = max(1, int(20 * H_m / config.IMAGE_H))
            buf_x = max(1, int(20 * W_m / config.IMAGE_W))
            valid[max(0, y1m - buf_y):min(H_m, y2m + buf_y),
                  max(0, x1m - buf_x):min(W_m, x2m + buf_x)] = False

        ys, xs = np.where(valid)
        if len(ys) == 0:
            logger.warning("[%s] no confident MASt3R pixels", stem)
            continue

        # Use ALL valid (confident, non-HUD) pixels as Stage-1 seeds.
        # Grid sampling is deferred to Stage 3 where it operates on confirmed
        # multi-view tracks, not on raw seeds that may not survive Stage 2.
        for y, x in zip(ys, xs):
            sampled_pts3d.append(pts3d[y, x])
            sampled_frame_stems.append(stem)
            sampled_pixels.append(np.array([float(x), float(y)]))
            sampled_confs.append(float(conf[y, x]))

    if not sampled_pts3d:
        logger.error("No valid 3D points extracted from MASt3R scene")
        return np.empty((0, 3)), [], dense_out

    all_pts  = np.stack(sampled_pts3d, axis=0)  # (N, 3)
    all_px   = sampled_pixels
    all_conf = sampled_confs
    all_stem = sampled_frame_stems

    logger.info("Sampled %d initial 3D points across %d frames", len(all_pts), n_frames)

    # ── Cross-frame observation augmentation ─────────────────────────────────
    # For each sampled 3D point, project into all frames and check if a
    # confident MASt3R pixel is nearby.

    # Two different intrinsics are needed here, for two different purposes:
    #   K_native — MASt3R's own per-camera K (scene.intrinsics), matching the
    #              pixel space that pts3d_views/masks_views/confs_views actually
    #              live in (≈512px-resized images). Used to test whether a
    #              point lands on a valid/confident pixel of view j.
    #   K_undist — this project's real calibration for the full undistorted
    #              image (1280×720). Used only for the pixel coordinate we
    #              *store*, since that's the space orientation_solver.py's
    #              MASt3RReprojCost expects (it's built with f.K_undist).
    # Projecting with K_undist and then bounds-checking against the small
    # native (h, w) arrays — as this used to do — means nearly every
    # projection lands outside [0, w)×[0, h) before any real match is even
    # attempted (K_undist's cx=640 alone exceeds a ~512px-wide native image).
    from pipeline.undistort import build_K_new, build_K

    K_default = build_K_new(build_K())

    def get_K_undist(stem: str) -> Optional[np.ndarray]:
        f = frame_map.get(stem)
        if f is not None and f.K_undist is not None:
            return f.K_undist
        return K_default

    def to_numpy_local(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.array(x)

    # index_of_stem → (pts3d, mask, conf, h, w, K_native) for observation lookup
    view_data = {}
    for fpath in filelist:
        stem = Path(fpath).stem
        idx  = stem_to_idx.get(stem)
        if idx is not None and idx < len(pts3d_views):
            h, w = pts3d_views[idx].shape[:2]
            K_native = to_numpy_local(scene.intrinsics[idx])
            view_data[stem] = (pts3d_views[idx], masks_views[idx], confs_views[idx], h, w, K_native)

    observations: list[Observation] = []
    point_to_obs_count = [0] * len(all_pts)  # track per-point multi-view count

    MATCH_DIST_PX = 5.0   # scale-invariant pixel-space backprojection threshold (native res)

    for pt_idx, (P, seed_stem, seed_px, seed_conf) in enumerate(
        zip(all_pts, all_stem, all_px, all_conf)
    ):
        for fpath in filelist:
            stem = Path(fpath).stem
            w2c = cam_poses.get(stem)
            vd  = view_data.get(stem)
            if w2c is None or vd is None:
                continue

            R34 = w2c[:3, :3]
            t3  = w2c[:3, 3]
            p_cam = R34 @ P + t3
            if p_cam[2] <= 0:  # behind camera
                continue

            pts_view, mask_view, conf_view, h, w, K_native = vd

            # Validity/confidence check in MASt3R's own native pixel space.
            u_native = K_native[0, 0] * p_cam[0] / p_cam[2] + K_native[0, 2]
            v_native = K_native[1, 1] * p_cam[1] / p_cam[2] + K_native[1, 2]
            ui, vi = int(round(u_native)), int(round(v_native))
            if not (0 <= ui < w and 0 <= vi < h):
                continue

            if not mask_view[vi, ui]:
                continue

            # Scale-invariant gate: project P_view back into the SOURCE frame and
            # check it lands near the seed pixel.  The old 3D-distance gate ran in
            # MASt3R's compressed coordinate space; with Sim(3) scale ~18.8x it was
            # equivalent to ~37 real metres, accepting almost every track including
            # franken-tracks (van in frame A paired with ground in frame B).
            # Backprojecting into the source frame works in pixels and is scale-invariant.
            # Confidence is still stored on the observation for Ceres weighting.
            P_view = pts_view[vi, ui]
            src_w2c = cam_poses.get(seed_stem)
            if src_w2c is None:
                continue
            p_view_src = src_w2c[:3, :3] @ P_view + src_w2c[:3, 3]
            if p_view_src[2] <= 0:
                continue
            K_src = view_data[seed_stem][5]   # K_native for source frame
            u_back = K_src[0, 0] * p_view_src[0] / p_view_src[2] + K_src[0, 2]
            v_back = K_src[1, 1] * p_view_src[1] / p_view_src[2] + K_src[1, 2]
            if (u_back - seed_px[0]) ** 2 + (v_back - seed_px[1]) ** 2 > MATCH_DIST_PX ** 2:
                continue

            # Pixel coordinate to store — in K_undist's full-resolution space,
            # since that's what MASt3RReprojCost reprojects against.
            K_undist = get_K_undist(stem)
            u_proj = K_undist[0, 0] * p_cam[0] / p_cam[2] + K_undist[0, 2]
            v_proj = K_undist[1, 1] * p_cam[1] / p_cam[2] + K_undist[1, 2]

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

    logger.info(
        "%d / %d 3D points have ≥2 observations",
        len(keep_idxs), len(all_pts),
    )

    if len(keep_idxs) == 0:
        # Fall back to single-view points to avoid empty Ceres problem
        logger.warning(
            "No multi-view points found; using all %d single-view seed points instead "
            "(MASt3RReprojCost will contribute little/no real cross-frame constraint)",
            len(all_pts),
        )
        keep_idxs = np.arange(len(all_pts))

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

    # ── Subsample to MAST3R_MAX_POINTS using spatial grid sampling ────────────
    if len(final_pts) > config.MAST3R_MAX_POINTS:
        final_pts, final_obs = _grid_subsample(final_pts, final_obs, config.MAST3R_MAX_POINTS)

    logger.info(
        "Final 3D cloud: %d points, %d observations",
        len(final_pts), len(final_obs),
    )
    return final_pts, final_obs, dense_out


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


def _grid_subsample(
    pts: np.ndarray,
    obs: list[Observation],
    target: int,
    cell_size: int = 32,
) -> tuple[np.ndarray, list[Observation]]:
    """Grid-subsample confirmed multi-view tracks by pixel coverage in K_undist space.

    Operates in config.IMAGE_W × config.IMAGE_H (1280×720), giving ~920 cells
    for cell_size=32.  Selects top-confidence tracks per cell, then fills the
    remaining budget in global confidence order.
    """
    if len(pts) <= target:
        return pts, obs

    from collections import defaultdict

    img_w = config.IMAGE_W
    img_h = config.IMAGE_H
    ncols = max(1, (img_w + cell_size - 1) // cell_size)

    # Best-confidence observation per point (for cell assignment and ranking)
    best_conf: dict[int, float] = {}
    best_uv:   dict[int, tuple[float, float]] = {}
    for o in obs:
        if o.point_idx not in best_conf or o.confidence > best_conf[o.point_idx]:
            best_conf[o.point_idx] = o.confidence
            best_uv[o.point_idx]   = (float(o.pixel_uv[0]), float(o.pixel_uv[1]))

    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for pt_idx in range(len(pts)):
        u, v = best_uv.get(pt_idx, (img_w / 2.0, img_h / 2.0))
        ci = int(np.clip(u, 0, img_w - 1)) // cell_size
        ri = int(np.clip(v, 0, img_h - 1)) // cell_size
        buckets[ri * ncols + ci].append((best_conf.get(pt_idx, 0.0), pt_idx))

    for b in buckets.values():
        b.sort(reverse=True)

    per_cell = max(1, target // max(1, len(buckets)))
    selected: list[int] = []
    for b in buckets.values():
        selected.extend(i for _, i in b[:per_cell])

    if len(selected) < target:
        used = set(selected)
        global_order = sorted(best_conf.items(), key=lambda kv: kv[1], reverse=True)
        selected += [k for k, _ in global_order if k not in used][: target - len(selected)]

    keep = set(selected[:target])
    old_to_new = {old: new for new, old in enumerate(sorted(keep))}
    new_pts = pts[sorted(keep)]
    new_obs = [
        Observation(o.frame_stem, old_to_new[o.point_idx], o.pixel_uv, o.confidence)
        for o in obs
        if o.point_idx in keep
    ]
    return new_pts, new_obs


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def _log_pairwise_match_diagnostics(
    pairs: list,
    filelist: list[str],
    model,
    cache_dir: str,
    device: str,
    path_to_stem: dict[str, str],
) -> None:
    """Run MASt3R's own pairwise matching pass up front and log per-pair match
    counts plus a connected-components check over the resulting graph.

    sparse_global_alignment() runs this exact same forward_mast3r() pass
    internally — calling it here first just primes the on-disk cache, so this
    costs no extra GPU work.

    Why this matters: MASt3R-SfM's sparse_scene_optimizer demotes any pair whose
    best correspondence confidence falls at/under `matching_conf_thr` from a
    rigid cross-view 3D constraint to a much weaker single-view loss — i.e. the
    two frames stop being tied together (see `matching_check()` in
    mast3r.cloud_opt.sparse_ga). If enough pairs fail this check, the global
    optimization can converge to several locally-consistent but globally
    disconnected sub-reconstructions ("islands" in the exported debug .glb).
    This logs exactly which pairs/frames are at risk, instead of only seeing
    the fragmented result after the fact.
    """
    import torch
    from mast3r.cloud_opt.sparse_ga import convert_dust3r_pairs_naming, forward_mast3r

    thr = config.MAST3R_MATCHING_CONF_THR

    # Mutates `pairs` in place (idempotent — sparse_global_alignment() does the
    # exact same conversion again right after this, from the same `idx` field).
    pairs = convert_dust3r_pairs_naming(filelist, pairs)
    res_paths, _cache_dir = forward_mast3r(pairs, model, cache_path=cache_dir, device=device)

    # res_paths has both (a,b) and (b,a) directions; dedupe to one row per
    # unordered pair.
    pair_stats: dict[tuple[str, str], tuple[int, float, float]] = {}
    for (inst1, inst2), (_forward_paths, path_corres) in res_paths.items():
        stem1 = path_to_stem.get(inst1, inst1)
        stem2 = path_to_stem.get(inst2, inst2)
        key = tuple(sorted((stem1, stem2)))
        if key in pair_stats:
            continue
        try:
            matching_score, corres = torch.load(path_corres, map_location="cpu")
        except FileNotFoundError:
            logger.warning("[%s ↔ %s] no cached correspondences", stem1, stem2)
            continue
        conf_score, _sum_confs, n_matches = matching_score
        _xy1, _xy2, confs = corres
        conf_max = float(confs.max()) if len(confs) else 0.0
        pair_stats[key] = (n_matches, conf_score, conf_max)

    rows = sorted(pair_stats.items(), key=lambda kv: kv[1][0])  # weakest (fewest matches) first
    logger.info("── MASt3R pairwise match diagnostics (%d pairs, matching_conf_thr=%.1f) ──",
                len(rows), thr)
    for (stem1, stem2), (n_matches, conf_score, conf_max) in rows:
        ok = conf_max > thr
        logger.info(
            "  %-32s ↔ %-32s  n_matches=%5d  conf_score=%6.2f  conf_max=%6.2f  matching_ok=%s",
            stem1, stem2, n_matches, conf_score, conf_max, ok,
        )

    # ── Connected-components check over the "matching_ok" edges ────────────
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    stems = sorted(path_to_stem.values())
    idx_of = {s: i for i, s in enumerate(stems)}
    row_idx, col_idx = [], []
    for (stem1, stem2), (_n, _conf_score, conf_max) in pair_stats.items():
        if conf_max > thr:
            row_idx.append(idx_of[stem1])
            col_idx.append(idx_of[stem2])
    graph = coo_matrix((np.ones(len(row_idx)), (row_idx, col_idx)), shape=(len(stems), len(stems)))
    n_components, labels = connected_components(graph, directed=False)

    if n_components > 1:
        groups: dict[int, list[str]] = {}
        for stem, label in zip(stems, labels):
            groups.setdefault(int(label), []).append(stem)
        logger.warning(
            "MASt3R pairwise match graph has %d disconnected components (matching_ok @ "
            "matching_conf_thr=%.1f) — sparse_global_alignment is likely to produce that many "
            "disjoint reconstruction 'islands':",
            n_components, thr,
        )
        for label, group_stems in sorted(groups.items()):
            logger.warning("  component %d (%d frames): %s", label, len(group_stems), group_stems)
    else:
        logger.info(
            "MASt3R pairwise match graph is fully connected (matching_ok @ matching_conf_thr=%.1f, "
            "%d frames).", thr, len(stems),
        )


def run_complete_graph(
    frames: list,
    save_mast3r_images: str | None = None,
    keep_dense: bool = False,
    export_raw_glb_dir: str | None = None,
) -> MASt3RResult:
    """
    Run MASt3R-SfM on all pairs of undistorted frames.

    Frames without an undistorted image are silently skipped.
    Returns a MASt3RResult with camera poses in MASt3R's internal coordinate
    frame and a filtered/subsampled point cloud with cross-frame observations.

    keep_dense          : also populate MASt3RResult.dense (per-view pointmap +
                           image + valid mask) for the Blender mesh exporter.
    export_raw_glb_dir   : if set, write MASt3R's raw reconstruction (its own
                           native frame, pre-Ceres/pre-Sim3) as a debug .glb
                           to this directory, using the `scene` object before
                           it's torn down. Works even with --run-matcher-only.
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

        if save_mast3r_images is not None:
            import shutil, sys
            out_dir = save_mast3r_images or os.path.join(config.OUTPUT_DIR, "debug", "mast3r_inputs")
            os.makedirs(out_dir, exist_ok=True)
            for fpath in filelist:
                shutil.copy(fpath, out_dir)
            logger.info("--save-mast3r-images: saved %d images to %s", len(filelist), out_dir)
            sys.exit(0)

        logger.info("Running MASt3R-SfM complete graph on %d frames …", len(filelist))

        # ── Load model ────────────────────────────────────────────────────────
        model = _load_mast3r_model(device)

        # ── MASt3R-SfM pipeline ───────────────────────────────────────────────
        from dust3r.utils.image import load_images
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from dust3r.image_pairs import make_pairs

        cache_dir = os.path.join(config.MODEL_CACHE_DIR, "mast3r_cache")
        os.makedirs(cache_dir, exist_ok=True)

        logger.info("Loading %d images at size=512 …", len(filelist))
        images = load_images(filelist, size=512, verbose=False)

        logger.info("Building complete-graph pairs …")
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        logger.info("%d pairs for %d frames", len(pairs), len(filelist))

        path_to_stem = {fpath: os.path.splitext(os.path.basename(fpath))[0] for fpath in filelist}
        try:
            _log_pairwise_match_diagnostics(pairs, filelist, model, cache_dir, device, path_to_stem)
        except Exception as e:
            logger.warning("Pairwise match diagnostics failed (continuing): %s", e)

        logger.info(
            "Running sparse global alignment (matching_conf_thr=%.1f, shared_intrinsics=%s) …",
            config.MAST3R_MATCHING_CONF_THR, config.MAST3R_SHARED_INTRINSICS,
        )
        scene = sparse_global_alignment(
            filelist,
            pairs,
            cache_dir,
            model,
            device=device,
            verbose=True,
            matching_conf_thr=config.MAST3R_MATCHING_CONF_THR,
            shared_intrinsics=config.MAST3R_SHARED_INTRINSICS,
        )

        _focals = scene.get_focals().detach().cpu().numpy()
        logger.info(
            "MASt3R SGA focals (native 512px space): min=%.1f  max=%.1f  "
            "[calibrated expected ≈156 px = 391.15 × (512/1280)]",
            float(_focals.min()), float(_focals.max()),
        )

        if export_raw_glb_dir is not None:
            try:
                export_raw_glb(
                    scene, filelist,
                    os.path.join(export_raw_glb_dir, "mast3r_raw_scene.glb"),
                )
            except Exception as e:
                logger.warning("Raw MASt3R .glb export failed (continuing): %s", e)

        # ── Extract results ───────────────────────────────────────────────────
        logger.info("Extracting camera poses …")
        cam_poses = _extract_poses(scene, filelist, frames)

        logger.info("Extracting 3D pointcloud and observations …")
        points_3d, observations, dense = _extract_pointcloud(
            scene, filelist, frames, cam_poses, keep_dense=keep_dense,
        )

        # Tear down model before returning (VRAM for Ceres host is fine; Ceres is CPU)
        del model, scene, images, pairs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("MASt3R VRAM released.")

    return MASt3RResult(
        camera_poses=cam_poses,
        points_3d=points_3d,
        observations=observations,
        dense=dense,
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


# ─────────────────────────────────────────────────────────────────────────────
# Blender .glb mesh export (debug visualization)
# ─────────────────────────────────────────────────────────────────────────────
#
# Both functions below reuse dust3r's own viz building blocks directly —
# the same ones that power the "download .glb" button in MASt3R/dust3r's
# official demo (dust3r.demo._convert_scene_output_to_glb does the same
# three steps). We don't import dust3r.demo itself: it imports gradio at
# module level, which we don't want as a pipeline dependency just to reuse
# one helper function.
#
#   pts3d_to_trimesh(img, pts3d, valid) — one view's dense per-pixel
#       pointmap → textured triangle mesh (quad-per-pixel, face-colored
#       from the RGB image).
#   cat_meshes(meshes)   — merges per-view meshes into one.
#   add_scene_cam(...)   — a camera-frustum gizmo with the actual photo
#       textured onto a card at the frustum's near plane.

def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.array(x)


def export_raw_glb(scene, filelist: list[str], out_path: str | Path) -> None:
    """
    Export MASt3R's raw reconstruction — its own native coordinate frame,
    pre-Ceres, pre-Sim(3) — as a single .glb: per-view textured mesh plus a
    reference card (the actual photo) at each camera's frustum.

    Must be called with the live `scene` object, before run_complete_graph
    tears it down. Independent of the Ceres solve — works even under
    --run-matcher-only.
    """
    import trimesh
    from dust3r.viz import pts3d_to_trimesh, cat_meshes, add_scene_cam
    from dust3r.utils.geometry import get_med_dist_between_poses

    imgs_f   = [_to_numpy(im) for im in scene.imgs]               # float [0,1], (H,W,3)
    imgs_u8  = [np.clip(im * 255.0, 0, 255).astype(np.uint8) for im in imgs_f]
    focals   = _to_numpy(scene.get_focals())                      # (N,)
    cams2w   = _to_numpy(scene.get_im_poses())                    # (N,4,4)
    pts3d_list, _depthmaps, confs_list = scene.get_dense_pts3d(clean_depth=True, subsample=8)

    meshes = []
    for i in range(len(filelist)):
        h, w = imgs_u8[i].shape[:2]
        pts   = _to_numpy(pts3d_list[i]).reshape(h, w, 3)
        conf  = _to_numpy(confs_list[i]).reshape(h, w).astype(np.float32)
        valid = _adaptive_conf_mask(conf)
        meshes.append(pts3d_to_trimesh(imgs_u8[i], pts, valid))

    scene_glb = trimesh.Scene()
    if meshes:
        scene_glb.add_geometry(trimesh.Trimesh(**cat_meshes(meshes)))

    # MASt3R's own units are arbitrary — size reference cards relative to
    # the scene's own scale rather than a fixed constant.
    cam_size = max(1e-3, 0.1 * float(get_med_dist_between_poses(cams2w)))
    for i in range(len(filelist)):
        add_scene_cam(
            scene_glb, cams2w[i], (255, 60, 60),
            image=imgs_u8[i], focal=float(focals[i]),
            imsize=(imgs_u8[i].shape[1], imgs_u8[i].shape[0]),
            screen_width=cam_size,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene_glb.export(str(out_path))
    logger.info("Raw MASt3R scene exported to %s (%d view(s))", out_path, len(filelist))


def _apply_sim3(P: np.ndarray, scale: float, R_sim: np.ndarray, t_sim: np.ndarray) -> np.ndarray:
    """Apply scale * R_sim @ P + t_sim to an array of points, shape (..., 3)."""
    flat = P.reshape(-1, 3)
    return (scale * (flat @ R_sim.T) + t_sim).reshape(P.shape)


def export_aligned_glb(
    mast3r_result: MASt3RResult,
    points_3d_solved: np.ndarray,
    sim3: tuple[float, np.ndarray, np.ndarray],
    frames: list,
    out_path: str | Path,
) -> None:
    """
    Export the full-pipeline result as a single .glb, all in real ENU meters:
    MASt3R's dense per-view mesh (mast3r_result.dense) rigidly moved into ENU
    via the same Sim(3) transform applied to the final solved cameras, the
    Ceres-refined sparse point cloud (points_3d_solved, also Sim(3)-mapped),
    and reference cards using each frame's FINAL solved pose + K_undist +
    actual undistorted image.

    Requires mast3r_result.dense to be populated (run_complete_graph(...,
    keep_dense=True)) and frames to already carry their final solved
    R/position_enu (i.e. called after align_to_telemetry_sim3).
    """
    import trimesh
    from dust3r.viz import pts3d_to_trimesh, cat_meshes, add_scene_cam

    scale, R_sim, t_sim = sim3

    scene_glb = trimesh.Scene()

    # Dense mesh, moved into ENU.
    meshes = []
    for stem, (pts3d_hw3, img_u8, valid) in mast3r_result.dense.items():
        pts_enu = _apply_sim3(pts3d_hw3, scale, R_sim, t_sim)
        meshes.append(pts3d_to_trimesh(img_u8, pts_enu, valid))
    if meshes:
        scene_glb.add_geometry(trimesh.Trimesh(**cat_meshes(meshes)))

    # Ceres-refined sparse point cloud, moved into ENU.
    if len(points_3d_solved):
        pts_enu = _apply_sim3(np.asarray(points_3d_solved), scale, R_sim, t_sim)
        scene_glb.add_geometry(trimesh.PointCloud(pts_enu, colors=(255, 220, 0)))

    # Reference cards at each frame's FINAL solved pose (post-Sim3) — same
    # poses solved_cameras.json carries, so this overlays directly against
    # what the raycast app actually uses.
    ready = [f for f in frames if f.ready]
    cam_size = max(0.1, float(config.MAST3R_GLB_CAM_SIZE_M))
    for f in ready:
        # p_cam = R @ (p_world - position_enu)  ⟹  camera-to-world:
        c2w = np.eye(4)
        c2w[:3, :3] = f.R.T
        c2w[:3, 3]  = f.position_enu

        img_rgb = cv2.cvtColor(f.undistorted, cv2.COLOR_BGR2RGB) if f.undistorted is not None else None
        focal   = float(f.K_undist[0, 0]) if f.K_undist is not None else None
        imsize  = (img_rgb.shape[1], img_rgb.shape[0]) if img_rgb is not None else None

        add_scene_cam(
            scene_glb, c2w, (60, 180, 255),
            image=img_rgb, focal=focal, imsize=imsize,
            screen_width=cam_size,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene_glb.export(str(out_path))
    logger.info("Aligned (ENU) scene exported to %s (%d camera(s))", out_path, len(ready))
