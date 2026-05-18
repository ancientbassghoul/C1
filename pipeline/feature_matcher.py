"""
pipeline/feature_matcher.py – Ground feature matching for camera orientation calibration.

Key design decisions
────────────────────
1. AGL filter: frames below MIN_AGL_M are skipped — perspective too extreme.

2. Van masking: detected van bbox is excluded from keypoint matches via
   post-match pixel filtering.  The van sits above the ground plane and
   would otherwise corrupt the ground-scatter geometry.

3. Van corner constraints: for frames listed in config.VAN_FRAMES, the
   GroundingDINO bbox is used to generate 3-D corner reprojection constraints
   that provide a crucial vertical anchor (known van height) the ground
   scatter alone cannot supply.

4. Single joint solve (pyceres): all cameras + van pose are solved together
   in one Ceres optimisation using pairwise ground scatter + van corner costs.
   The sparse block structure is exploited by SPARSE_NORMAL_CHOLESKY.

5. Manual overrides (config.GIMBAL_PITCH_OVERRIDES) are never modified.
"""

from __future__ import annotations

import logging
import os
from itertools import combinations

import cv2
import numpy as np

from pipeline.orientation_solver import solve as ceres_solve
from pipeline.van_corners import (
    get_van_observations, estimate_van_position,
)
from pipeline.frame import Frame
from pipeline.pose import build_rotation
import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
MIN_BASELINE_M       = 3.0
MIN_GROUND_MATCHES   = 6
RANSAC_THRESHOLD     = 4.0
INITIAL_PITCH_DEG    = 0.0

# Frames below this AGL are skipped — perspective is too extreme.
MIN_AGL_M            = 3.5



# ─────────────────────────────────────────────────────────────────────────────
# Enhancement toggle
# ─────────────────────────────────────────────────────────────────────────────
_ENHANCE = True

def set_enhance(on: bool) -> None:
    global _ENHANCE
    _ENHANCE = on


# ─────────────────────────────────────────────────────────────────────────────
# Debug visualisation toggle
# ─────────────────────────────────────────────────────────────────────────────
_DEBUG = False

def set_debug(on: bool) -> None:
    global _DEBUG
    _DEBUG = on


def _save_debug_matches(
    frame_a:    'Frame',
    frame_b:    'Frame',
    pts_a:      np.ndarray,
    pts_b:      np.ndarray,
    ground_a:   tuple[int, int],
    ground_b:   tuple[int, int],
    van_bbox_a: tuple | None,
    van_bbox_b: tuple | None,
) -> None:
    """
    Save a side-by-side debug image for one matched frame pair.

    Left panel  – frame_a:  ground region (green band), van bbox (red),
                             matched keypoints (yellow dots).
    Right panel – frame_b:  same annotations.
    Cyan lines connect corresponding matched pairs across the two panels.

    Saved to {OUTPUT_DIR}/debug/match_{stem_a}__{stem_b}.png.
    """
    import pathlib

    out_dir = pathlib.Path(config.OUTPUT_DIR) / 'debug'
    out_dir.mkdir(parents=True, exist_ok=True)

    img_a = frame_a.undistorted.copy()
    img_b = frame_b.undistorted.copy()
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]

    # ── Ground region band (semi-transparent green) ───────────────────────────
    for img, (g0, g1) in [(img_a, ground_a), (img_b, ground_b)]:
        overlay = img.copy()
        h = img.shape[0]
        cv2.rectangle(overlay, (0, g0), (img.shape[1], g1), (0, 180, 0), -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        cv2.line(img, (0, g0), (img.shape[1], g0), (0, 200, 0), 1)
        cv2.line(img, (0, g1), (img.shape[1], g1), (0, 200, 0), 1)

    # ── Van bboxes (red rectangle) ────────────────────────────────────────────
    for img, bbox in [(img_a, van_bbox_a), (img_b, van_bbox_b)]:
        if bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), 2)
            cv2.putText(img, 'van', (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA)

    # ── Matched keypoints (yellow dots) ──────────────────────────────────────
    for img, pts in [(img_a, pts_a), (img_b, pts_b)]:
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 4, (0, 220, 220), -1, cv2.LINE_AA)

    # ── Side-by-side canvas ───────────────────────────────────────────────────
    h_out = max(ha, hb)
    canvas = np.zeros((h_out, wa + wb, 3), dtype=np.uint8)
    canvas[:ha, :wa]      = img_a
    canvas[:hb, wa:wa+wb] = img_b

    # ── Connecting lines (cyan, subsampled for readability) ───────────────────
    step = max(1, len(pts_a) // 40)
    for (xa, ya), (xb, yb) in zip(pts_a[::step], pts_b[::step]):
        p1 = (int(xa),      int(ya))
        p2 = (int(xb) + wa, int(yb))
        cv2.line(canvas, p1, p2, (255, 200, 0), 1, cv2.LINE_AA)

    # ── Labels ────────────────────────────────────────────────────────────────
    for x, label in [(8, frame_a.stem[-20:]), (wa + 8, frame_b.stem[-20:])]:
        cv2.rectangle(canvas, (x - 2, 0), (x + len(label) * 9, 20), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    n = len(pts_a)
    summary = f'{n} matches'
    cv2.putText(canvas, summary, (wa // 2 - 40, h_out - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)

    fname = f'match_{frame_a.stem[-16:]}__{frame_b.stem[-16:]}.png'
    out_path = out_dir / fname
    cv2.imwrite(str(out_path), canvas)
    logger.debug('Debug match image saved: %s', out_path)

def _enhance_np(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE + unsharp mask for better SuperPoint keypoint detection."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    blur  = cv2.GaussianBlur(eq, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(eq, 1.8, blur, -0.8, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial grid subsampling
# ─────────────────────────────────────────────────────────────────────────────

def _grid_subsample(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    image_w: int,
    ground_rows: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep at most one match per spatial grid cell.

    The grid is defined over the source frame's (pts_a) ground region.
    Cell assignment is by pts_a pixel coordinates.  The first match found
    in each cell is kept; since LightGlue outputs matches in descending
    confidence order, this keeps the best match per cell.

    Grid dimensions are set by config.MATCH_GRID_COLS × MATCH_GRID_ROWS
    (default 5 × 4 = 20 cells maximum per frame pair).
    """
    ga0, ga1   = ground_rows
    n_cols     = config.MATCH_GRID_COLS
    n_rows     = config.MATCH_GRID_ROWS
    row_height = max(ga1 - ga0, 1)

    cell_first: dict[tuple[int, int], int] = {}
    for i, pa in enumerate(pts_a):
        col = int(pa[0] / max(image_w, 1) * n_cols)
        row = int((pa[1] - ga0) / row_height * n_rows)
        col = max(0, min(col, n_cols - 1))
        row = max(0, min(row, n_rows - 1))
        key = (col, row)
        if key not in cell_first:
            cell_first[key] = i

    if not cell_first:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    idx = list(cell_first.values())
    return pts_a[idx], pts_b[idx]


# ─────────────────────────────────────────────────────────────────────────────
# LightGlue lazy init
# ─────────────────────────────────────────────────────────────────────────────
_extractor = None
_matcher   = None
_device    = None

def _load_models():
    global _extractor, _matcher, _device
    if _extractor is not None:
        return
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
    except ImportError:
        raise ImportError(
            "\nLightGlue not installed. Run:\n"
            "  venv\\Scripts\\pip install git+https://github.com/cvg/LightGlue.git\n"
        )
    import torch
    _device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _extractor = SuperPoint(max_num_keypoints=1024).eval().to(_device)
    _matcher   = LightGlue(features='superpoint').eval().to(_device)
    logger.info("LightGlue loaded on %s", _device)


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _content_bounds(gray: np.ndarray) -> tuple[int, int]:
    rows = np.where((gray > 8).mean(axis=1) > 0.10)[0]
    return (int(rows[0]), int(rows[-1])) if len(rows) >= 20 else (0, gray.shape[0] - 1)


def _ground_rows(frame: Frame) -> tuple[int, int]:
    """Lower 55% of content area, excluding HUD strips."""
    gray = cv2.cvtColor(frame.undistorted, cv2.COLOR_BGR2GRAY)
    c_top, c_bot = _content_bounds(gray)
    h = c_bot - c_top
    if h < 50:
        return c_top, c_bot
    top = c_top + 70 + int(0.45 * h)
    bot = c_bot - int(0.05 * h)
    return max(top, c_top + 70), bot


def _to_tensor(img_bgr: np.ndarray, device):
    import torch
    if _ENHANCE:
        gray = _enhance_np(img_bgr)
        rgb  = np.stack([gray, gray, gray], axis=2)
    else:
        rgb = img_bgr[:, :, ::-1].copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).float().to(_device) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# LightGlue matching
# ─────────────────────────────────────────────────────────────────────────────

def _match(frame_a: Frame, frame_b: Frame,
           van_bbox_a: tuple | None = None,
           van_bbox_b: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Match ground features using SuperPoint + LightGlue.

    Keypoints inside the van bounding boxes are filtered out after matching
    via pixel-coordinate masking.  The ground region is restricted to the
    lower 55% of the content area to avoid matching sky features.
    """
    import torch
    from lightglue.utils import rbd

    _load_models()

    ga0, ga1 = _ground_rows(frame_a)
    gb0, gb1 = _ground_rows(frame_b)

    img_t_a = _to_tensor(frame_a.undistorted, _device)
    img_t_b = _to_tensor(frame_b.undistorted, _device)

    # Use autocast only on CUDA; it has no effect (and may error) on CPU.
    device_type = _device.type
    with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
        feats_a = _extractor.extract(img_t_a)
        feats_b = _extractor.extract(img_t_b)
        result  = _matcher({'image0': feats_a, 'image1': feats_b})

    feats_a, feats_b, result = [rbd(x) for x in [feats_a, feats_b, result]]
    matches = result['matches']
    if len(matches) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    kp_a = feats_a['keypoints'][matches[:, 0]].cpu().numpy()
    kp_b = feats_b['keypoints'][matches[:, 1]].cpu().numpy()

    # Keep only ground-region matches
    mask = (
        (kp_a[:, 1] >= ga0) & (kp_a[:, 1] <= ga1) &
        (kp_b[:, 1] >= gb0) & (kp_b[:, 1] <= gb1)
    )

    # Exclude van regions
    if van_bbox_a is not None:
        x1, y1, x2, y2 = van_bbox_a
        mask &= ~((kp_a[:, 0] >= x1) & (kp_a[:, 0] <= x2) &
                  (kp_a[:, 1] >= y1) & (kp_a[:, 1] <= y2))
    if van_bbox_b is not None:
        x1, y1, x2, y2 = van_bbox_b
        mask &= ~((kp_b[:, 0] >= x1) & (kp_b[:, 0] <= x2) &
                  (kp_b[:, 1] >= y1) & (kp_b[:, 1] <= y2))

    pts_a = kp_a[mask].astype(np.float32)
    pts_b = kp_b[mask].astype(np.float32)

    if len(pts_a) < MIN_GROUND_MATCHES:
        return np.empty((0, 2)), np.empty((0, 2))

    if len(pts_a) >= 8:
        _, rmask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, RANSAC_THRESHOLD)
        if rmask is not None:
            pts_a = pts_a[rmask.ravel().astype(bool)]
            pts_b = pts_b[rmask.ravel().astype(bool)]

    if len(pts_a) < MIN_GROUND_MATCHES:
        return np.empty((0, 2)), np.empty((0, 2))

    # Spatial grid subsampling — replaces random step subsampling
    pts_a, pts_b = _grid_subsample(pts_a, pts_b,
                                   frame_a.undistorted.shape[1],
                                   (ga0, ga1))

    if _DEBUG:
        _save_debug_matches(
            frame_a, frame_b, pts_a, pts_b,
            (ga0, ga1), (gb0, gb1),
            van_bbox_a, van_bbox_b,
        )

    return pts_a, pts_b





# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def refine_pitches(frames: list[Frame],
                   van_bboxes: dict | None = None) -> None:
    """
    Refine camera orientation (pitch, yaw offset, roll offset) jointly using
    pyceres with pairwise ground-scatter and van corner reprojection costs.

    van_bboxes : dict mapping frame.stem → (x1,y1,x2,y2) from GroundingDINO.
                 Used both to mask keypoint matches and to build van corner
                 reprojection constraints for frames in config.VAN_FRAMES.

    Frames listed in config.GIMBAL_PITCH_OVERRIDES are never modified.
    """
    vb = van_bboxes or {}

    # ── Eligible frames ───────────────────────────────────────────────────────
    ready = [
        f for f in frames
        if f.undistorted   is not None
        and f.position_enu is not None
        and f.K_undist     is not None
        and f.heading_deg  is not None
        and f.stem not in config.GIMBAL_PITCH_OVERRIDES
    ]

    if len(ready) < 2:
        logger.warning("refine_pitches: fewer than 2 eligible frames.")
        return

    # ── Feature matching ──────────────────────────────────────────────────────
    pairwise_features: list[tuple] = []   # (fi, pixel_i, fj, pixel_j)

    for fa, fb in combinations(ready, 2):
        if np.linalg.norm(fb.position_enu - fa.position_enu) < MIN_BASELINE_M:
            continue
        pts_a, pts_b = _match(fa, fb,
                               van_bbox_a=vb.get(fa.stem),
                               van_bbox_b=vb.get(fb.stem))
        if len(pts_a) < MIN_GROUND_MATCHES:
            continue
        logger.info("  %s ↔ %s : %d matches",
                    fa.stem[-12:], fb.stem[-12:], len(pts_a))
        for (ua, va), (ub, vb_) in zip(pts_a, pts_b):
            pairwise_features.append(
                (fa, (float(ua), float(va)),
                 fb, (float(ub), float(vb_)))
            )

    if not pairwise_features:
        logger.warning("No ground matches found — skipping refinement.")
        return

    logger.info("%d total ground feature pairs across %d frames.",
                len(pairwise_features), len(ready))

    # ── Van pose initialisation ───────────────────────────────────────────────
    # Find the best van frame with a bbox to seed the van position
    van_frames_with_bbox = [
        f for f in ready
        if f.stem in config.VAN_FRAMES and f.stem in vb
    ]

    van_east_init  = 0.0
    van_north_init = 0.0

    if van_frames_with_bbox:
        seed_frame = van_frames_with_bbox[0]
        van_east_init, van_north_init = estimate_van_position(
            seed_frame, vb[seed_frame.stem], van_z=config.VAN_Z_M,
        )
        logger.info(
            "Van pose seed: east=%.1f m  north=%.1f m  heading=%.1f°",
            van_east_init, van_north_init, config.VAN_HEADING_PRIOR_DEG,
        )
    else:
        logger.warning(
            "No VAN_FRAMES with bbox found — van corner constraints disabled."
        )

    van_pose_init = [van_east_init, van_north_init, config.VAN_HEADING_PRIOR_DEG]

    # ── Van corner observations ───────────────────────────────────────────────
    van_observations: list[tuple] = []

    for f in van_frames_with_bbox:
        obs = get_van_observations(
            f, vb[f.stem],
            van_east_init, van_north_init,
            config.VAN_HEADING_PRIOR_DEG,
            van_z=config.VAN_Z_M,
        )
        van_observations.extend((f, corner_local, pixel_obs)
                                 for corner_local, pixel_obs in obs)

    logger.info("%d van corner observations across %d frame(s).",
                len(van_observations), len(van_frames_with_bbox))

    # ── Joint Ceres solve ─────────────────────────────────────────────────────
    # Pass GeoCalib pitch estimates as seeds so Ceres uses a tight per-frame
    # window rather than the full physical pitch range.
    pitch_seeds = {
        f: f.gimbal_pitch_deg
        for f in ready
        if f.gimbal_pitch_deg is not None
    }
    cam_results, van_pose_solved, report = ceres_solve(
        ready, pairwise_features, van_observations, van_pose_init,
        pitch_seeds=pitch_seeds,
    )

    logger.info(
        "Van pose solved: east=%.1f m  north=%.1f m  heading=%.1f°",
        van_pose_solved[0], van_pose_solved[1], van_pose_solved[2],
    )

    # ── Apply results to frames ───────────────────────────────────────────────
    for f in ready:
        params = cam_results[f]
        pitch = float(params[0])
        yo    = float(params[1])
        ro    = float(params[2])
        logger.info(
            "  [%s] pitch → %.1f°  yaw_off → %+.2f°  roll_off → %+.2f°",
            f.stem[-12:], pitch, yo, ro,
        )
        f.gimbal_pitch_deg  = pitch
        f.heading_deg       = f.heading_deg     + yo
        f.camera_roll_deg   = f.camera_roll_deg + ro
        f.R = build_rotation(f.heading_deg, pitch, f.camera_roll_deg)
