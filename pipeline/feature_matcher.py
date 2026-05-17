"""
pipeline/feature_matcher.py – Ground feature matching for pitch calibration (SuperPoint + LightGlue).

Key design decisions
────────────────────
1. AGL filter: frames below MIN_AGL_M are skipped (too low, weird
   perspective, ground features look like horizon features in other cams).

2. Van masking: detected van bbox is excluded from keypoint matches
   via post-match pixel filtering.

3. Incremental solve:
   Pass 1 – solve "stable" frames globally (those with enough cross-frame
             ground matches, MIN_MATCHES_STABLE threshold).
   Pass 2 – for each remaining frame, fix all previously solved cameras
             and solve for this frame's pitch alone (1 unknown → robust).

4. Manual overrides (config.GIMBAL_PITCH_OVERRIDES) are never touched.
"""

from __future__ import annotations

import logging
import os
from itertools import combinations

import cv2
import numpy as np

from pipeline.orientation_solver import optimize_orientation
from pipeline.frame import Frame
from pipeline.pose import build_rotation
import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
MIN_BASELINE_M       = 3.0
MIN_GROUND_MATCHES   = 6
MAX_MATCHES_PAIR     = 80
RANSAC_THRESHOLD     = 4.0
INITIAL_PITCH_DEG    = 0.0

# Frames below this AGL are skipped — perspective is too extreme.
MIN_AGL_M            = 3.5

# Minimum matches a frame needs across ALL its pairs to be considered "stable"
# and included in the first global pass.
MIN_MATCHES_STABLE   = 12


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

    if len(pts_a) > MAX_MATCHES_PAIR:
        step  = len(pts_a) // MAX_MATCHES_PAIR
        pts_a = pts_a[::step][:MAX_MATCHES_PAIR]
        pts_b = pts_b[::step][:MAX_MATCHES_PAIR]

    if _DEBUG:
        _save_debug_matches(
            frame_a, frame_b, pts_a, pts_b,
            (ga0, ga1), (gb0, gb1),
            van_bbox_a, van_bbox_b,
        )

    return pts_a, pts_b


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cam_dict(frame: Frame) -> dict:
    return {
        'pos':  frame.position_enu.copy(),
        'yaw':  frame.heading_deg,
        'roll': frame.camera_roll_deg,
        'K':    frame.K_undist,
    }


def _run_optimizer(cameras, features, initial_pitches,
                   label='') -> tuple | None:
    if not features:
        logger.warning("No features for optimizer%s.", f' ({label})' if label else '')
        return None
    try:
        pitches, yaw_offs, roll_offs, result = optimize_orientation(
            cameras, features,
            initial_pitches=initial_pitches,
            z_ground=0.0,
            verbose=True,
        )
        pf = result.cost / max(len(features), 1)
        logger.info("Optimizer%s done: per_feature=%.2f m²  nfev=%d",
                    f' ({label})' if label else '', pf, result.nfev)
        return pitches, yaw_offs, roll_offs
    except Exception as exc:
        logger.error("Optimizer failed%s: %s",
                     f' ({label})' if label else '', exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def refine_pitches(frames: list[Frame],
                   van_bboxes: dict | None = None) -> None:
    """
    Refine gimbal pitch using SuperPoint + LightGlue ground-feature matching.

    van_bboxes : optional dict mapping frame.stem → (x1,y1,x2,y2) to exclude
                 the van region from feature matching.

    Two-pass incremental solve:
      Pass 1 – global solve on stable frames (enough cross-frame matches,
               ≥ MIN_MATCHES_STABLE total matched keypoints).
      Pass 2 – remaining frames solved one at a time against calibrated cams.

    Frames listed in config.GIMBAL_PITCH_OVERRIDES are never modified.
    """
    # Exclude only frames with manual overrides — optimizer handles everything else
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

    # ── Build match table ─────────────────────────────────────────────────────
    # pair_features[stem_a][stem_b] = list of {ia: pix_a, ib: pix_b}
    vb = van_bboxes or {}
    match_count: dict[str, int] = {f.stem: 0 for f in ready}
    pair_data: dict[tuple, list] = {}

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
        match_count[fa.stem] += len(pts_a)
        match_count[fb.stem] += len(pts_a)
        pair_data[(fa.stem, fb.stem)] = (fa, fb, pts_a, pts_b)

    if not pair_data:
        logger.warning("No ground matches found — skipping refinement.")
        return

    # ── Pass 1: global solve on stable frames ─────────────────────────────────
    stable   = [f for f in ready if match_count[f.stem] >= MIN_MATCHES_STABLE]
    unstable = [f for f in ready if match_count[f.stem] < MIN_MATCHES_STABLE]

    logger.info("Pass 1: %d stable, %d unstable frame(s).",
                len(stable), len(unstable))

    if len(stable) >= 2:
        s_idx     = {f.stem: i for i, f in enumerate(stable)}
        s_cams    = [_cam_dict(f) for f in stable]
        s_features: list[dict] = []
        for (sa, sb), (fa, fb, pts_a, pts_b) in pair_data.items():
            if fa.stem not in s_idx or fb.stem not in s_idx:
                continue
            ia, ib = s_idx[fa.stem], s_idx[fb.stem]
            for (ua, va), (ub, vb_) in zip(pts_a, pts_b):
                s_features.append({ia: (float(ua), float(va)),
                                    ib: (float(ub), float(vb_))})

        result_tuple = _run_optimizer(s_cams, s_features,
                                      [INITIAL_PITCH_DEG] * len(stable),
                                      label='pass-1')
        if result_tuple is not None:
            pitches, yaw_offs, roll_offs = result_tuple
            for i, f in enumerate(stable):
                p  = float(pitches[i])
                yo = float(yaw_offs[i])
                ro = float(roll_offs[i])
                logger.info("  [%s] pass-1 pitch → %.1f°  yaw_off → %+.2f°  roll_off → %+.2f°",
                            f.stem[-12:], p, yo, ro)
                f.gimbal_pitch_deg  = p
                f.heading_deg       = f.heading_deg      + yo
                f.camera_roll_deg   = f.camera_roll_deg  + ro
                f.R = build_rotation(f.heading_deg, p, f.camera_roll_deg)

    # ── Pass 2: solve each unstable frame against ALL calibrated cameras ──────
    calibrated = [f for f in ready
                  if f.gimbal_pitch_deg is not None
                  and f.stem not in [u.stem for u in unstable]]

    for fu in unstable:
        logger.info("Pass 2: solving [%s] against %d calibrated frame(s)…",
                    fu.stem[-12:], len(calibrated))
        features_u: list[dict] = []
        cams_u = [_cam_dict(fu)]   # index 0 = this frame (unknown pitch)

        for i, fc in enumerate(calibrated):
            key  = (fu.stem, fc.stem) if (fu.stem, fc.stem) in pair_data \
                   else (fc.stem, fu.stem) if (fc.stem, fu.stem) in pair_data \
                   else None
            if key is None:
                continue
            fa, fb, pts_a, pts_b = pair_data[key]
            # Determine which is fu and which is fc
            if fa.stem == fu.stem:
                pa, pb = pts_a, pts_b
            else:
                pa, pb = pts_b, pts_a

            cams_u.append(_cam_dict(fc))
            ci = len(cams_u) - 1   # index of this calibrated cam
            for (ua, va), (ub, vb_) in zip(pa, pb):
                features_u.append({0: (float(ua), float(va)),
                                    ci: (float(ub), float(vb_))})

        if not features_u:
            logger.info("  [%s] no matches with calibrated frames — skip.",
                        fu.stem[-12:])
            continue

        # Build initial pitches matching cams_u order:
        # index 0 is the unknown frame; indices 1+ are calibrated (fixed tightly).
        init = [INITIAL_PITCH_DEG] + [
            fc.gimbal_pitch_deg or INITIAL_PITCH_DEG
            for fc in calibrated
            if _cam_dict(fc) in cams_u[1:]
        ]

        pitches_u, yaw_offs_u, roll_offs_u, result_u = optimize_orientation(
            cams_u, features_u,
            initial_pitches=init,
            z_ground=0.0,
            verbose=False,
        )

        p  = float(pitches_u[0])
        yo = float(yaw_offs_u[0])
        ro = float(roll_offs_u[0])
        pf = result_u.cost / max(len(features_u), 1)
        logger.info("  [%s] pass-2 pitch → %.1f°  yaw_off → %+.2f°  roll_off → %+.2f°  per_feature=%.2f m²",
                    fu.stem[-12:], p, yo, ro, pf)
        fu.gimbal_pitch_deg  = p
        fu.heading_deg       = fu.heading_deg     + yo
        fu.camera_roll_deg   = fu.camera_roll_deg + ro
        fu.R = build_rotation(fu.heading_deg, p, fu.camera_roll_deg)
