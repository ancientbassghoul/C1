"""
pipeline/van.py – Van 3D model, YOLO detection, and joint self-calibration.

Uses the white van visible in multiple frames as a calibration target to
jointly solve for:
  - Effective focal length f_eff  (shared across all frames, same camera)
  - Van world position  (ENU x, y;  Z is pinned to the ground plane)
  - Van compass yaw
  - Gimbal pitch for each frame where the van is detected

Why this works
──────────────
The van is a rigid 3D box with known dimensions.  Its roof corners sit ~2 m
above the ground plane, breaking the scene planarity that causes the
Essential-matrix approach to fail.  A single YOLO-detected 2D bounding box
per frame gives 4 constraints.  Detected 2D Shi-Tomasi corners — validated
against the YOLO bbox so they never contradict it — add further constraints.
With 3+ frames the system is well over-determined.

After calibration
─────────────────
  - All frame.K_undist matrices are updated with the refined f_eff.
  - Pitch estimates for van-visible frames are replaced with optimised values.
  - The VanModel is returned for use in ray–box intersection when the user
    clicks the van in the interactive viewer.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from pipeline.frame import Frame
from pipeline.geometry import unproject_pixel, intersect_ground_plane
from pipeline.pose import build_rotation
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Van 3D model
# ─────────────────────────────────────────────────────────────────────────────

class VanModel:
    """
    Rigid oriented bounding box model of the white van.

    Local frame
    ───────────
    Origin  = centre of the van's base (ground contact point)
    X axis  = right (starboard), perpendicular to the nose direction
    Y axis  = forward (bow), the van's nose direction
    Z axis  = up

    Corner indexing (0-7)
    ─────────────────────
    Bottom face (Z=0):  0=rear-left   1=rear-right  2=front-right  3=front-left
    Top    face (Z=H):  4=rear-left   5=rear-right  6=front-right  7=front-left
    """

    CORNER_NAMES = [
        "rear-left-bot",  "rear-right-bot",  "front-right-bot", "front-left-bot",
        "rear-left-top",  "rear-right-top",  "front-right-top", "front-left-top",
    ]

    def __init__(
        self,
        width:  float = config.VAN_WIDTH_M,
        length: float = config.VAN_LENGTH_M,
        height: float = config.VAN_HEIGHT_M,
    ) -> None:
        self.width  = width    # metres, side-to-side
        self.length = length   # metres, nose-to-tail
        self.height = height   # metres, ground-to-roof

        # Set by calibrate()
        self.position_enu: Optional[np.ndarray] = None   # (3,) ENU
        self.yaw_deg:      Optional[float]       = None   # compass, CW from N

    @property
    def calibrated(self) -> bool:
        return self.position_enu is not None and self.yaw_deg is not None

    # ── Internal geometry ─────────────────────────────────────────────────────

    def _R_local_to_enu(self) -> np.ndarray:
        """
        3×3 rotation: van-local → ENU world.

        For van yaw θ (compass, CW from North):
          local X (right)   → ENU [cos θ, -sin θ, 0]
          local Y (forward) → ENU [sin θ,  cos θ, 0]
          local Z (up)      → ENU [0,       0,    1]

        Verification:
          θ=0  (North-facing):  right=East [1,0,0] ✓  forward=North [0,1,0] ✓
          θ=90 (East-facing):   right=South [0,-1,0] ✓  forward=East [1,0,0] ✓
        """
        θ = math.radians(self.yaw_deg)
        return np.array([
            [ math.cos(θ),  math.sin(θ), 0.0],
            [-math.sin(θ),  math.cos(θ), 0.0],
            [ 0.0,          0.0,         1.0],
        ], dtype=np.float64)

    def corners_local(self) -> np.ndarray:
        """(8, 3) corner coordinates in van-local frame."""
        W, L, H = self.width / 2, self.length / 2, self.height
        return np.array([
            [-W, -L, 0], [+W, -L, 0], [+W, +L, 0], [-W, +L, 0],   # bottom
            [-W, -L, H], [+W, -L, H], [+W, +L, H], [-W, +L, H],   # top
        ], dtype=np.float64)

    def corners_3d(self) -> np.ndarray:
        """(8, 3) corner coordinates in ENU world frame."""
        assert self.calibrated, "VanModel not yet calibrated."
        R = self._R_local_to_enu()
        return self.position_enu + (R @ self.corners_local().T).T

    # ── Projection ────────────────────────────────────────────────────────────

    def project_corners(
        self,
        camera_pos: np.ndarray,
        R_cam:      np.ndarray,
        K:          np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project all 8 corners through the camera.

        Returns
        ───────
        pixels  : (8, 2) float – (x, y) pixel for each corner (NaN if behind)
        visible : (8,)   bool  – True where the corner is in front of camera
        """
        corners_w = self.corners_3d()
        pixels  = np.full((8, 2), np.nan)
        visible = np.zeros(8, dtype=bool)

        for i, cw in enumerate(corners_w):
            p_cam = R_cam @ (cw - camera_pos)
            if p_cam[2] <= 0.1:
                continue
            pixels[i, 0] = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
            pixels[i, 1] = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
            visible[i]   = True

        return pixels, visible

    def predicted_bbox(
        self,
        camera_pos:  np.ndarray,
        R_cam:       np.ndarray,
        K:           np.ndarray,
        image_shape: tuple[int, int],
    ) -> Optional[tuple[float, float, float, float]]:
        """
        Predicted 2D bounding box (x1, y1, x2, y2).
        Returns None if fewer than 2 corners project in front of the camera.
        """
        pixels, visible = self.project_corners(camera_pos, R_cam, K)
        vis = pixels[visible]
        if len(vis) < 2:
            return None
        h, w = image_shape[:2]
        return (
            float(np.clip(vis[:, 0].min(), 0, w)),
            float(np.clip(vis[:, 1].min(), 0, h)),
            float(np.clip(vis[:, 0].max(), 0, w)),
            float(np.clip(vis[:, 1].max(), 0, h)),
        )

    # ── Ray–box intersection ──────────────────────────────────────────────────

    def intersect_ray(
        self,
        origin:    np.ndarray,
        direction: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Intersect a world-frame ray with this oriented bounding box.

        Transforms the ray into van-local frame, performs an AABB slab test,
        then transforms the hit point back to ENU.
        Returns the first intersection in ENU (in front of the camera), or None.
        """
        if not self.calibrated:
            return None

        R_l2e = self._R_local_to_enu()
        R_e2l = R_l2e.T

        o_l = R_e2l @ (origin    - self.position_enu)
        d_l = R_e2l @  direction

        W, L, H = self.width / 2, self.length / 2, self.height
        slabs = [
            ((-W - o_l[0]) / d_l[0] if abs(d_l[0]) > 1e-8 else None,
             (+W - o_l[0]) / d_l[0] if abs(d_l[0]) > 1e-8 else None,
             o_l[0], -W, +W),
            ((-L - o_l[1]) / d_l[1] if abs(d_l[1]) > 1e-8 else None,
             (+L - o_l[1]) / d_l[1] if abs(d_l[1]) > 1e-8 else None,
             o_l[1], -L, +L),
            ((0  - o_l[2]) / d_l[2] if abs(d_l[2]) > 1e-8 else None,
             (H  - o_l[2]) / d_l[2] if abs(d_l[2]) > 1e-8 else None,
             o_l[2],  0,   H),
        ]

        t_near = -np.inf
        t_far  =  np.inf

        for t1, t2, o_i, lo, hi in slabs:
            if t1 is None:          # ray parallel to this slab
                if o_i < lo or o_i > hi:
                    return None     # ray misses entirely
            else:
                t_near = max(t_near, min(t1, t2))
                t_far  = min(t_far,  max(t1, t2))

        if t_far < t_near or t_far < 0:
            return None

        t = t_near if t_near > 0 else t_far
        hit_local = o_l + t * d_l
        return self.position_enu + R_l2e @ hit_local


# ─────────────────────────────────────────────────────────────────────────────
# YOLO vehicle detector
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# White-blob fallback detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_white_blob(
    img: np.ndarray,
) -> Optional[tuple[float, float, float, float]]:
    """
    Fallback van detector based on colour: the van is the largest pure-white
    object in the scene outside the HUD zones.

    Strategy
    ────────
    1. Convert to HSV and threshold for high-value, low-saturation pixels
       (pure white in any lighting).
    2. Mask out the sky (top 35 % of image) and HUD strips (top/bottom 12 %).
    3. Find connected components; return the bounding box of the largest blob
       that has a plausible van aspect ratio (width/height between 0.5 and 4).

    Returns (x1, y1, x2, y2) or None.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # White: low saturation, high value
    mask = cv2.inRange(hsv,
                       np.array([0,   0, 200], dtype=np.uint8),
                       np.array([180, 60, 255], dtype=np.uint8))

    # Mask out sky (top 35 %) and HUD strips (top/bottom 12 %)
    mask[:int(h * 0.35), :] = 0
    mask[:int(h * 0.12), :] = 0
    mask[int(h * 0.88):, :] = 0

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_area = config.VAN_BLOB_MIN_AREA
    best_bbox = None

    for i in range(1, n):
        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < best_area:
            continue

        # Reject blobs that are too wide relative to the image — these are
        # sky/sun glare, not a van (which is a compact object).
        if bw > w * 0.25:
            continue

        # Reject unlikely aspect ratios (van is roughly box-shaped)
        aspect = bw / bh if bh > 0 else 0
        if not (0.4 <= aspect <= 3.0):
            continue

        # Require the blob centre to be in the lower 60% of the image.
        # Sky and glare live in the upper portion; the van is always on the ground.
        center_y = y + bh / 2
        if center_y < h * 0.40:
            continue

        best_area = area
        best_bbox = (float(x), float(y), float(x + bw), float(y + bh))

    return best_bbox


# ─────────────────────────────────────────────────────────────────────────────
# YOLO detector with white-blob fallback
# ─────────────────────────────────────────────────────────────────────────────

class VanDetector:
    """
    Primary: YOLOv8 vehicle detection.
    Fallback: white-blob colour segmentation when YOLO confidence is too low.

    The van is the largest pure-white object in the scene in every frame,
    making colour-based detection highly reliable as a backup.
    """

    VEHICLE_CLASSES = {2, 5, 7}   # COCO: car, bus, truck

    def __init__(self, model_name: str = config.YOLO_MODEL) -> None:
        self._model      = None
        self._model_name = model_name

    def _load(self) -> None:
        if self._model is None:
            from ultralytics import YOLO
            logger.info("Loading YOLO model '%s'…", self._model_name)
            self._model = YOLO(self._model_name)

    def detect(
        self, frame: Frame,
    ) -> Optional[tuple[float, float, float, float]]:
        """
        Detect the van in frame.undistorted.

        1. Try YOLO — accept if confidence ≥ YOLO_CONF_THRESH.
        2. If YOLO fails, try white-blob fallback.
        3. Return whichever succeeds first, or None.
        """
        self._load()
        if frame.undistorted is None:
            return None

        # ── YOLO ──────────────────────────────────────────────────────────────
        best_bbox = None
        best_conf = config.YOLO_CONF_THRESH

        results = self._model(frame.undistorted, verbose=False)
        for result in results:
            for box in result.boxes:
                cls  = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls in self.VEHICLE_CLASSES and conf > best_conf:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_bbox = (x1, y1, x2, y2)
                    best_conf = conf

        if best_bbox is not None:
            logger.info(
                "[%s] Van detected via YOLO  bbox=(%.0f,%.0f → %.0f,%.0f)  conf=%.2f",
                frame.stem, *best_bbox, best_conf,
            )
            return best_bbox

        # ── White-blob fallback ────────────────────────────────────────────────
        blob = _detect_white_blob(frame.undistorted)
        if blob is not None:
            logger.info(
                "[%s] Van detected via white-blob  bbox=(%.0f,%.0f → %.0f,%.0f)",
                frame.stem, *blob,
            )
        else:
            logger.warning("[%s] Van not detected by either method.", frame.stem)
        return blob

    def detect_all(
        self, frames: list[Frame],
    ) -> dict[Frame, tuple[float, float, float, float]]:
        """Run detection on all frames. Returns Frame → bbox for hits only."""
        detections: dict = {}
        for i, frame in enumerate(frames):
            logger.info("Van detection %d/%d: %s", i + 1, len(frames), frame.stem)
            bbox = self.detect(frame)
            if bbox is not None:
                detections[frame] = bbox
        yolo_count = sum(1 for f in detections)
        logger.info("Van detected in %d/%d frame(s).", len(detections), len(frames))
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Shi-Tomasi corner matching
# ─────────────────────────────────────────────────────────────────────────────

def detect_van_corners(
    frame:             Frame,
    bbox:              tuple,
    predicted_pixels:  np.ndarray,
    visible:           np.ndarray,
    threshold_px:      float = config.VAN_CORNER_MATCH_THRESHOLD_PX,
) -> list[tuple[int, float, float]]:
    """
    Match van corner predictions to Shi-Tomasi corners in the YOLO crop.

    Only predicted corners that lie *inside* the YOLO bbox are considered
    (the "don't contradict the bbox" check).  Detected corners outside the
    bbox are ignored even if they're close to a prediction.

    Returns a list of (corner_id, measured_x, measured_y) triples.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h_img, w_img   = frame.undistorted.shape[:2]

    pad  = 15
    cx0  = max(0,     x1 - pad);  cy0 = max(0,     y1 - pad)
    cx1  = min(w_img, x2 + pad);  cy1 = min(h_img, y2 + pad)
    crop = frame.undistorted[cy0:cy1, cx0:cx1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    detected = cv2.goodFeaturesToTrack(
        gray, maxCorners=40, qualityLevel=0.02, minDistance=6,
    )
    if detected is None:
        return []

    # Translate crop-local coords to full-image coords
    det_pts = [
        (float(c[0][0]) + cx0, float(c[0][1]) + cy0)
        for c in detected
    ]

    matches: list[tuple[int, float, float]] = []
    for cid in range(8):
        if not visible[cid]:
            continue
        pred_x, pred_y = predicted_pixels[cid]
        # Predicted corner must be inside the YOLO bbox
        if not (x1 <= pred_x <= x2 and y1 <= pred_y <= y2):
            continue

        best_d, best_pt = threshold_px, None
        for dx, dy in det_pts:
            if not (x1 <= dx <= x2 and y1 <= dy <= y2):
                continue   # detected corner outside bbox → ignore
            d = math.hypot(dx - pred_x, dy - pred_y)
            if d < best_d:
                best_d, best_pt = d, (dx, dy)

        if best_pt:
            matches.append((cid, best_pt[0], best_pt[1]))

    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap: rough initial van position from ray intersections
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_van_position(
    frames:     list[Frame],
    detections: dict[Frame, tuple],
) -> Optional[np.ndarray]:
    """
    Estimate the van's initial ENU (x, y) position by unprojecting the
    centre of each YOLO bbox and intersecting with the ground plane.

    This uses the current (imperfect) pitch estimates so the result is
    only an approximate seed for the optimiser.  Returns (2,) or None.
    """
    positions = []
    for frame, bbox in detections.items():
        if not frame.ready:
            continue
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        origin, direction = unproject_pixel(
            cx, cy, frame.K_undist, frame.R, frame.position_enu,
        )
        z = frame.terrain_offset_m if frame.terrain_offset_m is not None else 0.0
        pt = intersect_ground_plane(origin, direction, z=z)
        if pt is not None:
            positions.append(pt[:2])

    if not positions:
        return None
    arr = np.array(positions)
    median = np.median(arr, axis=0)
    logger.info(
        "Bootstrap van position: [%.1f, %.1f] ENU  (from %d frames)",
        median[0], median[1], len(positions),
    )
    return median


# ─────────────────────────────────────────────────────────────────────────────
# Residual function
# ─────────────────────────────────────────────────────────────────────────────

def _build_K(f_eff: float, frame: Frame) -> np.ndarray:
    """Camera matrix with updated focal length, keeping principal point."""
    K = frame.K_undist.copy()
    K[0, 0] = f_eff
    K[1, 1] = f_eff
    return K


# Type alias for pre-detected corners: list of (corner_id, meas_x, meas_y)
CornerList = list[tuple[int, float, float]]


def _residuals(
    params:          np.ndarray,
    van_frames:      list[Frame],
    detections:      dict[Frame, tuple],
    van_model:       VanModel,
    fixed_corners:   dict,   # Frame → CornerList (pre-detected, fixed size)
) -> np.ndarray:
    """
    Residual vector for scipy.optimize.least_squares.

    params = [f_eff, van_x, van_y, van_yaw_deg, pitch_0, …, pitch_N-1]

    Residuals (FIXED size — critical for scipy's Jacobian estimation):
      • 4 bbox residuals per frame         (x1, y1, x2, y2) predicted vs YOLO
      • 2 × M corner residuals per frame   using pre-detected corners (fixed)
    All normalised by image diagonal for scale invariance.
    """
    f_eff   = params[0]
    van_x   = params[1]
    van_y   = params[2]
    van_yaw = params[3]
    pitches = params[4:]

    resid = []

    for i, frame in enumerate(van_frames):
        bbox  = detections[frame]
        pitch = pitches[i]
        roll  = config.CAMERA_ROLL_OVERRIDES.get(frame.stem, config.CAMERA_ROLL_DEG)

        K = _build_K(f_eff, frame)
        R = build_rotation(frame.heading_deg, pitch, roll)

        z = frame.terrain_offset_m if frame.terrain_offset_m is not None else 0.0
        van_model.position_enu = np.array([van_x, van_y, z])
        van_model.yaw_deg      = van_yaw % 360.0

        pred = van_model.predicted_bbox(
            frame.position_enu, R, K, frame.undistorted.shape,
        )

        h, w = frame.undistorted.shape[:2]
        diag = math.hypot(w, h)
        x1, y1, x2, y2 = bbox

        if pred is None:
            # Heavy penalty — keep size fixed (4 residuals)
            resid.extend([100.0] * 4)
        else:
            px1, py1, px2, py2 = pred
            resid.extend([
                (px1 - x1) / diag,
                (py1 - y1) / diag,
                (px2 - x2) / diag,
                (py2 - y2) / diag,
            ])

        # Fixed corner residuals — pre-detected once before optimisation,
        # so the vector length never changes between Jacobian calls.
        pixels, _ = van_model.project_corners(frame.position_enu, R, K)
        for cid, mx, my in fixed_corners.get(frame, []):
            # 2 residuals per corner (x and y), always present
            pred_x = float(pixels[cid, 0]) if not np.isnan(pixels[cid, 0]) else mx
            pred_y = float(pixels[cid, 1]) if not np.isnan(pixels[cid, 1]) else my
            resid.extend([
                (pred_x - mx) / diag,
                (pred_y - my) / diag,
            ])

    return np.array(resid, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Joint calibration entry point
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(
    frames:      list[Frame],
    detections:  dict[Frame, tuple],
    van_model:   VanModel,
    use_corners: bool = True,
) -> bool:
    """
    Jointly optimise f_eff, van pose, and per-frame gimbal pitch.

    Requires at least 2 frames with the van detected.
    Updates van_model, all frame.K_undist, and pitch/R for van-visible frames.
    Returns True if the optimiser ran (even partial convergence is applied).
    """
    if len(detections) < 2:
        logger.warning(
            "Van calibration needs ≥ 2 frames with van detected (%d found) – skipping.",
            len(detections),
        )
        return False

    van_frames = [f for f in frames if f in detections]

    # ── Initial position guess ────────────────────────────────────────────────
    init_pos = bootstrap_van_position(frames, detections)
    if init_pos is None:
        logger.warning("Bootstrap failed — cannot seed van position.")
        return False

    f0 = float(van_frames[0].K_undist[0, 0])

    x0 = np.concatenate([
        [f0, init_pos[0], init_pos[1], 0.0],   # f, van_x, van_y, van_yaw
        [
            f.gimbal_pitch_deg
            if f.gimbal_pitch_deg is not None else config.GIMBAL_PITCH_FALLBACK_DEG
            for f in van_frames
        ],
    ])

    # ── Bounds ────────────────────────────────────────────────────────────────
    f_lo, f_hi = config.VAN_CALIB_F_BOUNDS
    p_lo, p_hi = config.VAN_CALIB_PITCH_BOUNDS
    n          = len(van_frames)

    lower = [f_lo, -np.inf, -np.inf,   0.0] + [p_lo] * n
    upper = [f_hi,  np.inf,  np.inf, 360.0] + [p_hi] * n

    logger.info(
        "Van calibration: f0=%.0f  van_seed=[%.1f, %.1f]  "
        "%d frame(s)  corners=%s",
        f0, init_pos[0], init_pos[1], n, use_corners,
    )

    # ── Pre-detect corners ONCE (fixed size for Jacobian estimation) ─────────
    # Scipy perturbs params many times; re-detecting corners each call
    # gives variable-length residuals → crash.  We detect once and fix them.
    fixed_corners: dict = {}
    if use_corners:
        # Seed van model with initial values for corner prediction
        init_z = float(np.median([
            f.terrain_offset_m for f in van_frames
            if f.terrain_offset_m is not None
        ] or [0.0]))
        van_model.position_enu = np.array([x0[1], x0[2], init_z])
        van_model.yaw_deg      = x0[3]

        for i, frame in enumerate(van_frames):
            K0 = _build_K(x0[0], frame)
            R0 = build_rotation(
                frame.heading_deg,
                x0[4 + i],
                config.CAMERA_ROLL_OVERRIDES.get(frame.stem, config.CAMERA_ROLL_DEG),
            )
            pixels0, vis0 = van_model.project_corners(frame.position_enu, R0, K0)
            corners = detect_van_corners(
                frame, detections[frame], pixels0, vis0,
            )
            fixed_corners[frame] = corners
            logger.info(
                "[%s] Pre-detected %d van corner(s).", frame.stem, len(corners),
            )

    # ── Optimise ──────────────────────────────────────────────────────────────
    result = least_squares(
        _residuals,
        x0,
        bounds   = (lower, upper),
        args     = (van_frames, detections, van_model, fixed_corners),
        method   = "trf",
        max_nfev = 3000,
        ftol     = 1e-5,
        xtol     = 1e-5,
        verbose  = 0,
    )

    # ── Extract results ───────────────────────────────────────────────────────
    f_opt   = float(result.x[0])
    van_x   = float(result.x[1])
    van_y   = float(result.x[2])
    van_yaw = float(result.x[3]) % 360.0
    pitches = result.x[4:]

    # Van Z: median terrain offset across van-visible frames
    terrain_vals = [
        f.terrain_offset_m for f in van_frames
        if f.terrain_offset_m is not None
    ]
    van_z = float(np.median(terrain_vals)) if terrain_vals else 0.0

    van_model.position_enu = np.array([van_x, van_y, van_z])
    van_model.yaw_deg      = van_yaw

    logger.info(
        "Calibration done: f_eff=%.1f (Δ%+.1f)  "
        "van=[%.1f E, %.1f N, %.1f U]  yaw=%.1f°  cost=%.5f",
        f_opt, f_opt - f0,
        van_x, van_y, van_z, van_yaw, result.cost,
    )

    # ── Apply to all frames ───────────────────────────────────────────────────
    for frame in frames:
        if frame.K_undist is not None:
            frame.K_undist[0, 0] = f_opt
            frame.K_undist[1, 1] = f_opt

    for i, frame in enumerate(van_frames):
        old = frame.gimbal_pitch_deg or 0.0
        new = float(pitches[i])
        frame.gimbal_pitch_deg = new
        roll  = config.CAMERA_ROLL_OVERRIDES.get(frame.stem, config.CAMERA_ROLL_DEG)
        frame.R = build_rotation(frame.heading_deg, new, roll)
        logger.info("[%s] pitch %.1f° → %.1f°", frame.stem, old, new)
        frame.pitch_from_van = True

    # Update config so future runs start from the calibrated focal length
    config.FOCAL_LENGTH = f_opt / config.UNDISTORT_SCALE

    return True
