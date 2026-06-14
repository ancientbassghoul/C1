"""
pipeline/pose.py – Camera pose estimation from telemetry.

For each frame we build:
  position_enu  – camera position in a local East-North-Up world frame
  R             – rotation matrix mapping world (ENU) → camera (OpenCV)
  gimbal_pitch  – camera tilt angle (degrees, negative = down)

Coordinate systems
──────────────────
World frame (ENU):
    X = East,  Y = North,  Z = Up
    Origin = GPS position of the first frame (arbitrary but consistent).

Camera frame (OpenCV convention):
    X = right,  Y = down,  Z = forward (out of the lens)

The rotation R satisfies:
    p_cam = R @ (p_world – position_enu)

GPS → ENU conversion
─────────────────────
We use the flat-Earth approximation, valid for scenes < ~10 km across:
    ΔEast  = Δlon · cos(lat_ref) · R_earth · π/180
    ΔNorth = Δlat                · R_earth · π/180

Gimbal pitch
────────────
The pitch (camera tilt) is not directly encoded in the HUD.

Priority order (applied in estimate_poses):
  1. Manual override from config.GIMBAL_PITCH_OVERRIDES  (always wins).
  2. GeoCalib batch estimate (shared_intrinsics=True across all frames).
  3. 0° fallback — the Ceres orientation solver refines from there.

Roll
────
Priority order:
  1. Manual override from config.CAMERA_ROLL_OVERRIDES.
  2. HUD bracket indicator detection.
  3. GeoCalib estimate (fallback when bracket detection fails).
  4. config.CAMERA_ROLL_DEG (default 0°).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

from pipeline.frame import Frame
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GPS → ENU
# ─────────────────────────────────────────────────────────────────────────────

def gps_to_enu(
    lat: float, lon: float, alt: float,
    lat_ref: float, lon_ref: float,
) -> np.ndarray:
    """
    Convert a WGS-84 GPS position to local ENU coordinates (metres).

    *lat_ref / lon_ref* define the ENU origin (typically the first frame).
    *alt* is the camera Z coordinate in the returned vector.
    """
    R = config.EARTH_RADIUS_M
    lat_rad = math.radians(lat_ref)

    east  = (lon - lon_ref) * math.cos(lat_rad) * R * math.pi / 180.0
    north = (lat - lat_ref)                      * R * math.pi / 180.0
    up    = alt

    return np.array([east, north, up], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Rotation matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """
    Build the 3×3 rotation matrix R_cam_from_world.

    Parameters
    ──────────
    yaw_deg   : camera compass heading (degrees, clockwise from North).
                For a stabilised gimbal this is the gimbal yaw; otherwise
                use the drone heading from telemetry.
    pitch_deg : camera elevation angle (degrees, negative = below horizon).
    roll_deg  : camera roll (degrees, positive = roll right).

    Returns R such that:  p_cam = R @ p_world_relative_to_camera
    (i.e. R maps ENU world vectors to OpenCV camera vectors)

    Derivation
    ──────────
    We construct the three camera-frame axes expressed in ENU world coords,
    then form R_world_from_cam = [right | down | fwd]  (columns).
    R_cam_from_world = R_world_from_cam.T   (orthonormal → transpose = inverse).

    Camera-frame axes in ENU:
      fwd   = direction the camera looks (ENU)
      right = camera +X axis (ENU); perpendicular to fwd and world-up
      down  = camera +Y axis (ENU); = fwd × right  (gives "into the ground")
    """
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll  = math.radians(roll_deg)

    # Forward vector in ENU  (East=X, North=Y, Up=Z)
    # Compass yaw: 0°=North, 90°=East  →  East component = sin(yaw), North = cos(yaw)
    fwd = np.array([
        math.sin(yaw) * math.cos(pitch),   # East
        math.cos(yaw) * math.cos(pitch),   # North
        math.sin(pitch),                   # Up  (negative for downward tilt)
    ])

    # Right vector: fwd × world_up  (degenerate at nadir/zenith)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, world_up)) > 0.999:
        # Gimbal lock – use drone heading to define "right"
        # At nadir (pitch=-90°), right = direction 90° CW from heading
        right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    else:
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)

    # Apply roll around the forward axis (Rodrigues)
    if roll != 0.0:
        c, s   = math.cos(roll), math.sin(roll)
        f_unit = fwd / np.linalg.norm(fwd)
        right  = right * c + np.cross(f_unit, right) * s
        right /= np.linalg.norm(right)

    # Down vector: fwd × right  → gives camera +Y (down) in ENU
    down = np.cross(fwd, right)
    down /= np.linalg.norm(down)

    # R_world_from_cam: columns = camera X, Y, Z axes in world coords
    R_world_from_cam = np.column_stack([right, down, fwd])

    # Invert: R_cam_from_world = R_world_from_cam.T
    return R_world_from_cam.T


# ─────────────────────────────────────────────────────────────────────────────
# Roll detection from artificial horizon indicator
# ─────────────────────────────────────────────────────────────────────────────

def _blob_rects_to_raw(
    label_indices: list,
    stats: np.ndarray,
    x0: int,
    y0: int,
) -> list:
    """Convert crop-local blob stats to raw-frame (y1,y2,x1,x2) tuples."""
    rects = []
    for li in label_indices:
        left   = int(stats[li, cv2.CC_STAT_LEFT])
        top    = int(stats[li, cv2.CC_STAT_TOP])
        width  = int(stats[li, cv2.CC_STAT_WIDTH])
        height = int(stats[li, cv2.CC_STAT_HEIGHT])
        rects.append((y0 + top, y0 + top + height, x0 + left, x0 + left + width))
    return rects


def detect_camera_roll(raw_img: np.ndarray):
    """
    Detect camera roll from the artificial horizon bracket indicator in the HUD.

    The HUD renders two fixed-geometry bracket symbols (⌐ and ¬) symmetrically
    around the central crosshair.  Their shape never changes — only their
    position and rotation relative to the frame centre.  They are pure-white
    overlays rendered by the flight controller directly onto the video stream,
    unaffected by lens distortion.

    Strategy (cascade — stops at first success)
    ────────────────────────────────────────────
    1. Two-blob:  find the most opposite, equidistant blob pair → most reliable.
    2. Left-blob: one clean blob left of centre → angle from blob to centre.
    3. Right-blob: one clean blob right of centre → angle from centre to blob.

    All three methods produce the same answer when blobs are perfectly symmetric,
    so there is no discontinuity between them.

    Returns (roll_deg, bracket_rects) where roll_deg is None if all strategies
    fail and bracket_rects is a list of (y1,y2,x1,x2) tuples in raw-frame coords
    for the winning blobs (used by suppress_center_overlays for blurring).
    """
    h, w = raw_img.shape[:2]
    cx, cy = w // 2, h // 2

    # Generous window — handles brackets at any roll angle including ±90°
    margin = config.BRACKET_SEARCH_MARGIN
    x0, x1 = max(0, cx - margin), min(w, cx + margin)
    y0, y1 = max(0, cy - margin), min(h, cy + margin)
    crop = raw_img[y0:y1, x0:x1]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Pure-white threshold — HUD overlays are (≈255,255,255); scene content
    # never reaches this, so no additional colour correction is needed.
    _, binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

    # Blank the crosshair "+" centre
    bh, bw = binary.shape
    inner = 35
    binary[bh//2 - inner : bh//2 + inner,
           bw//2 - inner : bw//2 + inner] = 0

    # Connected components — each bracket is one or two blobs
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )

    # Filter by area: bracket blobs are medium-sized (not noise, not huge banner)
    MIN_AREA = config.ROLL_BRACKET_MIN_AREA
    MAX_AREA = config.ROLL_BRACKET_MAX_AREA
    centre   = np.array([bw / 2.0, bh / 2.0])

    candidates = []  # (centroid, dist, area, label_index)
    for i in range(1, n_labels):   # skip label 0 (background)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if MIN_AREA <= area <= MAX_AREA:
            c = centroids[i]
            dist = float(np.linalg.norm(c - centre))
            if dist > inner:       # must be outside the blanked crosshair zone
                candidates.append((c, dist, area, i))

    # ── Strategy 1: two-blob (most reliable) ─────────────────────────────────
    if len(candidates) >= 2:
        best_score = -1.0
        best_pair  = None

        for i in range(len(candidates)):
            c1, d1, _, li1 = candidates[i]
            for j in range(i + 1, len(candidates)):
                c2, d2, _, li2 = candidates[j]

                v1 = c1 - centre
                v2 = c2 - centre
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-3 or n2 < 1e-3:
                    continue

                cos_theta     = float(np.dot(v1, v2) / (n1 * n2))
                dist_symmetry = 1.0 - abs(d1 - d2) / max(d1, d2)
                score         = (-cos_theta) * dist_symmetry

                if score > best_score:
                    best_score = score
                    best_pair  = (c1, c2, li1, li2)

        if best_pair is not None and best_score >= 0.5:
            c1, c2, li1, li2 = best_pair
            left, right = (c1, c2) if c1[0] < c2[0] else (c2, c1)
            angle = math.degrees(math.atan2(
                float(right[1] - left[1]),
                float(right[0] - left[0]),
            ))
            logger.debug(
                "Roll (two-blob): %.1f°  left=[%.0f,%.0f] right=[%.0f,%.0f]  score=%.2f",
                angle, left[0], left[1], right[0], right[1], best_score,
            )
            return float(angle), _blob_rects_to_raw([li1, li2], stats, x0, y0)

        logger.debug(
            "Roll two-blob failed (best score=%.2f) — trying single-blob fallback.",
            best_score,
        )

    # ── Strategy 2: left blob + crosshair centre ──────────────────────────────
    left_blobs = [(c, d, a, li) for c, d, a, li in candidates if c[0] < centre[0]]
    if left_blobs:
        c, dist, _, li = max(left_blobs, key=lambda x: x[2])   # largest blob
        angle = math.degrees(math.atan2(
            float(centre[1] - c[1]),
            float(centre[0] - c[0]),
        ))
        logger.debug(
            "Roll (left-blob fallback): %.1f°  blob=[%.0f,%.0f]  dist=%.1f",
            angle, c[0], c[1], dist,
        )
        return float(angle), _blob_rects_to_raw([li], stats, x0, y0)

    # ── Strategy 3: right blob + crosshair centre ─────────────────────────────
    right_blobs = [(c, d, a, li) for c, d, a, li in candidates if c[0] >= centre[0]]
    if right_blobs:
        c, dist, _, li = max(right_blobs, key=lambda x: x[2])  # largest blob
        angle = math.degrees(math.atan2(
            float(c[1] - centre[1]),
            float(c[0] - centre[0]),
        ))
        logger.debug(
            "Roll (right-blob fallback): %.1f°  blob=[%.0f,%.0f]  dist=%.1f",
            angle, c[0], c[1], dist,
        )
        return float(angle), _blob_rects_to_raw([li], stats, x0, y0)

    logger.debug("Roll detection: no usable blobs found — all strategies failed.")
    return None, []


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def estimate_poses(frames: list[Frame], skip_bracket_roll: bool | None = None) -> None:
    """
    Compute position_enu, R, and gimbal_pitch_deg for every frame in-place.

    skip_bracket_roll: if None (default), reads config.HORIZON_INDICATOR_READING
      to decide. Pass True/False explicitly to override the config (e.g. from a
      CLI flag).

    Frames without valid GPS/heading/altitude telemetry are skipped with a
    warning (they will not be usable for ray-casting).
    """
    if skip_bracket_roll is None:
        skip_bracket_roll = not config.HORIZON_INDICATOR_READING
    # ── Choose ENU origin ─────────────────────────────────────────────────
    origin_frame = next(
        (f for f in frames if f.lat is not None and f.lon is not None), None,
    )
    if origin_frame is None:
        raise RuntimeError("No frame has valid GPS telemetry – cannot establish ENU origin.")

    lat_ref = origin_frame.lat
    lon_ref = origin_frame.lon
    logger.info("ENU origin: lat=%.6f  lon=%.6f  (frame: %s)",
                lat_ref, lon_ref, origin_frame.stem)

    # ── GeoCalib batch estimation ─────────────────────────────────────────
    # Run once across all frames with shared focal length constraint.
    # Returns {frame: (pitch_deg, roll_deg)} for every eligible frame.
    geocalib: dict = {}
    if config.GEOCALIB_ENABLED:
        from pipeline.pitch_from_geocalib import calibrate_all_frames
        geocalib = calibrate_all_frames(frames)
        logger.info("GeoCalib estimated pitch+roll for %d/%d frame(s).",
                    len(geocalib), len(frames))

    # ── Per-frame pose ─────────────────────────────────────────────────────
    for frame in frames:
        if None in (frame.lat, frame.lon, frame.heading_deg, frame.alt_takeoff_ref_m):
            logger.warning("[%s] Incomplete telemetry – skipping pose.", frame.stem)
            continue

        _enu = gps_to_enu(
            frame.lat, frame.lon, frame.alt_takeoff_ref_m,
            lat_ref, lon_ref,
        )
        frame.position_enu = np.array([_enu[0], _enu[1], frame.alt_agl_m])

        # ── Roll ──────────────────────────────────────────────────────────────
        # Priority: manual override → bracket detection (unless skipped) → GeoCalib → default
        if frame.stem in config.CAMERA_ROLL_OVERRIDES:
            roll = config.CAMERA_ROLL_OVERRIDES[frame.stem]
            frame.camera_roll_deg = -roll   # override: same convention as bracket
            logger.info("[%s] Roll from manual override: %.1f°", frame.stem, roll)
        elif not skip_bracket_roll:
            _detected_roll, _rects = detect_camera_roll(frame.raw)
            frame.bracket_rects_raw = _rects
            if _detected_roll is not None:
                roll = _detected_roll
                frame.camera_roll_deg = -roll
                logger.info("[%s] Roll from horizon indicator: %.1f°", frame.stem, roll)
            elif frame in geocalib:
                _, roll = geocalib[frame]
                frame.camera_roll_deg = roll
                logger.warning(
                    "[%s] *** BRACKET DETECTION FAILED ON ALL STRATEGIES — "
                    "FALLING BACK TO GEOCALIB (roll=%.1f°) — "
                    "CHECK HUD FOR OBSTRUCTION ***",
                    frame.stem, roll,
                )
            else:
                roll = config.CAMERA_ROLL_DEG
                frame.camera_roll_deg = -roll
                logger.info("[%s] Roll: no source available – using default %.1f°",
                            frame.stem, roll)
        elif frame in geocalib:
            _, roll = geocalib[frame]
            frame.camera_roll_deg = roll
            logger.info("[%s] Roll from GeoCalib: %.1f°", frame.stem, roll)
        else:
            roll = config.CAMERA_ROLL_DEG
            frame.camera_roll_deg = -roll
            logger.info("[%s] Roll: no source available – using default %.1f°",
                        frame.stem, roll)

        # ── Pitch ─────────────────────────────────────────────────────────────
        # Priority: manual override → GeoCalib → 0° seed for Ceres
        if frame.stem in config.GIMBAL_PITCH_OVERRIDES:
            pitch = config.GIMBAL_PITCH_OVERRIDES[frame.stem]
            logger.info("[%s] Pitch from manual override: %.1f°", frame.stem, pitch)
        elif frame in geocalib:
            pitch, _ = geocalib[frame]
        else:
            pitch = 0.0
            logger.info("[%s] Pitch: no GeoCalib result – using 0° seed.", frame.stem)

        frame.gimbal_pitch_deg = pitch

        # ── Rotation matrix ───────────────────────────────────────────────────
        frame.R = build_rotation(
            yaw_deg   = frame.heading_deg,
            pitch_deg = pitch,
            roll_deg  = roll,
        )

        logger.info(
            "[%s] Pose: pos=[%.1f, %.1f, %.1f]m  hdg=%.0f°  pitch=%.1f°  roll=%.1f°",
            frame.stem, *frame.position_enu,
            frame.heading_deg, pitch, roll,
        )
