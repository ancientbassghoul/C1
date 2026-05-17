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

Priority order used by estimate_gimbal_pitch():
  1. Manual override from config.GIMBAL_PITCH_OVERRIDES  (always wins).
  2. 0° (horizontal) — the ground-scatter optimizer in refine.py then
     refines this to the correct value using SuperPoint + LightGlue
     ground-feature matches across frames.

Near-nadir frames or frames with poor feature overlap may still need a
manual pitch override in config.py if the optimizer cannot converge.
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
# Gimbal pitch
# ─────────────────────────────────────────────────────────────────────────────

def estimate_gimbal_pitch(frame: Frame, roll_deg: float = 0.0) -> float:
    """
    Return the camera gimbal pitch for *frame* (degrees, negative = down).

    Priority:
      1. Manual override from config.GIMBAL_PITCH_OVERRIDES (always wins).
      2. 0° — the ground-scatter optimizer (refine.py) refines this later.
    """
    if frame.stem in config.GIMBAL_PITCH_OVERRIDES:
        pitch = config.GIMBAL_PITCH_OVERRIDES[frame.stem]
        logger.info("[%s] Using manual pitch override: %.1f°", frame.stem, pitch)
        return pitch

    return 0.0


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

def detect_camera_roll(raw_img: np.ndarray) -> Optional[float]:
    """
    Detect camera roll from the artificial horizon bracket indicator in the HUD.

    The HUD renders two fixed-geometry bracket symbols (⌐ and ¬) symmetrically
    around the central crosshair.  Their shape never changes — only their
    position and rotation relative to the frame centre.  They are pure-white
    overlays rendered by the flight controller directly onto the video stream,
    unaffected by lens distortion.

    Algorithm
    ─────────
    1. Crop a generous window around the image centre (±200 px) — large enough
       to capture brackets at any roll angle including 90°.
    2. Threshold at ≥235 to isolate pure-white HUD pixels (scene content
       is never this bright, so no colour correction is needed).
    3. Blank the inner ±35 px (the crosshair "+" itself).
    4. Find connected components (blobs) and filter by area to get bracket-
       sized blobs only.
    5. Among all candidate blob pairs, pick the pair that is most nearly
       opposite each other about the image centre AND most equidistant from it.
       This is the geometric signature of the two bracket symbols.
    6. The roll angle is the angle of the line connecting the two blob centroids,
       normalised to (−90°, +90°].

    Returns roll in degrees (positive = roll right) or None on failure.
    """
    h, w = raw_img.shape[:2]
    cx, cy = w // 2, h // 2

    # Generous window — handles brackets at any roll angle including ±90°
    margin = 200
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

    candidates = []
    for i in range(1, n_labels):   # skip label 0 (background)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if MIN_AREA <= area <= MAX_AREA:
            c = centroids[i]
            dist = float(np.linalg.norm(c - centre))
            if dist > inner:       # must be outside the blanked crosshair zone
                candidates.append((c, dist, area))

    if len(candidates) < 2:
        logger.debug("Roll detection: fewer than 2 bracket blobs found – skip.")
        return None

    # Find the pair that is most opposite and equidistant about the centre.
    # Score = (−cos θ) × distance_symmetry, maximised for the ideal bracket pair.
    #   −cos θ → 1 when the two blobs are exactly opposite (θ = 180°)
    #   distance_symmetry → 1 when both are equidistant from centre
    best_score = -1.0
    best_pair  = None

    for i in range(len(candidates)):
        c1, d1, _ = candidates[i]
        for j in range(i + 1, len(candidates)):
            c2, d2, _ = candidates[j]

            v1 = c1 - centre
            v2 = c2 - centre
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-3 or n2 < 1e-3:
                continue

            cos_theta      = float(np.dot(v1, v2) / (n1 * n2))
            dist_symmetry  = 1.0 - abs(d1 - d2) / max(d1, d2)
            score          = (-cos_theta) * dist_symmetry

            if score > best_score:
                best_score = score
                best_pair  = (c1, c2)

    # Require the two blobs to be at least roughly opposite (score > 0.5)
    if best_pair is None or best_score < 0.5:
        logger.debug(
            "Roll detection: no sufficiently opposite blob pair found "
            "(best score=%.2f) – skip.", best_score,
        )
        return None

    c1, c2 = best_pair

    # Always compute vector from LEFT blob to RIGHT blob (by X coordinate).
    # This gives a deterministic sign: positive = right blob is lower = roll right,
    # negative = right blob is higher = roll left.
    left, right = (c1, c2) if c1[0] < c2[0] else (c2, c1)
    angle = math.degrees(math.atan2(
        float(right[1] - left[1]),   # dy: positive = right is lower in image
        float(right[0] - left[0]),   # dx: always positive (left→right)
    ))

    logger.debug(
        "Roll detected: %.1f°  left=[%.0f,%.0f] right=[%.0f,%.0f]  score=%.2f",
        angle, left[0], left[1], right[0], right[1], best_score,
    )
    return float(angle)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def estimate_poses(frames: list[Frame]) -> None:
    """
    Compute position_enu, R, and gimbal_pitch_deg for every frame in-place.

    Frames without valid GPS/heading/altitude telemetry are skipped with a
    warning (they will not be usable for ray-casting).
    """
    # ── Choose ENU origin ──────────────────────────────────────────────────
    # Use the first frame that has valid GPS as the world origin.
    origin_frame = next(
        (f for f in frames if f.lat is not None and f.lon is not None),
        None,
    )
    if origin_frame is None:
        raise RuntimeError("No frame has valid GPS telemetry – cannot establish ENU origin.")

    lat_ref = origin_frame.lat
    lon_ref = origin_frame.lon
    logger.info("ENU origin: lat=%.6f  lon=%.6f  (frame: %s)",
                lat_ref, lon_ref, origin_frame.stem)

    # ── Per-frame pose ─────────────────────────────────────────────────────
    for frame in frames:
        if None in (frame.lat, frame.lon, frame.heading_deg, frame.alt_takeoff_ref_m):
            logger.warning("[%s] Incomplete telemetry – skipping pose.", frame.stem)
            continue

        _enu = gps_to_enu(
            frame.lat, frame.lon, frame.alt_takeoff_ref_m,
            lat_ref, lon_ref,
        )
        # Ground plane is defined as Z = 0.
        # Camera Z = AGL altitude (direct radar measurement, most reliable).
        frame.position_enu = np.array([_enu[0], _enu[1], frame.alt_agl_m])

        # ── Roll — detected from HUD bracket indicator ────────────────────────
        if frame.stem in config.CAMERA_ROLL_OVERRIDES:
            roll = config.CAMERA_ROLL_OVERRIDES[frame.stem]
            logger.info("[%s] Using manual roll override: %.1f°", frame.stem, roll)
        else:
            detected_roll = detect_camera_roll(frame.raw)
            if detected_roll is not None:
                roll = detected_roll
                logger.info("[%s] Roll from horizon indicator: %.1f°",
                            frame.stem, roll)
            else:
                roll = config.CAMERA_ROLL_DEG
                logger.debug("[%s] Roll detection failed – using default %.1f°",
                             frame.stem, roll)

        frame.camera_roll_deg = -roll   # negated: detector convention is opposite to rotation convention

        # ── Pitch — manual override wins; otherwise 0° for the optimizer ──────
        pitch = estimate_gimbal_pitch(frame, roll_deg=roll)
        frame.gimbal_pitch_deg = pitch

        # ── Rotation matrix ───────────────────────────────────────────────────
        frame.R = build_rotation(
            yaw_deg   = frame.heading_deg,
            pitch_deg = pitch,
            roll_deg  = roll,
        )

        logger.info(
            "[%s] Pose: pos=[%.1f, %.1f, %.1f]m  hdg=%.0f°  pitch=%.1f°  roll=%.1f°",
            frame.stem,
            *frame.position_enu,
            frame.heading_deg, pitch, roll,
        )
