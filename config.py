"""
config.py – Central configuration for the Raycast Challenge pipeline.

All tunable parameters live here.  Start by running with defaults; if
re-projection accuracy is poor, adjust the CAMERA section first (focal
length and distortion coefficients have the biggest impact).
"""

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA INTRINSICS
# ─────────────────────────────────────────────────────────────────────────────
# These approximate a wide-angle FPV drone camera at 1280×720.
# The lens shows strong barrel / fisheye distortion, so k1 is strongly negative.
#
# Tuning guide:
#   1. Run raycast.py --preview-undistort and adjust until straight lines
#      (vegetation boundary, tyre tracks) look straight in the output.
#   2. Lower FOCAL_LENGTH → wider corrected FOV (more black edges).
#      Higher FOCAL_LENGTH → tighter corrected FOV (less wasted pixels).
#   3. Make k1 more negative if the image still looks barrel-distorted after
#      undistortion; less negative if it looks pincushion-distorted.

IMAGE_W = 1280
IMAGE_H = 720

# Estimated fisheye focal length (pixels).  For a ~140° diagonal-FOV lens
# with equidistant projection: f ≈ diag / (2 * FOV_rad) ≈ 660.
# Adjust in ±50 px increments.
FOCAL_LENGTH = 639.0

# Principal point – almost always the image centre.
CX = IMAGE_W / 2.0   # 640.0
CY = IMAGE_H / 2.0   # 360.0

# OpenCV fisheye distortion coefficients  D = [k1, k2, k3, k4].
# Typical range for a GoPro / DJI wide-angle: k1 ≈ -0.05 … -0.25.
# Start here and tune until straight real-world lines appear straight.
FISHEYE_K1 = -0.05
FISHEYE_K2 =  0.02
FISHEYE_K3 =  0.00
FISHEYE_K4 =  0.00

# Scale factor applied to the NEW camera matrix after undistortion.
# 0.5 → zoom out (more black border, but no cropping of valid pixels).
# 1.0 → fill the frame (some valid pixels get cropped at edges).
# Start at 0.6 for very wide lenses; increase if too much black border.
UNDISTORT_SCALE = 0.6

# ─────────────────────────────────────────────────────────────────────────────
# POSE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

# Set True to use the HUD horizon bracket indicator for roll detection.
# Set False to use GeoCalib roll estimates for all frames instead.
# Default is False — bracket detection is unreliable on heavily rolled frames.
HORIZON_INDICATOR_READING = False

# Earth radius used for the local flat-Earth (ENU) approximation.
# Valid for areas < ~10 km; this scene is well within that.
EARTH_RADIUS_M = 6_371_000.0

# ── Gimbal pitch overrides ────────────────────────────────────────────────────
# The orientation solver (feature_matcher.py) refines pitch automatically.
# Use these overrides only for frames where the solver cannot converge —
# e.g. a frame with no usable ground features and no overlap with other frames.
# Key = filename stem (no extension).
#
# Example:
#   GIMBAL_PITCH_OVERRIDES = {
#       "2026-02-15_16-35-56_06892": -15.0,
#       "2026-02-15_16-25-03_04569": -8.0,
#   }
GIMBAL_PITCH_OVERRIDES: dict[str, float] = {}

# ── Camera roll overrides ─────────────────────────────────────────────────────
# Roll is detected automatically from HUD bracket indicators.
# Override per frame if detection gives wrong results.
CAMERA_ROLL_DEG = 0.0
CAMERA_ROLL_OVERRIDES: dict[str, float] = {}

# ─────────────────────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────────────────────
OCR_LANGUAGES = ["en"]
OCR_GPU       = True    # Set True if a CUDA GPU is available.

# Pixel crop windows for each telemetry field [row_start, row_end, col_start, col_end].
# These match the HUD layout visible in the provided sample frames.
OCR_CROP_LAT_LON = (55,  135, 0,   290)   # "LAT: xx.xxx / LON: xx.xxx" (top-left)
OCR_CROP_HEADING = (5,   70,  510, 790)   # Large heading number (top-centre)
OCR_CROP_ALT     = (640, 718, 840, 1160)  # "XX.X  M  XX.X  M" (bottom-right)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./output"

# Fixed ground plane height in the shared ENU coordinate frame (metres).
# The terrain is flat; this single value is used for all ray-ground
# intersections in both the reprojection viewer and the Ceres solver.
# Set to 0.0 — the ENU origin sits on the ground at the takeoff point.
GROUND_Z_M = 0.0

# Path where the manual correspondence picker saves / loads its JSON file.
MANUAL_CORRESPONDENCES_FILE = "./output/manual_correspondences.json"

# Path where the automatic LightGlue matches are saved after each pipeline run.
# Saved by default so --show-scores can inspect them alongside manual picks.
AUTO_MATCHES_FILE = "./output/auto_matches.json"

# Weight of each manually-picked correspondence relative to a LightGlue match
# in the Ceres solver.  10 means one hand-picked point counts as 10 automatic
# matches.  Raise if the solver ignores manual picks; lower if they over-dominate.
MANUAL_CORRESPONDENCE_WEIGHT = 10.0

# Radius of the reprojection marker drawn on frames (pixels, in undistorted space).
MARKER_RADIUS    = 14
MARKER_COLOR_SRC = (0, 220, 0)    # Green  – source pick
MARKER_COLOR_DST = (0, 60,  220)  # Blue   – reprojected point
MARKER_THICKNESS = 3

# ─────────────────────────────────────────────────────────────────────────────
# VAN DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# Confidence threshold for GroundingDINO van detection (0–1).
# Detections below this score are discarded and the white-blob fallback is used.
YOLO_CONF_THRESH = 0.15

# Minimum white-blob area (pixels²) for the fallback blob detector.
# Increase if small bright patches (glare, HUD elements) cause false positives.
VAN_BLOB_MIN_AREA = 200

# ─────────────────────────────────────────────────────────────────────────────
# VAN GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
# Physical dimensions used by the manual van-feature correspondences.
# These are the only van geometry values consumed by the solver — the
# bbox-based 3-D corner approach has been removed in favour of manually-picked
# wheel centres, roof edge, and roof points.
VAN_WIDTH_M    = 1.92   # lateral (left–right) wheel centre distance
VAN_HEIGHT_M   = 1.94   # exterior roof height above ground
VAN_WHEELBASE_M = 3.275  # front-to-rear wheel centre distance
WHEEL_RADIUS_M  = 0.343   # wheel centre height above ground

# ─────────────────────────────────────────────────────────────────────────────
# GEOCALIB
# ─────────────────────────────────────────────────────────────────────────────
# GeoCalib estimates gimbal pitch from a single undistorted frame using
# learned line/vanishing-point detection.  Used as the initial pitch seed
# for each camera before the Ceres orientation solver refines it.
# Set False to use 0° seed for all frames (faster startup, worse convergence).
GEOCALIB_ENABLED = True

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE MATCHING
# ─────────────────────────────────────────────────────────────────────────────
# Spatial grid for match subsampling.  After LightGlue matching and RANSAC,
# the ground region is divided into MATCH_GRID_COLS × MATCH_GRID_ROWS cells
# and at most one match is kept per cell.  This ensures spatial diversity and
# limits matches to at most MATCH_GRID_COLS × MATCH_GRID_ROWS per frame pair.
#
# Spatial deduplication threshold for the global multi-frame-aware filter
# (pixels, undistorted image space).  Two match endpoints closer than this
# are considered the same physical point.  Increase to be more aggressive
# about removing near-duplicates; decrease to keep spatially similar matches.
MATCH_SPATIAL_DEDUP_THRESH = 15.0

# Grid subsampling is no longer used — replaced by _global_match_filter().
# These values are kept for reference but have no effect.
MATCH_GRID_COLS = 10
MATCH_GRID_ROWS = 8

# ─────────────────────────────────────────────────────────────────────────────
# ORIENTATION SOLVER
# ─────────────────────────────────────────────────────────────────────────────
# Per-camera bounds for the pyceres joint optimisation.
#
# Pitch window: Ceres searches ± SOLVER_PITCH_OFFSET degrees around the
#   GeoCalib seed for each frame.  The result is further clamped to
#   [SOLVER_PITCH_FLOOR, SOLVER_PITCH_CEILING] so no frame can land in a
#   physically impossible orientation regardless of the GeoCalib estimate.
#
# Yaw offset: correction on top of the OCR compass heading.
#   ±5° covers typical magnetometer drift and HUD rounding errors.
# Roll offset: correction on top of the bracket-detected roll.
#   ±1° covers residual HUD-detection error at near-level flight.
SOLVER_PITCH_OFFSET  =  20.0   # ± degrees around the GeoCalib seed
SOLVER_PITCH_FLOOR   = -89.0   # absolute minimum pitch (near-nadir clamp)
SOLVER_PITCH_CEILING =  30.0   # absolute maximum pitch (upward-tilt clamp)
SOLVER_YAW_OFFSET_RANGE  =  20.0   # ± degrees (widened: OCR compass can be ~15° off)
SOLVER_ROLL_OFFSET_RANGE =   5.0   # ± degrees (widened: GeoCalib roll can be off on oblique frames)

# Camera position correction bounds (metres).
# GPS accuracy is typically ±2-5m; Blender measurements show up to ~9m error.
# The solver is free to move each camera within this box around its GPS seed.
SOLVER_POSITION_RANGE_H  =  10.0   # ± metres horizontal (East, North)
SOLVER_POSITION_RANGE_V  =   3.0   # ± metres vertical (Up/altitude)

# Maximum Ceres solver iterations.  Increase if the solve terminates early
# without converging (check the summary BriefReport in the log).
SOLVER_MAX_ITERATIONS = 200

# ── Focal length estimation ──────────────────────────────────────────────────
# Set True to add focal length as a free parameter in the Ceres solve.
# Requires good manual correspondences spanning frames at different altitudes.
# When True, the solved value is logged — update FOCAL_LENGTH in config.py
# manually after a successful solve, then re-run with this set back to False.
ESTIMATE_FOCAL_LENGTH = False

# Search range as a fraction of FOCAL_LENGTH (e.g. 0.22 → ±22%).
# [660 × 0.78, 660 × 1.22] = [515, 805] for the default 660px seed.
FOCAL_LENGTH_RANGE = 0.22

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA POSE OVERRIDES
# ─────────────────────────────────────────────────────────────────────────────
# Manually calibrated camera poses (e.g. from Blender adjustment).
# Used when --cameras-init-from-config is passed to raycast.py or
# export_blender.py.  Key = last 5 digits of the frame filename stem.
# Values:  x, y, z     – ENU position in metres
#          heading      – compass bearing in degrees (0=N, 90=E, clockwise)
#          pitch        – degrees, negative = looking down
#          roll         – degrees, positive = roll right
#
# Leave empty ({}) to use GPS + GeoCalib for all frames.
CAMERA_POSE_OVERRIDES: dict[str, dict] = {
    '04681': {'x':    0.000, 'y':    0.000, 'z':    4.100,
              'heading':  145.40, 'pitch':    1.98, 'roll':    7.87},
    '04709': {'x':    8.225, 'y':   -9.110, 'z':    4.700,
              'heading':  152.00, 'pitch':   -0.05, 'roll':   16.41},
    '04752': {'x':   20.634, 'y':  -18.583, 'z':    6.717,
              'heading':  207.00, 'pitch':   -4.17, 'roll':   16.93},
    '07849': {'x':   23.477, 'y':   24.828, 'z':   13.600,
              'heading':  181.00, 'pitch':   -2.67, 'roll':    1.34},
    '10474': {'x':   22.043, 'y':   18.952, 'z':   14.500,
              'heading':  176.00, 'pitch':    1.50, 'roll':   -0.53},
    '10671': {'x':   25.897, 'y':   12.343, 'z':   14.800,
              'heading':  236.00, 'pitch':   -1.93, 'roll':    0.34},
    '12035': {'x':   28.989, 'y':  -18.239, 'z':   21.300,
              'heading':  226.40, 'pitch':  -30.79, 'roll':   -3.03},
}

# ─────────────────────────────────────────────────────────────────────────────
# HSV GROUND MASK  (experimental — see --preview-hsv-masks)
# ─────────────────────────────────────────────────────────────────────────────
# Alternative to GroundingDINO+SAM for exclusion masking.
# Uses HSV colour thresholding to identify sky and vegetation.
# All values in OpenCV HSV space: H 0-180, S 0-255, V 0-255.
#
# Each range is ((H_lo, S_lo, V_lo), (H_hi, S_hi, V_hi)).
# Multiple ranges are OR-combined.  Add more entries to cover edge cases
# (e.g. orange sunset sky, dark shadowed vegetation).
#
# Tune with --preview-hsv-masks — iterates in seconds, no model inference.

# Sky — blue sky + white/hazy sky separately
HSV_SKY_RANGES = [
    (( 95,   0, 130), (135, 110, 255)),   # blue sky
    ((  0,   0, 175), (180,  45, 255)),   # white / hazy / overcast sky
]

# Vegetation — green fields, trees, shrubs
HSV_VEG_RANGES = [
    # (( 30,  45,  25), ( 90, 255, 220)),   # green vegetation
    # ((32, 173, 61), (30, 179, 145)),
    ((25, 50, 25), (52, 255, 200)),
]

# Morphological cleanup kernel size (pixels) applied after HSV thresholding.
# Larger = smoother masks but blurs edges.
HSV_MORPH_KERNEL = 7

# ─────────────────────────────────────────────────────────────────────────────
# ROLL DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# Area bounds (pixels²) for white connected-component blobs to be considered
# a HUD bracket symbol.  Too small = noise;  too large = HUD banner or glare.
ROLL_BRACKET_MIN_AREA = 80
ROLL_BRACKET_MAX_AREA = 2500

# ─────────────────────────────────────────────────────────────────────────────
# GROUND MASK  (GroundedSAM)
# ─────────────────────────────────────────────────────────────────────────────
# Root of your Grounded-Segment-Anything clone.  All model paths are derived
# from this single directory — no other paths need to be set.
#
# Expected layout (standard repo clone):
#   <GROUNDED_SAM_DIR>/
#     sam_vit_h_4b8939.pth               ← SAM ViT-H weights
#     groundingdino_swint_ogc.pth        ← GroundingDINO weights
#     GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
GROUNDED_SAM_DIR = r"D:\EXTEND\Grounded-Segment-Anything"

# ── HUD regions (raw-frame pixel coordinates) ────────────────────────────────
# Each entry is (y1, y2, x1, x2) in the original 1280×720 RAW frame.
# These rectangles are undistorted geometrically at runtime — no ML involved.
# The four corners of each box are pushed through cv2.fisheye.undistortPoints
# (same camera model as undistort.py) and filled as a convex polygon.
#
# Defined from frame 07849 (representative HUD layout for this flight).
# Do NOT include the centre horizon brackets or crosshair — those overlap
# real-world content and are better left unmasked.
#
# Format: (y1, y2, x1, x2)  — top-row, bottom-row, left-col, right-col
HUD_REGIONS = [
    # Top-left: status icons (record dot, power, motor RPM, camera icon)
    (  0,  72,   0,  200),
    # Top-left: GPS coordinates (LAT / LON)
    ( 52, 122,   0,  200),
    # Top-centre: navigation bar (home icon, AGL altitude, heading ruler + values)
    (  0,  72, 325,  511),
    (  0,  72, 512,  697),
    (  0,  72, 698,  882),
    # Top-right: system icons + battery percentage + battery bar
    (  0,  65,1025, 1280),
    # Bottom-left: aircraft status rows (CHARLIE / BETA / DELTA)
    (548, 685,   0,  400),
    # Bottom-left: date, time and flight timer
    (678, 720,   0,  400),
    # Bottom-right: telemetry readout (alt, AGL, ground speed + unit labels)
    (650, 720, 920, 1170),
    # Bottom-right: throttle percentage gauge (circular)
    (633, 720,1170, 1280),
]

# ── Positive ground prompt ───────────────────────────────────────────────────
# GroundingDINO + SAM detects pixels that ARE ground.
# Intersected with the exclusion mask — a pixel must be positively identified
# as ground AND not identified as an exclusion to survive.
# If nothing is detected (e.g. near-nadir frame), falls back to exclusion-only.
# GROUND_INCLUDE_PROMPT         = "soil . dirt . ground . track . mud . path . bare earth"
GROUND_INCLUDE_PROMPT         = "soil . dirt . ground . track . mud . path . bare earth"
GROUND_INCLUDE_BOX_THRESHOLD  = 0.336
GROUND_INCLUDE_TEXT_THRESHOLD = 0.336

# ── Exclusion prompt ──────────────────────────────────────────────────────────
# GroundingDINO + SAM detects pixels that are NOT ground.
# Includes HUD text / graphics so we don't need hardcoded OCR pixel rectangles
# (which break after undistortion shifts the HUD elements).
# GROUND_EXCLUDE_PROMPT         = "sky . clouds . vegetation . grass . tree . shrub . van . vehicle"
GROUND_EXCLUDE_PROMPT         = "sky . clouds . tree . van . vehicle. shrub"
GROUND_EXCLUDE_BOX_THRESHOLD  = 0.35
GROUND_EXCLUDE_TEXT_THRESHOLD = 0.35
