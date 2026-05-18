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
FOCAL_LENGTH = 660.0

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
# Physical bounding box dimensions of the white transit van (metres).
VAN_LENGTH_M = 4.959
VAN_WIDTH_M  = 2.204
VAN_HEIGHT_M = 1.895

# Approximate terrain height at the van's location in the shared ENU frame.
# Derived from terrain_offset of the nearest camera frame (frame 04709: +1.3 m).
# Used as the fixed Z of the van base — tune if reprojection is vertically off.
VAN_Z_M = 1.0

# Frames whose van detections are used as 3-D corner reprojection constraints.
# Each must have a valid GroundingDINO bbox.  Selected for angular diversity:
#   12035 – near-nadir  (roof visible)
#   04752 – front-left  oblique
#   04681 – rear-left   oblique
#   04709 – pure left-side view
#   10474 – bridging angle between nadir and oblique
VAN_FRAMES: list[str] = [
    "2026-02-15_16-25-03_12035",
    "2026-02-15_16-25-03_04752",
    "2026-02-15_16-25-03_04681",
    "2026-02-15_16-25-03_04709",
    "2026-02-15_16-25-03_10474",
]

# Van heading prior (degrees, CW from North).
# Derived from frame 04709: drone heading 152°, van perpendicular with left
# face visible → right face outward normal = 152° → van heading = 62°.
VAN_HEADING_PRIOR_DEG = 62.0

# Solver is free to adjust van heading ± this many degrees from the prior.
VAN_HEADING_RANGE_DEG = 45.0

# Relative weight of each van corner reprojection residual vs a ground scatter
# residual.  Higher = van corners dominate; lower = ground scatter dominates.
# Van corners provide the crucial vertical constraint (known height), so they
# should be weighted above ground scatter.
VAN_CORNER_WEIGHT = 5.0

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
MATCH_GRID_COLS = 5   # default: 5 × 4 = 20 matches max per pair
MATCH_GRID_ROWS = 4

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
SOLVER_YAW_OFFSET_RANGE  =   5.0   # ± degrees
SOLVER_ROLL_OFFSET_RANGE =   1.0   # ± degrees

# Maximum Ceres solver iterations.  Increase if the solve terminates early
# without converging (check the summary BriefReport in the log).
SOLVER_MAX_ITERATIONS = 200

# ─────────────────────────────────────────────────────────────────────────────
# ROLL DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# Area bounds (pixels²) for white connected-component blobs to be considered
# a HUD bracket symbol.  Too small = noise;  too large = HUD banner or glare.
ROLL_BRACKET_MIN_AREA = 80
ROLL_BRACKET_MAX_AREA = 2500
