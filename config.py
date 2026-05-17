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
#   1. Run undistort.py --preview on one frame and adjust until straight lines
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

# ── Gimbal pitch ──────────────────────────────────────────────────────────────
# The camera's vertical tilt angle in degrees.
#   0°  = horizontal (looking at the horizon)
# -90°  = nadir (straight down)
# We auto-detect this per frame from the horizon position; override below
# for any frame where auto-detection fails (e.g. no sky visible).

GIMBAL_PITCH_AUTO = True          # Set False to use manual overrides only.
GIMBAL_PITCH_FALLBACK_DEG = -45.0 # Used when auto-detection is inconclusive.

# Per-frame manual overrides.  Key = filename stem (no extension).
# Example:
#   GIMBAL_PITCH_OVERRIDES = {
#       "2026-02-15_16-35-56_06892": -15.0,
#       "2026-02-15_16-25-03_04569": -8.0,
#   }
GIMBAL_PITCH_OVERRIDES: dict[str, float] = {
    # Calibrated via manual ground correspondences (manual_calibrate.py)
}

# ── Camera roll ───────────────────────────────────────────────────────────────
# Drone banking angle (degrees).  Positive = roll right.
# The artificial-horizon indicator in the HUD encodes this; currently read
# as 0 (level) for all frames.  Override per frame if you see a tilted horizon.
CAMERA_ROLL_DEG = 0.0
CAMERA_ROLL_OVERRIDES: dict[str, float] = {}

# ── Horizon detection ─────────────────────────────────────────────────────────
# Row-fraction search window for finding the sky/ground boundary.
# Excludes the top HUD strip and the bottom status bar.
HORIZON_SEARCH_TOP    = 0.10   # 10 % from top
HORIZON_SEARCH_BOTTOM = 0.80   # 80 % from top  (stop before bottom HUD)

# Minimum sky-fraction above the detected horizon for the result to be accepted.
# If the horizon is very close to the bottom of the search window (nadir view),
# auto-detection falls back to GIMBAL_PITCH_FALLBACK_DEG.
HORIZON_MIN_SKY_FRACTION = 0.05

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
MARKER_RADIUS   = 14
MARKER_COLOR_SRC = (0, 220, 0)    # Green  – source pick
MARKER_COLOR_DST = (0, 60,  220)  # Blue   – reprojected point
MARKER_THICKNESS = 3

# ─────────────────────────────────────────────────────────────────────────────
# VAN MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Approximate dimensions of the white transit van visible in the frames.
# Measure from the overhead shot (frame 12035) if you want to refine these.
VAN_WIDTH_M  = 2.0    # metres, side-to-side
VAN_LENGTH_M = 4.8    # metres, nose-to-tail
VAN_HEIGHT_M = 2.0    # metres, ground-to-roof

# ─────────────────────────────────────────────────────────────────────────────
# YOLO DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# Model is auto-downloaded by ultralytics on first run (~6 MB for nano).
YOLO_MODEL       = "yolov8s.pt"   # small model — better on small/unusual viewpoints
YOLO_CONF_THRESH = 0.15           # minimum detection confidence (0-1)

# ─────────────────────────────────────────────────────────────────────────────
# VAN CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
# Maximum pixel distance for matching a Shi-Tomasi corner to a predicted corner.
VAN_CORNER_MATCH_THRESHOLD_PX = 20

# Focal length search bounds [pixels] for the joint optimiser.
# The current estimate (FOCAL_LENGTH * UNDISTORT_SCALE) is used as the seed.
VAN_CALIB_F_BOUNDS     = (200.0, 1200.0)

# Pitch search bounds [degrees].
VAN_CALIB_PITCH_BOUNDS = (-90.0, 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# ORIENTATION SOLVER
# ─────────────────────────────────────────────────────────────────────────────
# Per-camera bounds used by orientation_solver.py when jointly refining
# pitch, yaw offset, and roll offset from ground-feature scatter.
#
# Pitch: full physical range (near-nadir to slightly above horizon).
# Yaw offset: small correction on top of the OCR compass heading.
#   ±5° handles typical magnetometer drift and HUD rounding errors.
# Roll offset: small correction on top of the bracket-detected roll.
#   ±1° handles residual HUD-detection error at near-level flight.

SOLVER_PITCH_MIN         = -89.0   # degrees
SOLVER_PITCH_MAX         =  15.0   # degrees
SOLVER_YAW_OFFSET_RANGE  =   5.0   # ± degrees
SOLVER_ROLL_OFFSET_RANGE =   1.0   # ± degrees

# ─────────────────────────────────────────────────────────────────────────────
# ROLL DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# Area bounds (pixels) for connected-component blobs to be considered a
# bracket.  Too small = noise;  too large = HUD banner or sky glare.
# Tune if roll detection fails on your specific footage.
# Minimum white-blob area (pixels²) to be considered the van in the fallback detector.
# Increase if small bright patches (glare, HUD) cause false positives.
VAN_BLOB_MIN_AREA = 200

ROLL_BRACKET_MIN_AREA = 80
ROLL_BRACKET_MAX_AREA = 2500
