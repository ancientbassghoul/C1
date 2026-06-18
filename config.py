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
FOCAL_LENGTH = 651.92

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
HORIZON_INDICATOR_READING = True

# Earth radius used for the local flat-Earth (ENU) approximation.
EARTH_RADIUS_M = 6_371_000.0

# ── Gimbal pitch overrides ────────────────────────────────────────────────────
# Use only for frames where the solver cannot converge.
GIMBAL_PITCH_OVERRIDES: dict[str, float] = {}

# ── Camera roll overrides ─────────────────────────────────────────────────────
CAMERA_ROLL_DEG = 0.0
CAMERA_ROLL_OVERRIDES: dict[str, float] = {}

# ─────────────────────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────────────────────
OCR_LANGUAGES = ["en"]
OCR_GPU       = True    # Set True if a CUDA GPU is available.

# Pixel crop windows for each telemetry field [row_start, row_end, col_start, col_end].
OCR_CROP_LAT_LON = (55,  135, 0,   290)   # "LAT: xx.xxx / LON: xx.xxx" (top-left)
OCR_CROP_HEADING = (5,   70,  510, 790)   # Large heading number (top-centre)
OCR_CROP_ALT     = (640, 718, 840, 1160)  # "XX.X  M  XX.X  M" (bottom-right)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./output"

# Fixed ground plane height in the shared ENU coordinate frame (metres).
GROUND_Z_M = 0.0

# Path where the manual correspondence picker saves / loads its JSON file.
MANUAL_CORRESPONDENCES_FILE = "./output/manual_correspondences.json"

# Path where MASt3R automatic matches are saved after --run-matcher-only.
AUTO_MATCHES_FILE = "./output/auto_matches.json"

# Path where the Qwen/CLIP anchor detection result is cached.
ANCHOR_CACHE_FILE = "./output/anchor_cache.json"

# Max fraction of total image area a single anchor bbox may cover.
# Detections larger than this threshold are treated as scene backgrounds
# (e.g. "field" covering 50% of the frame) and discarded.
MAX_ANCHOR_BBOX_FRACTION = 0.35

# Weight of each manually-picked correspondence relative to a MASt3R match
# in the Ceres solver.
MANUAL_CORRESPONDENCE_WEIGHT = 10.0

# Radius of the reprojection marker drawn on frames (pixels, in undistorted space).
MARKER_RADIUS    = 14
MARKER_COLOR_SRC = (0, 220, 0)    # Green  – source pick
MARKER_COLOR_DST = (0, 60,  220)  # Blue   – reprojected point
MARKER_THICKNESS = 3

# ─────────────────────────────────────────────────────────────────────────────
# VAN GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
# Physical dimensions consumed by the manual van-feature correspondences
# (AxisPairCost, roof_plane residuals in the Ceres solver).
VAN_WIDTH_M     = 1.92    # lateral (left–right) wheel centre distance
VAN_HEIGHT_M    = 1.94    # exterior roof height above ground
VAN_WHEELBASE_M = 3.275   # front-to-rear wheel centre distance
WHEEL_RADIUS_M  = 0.343   # wheel centre height above ground

# ─────────────────────────────────────────────────────────────────────────────
# GEOCALIB
# ─────────────────────────────────────────────────────────────────────────────
# GeoCalib estimates gimbal pitch from a single undistorted frame using
# learned line/vanishing-point detection.  Used as pitch seed before
# MASt3R-SfM camera poses are available.
GEOCALIB_ENABLED = True

# ─────────────────────────────────────────────────────────────────────────────
# HUD MASKING
# ─────────────────────────────────────────────────────────────────────────────
# After undistortion, these regions are painted solid black [0,0,0].
# MASt3R's transformer assigns zero confidence to zero-entropy black regions,
# excluding HUD pixels from feature matching automatically.
#
# Each entry is (y1, y2, x1, x2) in the raw/distorted frame coordinate space.
# Format: (top-row, bottom-row, left-col, right-col)
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
    (575, 685,   0,  400),
    # Bottom-left: date, time and flight timer
    (678, 720,   0,  400),
    # Bottom-right: telemetry readout (alt, AGL, ground speed + unit labels)
    (650, 720, 920, 1170),
    # Bottom-right: throttle percentage gauge (circular)
    (633, 720,1170, 1280),
]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PATHS & CACHING
# ─────────────────────────────────────────────────────────────────────────────
# All models download here — never to the pod's /root/.cache (wiped on pod stop).
# models/ is in .gitignore.
MODEL_CACHE_DIR = "./models"

# ─────────────────────────────────────────────────────────────────────────────
# ANCHOR DETECTION  (Qwen VL + CLIP)
# ─────────────────────────────────────────────────────────────────────────────
QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL  = "openai/clip-vit-large-patch14"

# CLIP cosine similarity thresholds for anchor weighting.
# w >= CLIP_ANCHOR_THRESHOLD  → rigid anchor frame (used as control point in Sim(3)).
# w <  CLIP_ANCHOR_MIN_WEIGHT → frame's AnchorRayCost skipped entirely.
CLIP_ANCHOR_THRESHOLD  = 0.20
CLIP_ANCHOR_MIN_WEIGHT = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# MAST3R-SFM
# ─────────────────────────────────────────────────────────────────────────────
MAST3R_MODEL          = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
MAST3R_CONF_THRESHOLD = 1.5    # DEPRECATED as a dense-pixel gate — superseded by the
                                # adaptive MAST3R_CONF_FLOOR / MAST3R_CONF_PERCENTILE knobs
                                # below. Still referenced only by an orientation_solver.py
                                # log string; kept to avoid churn.
                                #
                                # History: this was the absolute per-pixel confidence cutoff
                                # from SparseGA.get_dense_pts3d() (NOT a 0-1 score — SparseGA's
                                # own .show() uses `c > 1` as its validity heuristic). It was a
                                # fixed value here, which made it brittle: the gray HUD fill
                                # depresses dense confidence scene-wide, so on this low-texture
                                # field the whole distribution sits in the ~1.0–2.5 band and a
                                # fixed 1.5 cutoff wiped out ~all pixels in 11/13 frames,
                                # collapsing the reconstruction onto 2 frames. The gate is now
                                # per-frame and percentile-based (see below) so each frame
                                # contributes its own most-reliable pixels regardless of band.
MAST3R_MAX_POINTS     = 1200   # max 3D points passed to Ceres (subsampled if more)

# Adaptive per-pixel confidence gating (replaces the brittle absolute
# MAST3R_CONF_THRESHOLD comparison at the dense-extraction gate sites).
#
# The feathered gray HUD fill depresses MASt3R's dense per-pixel confidence
# scene-wide; on this low-texture agricultural field the whole distribution
# sits in the ~1.0–2.5 band, so a fixed 1.5 cutoff clipped ~every pixel in most
# frames (11/13 frames produced "no confident MASt3R pixels", collapsing the
# reconstruction onto 2 frames). Gating per frame on a percentile makes each
# frame contribute its own most-reliable pixels regardless of where its absolute
# confidence band lands; the floor still drops MASt3R's ~1.0 "no-information"
# baseline pixels.
MAST3R_CONF_FLOOR      = 1.02  # absolute floor, just above MASt3R's ~1.0 baseline
MAST3R_CONF_PERCENTILE = 60    # per-frame percentile → keep ~top 40% of each frame's pixels

# Passed into mast3r.cloud_opt.sparse_ga.sparse_global_alignment()'s `matching_conf_thr`.
# A pair whose best correspondence confidence falls at/under this is demoted by MASt3R-SfM from
# a rigid cross-view 3D constraint to a much weaker single-view loss — i.e. the two frames stop
# being tied together. MASt3R's own internal default is 5.0, which fragmented this scene's
# repetitive-agricultural-ground reconstruction into several disconnected "islands" (borderline-
# but-real pairs were landing under the cutoff and getting dropped to the weak loss). Dropping
# all the way to 1.0 fixed connectivity but let too much low-confidence "spaghetti" through,
# breaking the Ceres solve (see output/debug/LowThrsh_results/). 2.5 is the current sweet-spot
# attempt between the two failure modes — paired with the adaptive per-pixel gate above
# (MAST3R_CONF_FLOOR / MAST3R_CONF_PERCENTILE) to keep spaghetti out of Ceres regardless of
# what this lets through at the pair level. Also used
# by run_complete_graph()'s pairwise-match diagnostics for its `matching_ok` column, so the log
# stays consistent with what the solve actually uses.
MAST3R_MATCHING_CONF_THR = 2.0

# Force all frames to share one focal length during SGA instead of per-frame
# independent focals.  Prevents the optimizer from using focal flex as a
# shock absorber for noisy tracking links.
MAST3R_SHARED_INTRINSICS = True

# Reference-card size (metres) for the Sim(3)-aligned debug .glb export
# (--export-mesh). This scene is in real ENU units, unlike the raw MASt3R
# export (which auto-sizes itself off the scene's own arbitrary scale).
MAST3R_GLB_CAM_SIZE_M = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# GROUND-PLANE LEVELING
# ─────────────────────────────────────────────────────────────────────────────
# The telemetry Sim(3) fits camera centres only; on a near-constant-altitude
# arc those are nearly coplanar, so the recovered "up" axis is poorly
# constrained and the scene comes out tilted. After Sim(3) we RANSAC-fit the
# dominant ground plane to the solved sparse points and apply a corrective
# rigid transform that levels it to GROUND_Z_M (keeps GPS scale + horizontal
# alignment; fixes only tilt + Z offset).
LEVEL_GROUND_PLANE            = True
GROUND_RANSAC_ITERS           = 1000   # RANSAC sample iterations
GROUND_RANSAC_THRESH_M        = 0.30   # inlier orthogonal distance to plane (m)
GROUND_RANSAC_MIN_INLIER_FRAC = 0.30   # below this, no dominant plane → skip leveling
GROUND_LEVEL_MAX_TILT_DEG     = 45.0   # reject implausibly large corrections (likely not ground)

# ─────────────────────────────────────────────────────────────────────────────
# ORIENTATION SOLVER
# ─────────────────────────────────────────────────────────────────────────────
# Per-camera bounds for the pyceres joint optimisation.
# Seeds come from MASt3R-SfM camera poses (not GPS/GeoCalib).
# Bounds are applied around the MASt3R-seeded values.

SOLVER_PITCH_OFFSET  =  50.0   # ± degrees around the MASt3R/GeoCalib seed
SOLVER_PITCH_FLOOR   = -90.0   # absolute minimum pitch
SOLVER_PITCH_CEILING =  45.0   # absolute maximum pitch

SOLVER_YAW_OFFSET_RANGE  =  15.0   # ± degrees around seeded yaw
SOLVER_ROLL_OFFSET_RANGE =   5.0   # ± degrees around seeded roll

# Position search range around MASt3R-seeded position.
SOLVER_POSITION_RANGE_H  =  10.0   # ± metres East
SOLVER_POSITION_RANGE_N  =   5.0   # ± metres North
SOLVER_POSITION_RANGE_V  =   2.0   # ± metres vertical

# Maximum Ceres solver iterations.
SOLVER_MAX_ITERATIONS = 200

# ── Focal length estimation ──────────────────────────────────────────────────
# Set True to add focal length as a free parameter in the Ceres solve.
ESTIMATE_FOCAL_LENGTH = False
FOCAL_LENGTH_RANGE    = 0.22   # ±22% of FOCAL_LENGTH

# ─────────────────────────────────────────────────────────────────────────────
# ROLL DETECTION + CENTER OVERLAY SUPPRESSION
# ─────────────────────────────────────────────────────────────────────────────
# Area bounds (pixels²) for white connected-component blobs to be considered
# a HUD bracket symbol.
ROLL_BRACKET_MIN_AREA  = 80
ROLL_BRACKET_MAX_AREA  = 2500

# Half-width of the bracket search window (pixels from image center).
# Used by pose.py for detection and by undistort.suppress_center_overlays
# as the fallback blur window when no brackets are detected.
BRACKET_SEARCH_MARGIN  = 200

# Gaussian blur applied to brackets and crosshair in frame.undistorted
# before MASt3R and Qwen see the images.
OVERLAY_BLUR_KERNEL    = 51    # must be odd
OVERLAY_BLUR_PAD       = 10    # extra padding (px) around each detected bbox

# IoU threshold above which a Qwen discovery result is excluded as crosshair.
CROSSHAIR_IOU_EXCLUDE  = 0.3

# Pixel padding (original image space) painted solid black around the detected
# crosshair bbox in the temporary copy passed to _qwen_discover_frame.
# Large enough to cover the ⊕ waypoint indicator adjacent to the + crosshair;
# if the van also falls inside the masked region it is recovered by Phase 3b.
CROSSHAIR_MASK_PAD_PX  = 40

# IoU exclusion only applies when the candidate's bbox area is below this multiple
# of the crosshair area — prevents large van bboxes from being wrongly excluded.
CROSSHAIR_SIZE_RATIO   = 1.5

# A real status banner is one thin line of text (~20px tall, ~11:1 aspect ratio
# in the verified examples). Qwen occasionally false-positives on ground texture
# (e.g. tire tracks) that vaguely resembles text; reject implausibly-shaped
# "banner" detections outright rather than trust Qwen's bbox alone.
BANNER_MAX_HEIGHT_PX     = 50   # proc-px; real banner ≈20px, generous margin
BANNER_MIN_ASPECT_RATIO  = 3.0  # width/height; real banner ≈11:1, false positive was ≈2:1

# Confirmed real banner detections sit ~140px below the frame's vertical centre in
# proc-px (a fixed HUD layout slot). Qwen also frequently mistakes the central
# crosshair/bracket HUD cluster (which sits AT the centre, offset ≈0) for the banner.
# Reject anything whose vertical centre isn't clearly below the centre cluster.
BANNER_MIN_Y_OFFSET_FROM_CENTER_PX = 60

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
# Leave empty ({}) to use MASt3R-SfM camera poses for all frames.
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
