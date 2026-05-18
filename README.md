# Raycast Challenge

Spatial alignment pipeline for drone FPV frames.  
Select a pixel in any frame → the corresponding ground point is re-projected into all other frames with a target accuracy of ≤ 10 pixels.

---

## Quick start

```cmd
:: First time only — creates venv and installs everything
setup.bat

:: Check undistortion (tune config.py if lines still look curved)
venv\Scripts\python raycast.py --frames_dir ./frames --preview-undistort

:: Interactive viewer — click any frame to reproject
venv\Scripts\python raycast.py --frames_dir ./frames

:: Headless / batch mode (no GUI, saves proof sheet directly)
venv\Scripts\python raycast.py --frames_dir ./frames ^
    --batch ^
    --source-frame "2026-02-15_16-28-05_04752" ^
    --pick 640 400
```

**Interactive controls:**  click = pick pixel | `s` = save proof sheet | `r` = reset | `q` = quit

---

## CLI reference

All arguments to `raycast.py`:

| Argument | Default | Description |
|---|---|---|
| `--frames_dir` | *(required)* | Directory containing the drone frames (PNG/JPG). |
| `--batch` | off | Headless mode: reproject a fixed pixel and save the proof sheet without opening the GUI. Requires `--source-frame` and `--pick`. |
| `--source-frame` | — | *(Batch mode)* Filename stem of the source frame, e.g. `2026-02-15_16-28-05_04752`. |
| `--pick PX PY` | — | *(Batch mode)* Pixel coordinates to pick in the source frame, e.g. `640 400`. |
| `--height` | `tor` | Camera Z source: `tor` = takeoff-relative altitude (recommended), `agl` = above-ground-level only, `avg` = average of both. |
| `--output_dir` | from config.py | Override the output directory for proof sheets and debug images. |
| `--preview-undistort` | off | Show each undistorted frame one by one, then exit. Use to tune `FOCAL_LENGTH` and `FISHEYE_K*` in `config.py`. |
| `--preview-enhanced` | off | Show the CLAHE+unsharp frames fed to SuperPoint, then exit. Useful for diagnosing poor feature detection. |
| `--enhance` | off | Enable CLAHE+unsharp preprocessing before LightGlue feature matching. |
| `--no-refine` | off | Skip pitch refinement entirely; use only manual overrides from `config.py`. |
| `--feature-matcher-debug` | off | Save annotated side-by-side match images (keypoints, ground region band, van bbox, connecting lines) to `{output_dir}/debug/` for every matched frame pair. |

`export_blender.py` accepts the same `--height`, `--enhance`, and `--feature-matcher-debug` flags. Its equivalent of `--calculate-orientation` (runs the orientation solver before export) replaces the old `--optimize-pitch`.

---

## Setup

### Step 1 — Python 3.11

**Requires Python 3.11.** Python 3.12+ causes build failures for some dependencies. Install from https://www.python.org/downloads/release/python-3119/

```cmd
python --version
:: Should print: Python 3.11.x
```

### Step 2 — CUDA Toolkit 12.1

Required for GroundingDINO; also accelerates LightGlue and GeoCalib at runtime.

1. Download and install **CUDA Toolkit 12.1** from https://developer.nvidia.com/cuda-12-1-0-download-archive
   During installation choose **Custom** and uncheck the driver component if you already have a newer driver.
2. Add the `CUDA_HOME` environment variable (System → Advanced → Environment Variables → New):
   - **Name:** `CUDA_HOME`
   - **Value:** `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`

### Step 3 — Run SETUP_FIRST.bat

Installs the venv, PyTorch, all `requirements.txt` dependencies, LightGlue, and the local package.

```cmd
SETUP_FIRST.bat
```

### Step 4 — Install GroundingDINO  *(manual — requires VS Build Tools + CUDA)*

**a) Install Visual Studio Build Tools**

Download from https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022 and select **Desktop development with C++**. In the right-hand panel ensure **MSVC v143 – VS 2022 C++ x64/x86 build tools (v14.36)** is checked.

**b) Install GroundingDINO** — run in a plain `cmd.exe` window (not PowerShell):

```cmd
:: 1. Activate MSVC compiler for this session
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.36

:: 2. Tell distutils to use the SDK compiler
set DISTUTILS_USE_SDK=1

:: 3. Ensure build tools are fresh
venv\Scripts\pip install wheel setuptools

:: 4. Build and install (--no-build-isolation keeps our pinned transformers)
venv\Scripts\pip install git+https://github.com/IDEA-Research/GroundingDINO.git --no-build-isolation
```

The model weights (~700 MB) are downloaded automatically on first use.

### Step 5 — Run SETUP_SECOND.bat

Installs pyceres and GeoCalib (both are pre-built wheels, no compiler needed).

```cmd
SETUP_SECOND.bat
```

---

Nothing is installed globally. To start fresh, delete `venv` and re-run from Step 3
---

## Project structure

```
raycast_challenge/
├── raycast.py           Entry point & CLI (pipeline + mode dispatch)
├── config.py            All tunable parameters — start here if accuracy is poor
├── manual_calibrate.py  Interactive tool: click ground correspondences → pitch overrides
├── export_blender.py    Export camera poses as a Blender Python scene script
├── mask_van.py          Undistort frames and paint a black rectangle over the van
├── requirements.txt     Pip dependency list
├── setup.bat            One-click Windows environment setup
├── setup.py             Package definition (for reference / pip install -e .)
└── pipeline/
    ├── frame.py             Frame dataclass + bulk image loader
    ├── ocr.py               Telemetry extraction from HUD overlays (EasyOCR)
    ├── undistort.py         Fisheye lens correction (OpenCV fisheye model)
    ├── pose.py              Camera pose estimation: GPS → ENU, roll detection, R matrix
    ├── detect_van.py            Van detection (GroundingDINO → white-blob fallback)
    ├── van_corners.py           Van corner geometry, visibility, and bbox matching
    ├── pitch_from_geocalib.py   Batch pitch + roll estimation via GeoCalib (shared intrinsics)
    ├── orientation_solver.py  Pyceres cost functions + joint solve (ground scatter + van corners)
    ├── feature_matcher.py   Ground feature matching and pitch refinement orchestration
    ├── geometry.py          Ray–ground-plane intersection + reprojection math
    └── ui.py                Interactive grid viewer + proof-sheet export
```

---

## Pipeline overview

The pipeline runs in 6 sequential steps each time you launch `raycast.py`:

```
Step 1  Load frames         – read all images from --frames_dir
Step 2  OCR telemetry       – extract GPS, heading, altitudes from HUD
Step 3  Undistort           – correct fisheye lens distortion
Step 4  Estimate poses      – build camera positions + rotation matrices; GeoCalib batch-estimates pitch + roll for all frames
Step 5  Detect van          – GroundingDINO (→ white-blob fallback) locates the van in each frame
Step 6  Refine pitches      – ground-scatter matching, van region excluded from keypoints
        ↓
        Interactive viewer / batch output
```

---

## Methodology

### Step 1 — Lens undistortion

The drone camera has a wide-angle fisheye lens. This is immediately visible in any overhead shot: the vegetation boundary (a straight line on the ground) curves significantly across the raw frame. All geometry must be computed on undistorted images.

We use OpenCV's equidistant fisheye model:

```
r_distorted = f · θ · (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
```

where θ is the angle from the optical axis. After undistortion, every frame is a standard rectilinear (pinhole) projection suitable for ray-casting.

Parameters live in `config.py`. Use `--preview-undistort` to inspect results and tune until straight real-world lines (tyre tracks, vegetation boundary) look straight in the output.

### Step 2 — Telemetry OCR

EasyOCR reads each frame's HUD overlay and extracts three fields:

**GPS lat/lon** (top-left green text) → camera XY position in the world frame.

**Compass heading** (large number, top-centre) → camera yaw (0–360°, clockwise from North).

**Two altitude readings** (bottom-right). The HUD shows two values side by side:

| Position | Sensor | Meaning |
|---|---|---|
| Left (with ↓ icon) | Barometric | Height above the **takeoff/launch point** |
| Right | Radar / lidar | Height above the **ground directly below** (AGL) |

These are parsed **positionally** (left = takeoff-ref, right = AGL), not by min/max, so the correct value is always assigned even if AGL exceeds the takeoff-relative height (e.g. drone flying over terrain higher than its launch point).

The **takeoff-relative altitude** is used as the camera's Z coordinate in the shared world frame, because it references the same fixed point in every frame — giving a consistent altitude datum across the entire dataset.

The **AGL altitude** is used to compute the terrain offset: how high the ground sits under this frame relative to the launch point. This places the ground plane at the correct Z when casting rays (see Step 5).

### Step 3 — Camera pose estimation

All camera positions are converted from WGS-84 GPS to a local East-North-Up (ENU) Cartesian frame using the flat-Earth approximation, accurate to millimetres for scenes under ~10 km:

```
ΔEast  = Δlon · cos(lat_ref) · R_earth · π/180
ΔNorth = Δlat               · R_earth · π/180
ΔUp    = alt_takeoff_ref_m                       ← consistent Z datum
```

The ENU origin is the GPS position of the first frame.

**Camera roll** is detected from the artificial horizon bracket indicator in the HUD. Two symmetric white bracket symbols (⌐ ¬) are extracted by thresholding at ≥235 and finding the connected component pair that is most equidistant and opposite about the image centre. The angle of the line joining them gives the roll.

**Gimbal pitch and roll** are both estimated in a single GeoCalib batch call across all frames before the per-frame pose loop. GeoCalib is a neural camera calibration model (same cvg group as LightGlue) that estimates the gravity direction in the camera frame from visual cues (line distributions, vanishing points). Running with `shared_intrinsics=True` constrains all frames to share a single focal length — physically correct since all frames come from the same camera body. From the per-frame gravity vector `g_cam`:

```
pitch = arcsin(-g_cam[2])          # exact, roll-independent
roll  = arctan2(-g_cam[0], g_cam[1])  # exact, pitch-independent
```

The pitch estimate seeds the Ceres orientation solver in Step 5 and sets each camera's search window (±`SOLVER_PITCH_OFFSET`). The roll estimate is used as a fallback for frames where HUD bracket detection fails. Priority order:

| | Pitch | Roll |
|---|---|---|
| 1st | `GIMBAL_PITCH_OVERRIDES` | `CAMERA_ROLL_OVERRIDES` |
| 2nd | GeoCalib | HUD bracket detection |
| 3rd | 0° seed | GeoCalib |
| 4th | — | `CAMERA_ROLL_DEG` (0°) |

From position + yaw + pitch + roll we build the rotation matrix R that maps world vectors to camera vectors:

```
fwd   = [sin(yaw)·cos(pitch),  cos(yaw)·cos(pitch),  sin(pitch)]   # ENU
right = normalise(fwd × world_up)
down  = fwd × right
R_world_from_cam = [right | down | fwd]   (columns = camera axes in world)
R_cam_from_world = R_world_from_cam.T     (orthonormal → transpose = inverse)
```

### Step 4 — Van detection

Before feature matching, each undistorted frame is searched for the white van using **GroundingDINO** with the text prompt `"white van . delivery van . cargo van"`. If GroundingDINO returns no result above the confidence threshold, a **white-blob fallback** kicks in: the frame is converted to HSV, pixels with low saturation and high value are thresholded, and the largest blob with a plausible aspect ratio in the lower 60% of the image is returned as the bounding box.

The detected bounding box per frame is passed directly to the pitch refinement step so that van pixels are excluded from feature matching. The van sits above the ground plane, so any keypoints on it would produce ray–ground intersections with systematic Z errors — corrupting the pitch estimate. Detection results are logged; frames where neither method finds a van proceed to refinement without masking.

### Step 5 — Ground-scatter pitch refinement

The camera gimbal pitch is not available in the HUD. This step recovers it by minimising the scatter of ray–ground intersections for shared ground features across frame pairs.

**Feature matching** — SuperPoint keypoints are extracted from each frame and matched with LightGlue. Matches are filtered to the lower 55% of the image content area (ground region), any matches falling inside the van bounding box are excluded, and the remainder are validated with RANSAC homography.

**Orientation optimisation** — For each matched feature seen in two or more frames, each camera shoots a ray from its pixel through the ground plane. A perfect orientation would make all rays converge on the same point. The optimizer (`orientation_solver.py`, pyceres) minimises the 2D scatter (East, North) of these intersections with a Cauchy robust loss. It simultaneously solves for three unknowns per camera: **pitch**, **yaw offset** (±`SOLVER_YAW_OFFSET_RANGE`, correction on the OCR heading), and **roll offset** (±`SOLVER_ROLL_OFFSET_RANGE`, correction on the bracket-detected roll). Each camera's pitch search window is centred on its GeoCalib seed and extends ±`SOLVER_PITCH_OFFSET` degrees (default ±20°), clamped to [`SOLVER_PITCH_FLOOR`, `SOLVER_PITCH_CEILING`]. All bounds are set in `config.py`.

**Single joint solve (pyceres):** all cameras and the van pose are solved simultaneously in one Ceres optimisation. The sparse block structure — each ground residual touches exactly two camera parameter blocks — is exploited by Ceres's `SPARSE_NORMAL_CHOLESKY` linear solver, giving dramatically faster convergence than the previous dense scipy approach.

The key enabler is the **takeoff-relative altitude**: because all camera Z coordinates share the same datum, the GPS-derived positions are geometrically consistent across the dataset. This gives the optimizer a fixed scale reference and lifts the scale ambiguity that normally prevents pitch recovery from feature matches alone.

### Step 6 — Ray-casting (the re-projection pipeline)

```
Pick pixel (u, v) in source frame
  → normalised coords    p_n  = K⁻¹ · [u, v, 1]ᵀ
  → world ray direction  d    = R.T · p_n   (unit vector in ENU)
  → ground plane Z       z    = terrain_offset = alt_takeoff_ref − alt_agl
  → ground intersection  P    = origin + t·d   where t = (z − origin.Z) / d.Z
  → pixel in target      proj = K_t · R_t · (P − pos_t)  then perspective divide
```

The ground plane is placed at `Z = terrain_offset_m` (not hardcoded to 0). Both altitude readings are used together: the takeoff-relative altitude positions the camera in the shared coordinate system, and the AGL altitude tells us how high the terrain sits in that same system. The difference between the two — the terrain offset — is the correct ground plane Z for ray intersection.

---

## Tuning guide

| Symptom | Fix |
|---|---|
| Straight lines still curved after undistort | Increase `\|FISHEYE_K1\|` in config.py |
| Image over-corrected (pincushion) | Decrease `\|FISHEYE_K1\|` |
| Too much black border around undistorted frame | Increase `UNDISTORT_SCALE` toward 1.0 |
| Reprojection consistently offset in one direction | Pitch wrong for that frame; add entry to `GIMBAL_PITCH_OVERRIDES` |
| Near-nadir frames reproject poorly | Add manual pitch override (~-85°) if refinement fails to converge |
| Very few ground matches found | Not enough LightGlue inliers; lower `MIN_GROUND_MATCHES` in feature_matcher.py or increase `MATCH_GRID_COLS`/`MATCH_GRID_ROWS` in config.py |
| Reprojection correct for ground, wrong for van roof | Expected — van roof is ~2 m above the ground plane |
| OCR returns None for some frames | Check log output; glare may corrupt a crop; set manual overrides in config.py |

---

## Known limitations

- **Gimbal pitch is estimated, not measured.** The HUD does not expose it directly. GeoCalib provides the initial estimate; the Ceres solver refines it. Near-nadir frames with no usable ground features and poor overlap with other frames may still need a manual override in `config.py` under `GIMBAL_PITCH_OVERRIDES`.
- **Single flat ground plane.** Points on elevated objects (van roof, tree canopy) will reproject with a Z error equal to their height above ground (~2 m for the van), producing a visible pixel offset in the target frame. This is physically correct behaviour, not a bug.
- **OCR is the most fragile step.** Sun glare and HUD overlaps can corrupt individual field reads. The log prints every parsed value — check it on first run and add manual overrides in `config.py` for any frame that reads incorrectly.
- **Roll falls back to GeoCalib when bracket detection fails.** HUD bracket detection is the primary roll source; GeoCalib provides the fallback. Per-frame roll overrides can always be set in `config.py` under `CAMERA_ROLL_OVERRIDES`.
- **LightGlue requires a separate install.** See Setup above.
