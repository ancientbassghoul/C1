# Raycast Challenge

Spatial alignment pipeline for drone FPV frames.  
Select a pixel in any frame → the corresponding ground point is re-projected into all other frames with a target accuracy of ≤ 10 pixels.

---

## Quick start

```cmd
:: First time only — see the Setup section below before running anything
:: Check undistortion (tune config.py if lines still look curved)
venv\Scripts\python raycast.py --frames_dir ./frames --preview-undistort

:: Open manual correspondence picker (mark features, save JSON)
venv\Scripts\python raycast.py --frames_dir ./frames --manual-correspondences

:: Run full pipeline using the saved manual correspondences JSON
venv\Scripts\python raycast.py --frames_dir ./frames --manual-fm-json

:: Analyse telemetry vs solved camera state
venv\Scripts\python raycast.py --frames_dir ./frames --manual-fm-json --camera-deltas

:: Export solved cameras to Blender (manual-only solve, no LightGlue)
venv\Scripts\python export_blender.py --frames_dir ./frames --out_dir ./blender_out --manual-only
```

**Interactive controls:** click = pick pixel | scroll = zoom | mid-drag = pan | `s` = save proof sheet | `R` = reset | `q` = quit

---

## CLI reference

### `raycast.py`

| Argument | Default | Description |
|---|---|---|
| `--frames_dir` | *(required)* | Directory containing the drone frames (PNG/JPG). |
| `--height` | `tor` | Camera Z source: `tor` = takeoff-relative altitude (recommended), `agl` = above-ground-level, `avg` = average of both. |
| `--output_dir` | from config.py | Override the output directory for proof sheets and debug images. |
| `--manual-correspondences` | off | Open the interactive manual correspondence picker UI. Loads and undistorts frames, then shows the multi-frame grid for marking features. Saves to `MANUAL_CORRESPONDENCES_FILE`. Does **not** run the solver. |
| `--manual-fm-json [PATH]` | — | Run the full pipeline (OCR, undistort, pose, van detection) then feed manual correspondences to Ceres instead of LightGlue. PATH defaults to `MANUAL_CORRESPONDENCES_FILE` in config.py. |
| `--show-scores [PATH]` | — | Score correspondences and open the viewer coloured red→blue by quality. Without `--manual-correspondences`: scores `AUTO_MATCHES_FILE`. With `--manual-correspondences`: scores `MANUAL_CORRESPONDENCES_FILE`. Optional PATH overrides either default. |
| `--camera-deltas` | off | After solving, print a two-table report comparing final solved camera state to raw telemetry: position ΔEast/ΔNorth/Z and orientation Δyaw/pitch/roll. |
| `--no-refine` | off | Skip automatic pitch/orientation refinement entirely. Uses config overrides only (`GIMBAL_PITCH_OVERRIDES`, `CAMERA_ROLL_OVERRIDES`). |
| `--cameras-init-from-config` | off | Seed camera poses from `CAMERA_POSE_OVERRIDES` in config.py before running the orientation solver. Bypasses GPS + GeoCalib for overridden frames. |
| `--enhance` | off | Enable CLAHE + unsharp preprocessing before LightGlue feature matching. |
| `--feature-matcher-debug` | off | Open the **interactive feature matcher debug viewer** — a GUI showing SuperPoint keypoints per frame, coloured by ground mask, with click-to-trace matching. Skips pose estimation, van detection, and Ceres (load + undistort only). See *Feature matching inspection* below. |
| `--preview-undistort` | off | Show each undistorted frame one by one, then exit. Tune `FOCAL_LENGTH` and `FISHEYE_K*` in `config.py`. |
| `--preview-enhanced` | off | Show the CLAHE+unsharp frames fed to SuperPoint, then exit. |
| `--preview-ground-masks` | off | Compute and save GroundedSAM ground mask debug panels to `{output_dir}/debug/masks/`. |
| `--preview-hud-masks` | off | Apply geometric HUD masks and save debug images. No model inference — fast. |
| `--preview-hsv-masks` | off | Compute HSV-based sky/vegetation masks and save debug panels. |
| `--run-matcher-only` | off | Run steps 1–6 through LightGlue then exit after saving `auto_matches.json`. Skips Ceres. |
| `--batch` | off | Headless mode: reproject a fixed pixel and save the proof sheet. Requires `--source-frame` and `--pick`. |
| `--source-frame` | — | *(Batch mode)* Filename stem of the source frame, e.g. `2026-02-15_16-28-05_04752`. |
| `--pick PX PY` | — | *(Batch mode)* Pixel coordinates to pick in the source frame, e.g. `640 400`. |

### `export_blender.py`

| Argument | Default | Description |
|---|---|---|
| `--frames_dir` | *(required)* | Directory containing the drone frames. |
| `--out_dir` | *(required)* | Output directory for undistorted images and `blender_scene.py`. |
| `--manual-only` | off | Solve using manual correspondences JSON only — skips LightGlue and GroundedSAM entirely. Fast and deterministic. Requires a populated `manual_correspondences.json`. |
| `--calculate-orientation` | off | Run the full orientation solver (LightGlue + manual correspondences) before export. Without either this flag or `--manual-only`, no solver runs and cameras are exported from raw telemetry only. |
| `--height` | `tor` | Camera Z source: `tor` = takeoff-relative altitude (recommended), `agl` = above-ground-level, `avg` = average of both. |
| `--enhance` | off | Enable CLAHE + unsharp preprocessing before LightGlue feature matching. |

The generated `blender_scene.py` creates three camera collections:
- **`app_cameras`** — solved poses (R matrix, exact solver output)
- **`rigged_cameras`** — Euler hierarchy (yaw → pitch → roll empties), easier to manually adjust in Blender
- **`telemetry_cameras`** — raw telemetry only (GPS position with Z from `tor`, OCR heading, bracket roll, pitch = 0°), for direct comparison with solved cameras
- **`Van_BBox`** — wireframe bounding box derived from `roof_edge` and `wheel_axis` manual marks, with orange feature locators for each 2D mark back-projected to 3D

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

### Step 4 — Install GroundingDINO *(manual — requires VS Build Tools + CUDA)*

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

Nothing is installed globally. To start fresh, delete `venv` and re-run from Step 3.

---

## Project structure

```
raycast_challenge/
├── raycast.py                   Entry point & CLI (pipeline + mode dispatch)
├── export_blender.py            Export solved cameras + van bbox as a Blender Python scene script
├── config.py                    All tunable parameters — start here if accuracy is poor
├── camera_deltas.py             Helper for --camera-deltas: formats telemetry vs solved tables
├── manual_correspondence_ui.py  Multi-frame manual feature picker (ground, van features)
├── extract_camera_poses.py      Utility: dump final solved camera poses to JSON/CSV
├── SETUP_FIRST.bat              Step 1 of Windows environment setup
├── SETUP_SECOND.bat             Step 2 of Windows environment setup (pyceres + GeoCalib)
├── requirements.txt             Pip dependency list
├── setup.py                     Package definition (for pip install -e .)
└── pipeline/
    ├── frame.py                     Frame dataclass + bulk image loader
    ├── ocr.py                       Telemetry extraction from HUD overlays (EasyOCR)
    ├── undistort.py                 Fisheye lens correction (OpenCV fisheye model)
    ├── pose.py                      Camera pose estimation: GPS → ENU, roll detection cascade, R matrix
    ├── pitch_from_geocalib.py       Batch pitch + roll estimation via GeoCalib (shared intrinsics)
    ├── detect_van.py                Van detection (GroundingDINO → white-blob fallback)
    ├── van_corners.py               Van geometry constants and Ceres cost function helpers
    ├── feature_matcher.py           Ground feature matching and Ceres solve orchestration (two-stage)
    ├── orientation_solver.py        Ceres cost functions + joint solve (ground scatter + van constraints)
    ├── feature_matcher_debug_ui.py  Interactive feature match explorer (--feature-matcher-debug)
    ├── ground_mask.py               GroundedSAM ground segmentation (positive + exclusion passes)
    ├── hsv_ground_mask.py           HSV-based sky/vegetation masking (experimental)
    ├── correspondence_scorer.py     SuperPoint descriptor quality scoring for manual correspondences
    ├── geometry.py                  Ray–ground-plane intersection + reprojection math
    └── ui.py                        Interactive grid viewer + proof-sheet export
```

---

## Pipeline overview

The pipeline runs in 6 sequential steps each time `raycast.py` is launched:

```
Step 1  Load frames          – read all images from --frames_dir
Step 2  OCR telemetry        – extract GPS, heading, altitudes from HUD (EasyOCR)
Step 3  Undistort            – correct fisheye lens distortion → rectilinear
Step 4  Estimate poses       – GPS → ENU, bracket roll detection cascade,
                               GeoCalib pitch+roll, build initial R matrix per frame
Step 5  Detect van           – GroundingDINO (→ white-blob fallback) locates van bbox
Step 6  Refine orientation   – Ceres joint solve across all cameras
        ├── Stage 1: manual correspondences only  (fast anchor, ~6 iterations)
        └── Stage 2: manual + LightGlue auto-matches
        ↓
        Interactive viewer / batch output / Blender export
```

Steps 4–6 are skipped when using `--feature-matcher-debug`, `--preview-ground-masks`, or `--preview-hsv-masks`.

---

## Methodology

### Step 1 — Lens undistortion

The drone camera has a wide-angle fisheye lens. All geometry must be computed on undistorted images.

We use OpenCV's equidistant fisheye model:

```
r_distorted = f · θ · (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
```

where θ is the angle from the optical axis. After undistortion, every frame is a standard rectilinear (pinhole) projection suitable for ray-casting. With `FOCAL_LENGTH = 651` and `UNDISTORT_SCALE = 0.6`, the undistorted focal length is 390.6 px, giving a 117° horizontal FOV.

Parameters live in `config.py`. Use `--preview-undistort` to inspect results and tune until straight real-world lines (tyre tracks, vegetation boundary) look straight in the output.

### Step 2 — Telemetry OCR

EasyOCR reads each frame's HUD overlay and extracts:

**GPS lat/lon** (top-left green text) → camera XY position in world frame.

**Compass heading** (large number, top-centre) → camera yaw (0–360°, clockwise from North).

**Two altitude readings** (bottom-right):

| Position | Sensor | Meaning |
|---|---|---|
| Left (with ↓ icon) | Barometric | Height above **takeoff/launch point** (`alt_takeoff_ref_m`) |
| Right | Radar / lidar | Height **above ground directly below** (`alt_agl_m`) |

These are parsed positionally, not by min/max, so the correct value is always assigned.

The **takeoff-relative altitude** (`tor`) is used as the camera's Z coordinate — it references the same fixed point in every frame, giving a consistent altitude datum across the entire dataset.

### Step 3 — Camera pose estimation

GPS is converted to a local ENU (East-North-Up) Cartesian frame using the flat-Earth approximation, accurate to millimetres for scenes under ~10 km:

```
ΔEast  = Δlon · cos(lat_ref) · R_earth · π/180
ΔNorth = Δlat               · R_earth · π/180
ΔUp    = alt_takeoff_ref_m                       ← consistent Z datum
```

**Camera roll** is detected from the HUD artificial horizon bracket indicators (⌐ ¬) using a three-stage cascade:

1. **Two-blob**: find the most opposite, equidistant blob pair — most reliable.
2. **Left-blob fallback**: one clean left-of-centre blob → angle from blob to centre.
3. **Right-blob fallback**: one clean right-of-centre blob → angle from centre to blob.

If all three stages fail, roll falls back to GeoCalib and a **CAPITAL WARNING** is logged.

**Gimbal pitch and roll** are batch-estimated by GeoCalib across all frames before the per-frame pose loop, using `shared_intrinsics=True` (all frames share one camera body). GeoCalib provides the pitch seed for Ceres and the roll fallback.

Priority order:

| | Pitch | Roll |
|---|---|---|
| 1st | `GIMBAL_PITCH_OVERRIDES` in config | `CAMERA_ROLL_OVERRIDES` in config |
| 2nd | GeoCalib | HUD bracket detection cascade |
| 3rd | `SOLVER_PITCH_SEED_NEW` (−1.6°) | GeoCalib (with CAPITAL warning) |
| 4th | — | `CAMERA_ROLL_DEG` (0°) |

Roll source is controlled by `HORIZON_INDICATOR_READING` in `config.py` (default `True` — bracket detection active).

### Step 4 — Van detection

Each undistorted frame is searched for the white van using **GroundingDINO** with the prompt `"white van . delivery van . cargo van"`. If no result clears the confidence threshold, a **white-blob fallback** thresholds HSV and picks the largest blob with a plausible aspect ratio in the lower 60% of the image.

The detected bbox per frame is passed to the solver so van pixels are excluded from ground feature matching.

### Step 5 — Orientation solver (Ceres)

The full orientation solver runs in two stages, both using pyceres:

**Stage 1** — manual correspondences only (fast anchor, ~6 iterations). Establishes geometric consistency across calibrated frames.

**Stage 2** — manual + LightGlue auto-matches filtered by reprojection error. Extends coverage to frames with good automatic matching.

Each camera has 6 solver parameters: `[pitch, yaw_off, roll_off, dx, dy, dz]`.

Residual types:

- **`PlaneScatterCost`** (ground features) — each matched feature pair shoots rays to the ground plane; minimise 2D scatter of intersections with Cauchy robust loss.
- **`AxisPairCost`** (van wheel_axis, roof_edge) — pair marks constrain a known physical axis direction and distance.
- **`PlaneScatterCost`** (van roof_plane) — single marks on the van roof constrain the camera to agree that the ray hits the known roof height.

Solver seeds for uncalibrated (new) frames use empirical biases derived from `--camera-deltas` analysis: `SOLVER_YAW_SEED_OFFSET = -6.014°`, `SOLVER_DX_SEED = +1.385 m`, `SOLVER_DY_SEED = -0.600 m`, `SOLVER_DZ_SEED = -0.857 m`. Stage-1 (calibrated frames) always uses zero seeds with free bounds.

### Step 6 — Ray-casting (the re-projection pipeline)

```
Pick pixel (u, v) in source frame
  → normalised coords    p_n  = K⁻¹ · [u, v, 1]ᵀ
  → world ray direction  d    = R.T · p_n   (unit vector in ENU)
  → ground plane Z       z    = GROUND_Z_M  (default 0.0)
  → ground intersection  P    = origin + t·d   where t = (z − origin.Z) / d.Z
  → pixel in target      proj = K_t · R_t · (P − pos_t)  then perspective divide
```

---

## Tuning guide

| Symptom | Fix |
|---|---|
| Straight lines still curved after undistort | Increase `|FISHEYE_K1|` in config.py |
| Image over-corrected (pincushion) | Decrease `|FISHEYE_K1|` |
| Too much black border around undistorted frame | Increase `UNDISTORT_SCALE` toward 1.0 |
| Reprojection consistently offset in one direction | Pitch wrong for that frame; add entry to `GIMBAL_PITCH_OVERRIDES` |
| Near-nadir frames reproject poorly | Check GeoCalib pitch seed; Ceres should correct it if ground features exist |
| Very few ground matches found | Run `--feature-matcher-debug` to see which keypoints pass the ground mask. If the ground mask is too aggressive, tune `GROUND_INCLUDE_PROMPT` / `GROUND_EXCLUDE_PROMPT` thresholds in config.py (see `--preview-ground-masks`). Try `--enhance` to improve keypoint detection on low-contrast ground. Lower `MATCH_SPATIAL_DEDUP_THRESH` to keep more spatially similar matches. If automatic matching cannot be improved, use manual correspondences. |
| Reprojection correct for ground, wrong for van roof | Expected — van roof is ~2 m above the ground plane |
| OCR returns None for some frames | Check log; glare may corrupt a crop; set manual overrides in config.py |
| Stage-1 solve slow or high final cost | Seeds are probably wrong; check `--camera-deltas` output for systematic biases |
| Bracket roll N/A for a frame | Joystick icon or other HUD overlay obscuring one bracket; GeoCalib fallback used; CAPITAL warning in log |

---

## Manual calibration

When automatic feature matching produces poor solver convergence, use manual tools to inspect, annotate, and correct.

### Ground mask tuning

```cmd
:: Tune HUD regions (config.py → HUD_REGIONS)
venv\Scripts\python raycast.py --frames_dir ./frames --preview-hud-masks

:: Tune GroundedSAM prompts and thresholds
venv\Scripts\python raycast.py --frames_dir ./frames --preview-ground-masks

:: Tune HSV sky/vegetation ranges
venv\Scripts\python raycast.py --frames_dir ./frames --preview-hsv-masks
```

### Feature matching inspection

```cmd
venv\Scripts\python raycast.py --frames_dir ./frames --feature-matcher-debug
```

Opens an **interactive GUI** showing all frames. Skips pose estimation, van detection, and Ceres entirely (load + undistort only), so it's fast to launch.

**How it differs from `--show-scores`:**
- `--feature-matcher-debug` is *pre-hoc* — it shows raw SuperPoint keypoints and which ones pass the ground mask filter. Use it to diagnose *why* a frame is getting few or poor matches before you've committed to any correspondences.
- `--show-scores` is *post-hoc* — it scores existing saved correspondences (LightGlue or manual) by descriptor similarity. Use it to identify which correspondences are weak after matching has already run.

| Action | Effect |
|---|---|
| Click any frame | Make it the source; see its filtered ground keypoints (cyan dots) |
| Click a cyan dot | Trace that point — orange rings appear in every paired frame |
| Click a paired frame | Make it the new source |
| Scroll | Zoom centred on cursor |
| Middle-drag | Pan |
| Ctrl+Scroll | Adjust marker size |
| `q` / Esc | Quit |

### Manual correspondence picking

```cmd
venv\Scripts\python raycast.py --frames_dir ./frames --manual-correspondences
```

**Feature types** (press `T` to cycle):

| Type | Colour | Constraint | Geometry |
|---|---|---|---|
| `ground` | Cyan | Ray hits z = 0 | Single point per frame |
| `wheel_axis` | Magenta | Axis direction + wheelbase length (3.275 m) | Pair (front + rear wheel centre) |
| `roof_edge` | Orange/Yellow | Axis direction + van width (1.92 m) | Pair (left + right rear roof corner) |
| `roof_plane` | Green/Yellow | Ray hits z = VAN_HEIGHT_M (1.94 m) | Single point per frame |

**Workflow:**

1. Select the feature type with `T`.
2. Zoom into a distinctive feature in one frame; left-click to place a point (or point A of a pair).
3. For pair types, click the same frame again for point B.
4. Pan/zoom to the same feature in other frames and repeat.
5. Press **Enter** or **n** to save the correspondence.
6. Press **s** to write to JSON (also happens on clean quit).

**Controls:**

| Key / action | Effect |
|---|---|
| Left-click | Place / move this frame's point for the current correspondence |
| Enter / n | Save current correspondence (needs ≥ 2 frames) |
| `T` | Cycle feature type (ground → wheel_axis → roof_edge → roof_plane) |
| `[` | Enter edit mode on last saved of current type / move to previous |
| `]` | Move to next saved correspondence of current type |
| `d` | If a frame was clicked since entering edit: delete that frame's mark only. Otherwise: delete the entire correspondence |
| `c` | Cancel / restore current correspondence to pre-edit state |
| `s` | Write all to JSON immediately |
| `R` | Reset zoom/pan to fit all frames |
| Scroll | Zoom centred on cursor |
| Middle-drag | Pan |
| Ctrl+Scroll | Adjust marker size |
| `q` / Esc | Quit (warns on unsaved changes) |

Correspondences are saved to `MANUAL_CORRESPONDENCES_FILE` in `config.py` (default: `./output/manual_correspondences.json`). The JSON persists between sessions. **Important:** z-plane values are baked into the JSON at save time from `config.py`. If you update `VAN_HEIGHT_M` or other geometry constants, re-open the UI and press `s` to regenerate the JSON with the new values — otherwise the solver and Blender export will use stale z-planes.

### Using manual correspondences in the solver

```cmd
:: Use default JSON path from config.py
venv\Scripts\python raycast.py --frames_dir ./frames --manual-fm-json

:: Use a specific file
venv\Scripts\python raycast.py --frames_dir ./frames --manual-fm-json path\to\corr.json

:: Combine with config-seeded camera poses
venv\Scripts\python raycast.py --frames_dir ./frames --manual-fm-json --cameras-init-from-config
```

This runs the full pipeline (OCR, undistort, pose, van detection) then feeds your manual correspondences directly to the Ceres solver, bypassing GroundedSAM and LightGlue entirely. Each correspondence with N frames contributes N*(N-1)/2 pairwise constraints. Van feature constraints (wheel_axis, roof_edge, roof_plane) from the same JSON are also included.

### Scoring correspondences

```cmd
:: Score LightGlue matches (generate without Ceres first)
venv\Scripts\python raycast.py --frames_dir ./frames --run-matcher-only
venv\Scripts\python raycast.py --frames_dir ./frames --show-scores

:: Score manual correspondences (read-only colour view)
venv\Scripts\python raycast.py --frames_dir ./frames --show-scores output\manual_correspondences.json

:: Score + edit in one session
venv\Scripts\python raycast.py --frames_dir ./frames --manual-correspondences --show-scores
```

Markers are coloured **red → yellow → green → cyan → blue** (worst → best). Grey = no SuperPoint keypoint within 25 px (featureless location). Selected correspondence: **magenta ring** in all frames it appears in.

**Interpreting scores:** low scores across oblique frame pairs are expected — same physical point looks different from different angles. Flag correspondences scoring below ~0.25 across all their pairs, especially between nearby frames with similar viewing angles.

---

## Camera deltas analysis

After solving, use `--camera-deltas` to quantify the difference between telemetry and solved cameras:

```cmd
venv\Scripts\python raycast.py --frames_dir ./frames --manual-fm-json --camera-deltas
```

Prints two tables:

**Position table** — per frame: `ΔEast`, `ΔNorth` (solved XY minus GPS ENU), `Z_solved`, `Z_tor`, `Z_agl`.

**Orientation table** — per frame: `OCR_heading`, `yaw_implied`, `Δyaw`, `pitch_implied`, `pitch_solved`, `roll_brkt`.

Consistent Δyaw across frames → fixed compass bias (bake into `SOLVER_YAW_SEED_OFFSET`).  
Varying Δyaw → heading-dependent error (motor EMI on magnetometer).

---

## Known limitations

- **Gimbal pitch is estimated, not measured.** The HUD does not expose it directly. GeoCalib provides the initial estimate; Ceres refines it. Near-nadir frames may still need a manual override under `GIMBAL_PITCH_OVERRIDES` in `config.py`.
- **Single flat ground plane.** Points on elevated objects (van roof, trees) will reproject with a Z error equal to their height. This is physically correct behaviour, not a bug.
- **Van feature marks have limited accuracy at altitude.** The van occupies 40–150 px at 5–21 m altitude. Marks on blurry features (roof corners, wheel centres) have inherent sub-pixel uncertainty that limits world-constraint accuracy.
- **z-plane values are baked into the JSON.** See the note in the Manual correspondence section above.
- **LightGlue requires a separate install.** See Setup above.
