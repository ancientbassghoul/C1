# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Spatial alignment pipeline for drone FPV frames. Given a pixel picked in any frame, the corresponding ground point is re-projected into all other frames (target accuracy ≤ 10 px).

**Requires Python 3.11** (3.12+ breaks some dependencies). The virtualenv lives at `venv/`.

## Running the pipeline

All commands use `venv\Scripts\python` (Windows). The frames folder (containing PNG/JPG drone footage) must be provided via `--frames_dir`.

```cmd
:: Tune lens undistortion (inspect config.py FOCAL_LENGTH / FISHEYE_K*)
venv\Scripts\python raycast.py --frames_dir ./frames --preview-undistort

:: Open manual correspondence picker — mark features, save JSON
venv\Scripts\python raycast.py --frames_dir ./frames --manual-correspondences

:: Run full pipeline using saved manual correspondences
venv\Scripts\python raycast.py --frames_dir ./frames --manual-fm-json

:: Compare solved vs telemetry camera state
venv\Scripts\python raycast.py --frames_dir ./frames --manual-fm-json --camera-deltas

:: Export solved cameras + van bbox as a Blender Python scene script
venv\Scripts\python export_blender.py --frames_dir ./frames --out_dir ./blender_out --manual-only

:: Debug feature matching interactively (skips pose/van/Ceres, fast)
venv\Scripts\python raycast.py --frames_dir ./frames --feature-matcher-debug

:: Score existing correspondences (colour coded red→blue worst→best)
venv\Scripts\python raycast.py --frames_dir ./frames --show-scores output/manual_correspondences.json

:: Run LightGlue matching only (saves auto_matches.json, skips Ceres)
venv\Scripts\python raycast.py --frames_dir ./frames --run-matcher-only
```

Interactive viewer controls: click = pick pixel | scroll = zoom | mid-drag = pan | `s` = save proof sheet | `R` = reset | `q` = quit.

## Architecture

### Entry points

- **`raycast.py`** — CLI dispatcher. Parses args, runs the 6-step pipeline, then opens the interactive viewer or batch-outputs a proof sheet.
- **`export_blender.py`** — Runs the solver (or reads manual correspondences), then writes `blender_scene.py` (a Blender Python script) with four collections: `app_cameras`, `rigged_cameras`, `telemetry_cameras`, `Van_BBox`.

### Central config

**`config.py`** is the single source of truth for every tunable parameter: camera intrinsics, OCR crop windows, solver bounds, van geometry, GroundedSAM prompts, HUD mask regions, and per-frame overrides (`GIMBAL_PITCH_OVERRIDES`, `CAMERA_ROLL_OVERRIDES`, `CAMERA_POSE_OVERRIDES`).

When behaviour seems wrong, start with `config.py`. Van feature z-plane values are **baked into the manual correspondences JSON at save time** — if you change `VAN_HEIGHT_M` or similar constants, re-open `--manual-correspondences` and press `s` to regenerate.

### The 6-step pipeline (`pipeline/`)

```
Step 1  frame.py               Load all images from --frames_dir
Step 2  ocr.py                 EasyOCR extracts GPS lat/lon, heading, and two altitude readings from HUD
Step 3  undistort.py           OpenCV fisheye equidistant model → rectilinear
Step 4  pose.py                GPS → ENU, bracket-roll cascade, GeoCalib pitch, build R matrix per frame
Step 5  detect_van.py          GroundingDINO → white-blob fallback → van bbox per frame
Step 6  feature_matcher.py     LightGlue + manual correspondences → Ceres solve (two-stage)
                └── orientation_solver.py   pyceres joint optimisation
```

After Step 6, `geometry.py` provides the three-step ray-cast: pixel → world ray → ground intersection → pixel in target frame.

### Coordinate system

World frame is **ENU** (East-North-Up), origin = GPS position of the first frame. Camera frame is **OpenCV** (X right, Y down, Z forward). The rotation matrix `R` satisfies `p_cam = R @ (p_world − position_enu)`.

### Ceres solver (`pipeline/orientation_solver.py`)

Optimises 6 parameters per camera: `[pitch, yaw_off, roll_off, dx, dy, dz]` plus one global `van_heading_deg`.

**Two-stage solve:**
- Stage 1: manual correspondences only (~6 iterations, fast anchor).
- Stage 2: manual + LightGlue auto-matches filtered by Stage-1 reprojection error (threshold in `LIGHTGLUE_FILTER_THRESHOLD_PX`).

**Residual types:**
- `PlaneScatterCost` — matched ground feature pairs; rays must intersect the ground plane at the same point.
- `AxisPairCost` — wheel_axis / roof_edge manual marks; constrains physical axis direction and known length.
- `PlaneScatterCost` (roof_plane) — roof marks; ray must hit `VAN_HEIGHT_M`.

New frames are seeded with empirical biases derived from `--camera-deltas`: `SOLVER_YAW_SEED_OFFSET`, `SOLVER_DX/DY/DZ_SEED`.

### Manual correspondences (`pipeline/manual_correspondence_ui.py`)

Four feature types (cycle with `T`):

| Type | Colour | Constraint |
|---|---|---|
| `ground` | Cyan | Ray hits z = 0 |
| `wheel_axis` | Magenta | Axis direction + wheelbase (3.275 m) |
| `roof_edge` | Orange/Yellow | Axis direction + van width (1.92 m) |
| `roof_plane` | Green/Yellow | Ray hits z = VAN_HEIGHT_M (1.94 m) |

Saved to `MANUAL_CORRESPONDENCES_FILE` (default `./output/manual_correspondences.json`). `[`/`]` navigate correspondences; `d` deletes a frame's mark or the whole correspondence; `s` writes JSON immediately.

### Key output files

- `output/manual_correspondences.json` — hand-picked feature correspondences (ground truth for solver).
- `output/auto_matches.json` — LightGlue automatic matches saved after `--run-matcher-only`.
- `output/debug/` — debug images from `--preview-ground-masks`, `--feature-matcher-debug`, etc.

### External dependencies not in requirements.txt

These must be installed manually (see README.md Setup section):
- **LightGlue + SuperPoint** — installed via `SETUP_FIRST.bat`
- **GroundingDINO** — requires CUDA 12.1 and VS Build Tools; built from source in `cmd.exe`
- **pyceres** + **GeoCalib** — installed via `SETUP_SECOND.bat`
- **SAM weights** (`sam_vit_h_4b8939.pth`) and **GroundingDINO weights** (`groundingdino_swint_ogc.pth`) live in the directory pointed to by `config.GROUNDED_SAM_DIR` (currently `D:\EXTEND\Grounded-Segment-Anything`)
