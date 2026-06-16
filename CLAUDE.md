# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working style

- **Always create a task list** at the start of any multi-step implementation, using the TaskCreate tool. Mark each task `in_progress` when you start it and `completed` as soon as it's done. This lets the user see live progress.
- Add log lines freely when diagnosing issues — the user is happy to re-run and share output.
- Never commit unless the user explicitly asks.
- **The pipeline is meant to run automatically and autonomously, without manual correspondences.** Don't suggest `--manual-fm-json` (or `--manual-correspondences`) as part of a normal/default `raycast.py` invocation. Only bring it up when the conversation is specifically about strengthening a weak/poorly-constrained frame with hand-picked features.

## Git / GitHub

Remote: `https://github.com/ancientbassghoul/C1`

Push command (requires a GitHub personal access token):
```bash
git push https://<TOKEN>@github.com/ancientbassghoul/C1.git master
```

To let Claude push without exposing the token in chat, the user sets it as an env var first:
```bash
export GITHUB_TOKEN=<token>   # paste in terminal, not in chat
```
Then Claude can run: `git push https://$GITHUB_TOKEN@github.com/ancientbassghoul/C1.git master`

## Overview

Spatial alignment pipeline for drone FPV frames. Given a pixel picked in any frame, the corresponding ground point is re-projected into all other frames (target accuracy ≤ 10 px).

The scene: a stationary van on highly repetitive agricultural ground. Several frames have severe motion/defocus blur. The pipeline uses a **probabilistic, confidence-weighted Ceres solve** rather than deterministic hard-constraint geometry.

**Requires Python 3.11** (3.12+ breaks some dependencies). The virtualenv lives at `venv/`.

---

## Setup

MASt3R and Qwen must be installed first (from source, into `venv/`) — see §External dependencies below. Once that venv exists, install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

`torch` is not listed in `requirements.txt` — it is already present as a MASt3R dependency.

---

## Running the pipeline

All commands use `venv\Scripts\python` (Windows). The frames folder (containing PNG/JPG drone footage) must be provided via `--frames_dir`.

```cmd
:: Tune lens undistortion (inspect config.py FOCAL_LENGTH / FISHEYE_K*)
venv\Scripts\python raycast.py --frames_dir ./frames --preview-undistort

:: Open manual correspondence picker — mark features, save JSON
venv\Scripts\python raycast.py --frames_dir ./frames --manual-correspondences

:: Run full pipeline (automatic anchor discovery + MASt3R) — the default, autonomous path
venv\Scripts\python raycast.py --frames_dir ./frames

:: Reuse a cached Qwen/CLIP anchor result instead of re-running Qwen (output/anchor_cache.json)
venv\Scripts\python raycast.py --frames_dir ./frames --use-saved-qwen

:: Compare solved vs telemetry camera state
venv\Scripts\python raycast.py --frames_dir ./frames --camera-deltas

:: Export solved cameras + van bbox as a Blender Python scene script
venv\Scripts\python export_blender.py --frames_dir ./frames --out_dir ./blender_out --manual-only

:: Score existing correspondences (colour coded red→blue worst→best)
venv\Scripts\python raycast.py --frames_dir ./frames --show-scores output/manual_correspondences.json

:: Run MASt3R matching only (saves auto_matches.json, skips Ceres) — runs on RunPod
venv\Scripts\python raycast.py --frames_dir ./frames --run-matcher-only

:: Headless solve on RunPod — run full pipeline, export solved cameras, no GUI
venv\Scripts\python raycast.py --frames_dir ./frames --use-saved-qwen --export-solve

:: Same run, also dump debug .glb scenes for inspecting the MASt3R reconstruction in Blender
venv\Scripts\python raycast.py --frames_dir ./frames --use-saved-qwen --export-solve --export-mesh

:: Local GUI from RunPod solve — skip pipeline, load JSON, open viewer
venv\Scripts\python raycast.py --frames_dir ./frames --import-solve
```

`--manual-fm-json` / `--manual-correspondences` exist for strengthening a specific
poorly-constrained frame (see §Manual correspondences below) — they are **not** part
of the normal/default workflow and shouldn't be suggested unless that's explicitly
what's being discussed.

Interactive viewer controls: click = pick pixel | scroll = zoom | mid-drag = pan | `s` = save proof sheet | `R` = reset | `q` = quit.

---

## Architecture

### Entry points

- **`raycast.py`** — CLI dispatcher. Parses args, runs the 6-step pipeline, then opens the interactive viewer or batch-outputs a proof sheet.
- **`export_blender.py`** — Runs the solver (or reads manual correspondences), then writes `blender_scene.py` (a Blender Python script) with four collections: `app_cameras`, `rigged_cameras`, `telemetry_cameras`, `Van_BBox`.

### Central config

**`config.py`** is the single source of truth for every tunable parameter: camera intrinsics, OCR crop windows, solver bounds, van geometry, HUD mask regions, Qwen/CLIP thresholds, MASt3R settings, and per-frame overrides (`GIMBAL_PITCH_OVERRIDES`, `CAMERA_ROLL_OVERRIDES`, `CAMERA_POSE_OVERRIDES`).

When behaviour seems wrong, start with `config.py`. Van feature z-plane values are **baked into the manual correspondences JSON at save time** — if you change `VAN_HEIGHT_M` or similar constants, re-open `--manual-correspondences` and press `s` to regenerate.

### The pipeline (`pipeline/`)

```
Step 1  frame.py               Load all images from --frames_dir
Step 2  ocr.py                 EasyOCR extracts GPS lat/lon, heading, and two altitude readings from HUD
Step 3  undistort.py           OpenCV fisheye equidistant model → rectilinear; then HUD hard-mask (paint black)
Step 4  pose.py                GPS → ENU, bracket-roll cascade, GeoCalib pitch, build R matrix per frame
                               Telemetry is a weak prior only — not a hard anchor (see §Telemetry below)
Step 5  detect_anchor.py        Qwen VL per-frame object discovery → shared-object selection
                               → CLIP correlation heatmap → weighted centroid (soft ray)
Step 6  feature_matcher.py     MASt3R-SfM complete graph (N×(N-1)/2 pairs) → dense metric pointmaps
                               → single-stage Ceres solve
                └── orientation_solver.py   pyceres joint optimisation
```

After Step 6, `geometry.py` provides the three-step ray-cast: pixel → world ray → ground intersection → pixel in target frame.

### HUD hard-masking (Step 3)

MASt3R's transformer treats zero-entropy solid-black regions as zero-confidence pixels and excludes them from feature matching automatically. Before passing frames to MASt3R or Qwen, paint all HUD overlay regions solid black `[0,0,0]`. HUD coordinates live in `config.HUD_REGIONS`.

### The two data sources fed into Ceres

**MASt3R-SfM — terrain structure + initial camera poses**
- MASt3R-SfM wrapper runs a **complete graph** (all N×(N-1)/2 pairs, no manual pairing or quality sorting). Outputs: initial camera poses and a dense 3D point cloud with per-point confidence scores.
- MASt3R-SfM is used as a **sensor**, not as the final solver. Its camera poses seed Ceres. Its 3D points become `MASt3RReprojCost` residuals (standard reprojection error: each 3D point must project to its observed 2D pixel in every frame that sees it, weighted by confidence).
- **Do not inject P_anchor or any sparse constraint into MASt3R's internals.** MASt3R-SfM's `sparse_global_alignment` is a raw PyTorch Adam/LBFGS loop with no external constraint API. Hacking it leads to broken autograd gradients. Ceres handles all multi-source fusion.
- `PlaneScatterCost` (z=0 ground plane) is **dead**. MASt3R reconstructs real 3D topography (crop heights, furrows); z=0 would fight its reconstruction.

**Qwen/CLIP — anchor object discovery + soft rays**
- Qwen VL discovers the shared anchor object across frames (open-world, no hardcoded name). Label normalization hierarchy: generic (`vehicle`) → type (`vehicle-truck`) → color → model. Consolidation pass ensures cross-frame label consistency before computing coverage intersection.
- CLIP computes per-frame cosine similarity weight `w` for the anchor crop.
- Output: `AnchorResult` — per-frame centroid pixels + CLIP weights → `AnchorRayCost` residuals in Ceres. `P_anchor` is a free 3-vector in Ceres, initialized from MASt3R's point cloud at the anchor bbox region. This is what prevents the MASt3R reconstruction from drifting — the anchor point acts as a global fixed landmark inside Ceres, not inside MASt3R.

### Coordinate system

MASt3R-SfM operates in its own arbitrary coordinate frame. Ceres runs in this frame. After the Ceres solve, `align_to_telemetry_sim3()` applies a single Umeyama Sim(3) transform (scale + rotation + translation) to map solved camera positions to GPS ENU. GPS enters the pipeline exactly once, here, without polluting the visual solve.

Camera frame is **OpenCV** (X right, Y down, Z forward). The rotation matrix `R` satisfies `p_cam = R @ (p_world − position_enu)`.

### Ceres solver (`pipeline/orientation_solver.py`)

**Single-stage solve.** Seeds from MASt3R-SfM camera poses (not GPS/GeoCalib).

**Residual types:**
- `MASt3RReprojCost` — for each (frame, 3D point, observed pixel) from MASt3R: the 3D point must reproject to its observed pixel given the solved camera pose. Weight = MASt3R confidence. **3D points are free parameters (full BA, not motion-only).** Fixing them would cause the rigid ground points to fight `AnchorRayCost` and domain constraints, corrupting camera orientation. Full BA lets the topography flex to absorb MASt3R's imperfections (~15,000 params total — trivially fast in pyceres).
- `AnchorRayCost` (soft) — Qwen/CLIP anchor centroid ray must pass through shared `P_anchor` (free 3-vector, initialized via multi-view projection of MASt3R point cloud into anchor bboxes across ≥2 frames). Blurry frames contribute down-weighted residuals; `w < CLIP_ANCHOR_MIN_WEIGHT` frames are skipped.
- `ManualReprojCost` — manually-picked pixels treated as shared 3D points (weight = `MANUAL_CORRESPONDENCE_WEIGHT`); cross-frame reprojection.
- `VanMetricCost` — wheel_axis / roof_edge pairs: known physical distance constraint between two 3D point free parameters.

**Implementation notes:**
- **CLIP prompt template** — use `f"a crisp, clear aerial photograph of a {label}"`, never raw label text. Raw single-word embeddings have compressed variance; the template stabilizes weight scores across sharp vs. blurry frames.
- **Qwen bbox coordinates** — Qwen VL returns bbox tokens in a custom resolution window (relative to the model's internal padding, typically ~1000 px). Always use Qwen's native coordinate-scaling utilities or manually account for aspect-ratio letterboxing when mapping to pixel space.
- **Sim(3) uses only high-confidence frames** — `align_to_telemetry_sim3()` estimates the Umeyama transform using only cameras with `w >= CLIP_ANCHOR_THRESHOLD`. One GPS outlier from a blurry frame can corrupt the global scale; high-confidence frames are the control set. The resulting transform is applied to all cameras.

### Telemetry handling

GPS telemetry has multi-metre translation errors. **Do not inject it into Ceres as a residual.** The solve runs purely on visual constraints. GPS is used only as the target for a post-solve Sim(3) alignment (Umeyama's method) that maps the MASt3R coordinate frame to ENU. This prevents GPS errors from warping the ground plane or introducing artificial rotation.

### Manual correspondences (`pipeline/manual_correspondence_ui.py`) — optional

These are no longer required for the primary solve. Use them when the automatic solve leaves a specific frame poorly constrained. Four feature types (cycle with `T`):

| Type | Colour | Constraint |
|---|---|---|
| `ground` | Cyan | Ray hits z = 0 |
| `wheel_axis` | Magenta | Axis direction + wheelbase (3.275 m) |
| `roof_edge` | Orange/Yellow | Axis direction + van width (1.92 m) |
| `roof_plane` | Green/Yellow | Ray hits z = VAN_HEIGHT_M (1.94 m) |

Saved to `MANUAL_CORRESPONDENCES_FILE` (default `./output/manual_correspondences.json`). `[`/`]` navigate correspondences; `d` deletes a frame's mark or the whole correspondence; `s` writes JSON immediately.

### RunPod headless workflow

Heavy compute (MASt3R, Qwen) runs on RunPod (no display). The local machine handles the interactive viewer.

```
RunPod:  python raycast.py --frames_dir ./frames --use-saved-qwen --export-solve
         → writes output/solved_cameras.json

Local:   (copy solved_cameras.json from pod)
         python raycast.py --frames_dir ./frames --import-solve
         → skips OCR/pose/solver, loads JSON, opens viewer
```

`--import-solve` runs only load + undistort (a few seconds) then injects the solved poses. Default path is `output/solved_cameras.json` for both flags; override with an explicit path argument.

### Debug mesh export (`--export-mesh`) — inspecting the MASt3R reconstruction in Blender

`solved_cameras.json` only carries camera poses — it has no view into the actual MASt3R 3D
reconstruction that fed the Ceres solve. `--export-mesh [DIR]` (default `output/debug/`) writes
two self-contained `.glb` scenes, built from `pipeline/mast3r_matcher.py`'s
`export_raw_glb()` / `export_aligned_glb()`, both of which reuse `dust3r.viz`'s own building
blocks directly (`pts3d_to_trimesh`, `cat_meshes`, `add_scene_cam` — the same ones behind the
"download .glb" button in MASt3R/dust3r's official demo; **not** `dust3r.demo`, which imports
`gradio` at module level):

- `mast3r_raw_scene.glb` — MASt3R's raw per-view dense mesh + camera reference cards, in
  MASt3R's own native (arbitrary-scale, pre-Ceres, pre-Sim(3)) coordinate frame. Written inside
  `run_complete_graph` right after `sparse_global_alignment`, before teardown — works even with
  `--run-matcher-only` (Ceres never has to run).
- `mast3r_aligned_scene.glb` — the same dense mesh plus the Ceres-refined sparse point cloud,
  both rigidly moved into ENU via the identical Sim(3) transform applied to the final cameras,
  with reference cards at each frame's *final solved* pose (the same ones in
  `solved_cameras.json`). This is the one that answers "does the reconstructed terrain actually
  sit where the solved cameras say it should" — open it alongside the `app_cameras` collection
  from `export_blender.py`'s output to compare directly. Requires a real Ceres/Sim(3) result;
  not written under `--run-matcher-only` or an aborted solve.

Open either with Blender's **File > Import > glTF 2.0**. No custom Blender script is needed for
this part — it's a self-contained snapshot, separate from `export_blender.py`'s camera-rig
script (which you can still run and import alongside it).

### Key output files

- `output/manual_correspondences.json` — hand-picked feature correspondences (ground truth for solver).
- `output/auto_matches.json` — MASt3R automatic matches saved after `--run-matcher-only`.
- `output/solved_cameras.json` — serialized solved camera state (position_enu, heading, pitch, roll, K_undist) for the RunPod → local handoff.
- `output/debug/mast3r_raw_scene.glb` / `mast3r_aligned_scene.glb` — debug mesh exports from `--export-mesh` (see above).
- `output/debug/anchor/` — anchor detection debug images from `--preview-anchor`.
- `output/debug/hud_masks/` — HUD masking before/after from `--preview-hud-masks`.
- `models/` — downloaded model weights (`./models/`, gitignored).

### External dependencies not in requirements.txt

These must be installed manually (from source, into the same `venv/`):
- **MASt3R-SfM** (`mast3r`, `dust3r`) — requires CUDA; install from source on RunPod
- **Qwen VL** (`Qwen2.5-VL-7B-Instruct`) — requires CUDA; loaded via `transformers>=4.49`
- **CLIP** (`openai/clip-vit-large-patch14`) — downloaded at runtime via `transformers`

`pyceres` and `geocalib` are now in `requirements.txt` and install automatically.

`trimesh` is not listed in `requirements.txt` either — like `torch`, it's already present as a
dust3r dependency (declared in `dust3r/requirements.txt`), installed when MASt3R/dust3r is set
up from source. Only needed for `--export-mesh`.

**Model download path:** All `from_pretrained()` calls must pass `cache_dir=config.MODEL_CACHE_DIR` (default `./models/`). Never let models download to the pod's `/root/.cache` — it is wiped when the pod stops. `models/` is in `.gitignore`.

**VRAM management:** Qwen (~7B params) and MASt3R ViT-Large both consume significant VRAM. `detect_anchor.py` explicitly tears down both Qwen and CLIP before returning (`del model; del processor; torch.cuda.empty_cache()`) so MASt3R can load without OOM on 16–24 GB pods. Never hold both in VRAM simultaneously.

**Qwen bbox coordinates:** Qwen2.5-VL returns bbox tokens as integers in [0, 1000], normalized to its internal processing window (with possible aspect-ratio letterboxing). Scale by `img_h / 1000` and `img_w / 1000` independently — do not assume 1:1 mapping to image dimensions.
