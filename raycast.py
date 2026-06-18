"""
raycast.py – Raycast Challenge entry point.

Usage
─────
Interactive viewer (click any frame to reproject):
    venv\\Scripts\\python raycast.py --frames_dir ./frames

Preview undistortion only (for tuning config.py intrinsics):
    venv\\Scripts\\python raycast.py --frames_dir ./frames --preview-undistort

Preview anchor detection (Qwen VL + CLIP bboxes/weights):
    venv\\Scripts\\python raycast.py --frames_dir ./frames --preview-anchor

Headless batch run (no GUI, outputs proof sheet for a fixed pixel):
    venv\\Scripts\\python raycast.py --frames_dir ./frames \\
        --batch --source-frame "2026-02-15_16-28-05_04752" \\
        --pick 640 400
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("raycast")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Drone-frame spatial alignment and 2D ray-cast re-projection."
    )
    p.add_argument(
        "--frames_dir", required=True,
        help="Directory containing the drone frames (PNG/JPG).",
    )
    p.add_argument(
        "--preview-undistort", action="store_true",
        help="Show undistorted frames one by one, then exit. "
             "Use to tune FOCAL_LENGTH and FISHEYE_K* in config.py.",
    )
    p.add_argument(
        "--batch", action="store_true",
        help="Headless mode: compute re-projections for a fixed pixel "
             "and save the proof sheet without opening a GUI window.",
    )
    p.add_argument(
        "--source-frame", default=None,
        help="(Batch mode) Filename stem of the source frame.",
    )
    p.add_argument(
        "--pick", nargs=2, type=float, metavar=("PX", "PY"),
        help="(Batch mode) Pixel to pick in the source frame, e.g. '640 400'.",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Override output directory from config.py.",
    )
    p.add_argument(
        "--height", choices=["agl","avg","tor"], default="tor",
        help="Camera Z: tor=takeoff_ref (default), agl=AGL only, avg=average both.",
    )
    p.add_argument(
        "--no-refine", action="store_true",
        help="Skip MASt3R + Ceres refinement (use config overrides only).",
    )
    p.add_argument(
        "--cameras-init-from-config", action="store_true",
        dest="cameras_init_from_config",
        help="Seed camera poses from CAMERA_POSE_OVERRIDES in config.py before "
             "running the orientation solver.",
    )
    p.add_argument(
        "--preview-anchor", action="store_true", dest="preview_anchor",
        help="Run Qwen VL + CLIP anchor detection, save annotated images showing "
             "detected anchor bboxes and CLIP weights per frame to "
             "{OUTPUT_DIR}/debug/anchor/, then exit.",
    )
    p.add_argument(
        "--preview-hud-masks", action="store_true", dest="preview_hud_masks",
        help="Load + undistort frames (HUD masking applied), save side-by-side "
             "debug images to {OUTPUT_DIR}/debug/hud_masks/, then exit.  "
             "Use to verify HUD_REGIONS coordinates in config.py.",
    )
    p.add_argument(
        "--manual-correspondences", action="store_true",
        dest="manual_correspondences",
        help="Open the manual correspondence picker.  No OCR, pose, or model "
             "inference needed.  Correspondences saved to "
             "MANUAL_CORRESPONDENCES_FILE in config.py.",
    )
    p.add_argument(
        "--show-scores", nargs="?", const="", default=None,
        dest="show_scores", metavar="PATH",
        help="Score correspondences with CLIP and open the viewer coloured "
             "red→blue by appearance quality.  Optional PATH overrides the JSON "
             "file to score.",
    )
    p.add_argument(
        "--manual-fm-json", nargs="?", const="", default=None,
        dest="manual_fm_json", metavar="PATH",
        help="Load manual correspondences from PATH (default: "
             "MANUAL_CORRESPONDENCES_FILE) and inject them as high-weight "
             "residuals into the Ceres solver alongside MASt3R observations.",
    )
    p.add_argument(
        "--run-matcher-only", action="store_true", dest="run_matcher_only",
        help="Run steps 1–6 (load → OCR → undistort → pose → anchor → MASt3R) "
             "then exit after saving auto_matches.json.  Skips the Ceres solve.",
    )
    p.add_argument(
        "--camera-deltas", action="store_true", dest="camera_deltas",
        help="After the full solve, print a table comparing solved camera "
             "orientations to raw telemetry, then exit.",
    )
    p.add_argument(
        "--export-solve", nargs="?", const="", default=None,
        dest="export_solve", metavar="PATH",
        help="After the solve, write camera poses to PATH "
             "(default: output/solved_cameras.json) and exit without opening "
             "the GUI.  Compatible with --manual-fm-json for headless RunPod runs.",
    )
    p.add_argument(
        "--import-solve", nargs="?", const="", default=None,
        dest="import_solve", metavar="PATH",
        help="Skip OCR / pose / solver.  Load solved camera state from PATH "
             "(default: output/solved_cameras.json) and open the interactive "
             "viewer.  Use locally after copying the JSON from RunPod.",
    )
    p.add_argument(
        "--use-saved-qwen", action="store_true", dest="use_saved_qwen",
        help="Load Qwen/CLIP anchor results from the cache file "
             "(output/anchor_cache.json) instead of re-running Qwen.  "
             "If the cache does not exist, Qwen runs and saves it as usual.",
    )
    p.add_argument(
        "--save-mast3r-images", nargs="?", const="", default=None,
        dest="save_mast3r_images", metavar="DIR",
        help="Save the HUD-masked undistorted images that would be fed to MASt3R "
             "to DIR (default: output/debug/mast3r_inputs/), then exit without "
             "running MASt3R or Ceres.",
    )
    p.add_argument(
        "--skip-ceres", action="store_true", dest="skip_ceres",
        help="Run steps 1–6 through MASt3R then exit: saves auto_matches.json and "
             "mast3r_raw_scene.glb (to output/debug/ or the dir given by --export-mesh). "
             "Skips the Ceres solve and does not open the GUI.",
    )
    p.add_argument(
        "--export-mesh", nargs="?", const="", default=None,
        dest="export_mesh", metavar="DIR",
        help="After the solve, write two debug .glb scenes to DIR (default: "
             "output/debug/): mast3r_raw_scene.glb (MASt3R's raw reconstruction, "
             "its own native coordinate frame) and mast3r_aligned_scene.glb "
             "(the same mesh + Ceres-refined point cloud, moved into ENU via "
             "the solved Sim(3), with reference cards at the final solved "
             "camera poses). Open either in Blender via File > Import > glTF. "
             "Exits without opening the GUI. With --run-matcher-only, only the "
             "raw scene is produced (Ceres never runs).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    frames_dir: str,
    no_refine: bool = False,
    height_mode: str = "tor",
    cameras_init_from_config: bool = False,
    use_manual_features: bool = False,
    run_matcher_only: bool = False,
    use_saved_qwen: bool = False,
    save_mast3r_images: str | None = None,
    keep_dense: bool = False,
    export_raw_glb_dir: str | None = None,
) -> tuple[list, "RefineResult | None"]:
    """
    Full pipeline: load → OCR → undistort → pose → detect_anchor → MASt3R+Ceres.
    Returns (frames, refine_result) — check f.ready on the frames for which
    have solved poses. refine_result is None if refinement was skipped or
    aborted early (e.g. --no-refine, --run-matcher-only, MASt3R failure); see
    pipeline.feature_matcher.RefineResult.
    """
    from pipeline.frame     import load_frames
    from pipeline.ocr       import extract_telemetry_all
    from pipeline.undistort import undistort_all
    from pipeline.pose      import estimate_poses

    logger.info("═" * 60)
    logger.info("Step 1/6 – Loading frames from %s", frames_dir)
    frames = load_frames(frames_dir)

    logger.info("═" * 60)
    logger.info("Step 2/6 – Extracting telemetry via OCR")
    extract_telemetry_all(frames)

    logger.info("═" * 60)
    logger.info("Step 3/6 – Undistorting frames + applying HUD mask")
    undistort_all(frames)

    logger.info("═" * 60)
    logger.info("Step 4/6 – Estimating camera poses from telemetry")
    estimate_poses(frames, skip_bracket_roll=not config.HORIZON_INDICATOR_READING)

    # Snapshot OCR headings and GPS positions before the solver mutates them
    for f in frames:
        f._ocr_heading_deg  = f.heading_deg
        f._gps_position_enu = (
            f.position_enu.copy() if f.position_enu is not None else None
        )

    # Apply --height override to camera Z
    for f in frames:
        if f.position_enu is None:
            continue
        alt_ref = f.alt_takeoff_ref_m or 0.0
        alt_agl = f.alt_agl_m         or 0.0
        if height_mode == "avg":
            f.position_enu[2] = (alt_ref + alt_agl) / 2.0
        elif height_mode == "tor":
            f.position_enu[2] = alt_ref
        # 'agl' already set by estimate_poses

    logger.info("═" * 60)
    logger.info("Step 5/6 – Anchor detection (Qwen VL + CLIP)")
    from pipeline.detect_anchor import detect_anchor, load_anchor_result
    import os as _os
    if use_saved_qwen and _os.path.isfile(config.ANCHOR_CACHE_FILE):
        logger.info("--use-saved-qwen: loading anchor cache from %s", config.ANCHOR_CACHE_FILE)
        anchor_result = load_anchor_result(config.ANCHOR_CACHE_FILE)
        # Restore crosshair/banner bboxes that were detected during the original Qwen run
        frame_map = {f.stem: f for f in frames}
        for stem, bbox in anchor_result.crosshair_bboxes.items():
            if stem in frame_map:
                frame_map[stem].crosshair_bbox_px = bbox
        for stem, bbox in anchor_result.banner_bboxes.items():
            if stem in frame_map:
                frame_map[stem].banner_bbox_px = bbox
    else:
        if use_saved_qwen:
            logger.warning(
                "--use-saved-qwen: no cache at %s — running Qwen", config.ANCHOR_CACHE_FILE
            )
        anchor_result = detect_anchor(frames)

    logger.info("Step 5b – Suppressing bracket + crosshair + banner overlays in undistorted frames")
    from pipeline.undistort import suppress_center_overlays
    suppress_center_overlays(frames)

    refine_result = None
    if no_refine:
        logger.info("Step 6/6 – Refinement SKIPPED (--no-refine)")
    else:
        logger.info("═" * 60)
        logger.info("Step 6/6 – MASt3R-SfM + Ceres full-BA")
        from pipeline.feature_matcher import refine_pitches
        refine_result = refine_pitches(
            frames,
            anchor_result=anchor_result,
            cameras_init_from_config=cameras_init_from_config,
            use_manual_features=use_manual_features,
            matcher_only=run_matcher_only,
            save_mast3r_images=save_mast3r_images,
            keep_dense=keep_dense,
            export_raw_glb_dir=export_raw_glb_dir,
        )

    ready = [f for f in frames if f.ready]
    logger.info("═" * 60)
    logger.info(
        "Pipeline complete.  %d/%d frame(s) ready for ray-casting.",
        len(ready), len(frames),
    )
    for f in (f for f in frames if not f.ready):
        logger.warning("  NOT READY: %s  (check OCR output above)", f.stem)

    return frames, refine_result


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def preview_undistort(frames: list) -> None:
    """Save undistorted frames to output/debug/undistort/ (headless-safe)."""
    out_dir = Path(config.OUTPUT_DIR) / "debug" / "undistort"
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        if frame.undistorted is None:
            continue
        out_path = out_dir / f"{frame.stem}_undistorted.png"
        cv2.imwrite(str(out_path), frame.undistorted)
        logger.info("  Saved: %s", out_path)
    print(f"\nUndistorted frames saved to: {out_dir}\n")


def preview_hud_masks_cmd(frames_dir: str) -> None:
    """
    Load frames and save side-by-side debug images showing before/after HUD masking.
    Left = undistorted raw (no mask).  Right = mask-then-undistort (correct order).
    """
    import numpy as np
    from pipeline.frame     import load_frames
    from pipeline.undistort import undistort_image, undistort_frame, _apply_hud_mask_raw, build_K, build_D, build_K_new
    import copy

    logger.info("Loading frames from %s", frames_dir)
    frames = load_frames(frames_dir)

    out_dir = Path(config.OUTPUT_DIR) / "debug" / "hud_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    K, D, K_new = build_K(), build_D(), build_K_new(build_K())

    for f in frames:
        if f.raw is None:
            continue
        # Left: undistorted with no masking (for comparison)
        raw_undist = undistort_image(f.raw, K, D, K_new)

        # Right: mask raw first, then undistort (correct pipeline order)
        f_copy = copy.copy(f)
        f_copy.raw = f.raw.copy()
        _apply_hud_mask_raw(f_copy)
        undistort_frame(f_copy, K, D, K_new)
        masked = f_copy.undistorted

        canvas = np.hstack([raw_undist, masked])
        cv2.putText(canvas, "Before HUD mask (raw undistorted)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(canvas, "After: mask-raw then undistort",  (raw_undist.shape[1] + 10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out_path = out_dir / f"{f.stem}_hud.png"
        cv2.imwrite(str(out_path), canvas)
        logger.info("  Saved: %s", out_path)

    print(f"\nHUD mask previews saved to: {out_dir}\n")
    print("Left = undistorted frame (no mask).  Right = mask applied to raw before undistort.")
    print("Adjust HUD_REGIONS in config.py and re-run.\n")


def preview_anchor_cmd(frames_dir: str) -> None:
    """
    Run Qwen VL + CLIP anchor detection on undistorted frames and save annotated
    debug images showing bboxes and CLIP weights to {OUTPUT_DIR}/debug/anchor/.
    """
    import numpy as np
    from pipeline.frame     import load_frames
    from pipeline.undistort import undistort_all
    from pipeline.detect_anchor import detect_anchor

    logger.info("Loading frames from %s", frames_dir)
    frames = load_frames(frames_dir)

    logger.info("Undistorting %d frame(s)…", len(frames))
    undistort_all(frames)

    logger.info("Running anchor detection (Qwen VL + CLIP) …")
    anchor = detect_anchor(frames)

    out_dir = Path(config.OUTPUT_DIR) / "debug" / "anchor"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in frames:
        if f.undistorted is None:
            continue
        vis = f.undistorted.copy()
        stem = f.stem

        bbox = anchor.bboxes.get(stem)
        if bbox is not None:
            y1, x1, y2, x2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = anchor.label
            cv2.putText(vis, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        w = anchor.weights.get(stem, 0.0)
        threshold_color = (0, 220, 0) if w >= config.CLIP_ANCHOR_THRESHOLD else (0, 80, 220)
        cv2.putText(vis, f"CLIP w={w:.3f}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, threshold_color, 2)

        centroid = anchor.centroids.get(stem)
        if centroid is not None:
            u, v = int(centroid[0]), int(centroid[1])
            cv2.drawMarker(vis, (u, v), (0, 200, 255),
                           cv2.MARKER_CROSS, 20, 2)

        out_path = out_dir / f"{f.stem}_anchor.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.info("  [%s] CLIP=%.3f  saved %s", stem[-12:], w, out_path.name)

    print(f"\nAnchor preview images saved to: {out_dir}")
    print(f"Anchor label: '{anchor.label}'")
    print(f"Frames with bbox: {len(anchor.bboxes)}")
    print(f"Frames above threshold ({config.CLIP_ANCHOR_THRESHOLD}): "
          f"{sum(1 for w in anchor.weights.values() if w >= config.CLIP_ANCHOR_THRESHOLD)}\n")


def run_interactive(frames: list) -> None:
    """Launch the click-to-reproject GUI."""
    from pipeline.ui import ReprojectionViewer
    viewer = ReprojectionViewer(frames)
    viewer.run()


def run_batch(
    frames: list,
    source_stem: str,
    pick: tuple[float, float],
) -> None:
    """Headless: reproject *pick* from *source_stem* and save proof sheet."""
    from pipeline.geometry import reproject_pick
    from pipeline.ui       import save_proof_sheet

    source = next((f for f in frames if f.stem == source_stem), None)
    if source is None:
        logger.error(
            "Source frame '%s' not found.  Available stems:\n  %s",
            source_stem, "\n  ".join(f.stem for f in frames),
        )
        sys.exit(1)

    if not source.ready:
        logger.error("Source frame '%s' is not ready (OCR/pose failed).", source_stem)
        sys.exit(1)

    px, py = pick
    logger.info("Batch reproject: source=%s  pick=(%.0f, %.0f)", source_stem, px, py)
    results = reproject_pick(px, py, source, frames)
    if not results:
        logger.error("No successful reprojections.  Check telemetry and config.py.")
        sys.exit(1)

    out_path = save_proof_sheet(frames, source, (px, py), results)
    print(f"\nProof sheet written: {out_path}\n")


def manual_correspondences_cmd(frames_dir: str) -> None:
    """Open the manual correspondence picker (load + undistort only)."""
    from pipeline.frame                    import load_frames
    from pipeline.undistort                import undistort_all
    from pipeline.manual_correspondence_ui import ManualCorrespondenceViewer

    logger.info("Loading frames from %s", frames_dir)
    frames = load_frames(frames_dir)
    logger.info("Undistorting %d frame(s)...", len(frames))
    undistort_all(frames)

    json_path = Path(config.MANUAL_CORRESPONDENCES_FILE)
    logger.info("Opening correspondence picker.  JSON: %s", json_path)
    viewer = ManualCorrespondenceViewer(frames, json_path)
    viewer.run()

    if json_path.exists():
        print(f"\nCorrespondences saved to: {json_path}")
        print("Run the normal pipeline (without --manual-correspondences) to use them.\n")


def show_scores_cmd(frames_dir: str, json_path: str,
                    edit_mode: bool = False) -> None:
    """
    Score correspondences with CLIP cosine similarity, then open the viewer
    coloured red→blue by appearance quality.
    edit_mode=True: combined --manual-correspondences + --show-scores.
    """
    from pipeline.frame                    import load_frames
    from pipeline.undistort                import undistort_all
    from pipeline.correspondence_scorer    import score_correspondences
    from pipeline.manual_correspondence_ui import ManualCorrespondenceViewer

    logger.info("Loading frames from %s", frames_dir)
    frames = load_frames(frames_dir)
    logger.info("Undistorting %d frame(s)...", len(frames))
    undistort_all(frames)

    path = Path(json_path)
    if not path.exists():
        print(f"Score file not found: {path}")
        return

    logger.info("Scoring correspondences in %s...", path)
    scores = score_correspondences(frames, path)
    if not scores:
        print(f"No correspondences found in {path}.")
        return

    viewer = ManualCorrespondenceViewer(
        frames, path,
        scores=scores,
        score_mode=True,
    )
    viewer.run()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir

    # HUD mask preview — no pipeline needed beyond undistort
    if args.preview_hud_masks:
        preview_hud_masks_cmd(args.frames_dir)
        return

    # Anchor detection preview
    if args.preview_anchor:
        preview_anchor_cmd(args.frames_dir)
        return

    # Resolve --show-scores json path
    if args.show_scores is not None:
        if args.show_scores:
            _score_path = args.show_scores
        elif args.manual_correspondences:
            _score_path = config.MANUAL_CORRESPONDENCES_FILE
        else:
            _score_path = config.AUTO_MATCHES_FILE
    else:
        _score_path = None

    # Manual correspondence picker
    if args.manual_correspondences:
        if _score_path is not None:
            show_scores_cmd(args.frames_dir, _score_path, edit_mode=True)
        else:
            manual_correspondences_cmd(args.frames_dir)
        return

    # Score-only mode
    if _score_path is not None:
        show_scores_cmd(args.frames_dir, _score_path, edit_mode=False)
        return

    # Import-solve: skip pipeline, load JSON, open GUI
    if args.import_solve is not None:
        from pipeline.frame     import load_frames
        from pipeline.undistort import undistort_all
        from pipeline.solve_io  import import_solve
        _frames = load_frames(args.frames_dir)
        undistort_all(_frames)
        _path = args.import_solve or os.path.join(config.OUTPUT_DIR, "solved_cameras.json")
        n = import_solve(_frames, _path)
        logger.info("Imported solve for %d frame(s) from %s", n, _path)
        run_interactive(_frames)
        return

    # Preview undistort only
    if args.preview_undistort:
        from pipeline.frame     import load_frames
        from pipeline.undistort import undistort_all
        _frames = load_frames(args.frames_dir)
        undistort_all(_frames)
        preview_undistort(_frames)
        return

    # Run the shared pipeline
    _use_manual = args.manual_fm_json is not None
    _want_mesh = args.export_mesh is not None or args.skip_ceres
    _export_mesh_dir = (
        args.export_mesh or os.path.join(config.OUTPUT_DIR, "debug")
        if _want_mesh else None
    )
    frames, refine_result = run_pipeline(
        args.frames_dir,
        no_refine=args.no_refine,
        height_mode=args.height,
        cameras_init_from_config=args.cameras_init_from_config,
        use_manual_features=_use_manual,
        run_matcher_only=args.run_matcher_only or args.skip_ceres,
        use_saved_qwen=args.use_saved_qwen,
        save_mast3r_images=args.save_mast3r_images,
        keep_dense=_want_mesh,
        export_raw_glb_dir=_export_mesh_dir,
    )

    # Camera deltas analysis
    if args.camera_deltas:
        camera_deltas_cmd(frames)
        return

    # Matcher-only / skip-ceres: auto_matches.json (and the raw .glb) already
    # written inside run_pipeline / run_complete_graph before Ceres would run.
    if args.run_matcher_only or args.skip_ceres:
        print(f"\nDone.  Matches saved to: {config.AUTO_MATCHES_FILE}")
        print("Run --show-scores to inspect them.\n")
        if _export_mesh_dir is not None:
            print(f"Raw MASt3R scene written to: "
                  f"{os.path.join(_export_mesh_dir, 'mast3r_raw_scene.glb')}\n")
        return

    # Export-solve / export-mesh: serialize debug artifacts and exit (no GUI).
    # Both can be combined in one invocation for a single RunPod round-trip.
    did_export = False

    if args.export_solve is not None:
        from pipeline.solve_io import export_solve, export_mast3r_cameras
        _path = args.export_solve or os.path.join(config.OUTPUT_DIR, "solved_cameras.json")
        export_solve(frames, _path)
        print(f"\nSolved cameras written to: {_path}")

        if refine_result is not None and refine_result.mast3r_cameras_enu:
            _mast3r_path = os.path.join(config.OUTPUT_DIR, "solved_camera_mast3r.json")
            export_mast3r_cameras(refine_result.mast3r_cameras_enu, _mast3r_path)
            print(f"MASt3R seed cameras written to: {_mast3r_path}")

        print("Copy these files locally and run:")
        print(f"  python raycast.py --frames_dir ./frames --import-solve {_path}\n")
        did_export = True

    if args.export_mesh is not None:
        # The raw .glb was already written inside run_complete_graph (before
        # Ceres ran). The aligned .glb needs the post-Ceres/Sim3 result, so
        # it's built here.
        _raw_path = os.path.join(_export_mesh_dir, "mast3r_raw_scene.glb")
        print(f"\nRaw MASt3R scene written to: {_raw_path}")

        if (refine_result is not None
                and refine_result.sim3 is not None
                and refine_result.mast3r_result.dense):
            from pipeline.mast3r_matcher import export_aligned_glb
            _aligned_path = os.path.join(_export_mesh_dir, "mast3r_aligned_scene.glb")
            export_aligned_glb(
                refine_result.mast3r_result,
                refine_result.points_3d_solved,
                refine_result.sim3,
                frames,
                _aligned_path,
            )
            print(f"Aligned (ENU) scene written to: {_aligned_path}")
        else:
            print("Aligned scene NOT written — no Ceres/Sim(3) result available "
                  "(check the log above for an aborted solve).")
        print("Open either in Blender via File > Import > glTF 2.0.\n")
        did_export = True

    if did_export:
        return

    # Mode dispatch
    if args.batch:
        if not args.source_frame or not args.pick:
            print("Batch mode requires --source-frame and --pick.  See --help.")
            sys.exit(1)
        run_batch(frames, args.source_frame, tuple(args.pick))
    else:
        run_interactive(frames)


# ─────────────────────────────────────────────────────────────────────────────
# Camera deltas analysis
# ─────────────────────────────────────────────────────────────────────────────

def camera_deltas_cmd(frames: list) -> None:
    """
    Print two tables comparing solved camera state to raw telemetry.

    POSITION TABLE (per frame)
      ΔEast, ΔNorth – solved XY minus GPS-derived ENU XY (metres).
      Z_solved      – camera Z used by the solver (controlled by --height flag).
      Z_tor         – alt_takeoff_ref_m: barometric height above launch point.
      Z_agl         – alt_agl_m: radar/lidar height above terrain below.

    ORIENTATION TABLE (per frame)
      The forward vector fwd = R_solved.T[:, 2] is roll-invariant, so we read
      yaw and pitch directly from it:
          pitch_implied = asin(fwd.Z)
          yaw_implied   = atan2(fwd.E, fwd.N)
      Δyaw compares yaw_implied to the OCR compass heading snapshotted before
      the solve.
    """
    import math
    import numpy as np
    from pipeline.pose import detect_camera_roll

    def _decompose(R_solved):
        fwd_world     = R_solved.T[:, 2]
        pitch_implied = math.degrees(
            math.asin(float(np.clip(fwd_world[2], -1.0, 1.0)))
        )
        yaw_implied   = math.degrees(
            math.atan2(fwd_world[0], fwd_world[1])
        ) % 360.0
        return yaw_implied, pitch_implied

    def _angle_diff(a, b):
        d = (a - b) % 360.0
        return d - 360.0 if d > 180.0 else d

    ready = [
        f for f in frames
        if f.R is not None and hasattr(f, "_ocr_heading_deg")
    ]

    # ── POSITION TABLE ────────────────────────────────────────────────────────
    print()
    print("Position delta — solved XY vs GPS telemetry  |  height cross-check")
    pos_hdr = (
        f"{'Frame':<22}  {'ΔEast(m)':>9}  {'ΔNorth(m)':>10}  "
        f"{'Z_solved(m)':>11}  {'Z_tor(m)':>9}  {'Z_agl(m)':>9}"
    )
    pos_sep = "─" * len(pos_hdr)
    print(pos_sep)
    print(pos_hdr)
    print(pos_sep)

    for f in ready:
        if f.position_enu is None:
            continue
        gps_pos = getattr(f, "_gps_position_enu", None)
        if gps_pos is not None:
            de_str = f"{f.position_enu[0] - gps_pos[0]:+.2f}"
            dn_str = f"{f.position_enu[1] - gps_pos[1]:+.2f}"
        else:
            de_str = dn_str = "   N/A"

        z_solved = f.position_enu[2]
        z_tor    = f.alt_takeoff_ref_m if f.alt_takeoff_ref_m is not None else float("nan")
        z_agl    = f.alt_agl_m         if f.alt_agl_m is not None else float("nan")
        tor_str  = f"{z_tor:8.1f}m" if not math.isnan(z_tor) else "     N/A"
        agl_str  = f"{z_agl:8.1f}m" if not math.isnan(z_agl) else "     N/A"

        print(
            f"{f.stem[-22:]:<22}  {de_str:>9}  {dn_str:>10}  "
            f"{z_solved:>10.1f}m  {tor_str}  {agl_str}"
        )

    print(pos_sep)
    print()
    print("  ΔEast / ΔNorth : solved position minus GPS-telemetry ENU")
    print("  Z_tor          : alt_takeoff_ref_m — barometric, shared datum, consistent")
    print("  Z_agl          : alt_agl_m         — radar/lidar, varies with terrain")
    print()

    # ── ORIENTATION TABLE ─────────────────────────────────────────────────────
    print("Orientation delta — solved yaw vs OCR compass heading")
    ori_hdr = (
        f"{'Frame':<22}  {'OCR hdg':>8}  {'yaw_impl':>9}  {'Δyaw':>7}  "
        f"{'pitch_impl':>11}  {'pitch_solv':>11}  {'roll_brkt':>10}"
    )
    ori_sep = "─" * len(ori_hdr)
    print(ori_sep)
    print(ori_hdr)
    print(ori_sep)

    for f in ready:
        roll_raw = detect_camera_roll(f.raw)
        if roll_raw is not None:
            roll_bracket = -roll_raw
            roll_str     = f"{roll_bracket:+.1f}°"
        else:
            roll_bracket = 0.0
            roll_str     = "N/A"

        yaw_impl, pitch_impl = _decompose(f.R)
        ocr_hdg  = f._ocr_heading_deg
        delta    = _angle_diff(yaw_impl, ocr_hdg) if ocr_hdg is not None else float("nan")
        ocr_str  = f"{ocr_hdg:7.1f}°" if ocr_hdg is not None else "    N/A"
        delta_str = f"{delta:+6.1f}°"  if not math.isnan(delta) else "    N/A"

        print(
            f"{f.stem[-22:]:<22}  {ocr_str}  {yaw_impl:8.1f}°  {delta_str}  "
            f"{pitch_impl:+10.1f}°  {f.gimbal_pitch_deg:+10.1f}°  {roll_str:>10}"
        )

    print(ori_sep)
    print()
    print("  Δyaw = yaw_implied − OCR_heading")
    print("    positive → solved camera points further CW than compass reported")
    print("    consistent Δyaw → fixed compass bias")
    print("    varying   Δyaw → heading-dependent error (motor/EMI?)")
    print()


if __name__ == "__main__":
    main()
