"""
raycast.py – Raycast Challenge entry point.

Usage
─────
Interactive viewer (click any frame to reproject):
    pipenv run python raycast.py --frames_dir ./frames

Preview undistortion only (for tuning config.py intrinsics):
    pipenv run python raycast.py --frames_dir ./frames --preview-undistort

Headless batch run (no GUI, outputs proof sheet for a fixed pixel):
    pipenv run python raycast.py --frames_dir ./frames \\
        --batch --source-frame "2026-02-15_16-28-05_04752" \\
        --pick 640 400

Quick-start workflow
─────────────────────
1.  Put all 13 frames in a folder, e.g. ./frames/
2.  pipenv install            (first time only)
3.  pipenv run python raycast.py --frames_dir ./frames --preview-undistort
    → Inspect the undistorted frames.  If lines are still curved, edit
      config.py (FOCAL_LENGTH, FISHEYE_K1) and repeat.
4.  pipenv run python raycast.py --frames_dir ./frames
    → Click any frame.  Green dot = pick, blue dots = reprojections.
    → Press 's' to save proof_sheet.png to ./output/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `pipeline` is always importable,
# regardless of how or from where the script is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import config

# ── Logging ───────────────────────────────────────────────────────────────────
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
        help="Directory containing the 13 drone frames (PNG/JPG).",
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
        help="(Batch mode) Filename stem of the source frame, e.g. "
             "'2026-02-15_16-28-05_04752'.",
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
        "--enhance", action="store_true",
        help="Enable CLAHE+unsharp preprocessing before LightGlue (off by default).",
    )
    p.add_argument(
        "--preview-enhanced", action="store_true",
        help="Show edge-enhanced frames used for SuperPoint, then exit.",
    )
    p.add_argument(
        "--no-refine", action="store_true",
        help="Skip automatic pitch refinement (use config overrides only).",
    )
    p.add_argument(
        "--cameras-init-from-config", action="store_true",
        dest="cameras_init_from_config",
        help="Seed camera poses from CAMERA_POSE_OVERRIDES in config.py before running "
             "the orientation solver. Bypasses GPS + GeoCalib for overridden frames.",
    )
    p.add_argument(
        "--feature-matcher-debug", action="store_true", dest="feature_matcher_debug",
        help="Save annotated match images (keypoints, ground region, van bbox) "
             "to {OUTPUT_DIR}/debug/ for every matched frame pair.",
    )
    p.add_argument(
        "--preview-ground-masks", action="store_true", dest="preview_ground_masks",
        help="Compute GroundedSAM ground masks for every frame, save three-panel "
             "debug images (positive / exclusion / final) to {OUTPUT_DIR}/debug/masks/, "
             "then exit.  Use to tune GROUND_INCLUDE/EXCLUDE prompts and thresholds "
             "in config.py without running the full feature-matching pipeline.",
    )
    p.add_argument(
        "--preview-hud-masks", action="store_true", dest="preview_hud_masks",
        help="Detect HUD overlays on raw frames using GroundedSAM, save two-panel "
             "debug images to {OUTPUT_DIR}/debug/hud_masks/, then exit.  "
             "Only loads raw frames — no OCR, undistort, or matching needed.  "
             "Use to tune HUD_REGIONS coordinates in config.py.",
    )
    p.add_argument(
        "--preview-hsv-masks", action="store_true", dest="preview_hsv_masks",
        help="Compute HSV-based ground masks and save 5-panel debug images "
             "to {OUTPUT_DIR}/debug/hsv_masks/.  Runs load + undistort only — "
             "no OCR, pose, or model inference.  Use to tune HSV_SKY_RANGES "
             "and HSV_VEG_RANGES in config.py.",
    )
    p.add_argument(
        "--manual-correspondences", action="store_true",
        dest="manual_correspondences",
        help="Open the manual correspondence picker.  Load and undistort frames, "
             "then show an interactive grid where you can click to mark matching "
             "ground features across frames.  Correspondences are saved to "
             "MANUAL_CORRESPONDENCES_FILE in config.py and consumed by the "
             "solver in the normal pipeline run.  No OCR, pose, or model "
             "inference needed.",
    )
    p.add_argument(
        "--show-scores", nargs="?", const="", default=None,
        dest="show_scores", metavar="PATH",
        help="Score correspondences and open the viewer coloured red→blue "
             "by appearance quality.  Optional PATH overrides the JSON "
             "file to score.  Without PATH: defaults to AUTO_MATCHES_FILE "
             "when used alone, or MANUAL_CORRESPONDENCES_FILE when combined "
             "with --manual-correspondences.",
    )
    p.add_argument(
        "--manual-fm-json", nargs="?", const="", default=None,
        dest="manual_fm_json", metavar="PATH",
        help="Replace LightGlue feature matching with manually picked "
             "correspondences.  PATH defaults to MANUAL_CORRESPONDENCES_FILE "
             "in config.py if not specified.  Runs the full pipeline "
             "(OCR, undistort, pose, van detection) then feeds manual "
             "features directly to the Ceres solver.",
    )
    p.add_argument(
        "--run-matcher-only", action="store_true", dest="run_matcher_only",
        help="Run steps 1–6 (load → OCR → undistort → pose → van → LightGlue) "
             "then exit after saving auto_matches.json.  Skips the Ceres solve. "
             "Use to generate the match file for --show-scores without "
             "waiting for the full orientation solve.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def preview_enhanced(frames: list) -> None:
    """Show the CLAHE+unsharp edge-enhanced frames used for SuperPoint matching."""
    from pipeline.feature_matcher import _enhance_np
    import cv2
    for frame in frames:
        if frame.undistorted is None:
            continue
        enhanced = _enhance_np(frame.undistorted)
        # Show side by side: original (gray) and enhanced
        orig_gray = cv2.cvtColor(frame.undistorted, cv2.COLOR_BGR2GRAY)
        side = cv2.hconcat([orig_gray, enhanced])
        cv2.imshow(f"Original  |  Enhanced  —  {frame.stem}", side)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == ord('q'):
            break


def run_pipeline(frames_dir: str, no_refine: bool = False, no_enhance: bool = False, height_mode: str = 'tor', feature_matcher_debug: bool = False, preview_ground_masks: bool = False, preview_hsv_masks: bool = False, cameras_init_from_config: bool = False, manual_fm_json: str | None = None,
                 run_matcher_only: bool = False) -> list:
    """Load, OCR, undistort, pose-estimate, detect van, and refine pitches. Return ready frames."""
    from pipeline.frame     import load_frames
    from pipeline.ocr       import extract_telemetry_all
    from pipeline.undistort import undistort_all
    from pipeline.pose      import estimate_poses
    from pipeline.detect_van import VanDetector

    logger.info("═" * 60)
    logger.info("Step 1/6 – Loading frames from %s", frames_dir)
    frames = load_frames(frames_dir)

    logger.info("═" * 60)
    logger.info("Step 2/6 – Extracting telemetry via OCR")
    extract_telemetry_all(frames)

    logger.info("═" * 60)
    logger.info("Step 3/6 – Undistorting frames (fisheye model)")
    undistort_all(frames)

    skip_heavy = feature_matcher_debug or preview_ground_masks or preview_hsv_masks
    if skip_heavy:
        logger.info("═" * 60)
        logger.info("Step 4/6 – SKIPPED (pose not needed for this mode)")
    else:
        logger.info("═" * 60)
        logger.info("Step 4/6 – Estimating camera poses from telemetry")
        estimate_poses(frames, skip_bracket_roll=not config.HORIZON_INDICATOR_READING)
        # Override camera Z according to --height
        for f in frames:
            if f.position_enu is None: continue
            alt_ref = f.alt_takeoff_ref_m or 0.0
            alt_agl = f.alt_agl_m         or 0.0
            if height_mode == 'avg':
                f.position_enu[2] = (alt_ref + alt_agl) / 2.0
            elif height_mode == 'tor':
                f.position_enu[2] = alt_ref
            # 'agl' is already set by estimate_poses — no change needed

    if skip_heavy:
        logger.info("═" * 60)
        logger.info("Steps 5-6 – SKIPPED (not needed for this mode)")
        van_bboxes = {}
    else:
        logger.info("═" * 60)
        logger.info("Step 5/6 – Detecting van (GroundingDINO → white-blob fallback)")
        detector   = VanDetector()
        detections = detector.detect_all(frames)
        # Convert Frame-keyed dict to stem-keyed dict for refine_pitches
        van_bboxes = {f.stem: bbox for f, bbox in detections.items()}

    # Load manual correspondences if requested
    _manual_pf = None
    if manual_fm_json is not None and not skip_heavy:
        from pipeline.manual_correspondence_ui import ManualCorrespondenceViewer
        import json as _json
        from pathlib import Path as _Path
        import config as _cfg
        _path = Path(manual_fm_json) if manual_fm_json else _Path(_cfg.MANUAL_CORRESPONDENCES_FILE)
        if not _path.exists():
            raise FileNotFoundError(f"Manual correspondences not found: {_path}")
        _data  = _json.loads(_path.read_text())
        _stems = {f.stem: f for f in frames}
        _manual_pf = []
        for entry in _data.get("correspondences", []):
            # Only ground-type single-point entries feed the cross-frame scatter.
            # Van feature entries (wheel_axis, roof_edge, roof) are loaded
            # separately by _load_van_features() in feature_matcher.py.
            if entry.get("type", "ground") != "ground":
                continue
            if entry.get("is_pair", False):
                continue
            pts = entry.get("points", {})
            stems = list(pts.keys())
            from itertools import combinations as _comb
            for sa, sb in _comb(stems, 2):
                fa = _stems.get(sa)
                fb = _stems.get(sb)
                if fa is None or fb is None:
                    continue
                if not (fa.ready and fb.ready):
                    continue
                xa, ya = pts[sa]
                xb, yb = pts[sb]
                _manual_pf.append((fa, (float(xa), float(ya)),
                                   fb, (float(xb), float(yb))))
        logger.info("Loaded %d manual feature pairs from %s",
                    len(_manual_pf), _path)

    if no_refine and not skip_heavy:
        logger.info("Step 6/6 – Pitch refinement SKIPPED (--no-refine)")
    elif not skip_heavy:
        logger.info("Step 6/6 – Refining gimbal pitch via ground-scatter (LightGlue)")
        from pipeline.feature_matcher import refine_pitches, set_enhance
        set_enhance(not no_enhance)
        refine_pitches(frames, van_bboxes=van_bboxes,
                       cameras_init_from_config=cameras_init_from_config,
                       manual_pairwise_features=_manual_pf,
                       matcher_only=run_matcher_only)

    ready = [f for f in frames if f.ready]
    logger.info("═" * 60)
    logger.info(
        "Pipeline complete.  %d/%d frame(s) ready for ray-casting.",
        len(ready), len(frames),
    )

    not_ready = [f for f in frames if not f.ready]
    for f in not_ready:
        logger.warning("  NOT READY: %s  (check OCR output above)", f.stem)

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def preview_undistort(frames: list) -> None:
    """Display each undistorted frame; press any key to advance, q to quit."""
    print("\nUndistortion preview.  Press any key → next frame | q → quit\n")
    for frame in frames:
        if frame.undistorted is None:
            continue
        label = f"[undistorted] {frame.stem}"
        cv2.namedWindow(label, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(label, 1280, 720)
        cv2.imshow(label, frame.undistorted)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            break


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
        stems = [f.stem for f in frames]
        logger.error(
            "Source frame '%s' not found.  Available stems:\n  %s",
            source_stem, "\n  ".join(stems),
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


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def preview_ground_masks(frames: list) -> None:
    """
    Compute GroundedSAM ground masks and save three-panel debug images.
    Runs steps 1-3 only (load → OCR → undistort) then exits.
    Outputs go to {OUTPUT_DIR}/debug/masks/<stem>_masks.png.
    """
    from pipeline.ground_mask import get_ground_masks
    import config as cfg
    from pathlib import Path

    ready = [f for f in frames if f.undistorted is not None]
    logger.info("Computing ground masks for %d frame(s)…", len(ready))
    get_ground_masks(ready, save_debug=True)

    out = Path(cfg.OUTPUT_DIR) / "debug" / "masks"
    print(f"\nGround mask debug images saved to: {out}\n")
    print("Each image shows three panels:")
    print("  BLUE   = positive mask  (what GroundedSAM thinks IS ground)")
    print("  RED    = exclusion mask (what GroundedSAM thinks is NOT ground)")
    print("  GREEN  = final mask     (positive AND NOT exclusion AND NOT black border)")
    print("\nTune GROUND_INCLUDE_PROMPT / GROUND_EXCLUDE_PROMPT and thresholds in config.py,")
    print("then re-run --preview-ground-masks to iterate without running feature matching.\n")


def preview_hud_masks_cmd(frames_dir: str) -> None:
    """
    Load + undistort frames, apply geometric HUD mask, save debug PNGs, exit.
    No model inference — just geometry from config.HUD_REGIONS.
    Fast enough to iterate in seconds when tuning region coordinates.
    """
    from pipeline.frame    import load_frames
    from pipeline.undistort import undistort_all
    from pipeline.ground_mask import save_hud_mask_preview
    import config as cfg
    from pathlib import Path

    logger.info("Loading frames from %s", frames_dir)
    frames = load_frames(frames_dir)

    logger.info("Undistorting %d frame(s)…", len(frames))
    undistort_all(frames)

    logger.info("Building geometric HUD masks and saving previews…")
    save_hud_mask_preview(frames)

    out = Path(cfg.OUTPUT_DIR) / "debug" / "hud_masks"
    print(f"\nHUD mask preview images saved to: {out}\n")
    print("Each image shows two panels:")
    print("  LEFT   = undistorted frame")
    print("  RIGHT  = undistorted frame with HUD mask overlay (magenta)")
    print("\nAdjust HUD_REGIONS coordinates in config.py and re-run — no model")
    print("inference needed, so iteration is near-instant.\n")


def preview_hsv_masks_cmd(frames_dir: str) -> None:
    """
    Load + undistort frames, compute HSV exclusion masks, save debug PNGs, exit.
    No model inference — just HSV thresholding + geometric HUD mask.
    Iterates in a few seconds.
    """
    from pipeline.frame          import load_frames
    from pipeline.undistort      import undistort_all
    from pipeline.hsv_ground_mask import get_hsv_ground_masks
    from pathlib import Path
    import config as cfg

    logger.info("Loading frames from %s", frames_dir)
    frames = load_frames(frames_dir)

    logger.info("Undistorting %d frame(s)…", len(frames))
    undistort_all(frames)

    ready = [f for f in frames if f.undistorted is not None]
    logger.info("Computing HSV ground masks for %d frame(s)…", len(ready))
    # Van detection skipped during preview — add bbox exclusion manually via
    # VanDetector if needed, or test without van first.
    get_hsv_ground_masks(ready, van_detections=None, save_debug=True)

    out = Path(cfg.OUTPUT_DIR) / "debug" / "hsv_masks"
    print(f"\nHSV ground mask debug images saved to: {out}\n")
    print("Each image shows five panels:")
    print("  RED    = sky mask")
    print("  GREEN  = vegetation mask")
    print("  YELLOW = van + HUD mask")
    print("  ORANGE = combined exclusion")
    print("  TEAL   = final ground mask")
    print("\nTune HSV_SKY_RANGES, HSV_VEG_RANGES, HSV_MORPH_KERNEL in config.py")
    print("and re-run -- no model loading needed, iterates in seconds.\n")


def manual_correspondences_cmd(frames_dir: str) -> None:
    """
    Open the manual correspondence picker.
    Needs only load + undistort — no OCR, pose, or model inference.
    """
    from pipeline.frame                   import load_frames
    from pipeline.undistort               import undistort_all
    from pipeline.manual_correspondence_ui import ManualCorrespondenceViewer
    import config as cfg
    from pathlib import Path

    logger.info("Loading frames from %s", frames_dir)
    frames = load_frames(frames_dir)

    logger.info("Undistorting %d frame(s)...", len(frames))
    undistort_all(frames)

    json_path = Path(cfg.MANUAL_CORRESPONDENCES_FILE)
    logger.info("Opening correspondence picker.  JSON: %s", json_path)

    viewer = ManualCorrespondenceViewer(frames, json_path)
    viewer.run()

    if json_path.exists():
        print(f"\nCorrespondences saved to: {json_path}")
        print("Run the normal pipeline (without --manual-correspondences) to use them.\n")


def show_scores_cmd(frames_dir: str, json_path: str,
                    edit_mode: bool = False) -> None:
    """
    Score correspondences in json_path with SuperPoint descriptors,
    then open the viewer in score mode.
    edit_mode=True: fully editable (--manual-correspondences + --show-scores).
    edit_mode=False: score display only.
    """
    from pipeline.frame                    import load_frames
    from pipeline.undistort                import undistort_all
    from pipeline.correspondence_scorer    import score_correspondences
    from pipeline.manual_correspondence_ui import ManualCorrespondenceViewer
    from pathlib import Path

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

    mode_str = "score+edit" if edit_mode else "score"
    logger.info("Opening %s viewer...", mode_str)
    viewer = ManualCorrespondenceViewer(
        frames, path,
        scores=scores,
        score_mode=True,
    )
    viewer.run()


def main() -> None:
    args = parse_args()

    # Optional config overrides
    if args.output_dir:
        import config
        config.OUTPUT_DIR = args.output_dir

    # HUD mask preview — needs only raw frames, no pipeline at all
    if args.preview_hud_masks:
        preview_hud_masks_cmd(args.frames_dir)
        return

    # HSV ground mask preview — load + undistort only, no model inference
    if args.preview_hsv_masks:
        preview_hsv_masks_cmd(args.frames_dir)
        return

    # Resolve --show-scores json path
    import config as _cfg_main
    if args.show_scores is not None:
        if args.show_scores:                          # explicit path supplied
            _score_path = args.show_scores
        elif args.manual_correspondences:             # --manual-correspondences default
            _score_path = _cfg_main.MANUAL_CORRESPONDENCES_FILE
        else:                                         # --show-scores alone default
            _score_path = _cfg_main.AUTO_MATCHES_FILE
    else:
        _score_path = None

    # Manual correspondence picker
    if args.manual_correspondences:
        if _score_path is not None:
            # Score first, then open picker in score+edit mode
            show_scores_cmd(args.frames_dir, _score_path, edit_mode=True)
        else:
            manual_correspondences_cmd(args.frames_dir)
        return

    # Score-only mode
    if _score_path is not None:
        show_scores_cmd(args.frames_dir, _score_path, edit_mode=False)
        return

    # Run the shared pipeline
    frames = run_pipeline(args.frames_dir, no_refine=args.no_refine, no_enhance=not args.enhance, height_mode=args.height, feature_matcher_debug=args.feature_matcher_debug, preview_ground_masks=args.preview_ground_masks, preview_hsv_masks=args.preview_hsv_masks, cameras_init_from_config=args.cameras_init_from_config, manual_fm_json=args.manual_fm_json,
                        run_matcher_only=args.run_matcher_only)

    # Matcher-only mode exits here — auto_matches.json already written inside run_pipeline
    if args.run_matcher_only:
        import config as _cfg_mo
        print(f"\nDone.  Matches saved to: {_cfg_mo.AUTO_MATCHES_FILE}")
        print("Run --show-scores to inspect them.\n")
        return

    # Feature-matcher debug UI (replaces the old per-pair PNG file dumps)
    if args.feature_matcher_debug:
        from pipeline.feature_matcher_debug_ui import FeatureMatcherDebugViewer
        viewer = FeatureMatcherDebugViewer(frames, enhance=args.enhance)
        viewer.run()
        return

    # Ground mask preview — saves debug PNGs and exits
    if args.preview_ground_masks:
        preview_ground_masks(frames)
        return

    # Mode dispatch
    if args.preview_enhanced:
        preview_enhanced(frames)
        return

    if args.preview_undistort:
        preview_undistort(frames)
    elif args.batch:
        if not args.source_frame or not args.pick:
            print("Batch mode requires --source-frame and --pick.  See --help.")
            sys.exit(1)
        run_batch(frames, args.source_frame, tuple(args.pick))

    else:
        run_interactive(frames)


if __name__ == "__main__":
    main()
