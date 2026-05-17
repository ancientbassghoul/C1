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
        "--feature-matcher-debug", action="store_true", dest="feature_matcher_debug",
        help="Save annotated match images (keypoints, ground region, van bbox) "
             "to {OUTPUT_DIR}/debug/ for every matched frame pair.",
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


def run_pipeline(frames_dir: str, no_refine: bool = False, no_enhance: bool = False, height_mode: str = 'tor', feature_matcher_debug: bool = False) -> list:
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

    logger.info("═" * 60)
    logger.info("Step 4/6 – Estimating camera poses from telemetry")
    estimate_poses(frames)
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

    logger.info("═" * 60)
    logger.info("Step 5/6 – Detecting van (GroundingDINO → white-blob fallback)")
    detector   = VanDetector()
    detections = detector.detect_all(frames)
    # Convert Frame-keyed dict to stem-keyed dict for refine_pitches
    van_bboxes = {f.stem: bbox for f, bbox in detections.items()}

    if no_refine:
        logger.info("Step 6/6 – Pitch refinement SKIPPED (--no-refine)")
    else:
        logger.info("Step 6/6 – Refining gimbal pitch via ground-scatter (LightGlue)")
        from pipeline.feature_matcher import refine_pitches, set_enhance, set_debug
        set_enhance(not no_enhance)
        set_debug(feature_matcher_debug)
        refine_pitches(frames, van_bboxes=van_bboxes)

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

def main() -> None:
    args = parse_args()

    # Optional config overrides
    if args.output_dir:
        import config
        config.OUTPUT_DIR = args.output_dir

    # Run the shared pipeline
    frames = run_pipeline(args.frames_dir, no_refine=args.no_refine, no_enhance=not args.enhance, height_mode=args.height, feature_matcher_debug=args.feature_matcher_debug)

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
