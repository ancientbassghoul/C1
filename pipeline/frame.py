"""
pipeline/frame.py – Frame dataclass and bulk loader.

A Frame bundles the raw image, its undistorted counterpart, and all
telemetry extracted from the HUD overlay.  The pose (R, t) is added later
by pipeline/pose.py once all frames are loaded and a common ENU origin is
established.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


@dataclass(eq=False)
class Frame:
    """All data associated with a single drone image."""

    # ── Identity ──────────────────────────────────────────────────────────────
    path: Path
    stem: str                          # filename without extension

    # ── Images ───────────────────────────────────────────────────────────────
    raw: np.ndarray                    # BGR, original pixels
    undistorted: Optional[np.ndarray] = None  # BGR, after lens correction

    # ── Telemetry (from OCR) ──────────────────────────────────────────────────
    lat: Optional[float] = None          # WGS-84 latitude  (degrees)
    lon: Optional[float] = None          # WGS-84 longitude (degrees)
    heading_deg: Optional[float] = None  # Drone compass heading (0-360°, CW from N)

    # The HUD bottom-right shows TWO altitude readings:
    #   alt_takeoff_ref_m  – height above the takeoff/launch point (barometric).
    #                        Consistent Z reference across ALL frames since the
    #                        launch point never moves.  Use this for the camera's
    #                        Z position in the shared world frame.
    #   alt_agl_m          – height above the ground directly below (radar/lidar).
    #                        Varies with terrain; use this for ground-plane Z offset.
    alt_takeoff_ref_m: Optional[float] = None
    alt_agl_m:         Optional[float] = None

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def terrain_offset_m(self) -> Optional[float]:
        """
        Elevation of the ground directly below this frame relative to the
        shared takeoff-point datum.

          terrain_offset = alt_takeoff_ref – alt_agl

        Positive → terrain here is HIGHER than the launch point.
        Used to place the ground plane at the correct Z in world space.
        """
        if self.alt_takeoff_ref_m is not None and self.alt_agl_m is not None:
            return self.alt_takeoff_ref_m - self.alt_agl_m
        return None

    # ── Pose (set by pose.py) ─────────────────────────────────────────────────
    # Position of the camera in the local ENU world frame (metres).
    position_enu: Optional[np.ndarray] = None   # shape (3,)  [East, North, Up]

    # Rotation matrix from world (ENU) to camera (OpenCV convention).
    # p_cam = R @ (p_world - position_enu)
    R: Optional[np.ndarray] = None              # shape (3, 3)

    # Estimated gimbal pitch used for this frame (degrees, negative = down).
    gimbal_pitch_deg: Optional[float] = None

    # Camera intrinsic matrix used after undistortion (set by undistort.py).
    K_undist: Optional[np.ndarray] = None       # shape (3, 3)

    # Set True by van.py after pitch is refined via van geometry.
    # Prevents homography refinement from overriding a good van-based estimate.
    pitch_from_van: bool = False

    # Camera roll detected from HUD bracket indicator (degrees).
    # Stored here so refine.py can access it without re-detecting.
    camera_roll_deg: float = 0.0

    # Bounding boxes of detected bracket blobs in raw/distorted pixel coords.
    # Populated by pose.py; used by undistort.suppress_center_overlays for blurring.
    bracket_rects_raw: list = field(default_factory=list)  # list of (y1,y2,x1,x2)

    # Crosshair bbox from Qwen in [0,1000] normalized coords (ymin,xmin,ymax,xmax).
    # Populated by detect_anchor; used by suppress_center_overlays.
    crosshair_bbox_norm: Optional[tuple] = None

    # ── Convenience ───────────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        """True when the frame has been fully processed and is ready for raycasting."""
        return (
            self.undistorted is not None
            and self.R is not None
            and self.position_enu is not None
            and self.K_undist is not None
        )

    @property
    def display_name(self) -> str:
        return self.stem[-20:] if len(self.stem) > 20 else self.stem


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_frames(frames_dir: str | Path) -> list[Frame]:
    """
    Read all supported image files from *frames_dir* and return a sorted
    list of Frame objects (raw image loaded, everything else pending).
    """
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not paths:
        raise FileNotFoundError(
            f"No images found in {frames_dir}. "
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )

    frames: list[Frame] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("Could not read image: %s – skipping.", p)
            continue
        frames.append(Frame(path=p, stem=p.stem, raw=img))
        logger.info("Loaded: %s  (%dx%d)", p.name, img.shape[1], img.shape[0])

    logger.info("Loaded %d frame(s) from %s", len(frames), frames_dir)
    return frames
