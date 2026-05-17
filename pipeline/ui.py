"""
pipeline/ui.py – Interactive re-projection viewer and proof-sheet export.

Layout
──────
All undistorted frames are tiled in a grid (≤4 columns).
Click any frame to pick a target pixel.  The pipeline:
  1. Marks the picked pixel in green.
  2. Computes the ground-plane intersection.
  3. Draws the re-projected point in blue on every other frame.

Key bindings
────────────
  click     – pick a pixel in any frame
  s         – save the current annotated grid as a proof-sheet PNG
  r         – reset all markers
  q / Esc   – quit
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pipeline.frame import Frame
from pipeline.geometry import reproject_pick
import config

logger = logging.getLogger(__name__)

# Maximum columns in the display grid
MAX_COLS = 4
# Size of each thumbnail in the interactive window (width, height)
THUMB_W = 640
THUMB_H = 360


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_marker(
    img: np.ndarray,
    pt: tuple[float, float],
    color: tuple[int, int, int],
    radius: int = None,
    thickness: int = None,
    label: str = "",
) -> np.ndarray:
    img = img.copy()
    r = radius   or config.MARKER_RADIUS
    t = thickness or config.MARKER_THICKNESS
    x, y = int(round(pt[0])), int(round(pt[1]))

    # Outer circle
    cv2.circle(img, (x, y), r,     color,    t,           cv2.LINE_AA)
    # Inner dot
    cv2.circle(img, (x, y), r // 3, color,   cv2.FILLED,  cv2.LINE_AA)
    # Cross-hair lines
    cv2.line(img, (x - r - 4, y), (x + r + 4, y), color, max(1, t - 1), cv2.LINE_AA)
    cv2.line(img, (x, y - r - 4), (x, y + r + 4), color, max(1, t - 1), cv2.LINE_AA)

    if label:
        cv2.putText(img, label, (x + r + 4, y - r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img


def _frame_overlay(frame: Frame, src_pt=None, dst_pt=None) -> np.ndarray:
    """
    Return a copy of the frame's undistorted image with markers and HUD info.
    """
    img = frame.undistorted.copy() if frame.undistorted is not None else frame.raw.copy()

    # Telemetry banner at the top
    info = (
        f"hdg={frame.heading_deg:.0f}° "
        f"alt={frame.alt_agl_m:.1f}m "
        f"pitch={frame.gimbal_pitch_deg:.0f}°"
        if all(v is not None for v in [frame.heading_deg, frame.alt_agl_m, frame.gimbal_pitch_deg])
        else "no telemetry"
    )
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, f"{frame.display_name}  {info}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    if src_pt is not None:
        img = _draw_marker(img, src_pt, config.MARKER_COLOR_SRC,
                           label="SRC", radius=config.MARKER_RADIUS + 4)
    if dst_pt is not None:
        img = _draw_marker(img, dst_pt, config.MARKER_COLOR_DST,
                           label="PROJ")
    return img


def _build_grid(frames: list[Frame], annotations: dict) -> np.ndarray:
    """
    Tile annotated frames into a grid image.

    *annotations* maps Frame → {"src": (x,y) | None, "dst": (x,y) | None}
    """
    n_cols = min(MAX_COLS, len(frames))
    n_rows = math.ceil(len(frames) / n_cols)

    cell_h, cell_w = THUMB_H, THUMB_W
    grid = np.zeros((n_rows * cell_h, n_cols * cell_w, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row, col = divmod(idx, n_cols)
        ann  = annotations.get(frame, {})
        cell = _frame_overlay(frame, ann.get("src"), ann.get("dst"))
        cell = cv2.resize(cell, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

        r0, r1 = row * cell_h, (row + 1) * cell_h
        c0, c1 = col * cell_w, (col + 1) * cell_w
        grid[r0:r1, c0:c1] = cell

    # Draw grid lines
    for r in range(1, n_rows):
        cv2.line(grid, (0, r * cell_h), (grid.shape[1], r * cell_h), (60, 60, 60), 1)
    for c in range(1, n_cols):
        cv2.line(grid, (c * cell_w, 0), (c * cell_w, grid.shape[0]), (60, 60, 60), 1)

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Interactive viewer
# ─────────────────────────────────────────────────────────────────────────────

class ReprojectionViewer:
    """
    OpenCV-based interactive re-projection viewer.

    Usage::

        viewer = ReprojectionViewer(frames, van_model=van_model)
        viewer.run()
    """

    WINDOW = "Raycast Challenge  [click=pick | s=save | r=reset | q=quit]"

    def __init__(self, frames: list[Frame], van_model=None) -> None:
        self.frames      = [f for f in frames if f.ready]
        self.van_model   = van_model   # Optional[VanModel] for ray-box intersection
        self.annotations: dict[Frame, dict] = {f: {} for f in self.frames}
        self._grid       = self._rebuild_grid()
        self._n_cols     = min(MAX_COLS, len(self.frames))
        self._last_save  = None

        if not self.frames:
            raise RuntimeError("No ready frames to display.")

    # ── Grid ──────────────────────────────────────────────────────────────────

    def _rebuild_grid(self) -> np.ndarray:
        return _build_grid(self.frames, self.annotations)

    def _add_status_bar(self, grid: np.ndarray, msg: str) -> np.ndarray:
        bar = np.zeros((28, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, msg, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 1, cv2.LINE_AA)
        return np.vstack([grid, bar])

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _on_click(self, event, gx: int, gy: int, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Which frame cell was clicked?
        col = gx // THUMB_W
        row = gy // THUMB_H
        idx = row * self._n_cols + col

        if idx >= len(self.frames):
            return

        source = self.frames[idx]

        # Map grid click → undistorted image pixel
        thumb_x = gx % THUMB_W
        thumb_y = gy % THUMB_H
        h, w    = source.undistorted.shape[:2]
        px = thumb_x * w / THUMB_W
        py = thumb_y * h / THUMB_H

        logger.info("Click on frame %d [%s] at undist pixel (%.1f, %.1f)",
                    idx, source.stem, px, py)

        # Run the pipeline
        self.annotations = {f: {} for f in self.frames}
        self.annotations[source]["src"] = (px, py)

        results = reproject_pick(px, py, source, self.frames, van_model=self.van_model)
        for tf, proj in results.items():
            self.annotations[tf]["dst"] = proj

        n_ok = len(results)
        status = (f"Picked ({px:.0f}, {py:.0f}) in '{source.display_name}'  →  "
                  f"reprojected into {n_ok}/{len(self.frames)-1} frame(s).")
        self._status = status
        logger.info(status)
        self._grid = self._rebuild_grid()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self._grid.shape[1], self._grid.shape[0] + 28)
        cv2.setMouseCallback(self.WINDOW, self._on_click)

        self._status = "Click any frame to pick a target pixel."

        while True:
            display = self._add_status_bar(self._grid, self._status)
            cv2.imshow(self.WINDOW, display)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):    # q or Esc
                break
            elif key == ord("s"):
                self._save_proof_sheet()
            elif key == ord("r"):
                self.annotations = {f: {} for f in self.frames}
                self._grid       = self._rebuild_grid()
                self._status     = "Markers reset.  Click any frame to pick."

        cv2.destroyAllWindows()

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_proof_sheet(self) -> None:
        out_dir = Path(config.OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "proof_sheet.png"
        cv2.imwrite(str(path), self._grid)
        self._status = f"Proof sheet saved → {path}"
        logger.info("Proof sheet saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Proof-sheet (non-interactive batch export)
# ─────────────────────────────────────────────────────────────────────────────

def save_proof_sheet(
    frames: list[Frame],
    source_frame: Frame,
    src_pixel: tuple[float, float],
    reprojections: dict,
    filename: str = "proof_sheet.png",
) -> Path:
    """
    Save a single-image proof sheet showing the source pick and all
    re-projected points.  Suitable for the challenge deliverable.
    """
    annotations: dict[Frame, dict] = {}
    for f in frames:
        annotations[f] = {}
    annotations[source_frame]["src"] = src_pixel
    for tf, proj in reprojections.items():
        annotations[tf]["dst"] = proj

    grid = _build_grid([f for f in frames if f.ready], annotations)

    # Legend
    legend_h = 60
    legend   = np.zeros((legend_h, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "GREEN = source pick    BLUE = re-projected point",
                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (180, 220, 180), 2, cv2.LINE_AA)
    proof = np.vstack([grid, legend])

    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), proof)
    logger.info("Proof sheet written: %s", out_path)
    return out_path
