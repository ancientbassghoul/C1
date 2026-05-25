"""
pipeline/ui.py – Interactive re-projection viewer and proof-sheet export.

Layout
──────
All undistorted frames are tiled in a grid (≤4 columns).
Click any frame to pick a target pixel.  The pipeline:
  1. Marks the picked pixel in green.
  2. Computes the ground-plane intersection.
  3. Draws the re-projected point in orange on every other frame.

Navigation
──────────
  Scroll              Zoom in / out, centred on the cursor
  Middle-drag         Pan
  R                   Reset view to fit all frames
  Ctrl + Scroll       Increase / decrease marker size

Key bindings
────────────
  click     – pick a pixel in any frame
  s         – save the current annotated grid as a proof-sheet PNG
  r / R     – reset view (first press); double-r resets markers too
  q / Esc   – quit
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

from pipeline.frame import Frame
from pipeline.geometry import reproject_pick
import config

logger = logging.getLogger(__name__)

MAX_COLS = 4
THUMB_W  = 640
THUMB_H  = 360

_ZOOM_STEP     = 0.12
_ZOOM_MIN      = 0.10
_ZOOM_MAX      = 12.0
_MARKER_R_DEFAULT = 14
_MARKER_R_MAX     = 60
_DOT_FRAC         = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers (used for proof-sheet only — viewer draws in screen space)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_marker(img, pt, color, radius=None, thickness=None, label=""):
    img = img.copy()
    r = radius    or config.MARKER_RADIUS
    t = thickness or config.MARKER_THICKNESS
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(img, (x, y), r,      color, t,          cv2.LINE_AA)
    cv2.circle(img, (x, y), r // 3, color, cv2.FILLED, cv2.LINE_AA)
    cv2.line(img, (x - r - 4, y), (x + r + 4, y), color, max(1, t - 1), cv2.LINE_AA)
    cv2.line(img, (x, y - r - 4), (x, y + r + 4), color, max(1, t - 1), cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (x + r + 4, y - r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img


def _frame_banner(frame: Frame) -> np.ndarray:
    """Return the undistorted image with only the telemetry banner (no markers)."""
    img = frame.undistorted.copy() if frame.undistorted is not None else frame.raw.copy()
    info = (
        f"hdg={frame.heading_deg:.0f}°  "
        f"alt={frame.alt_agl_m:.1f}m  "
        f"pitch={frame.gimbal_pitch_deg:.0f}°"
        if all(v is not None for v in
               [frame.heading_deg, frame.alt_agl_m, frame.gimbal_pitch_deg])
        else "no telemetry"
    )
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, f"{frame.display_name}  {info}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def _build_grid(frames, annotations) -> np.ndarray:
    """Build a thumbnail grid with markers baked in (for proof-sheet export)."""
    n_cols = min(MAX_COLS, len(frames))
    n_rows = math.ceil(len(frames) / n_cols)
    grid   = np.zeros((n_rows * THUMB_H, n_cols * THUMB_W, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row, col = divmod(idx, n_cols)
        ann  = annotations.get(frame, {})
        img  = _frame_banner(frame)
        if ann.get("src"):
            img = _draw_marker(img, ann["src"], config.MARKER_COLOR_SRC,
                               label="SRC", radius=config.MARKER_RADIUS + 4)
        if ann.get("dst"):
            img = _draw_marker(img, ann["dst"], config.MARKER_COLOR_DST,
                               label="PROJ")
        cell = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        r0, r1 = row * THUMB_H, (row + 1) * THUMB_H
        c0, c1 = col * THUMB_W, (col + 1) * THUMB_W
        grid[r0:r1, c0:c1] = cell
    for r in range(1, n_rows):
        cv2.line(grid, (0, r * THUMB_H), (grid.shape[1], r * THUMB_H), (60, 60, 60), 1)
    for c in range(1, n_cols):
        cv2.line(grid, (c * THUMB_W, 0), (c * THUMB_W, grid.shape[0]), (60, 60, 60), 1)
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Interactive viewer with zoom / pan
# ─────────────────────────────────────────────────────────────────────────────

class ReprojectionViewer:
    """
    OpenCV-based interactive re-projection viewer with zoom / pan / marker scale.
    """

    WINDOW = ("Raycast  [click=pick | scroll=zoom | mid-drag=pan | "
              "ctrl+scroll=marker | R=reset | s=save | q=quit]")

    def __init__(self, frames: list[Frame]) -> None:
        self.frames = [f for f in frames if f.ready]
        if not self.frames:
            raise RuntimeError("No ready frames to display.")

        self._n_cols    = min(MAX_COLS, len(self.frames))
        self._frame_idx = {f: i for i, f in enumerate(self.frames)}

        # Annotations: Frame → {"src": (x,y), "dst": (x,y)}
        self.annotations: dict[Frame, dict] = {f: {} for f in self.frames}

        # Marker radius in screen pixels
        self._r = _MARKER_R_DEFAULT

        # View transform
        self._scale : float = 1.0
        self._ox    : float = 0.0
        self._oy    : float = 0.0

        # Pan state
        self._panning     = False
        self._pan_start_w = (0, 0)
        self._pan_start_o = (0.0, 0.0)

        # Window size (updated each frame)
        n_rows = math.ceil(len(self.frames) / self._n_cols)
        self._win_w = self._n_cols * THUMB_W
        self._win_h = n_rows * THUMB_H + 28

        self._status    = "Click any frame to pick a target pixel."
        self._last_save = None

        # Build the static clean canvas (no markers)
        self._canvas = self._build_canvas()
        self._reset_view()

    # ── Canvas (clean, no markers) ────────────────────────────────────────────

    def _build_canvas(self) -> np.ndarray:
        n_rows = math.ceil(len(self.frames) / self._n_cols)
        canvas = np.zeros((n_rows * THUMB_H, self._n_cols * THUMB_W, 3), dtype=np.uint8)
        for idx, frame in enumerate(self.frames):
            row, col = divmod(idx, self._n_cols)
            cell = cv2.resize(_frame_banner(frame), (THUMB_W, THUMB_H),
                               interpolation=cv2.INTER_AREA)
            r0, r1 = row * THUMB_H, (row + 1) * THUMB_H
            c0, c1 = col * THUMB_W, (col + 1) * THUMB_W
            canvas[r0:r1, c0:c1] = cell
        for r in range(1, n_rows):
            cv2.line(canvas, (0, r * THUMB_H), (canvas.shape[1], r * THUMB_H), (60, 60, 60), 1)
        for c in range(1, self._n_cols):
            cv2.line(canvas, (c * THUMB_W, 0), (c * THUMB_W, canvas.shape[0]), (60, 60, 60), 1)
        return canvas

    # ── View helpers ──────────────────────────────────────────────────────────

    def _reset_view(self):
        ch, cw = self._canvas.shape[:2]
        sx = self._win_w / cw
        sy = (self._win_h - 28) / ch
        self._scale = min(sx, sy)
        self._ox    = (self._win_w - cw * self._scale) / 2
        self._oy    = ((self._win_h - 28) - ch * self._scale) / 2

    def _win_to_canvas(self, wx, wy):
        return ((wx - self._ox) / self._scale,
                (wy - self._oy) / self._scale)

    def _img_to_screen(self, frame: Frame, px: float, py: float):
        h_img, w_img = frame.undistorted.shape[:2]
        idx          = self._frame_idx[frame]
        row, col     = divmod(idx, self._n_cols)
        cx = (col * THUMB_W + px * THUMB_W / w_img) * self._scale + self._ox
        cy = (row * THUMB_H + py * THUMB_H / h_img) * self._scale + self._oy
        return int(cx), int(cy)

    # ── Compose display ───────────────────────────────────────────────────────

    def _compose(self) -> np.ndarray:
        ch, cw    = self._canvas.shape[:2]
        ww        = self._win_w
        wh_grid   = self._win_h - 28
        display   = np.zeros((wh_grid, ww, 3), dtype=np.uint8)

        cx0 = -self._ox / self._scale
        cy0 = -self._oy / self._scale
        src_x0 = max(0, int(cx0));      src_y0 = max(0, int(cy0))
        src_x1 = min(cw, int(cx0 + ww / self._scale) + 1)
        src_y1 = min(ch, int(cy0 + wh_grid / self._scale) + 1)

        if src_x1 > src_x0 and src_y1 > src_y0:
            patch        = self._canvas[src_y0:src_y1, src_x0:src_x1]
            dst_x0       = int(src_x0 * self._scale + self._ox)
            dst_y0       = int(src_y0 * self._scale + self._oy)
            scaled_patch = cv2.resize(
                patch,
                (int(patch.shape[1] * self._scale),
                 int(patch.shape[0] * self._scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            sp_y0 = max(0, dst_y0 - dst_y0 if dst_y0 >= 0 else -dst_y0)
            sp_x0 = max(0, -dst_x0 if dst_x0 < 0 else 0)
            dst_y0c = max(0, dst_y0);  dst_x0c = max(0, dst_x0)
            dst_y1c = min(wh_grid, dst_y0 + scaled_patch.shape[0])
            dst_x1c = min(ww,      dst_x0 + scaled_patch.shape[1])
            if dst_y1c > dst_y0c and dst_x1c > dst_x0c:
                ph = dst_y1c - dst_y0c
                pw = dst_x1c - dst_x0c
                sp = scaled_patch[sp_y0:sp_y0 + ph, sp_x0:sp_x0 + pw]
                display[dst_y0c:dst_y0c + sp.shape[0],
                        dst_x0c:dst_x0c + sp.shape[1]] = sp

        # Markers in screen space
        self._draw_markers_screen(display)

        # Status bar
        bar = np.zeros((28, ww, 3), dtype=np.uint8)
        zoom_pct = int(self._scale * 100)
        msg = f"{self._status}   |  zoom {zoom_pct}%  marker r={self._r}"
        cv2.putText(bar, msg, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 215, 150), 1, cv2.LINE_AA)
        return np.vstack([display, bar])

    def _draw_markers_screen(self, display: np.ndarray):
        r     = max(1, self._r)
        dot_r = max(1, int(r * _DOT_FRAC))
        for frame in self.frames:
            ann = self.annotations.get(frame, {})
            if ann.get("src"):
                sx, sy = self._img_to_screen(frame, *ann["src"])
                c = config.MARKER_COLOR_SRC
                cv2.circle(display, (sx, sy), r,     c, 2,          cv2.LINE_AA)
                cv2.circle(display, (sx, sy), dot_r, c, cv2.FILLED, cv2.LINE_AA)
                arm = r + 5
                cv2.line(display, (sx-arm, sy), (sx+arm, sy), c, 1, cv2.LINE_AA)
                cv2.line(display, (sx, sy-arm), (sx, sy+arm), c, 1, cv2.LINE_AA)
            if ann.get("dst"):
                sx, sy = self._img_to_screen(frame, *ann["dst"])
                c = config.MARKER_COLOR_DST
                cv2.circle(display, (sx, sy), r,     c, 2,          cv2.LINE_AA)
                cv2.circle(display, (sx, sy), dot_r, c, cv2.FILLED, cv2.LINE_AA)

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _on_mouse(self, event, wx: int, wy: int, flags, param):
        ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)

        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            if ctrl:
                self._r = max(1, min(_MARKER_R_MAX, self._r + delta * 2))
            else:
                factor    = 1 + _ZOOM_STEP * delta
                new_scale = max(_ZOOM_MIN, min(_ZOOM_MAX, self._scale * factor))
                cx, cy    = self._win_to_canvas(wx, wy)
                self._scale = new_scale
                self._ox    = wx - cx * self._scale
                self._oy    = wy - cy * self._scale
            return

        if event == cv2.EVENT_MBUTTONDOWN:
            self._panning     = True
            self._pan_start_w = (wx, wy)
            self._pan_start_o = (self._ox, self._oy)
            return
        if event == cv2.EVENT_MOUSEMOVE and self._panning:
            self._ox = self._pan_start_o[0] + wx - self._pan_start_w[0]
            self._oy = self._pan_start_o[1] + wy - self._pan_start_w[1]
            return
        if event == cv2.EVENT_MBUTTONUP:
            self._panning = False
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Map window → canvas → frame
        cx, cy = self._win_to_canvas(wx, wy)
        if cx < 0 or cy < 0:
            return
        col = int(cx // THUMB_W)
        row = int(cy // THUMB_H)
        idx = row * self._n_cols + col
        if idx >= len(self.frames) or col >= self._n_cols:
            return

        source  = self.frames[idx]
        tx, ty  = cx % THUMB_W, cy % THUMB_H
        h, w    = source.undistorted.shape[:2]
        px      = tx * w / THUMB_W
        py      = ty * h / THUMB_H

        logger.info("Click on frame %d [%s] at undist pixel (%.1f, %.1f)",
                    idx, source.stem, px, py)

        self.annotations = {f: {} for f in self.frames}
        self.annotations[source]["src"] = (px, py)
        results = reproject_pick(px, py, source, self.frames)
        for tf, proj in results.items():
            self.annotations[tf]["dst"] = proj

        n_ok = len(results)
        self._status = (f"Picked ({px:.0f}, {py:.0f}) in '{source.display_name}'  "
                        f"???  reprojected into {n_ok}/{len(self.frames)-1} frame(s).")
        logger.info(self._status.replace("???", "→"))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self._win_w, self._win_h)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)

        while True:
            try:
                rect = cv2.getWindowImageRect(self.WINDOW)
                if rect[2] > 0 and rect[3] > 0:
                    self._win_w = rect[2]
                    self._win_h = rect[3]
            except Exception:
                pass

            cv2.imshow(self.WINDOW, self._compose())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key in (ord('r'), ord('R')):
                self._reset_view()
            elif key == ord('s'):
                self._save_proof_sheet()

        cv2.destroyAllWindows()

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_proof_sheet(self) -> None:
        out_dir = Path(config.OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "proof_sheet.png"
        # Build grid with markers baked in for the saved image
        grid = _build_grid(self.frames, self.annotations)
        cv2.imwrite(str(path), grid)
        self._status = f"Proof sheet saved → {path}"
        logger.info("Proof sheet saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Proof-sheet (non-interactive batch export) — unchanged
# ─────────────────────────────────────────────────────────────────────────────

def save_proof_sheet(frames, source_frame, src_pixel, reprojections,
                     filename="proof_sheet.png") -> Path:
    annotations: dict[Frame, dict] = {f: {} for f in frames}
    annotations[source_frame]["src"] = src_pixel
    for tf, proj in reprojections.items():
        annotations[tf]["dst"] = proj
    grid = _build_grid([f for f in frames if f.ready], annotations)
    legend_h = 60
    legend   = np.zeros((legend_h, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "GREEN = source pick    BLUE = re-projected point",
                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 180), 2, cv2.LINE_AA)
    proof    = np.vstack([grid, legend])
    out_dir  = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), proof)
    logger.info("Proof sheet written: %s", out_path)
    return out_path
