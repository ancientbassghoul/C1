"""
pipeline/manual_correspondence_ui.py – Manual multi-frame correspondence picker.

Navigation
──────────
  Scroll              Zoom in / out, centred on the cursor
  Middle-drag         Pan
  R                   Reset view to fit all frames
  Ctrl + Scroll       Increase / decrease marker size

Correspondence editing
──────────────────────
  Left-click          Place / move this frame's point for the current
                      in-progress or being-edited correspondence
  Right-click         Remove this frame's point from the current
                      in-progress or being-edited correspondence
  Enter  /  n         Finalise current correspondence (needs ≥ 2 frames)
  [  /  ]             Enter edit mode on last saved; navigate older / newer
  d                   Delete last saved correspondence (whole)
  c                   Clear / cancel current in-progress correspondence
  s                   Write to JSON immediately
  q  /  Esc           Quit (warns on unsaved changes; press twice to force)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from pipeline.frame import Frame
import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Grid layout  (native / unzoomed)
# ─────────────────────────────────────────────────────────────────────────────
MAX_COLS = 4
THUMB_W  = 640
THUMB_H  = 360

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
_MARKER_R_DEFAULT = 13     # outer ring radius at startup

# Van feature types: (label, config_z_attr, BGR_color)
# (name, z_config_attr, BGR_color, is_pair, distance_config_attr, axis)
_FEATURE_TYPES = [
    ("ground",     "GROUND_Z_M",     (200, 200,   0), False, None,              None),
    ("roof",       "VAN_HEIGHT_M",   ( 40, 200, 200), False, None,              None),
    ("wheel_axis", "WHEEL_RADIUS_M", (200,  80, 200), True,  "VAN_WHEELBASE_M", "forward"),
    ("roof_edge",  "VAN_HEIGHT_M",   (100, 200, 255), True,  "VAN_WIDTH_M",     "lateral"),
]
_F_NAME, _F_Z, _F_COL, _F_PAIR, _F_DIST, _F_AXIS = range(6)
_MARKER_R_MAX     = 40
_DOT_FRAC         = 0.35   # inner dot = this fraction of ring radius

_ZOOM_STEP        = 0.12   # fractional zoom per scroll tick
_ZOOM_MIN         = 0.15
_ZOOM_MAX         = 8.0

# ─────────────────────────────────────────────────────────────────────────────
# Colours  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
_C_SAVED        = (200, 200,   0)
_C_CURRENT      = (  0, 230, 230)
_C_BANNER_CLEAN = (  0,   0,   0)
_C_BANNER_DIRTY = ( 30,  60, 140)
_C_TEXT         = (210, 210, 210)
_C_DIVIDER      = ( 55,  55,  55)

def _score_to_bgr(score: float) -> tuple[int, int, int]:
    """Map score 0→1 to BGR colour: red → yellow → green → cyan → blue."""
    h = int(score * 120)   # OpenCV hue 0=red, 60=green, 120=blue
    hsv = np.array([[[h, 255, 210]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2]))

_C_NO_SCORE = (120, 120, 120)   # grey — score unavailable


class ManualCorrespondenceViewer:
    """
    Interactive manual ground-feature correspondence picker with zoom / pan.
    """

    WINDOW = ("Manual Correspondences  "
              "[click=pick | RClick=del point | Enter/n=save | [=prev ]=next | d=del corr | c=clear | "
              "s=write | R=reset | q=quit]")

    def __init__(self, frames: list[Frame], json_path: Path,
                 scores: dict | None = None,
                 score_mode: bool = False) -> None:
        self.frames = [f for f in frames if f.undistorted is not None]
        if not self.frames:
            raise RuntimeError("No undistorted frames to display.")

        self._n_cols    = min(MAX_COLS, len(self.frames))
        self._json_path = Path(json_path)
        self._by_stem   = {f.stem: f for f in self.frames}
        self._frame_idx = {f: i for i, f in enumerate(self.frames)}

        # Correspondence data
        self._saved  : list[dict[Frame, tuple[float, float]]] = []
        self._current: dict[Frame, tuple[float, float]]        = {}
        self._dirty      = False
        self._score_mode = score_mode
        self._scores     : dict[int, float | None] = scores or {}
        self._sel_corr   : int | None              = None
        self._status  = ("SCORE MODE — zoom/pan freely.  "
                         "[ / ] to cycle correspondences."
                         if score_mode else
                         "Click any frame to start a new correspondence.")

        # Edit-mode state
        # When _edit_mode is True, _edit_idx points into self._saved and
        # self._current holds the working copy of that correspondence.
        # _edit_backup holds the original so 'c' can restore it.
        # _edit_last_frame is set when the user left-clicks a frame while in
        # edit mode; 'd' uses it to decide single-frame vs whole-corr delete.
        self._edit_mode      : bool             = False
        self._edit_idx       : int              = 0
        self._edit_backup    : dict             = {}
        self._edit_last_frame                   = None   # Frame | None

        # Feature type for next correspondence (T cycles)
        self._type_idx   : int       = 0
        self._saved_types: list[int] = []

        # Marker size (outer ring radius, in thumbnail pixels)
        self._r = _MARKER_R_DEFAULT

        # ── View transform ────────────────────────────────────────────────────
        # The full unzoomed grid is rendered into self._canvas (fixed size).
        # _scale and _offset (ox, oy) map canvas pixels to window pixels:
        #   win_x = canvas_x * _scale + _ox
        #   win_y = canvas_y * _scale + _oy
        # Inverse: canvas_x = (win_x - _ox) / _scale
        self._scale : float       = 1.0
        self._ox    : float       = 0.0
        self._oy    : float       = 0.0

        # Middle-mouse pan state
        self._panning     = False
        self._pan_start_w = (0, 0)   # window coords at pan start
        self._pan_start_o = (0.0, 0.0)  # _ox/_oy at pan start

        # Ctrl key state (OpenCV flags bit 8)
        self._ctrl = False

        # Window size — set once on first show, updated on resize
        n_rows = math.ceil(len(self.frames) / self._n_cols)
        self._win_w = self._n_cols * THUMB_W
        self._win_h = n_rows * THUMB_H + 28   # +28 status bar

        # Build the static (unzoomed) canvas once; re-rendered on annotation change
        self._canvas : np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)
        self._rebuild_canvas()

        self._load()
        self._rebuild_canvas()
        self._reset_view()


    def _img_to_screen(self, frame: Frame, px: float, py: float) -> tuple[int, int]:
        """Convert full-image pixel coords to window screen pixel coords."""
        h_img, w_img = frame.undistorted.shape[:2]
        idx          = self._frame_idx[frame]
        row, col     = divmod(idx, self._n_cols)
        cx = (col * THUMB_W + px * THUMB_W / w_img) * self._scale + self._ox
        cy = (row * THUMB_H + py * THUMB_H / h_img) * self._scale + self._oy
        return int(cx), int(cy)

    def _draw_markers_screen(self, display: np.ndarray) -> None:
        """Draw all markers in screen space. self._r is in screen pixels."""
        r     = max(1, self._r)
        dot_r = max(1, int(r * _DOT_FRAC))

        def _draw_one(sx, sy, col, label=""):
            cv2.circle(display, (sx, sy), r,     col, 2,          cv2.LINE_AA)
            cv2.circle(display, (sx, sy), dot_r, col, cv2.FILLED, cv2.LINE_AA)
            if label:
                cv2.putText(display, label, (sx + r + 3, sy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

        for frame in self.frames:
            for i, corr in enumerate(self._saved):
                if frame not in corr:
                    continue
                tidx    = self._saved_types[i] if i < len(self._saved_types) else 0
                if not self._score_mode and tidx != self._type_idx:
                    continue   # hide markers from other type modes
                ft      = _FEATURE_TYPES[tidx]
                is_pair = ft[_F_PAIR]
                col     = ft[_F_COL] if not self._score_mode else None
                if is_pair:
                    pts = corr[frame]   # list of 2 (x,y)
                    if len(pts) < 2:
                        continue
                    sxa, sya = self._img_to_screen(frame, *pts[0])
                    sxb, syb = self._img_to_screen(frame, *pts[1])
                    c = ft[_F_COL]
                    cv2.line(display, (sxa, sya), (sxb, syb), c, 1, cv2.LINE_AA)
                    _draw_one(sxa, sya, c, f"{i}A")
                    _draw_one(sxb, syb, c, f"{i}B")
                    if i == self._sel_corr:
                        mag = (255, 0, 255)
                        cv2.circle(display, (sxa, sya), r+5, mag, 2, cv2.LINE_AA)
                        cv2.circle(display, (sxb, syb), r+5, mag, 2, cv2.LINE_AA)
                else:
                    sx, sy = self._img_to_screen(frame, *corr[frame])
                    if self._score_mode:
                        sc  = self._scores.get(i)
                        c   = _score_to_bgr(sc) if sc is not None else _C_NO_SCORE
                        lbl = f"{i}:{sc:.2f}" if sc is not None else f"{i}:?"
                    else:
                        c   = ft[_F_COL]
                        lbl = str(i)
                    _draw_one(sx, sy, c, lbl)
                    if i == self._sel_corr:
                        mag = (255, 0, 255)
                        cv2.circle(display, (sx, sy), r+5, mag, 2, cv2.LINE_AA)
                        cv2.circle(display, (sx, sy), dot_r, mag, cv2.FILLED, cv2.LINE_AA)

            if frame in self._current and not self._score_mode:
                cur_tidx = (self._saved_types[self._edit_idx]
                            if self._edit_mode else self._type_idx)
                cur_ft   = _FEATURE_TYPES[cur_tidx]
                cur_pts  = self._current[frame]
                if cur_ft[_F_PAIR]:
                    # Draw partial pair in progress
                    labels = ["A", "B"]
                    screens = [self._img_to_screen(frame, *p) for p in cur_pts]
                    for k, (sx, sy) in enumerate(screens):
                        _draw_one(sx, sy, _C_CURRENT, labels[k])
                    if len(screens) == 2:
                        cv2.line(display, screens[0], screens[1],
                                 _C_CURRENT, 1, cv2.LINE_AA)
                else:
                    sx, sy = self._img_to_screen(frame, *cur_pts)
                    cv2.circle(display, (sx, sy), r,     _C_CURRENT, 2,          cv2.LINE_AA)
                    cv2.circle(display, (sx, sy), dot_r, _C_CURRENT, cv2.FILLED, cv2.LINE_AA)
                    arm = r + 6
                    cv2.line(display, (sx-arm, sy), (sx+arm, sy), _C_CURRENT, 1, cv2.LINE_AA)
                    cv2.line(display, (sx, sy-arm), (sx, sy+arm), _C_CURRENT, 1, cv2.LINE_AA)
                    cv2.putText(display, "new", (sx + r + 3, sy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, _C_CURRENT, 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    # Marker size helpers
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def _r_min(self) -> int:
        """
        Minimum marker ring radius (thumbnail pixels) such that the ring is
        Minimum: 1 screen pixel (markers are drawn in screen space).
        """
        return 1

    # ─────────────────────────────────────────────────────────────────────────
    # JSON I/O
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._json_path.exists():
            return
        try:
            data = json.loads(self._json_path.read_text())
            n = 0
            for entry in data.get("correspondences", []):
                tname  = entry.get("type", "ground")
                tidx   = next((i for i, ft in enumerate(_FEATURE_TYPES)
                               if ft[_F_NAME] == tname), 0)
                is_pair = _FEATURE_TYPES[tidx][_F_PAIR]
                corr: dict = {}
                for stem, xy in entry.get("points", {}).items():
                    f = self._by_stem.get(stem)
                    if f is None:
                        continue
                    if is_pair:
                        # xy is [[xa,ya],[xb,yb]]
                        corr[f] = [(float(p[0]), float(p[1])) for p in xy]
                    else:
                        corr[f] = (float(xy[0]), float(xy[1]))
                min_frames = 1 if is_pair else 2
                if len(corr) >= min_frames:
                    self._saved.append(corr)
                    self._saved_types.append(tidx)
                    n += 1
            logger.info("Loaded %d correspondence(s) from %s", n, self._json_path)
            self._status = f"Loaded {n} saved correspondence(s)."
        except Exception as exc:
            logger.warning("Failed to load correspondences: %s", exc)

    def _write(self) -> None:
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        import config as _cw
        entries = []
        for i, corr in enumerate(self._saved):
            tidx = self._saved_types[i] if i < len(self._saved_types) else 0
            ft    = _FEATURE_TYPES[tidx]
            tname = ft[_F_NAME]
            z_val = getattr(_cw, ft[_F_Z], 0.0)
            dist_val = getattr(_cw, ft[_F_DIST], None) if ft[_F_DIST] else None
            entry = {"id": i, "type": tname,
                     "z_plane": round(float(z_val), 4),
                     "is_pair": ft[_F_PAIR]}
            if dist_val is not None:
                entry["distance_m"] = round(float(dist_val), 4)
            if ft[_F_AXIS]:
                entry["axis"] = ft[_F_AXIS]
            if ft[_F_PAIR]:
                pts_dict = {}
                for f, pts in corr.items():
                    pts_dict[f.stem] = [[float(p[0]), float(p[1])] for p in pts]
            else:
                pts_dict = {f.stem: [float(x), float(y)]
                            for f, (x, y) in corr.items()}
            entry["points"] = pts_dict
            entries.append(entry)
        self._json_path.write_text(
            json.dumps({"correspondences": entries}, indent=2)
        )
        self._dirty  = False
        self._status = (f"Written {len(self._saved)} correspondence(s) → "
                        f"{self._json_path}")
        logger.info("Saved %d correspondences to %s",
                    len(self._saved), self._json_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Canvas rendering  (full unzoomed grid, with annotations)
    # ─────────────────────────────────────────────────────────────────────────

    def _make_cell(self, frame: Frame) -> np.ndarray:
        full   = frame.undistorted
        h, w   = full.shape[:2]
        cell = cv2.resize(full.copy(), (THUMB_W, THUMB_H),
                          interpolation=cv2.INTER_AREA)

        # Banner
        n_saved = sum(1 for c in self._saved if frame in c)
        has_cur = frame in self._current
        bg      = _C_BANNER_DIRTY if self._dirty else _C_BANNER_CLEAN
        cv2.rectangle(cell, (0, 0), (cell.shape[1], 20), bg, cv2.FILLED)
        marker = "\u25cf " if has_cur else ""
        cv2.putText(cell,
                    f"{marker}{frame.display_name}   {n_saved} saved",
                    (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, _C_TEXT, 1, cv2.LINE_AA)
        return cell

    def _rebuild_canvas(self) -> None:
        n_rows = math.ceil(len(self.frames) / self._n_cols)
        grid   = np.zeros((n_rows * THUMB_H, self._n_cols * THUMB_W, 3),
                          dtype=np.uint8)
        for idx, f in enumerate(self.frames):
            row, col = divmod(idx, self._n_cols)
            cell = self._make_cell(f)
            r0, r1 = row * THUMB_H, (row + 1) * THUMB_H
            c0, c1 = col * THUMB_W, (col + 1) * THUMB_W
            grid[r0:r1, c0:c1] = cell
        for r in range(1, n_rows):
            cv2.line(grid, (0, r * THUMB_H), (grid.shape[1], r * THUMB_H),
                     _C_DIVIDER, 1)
        for c in range(1, self._n_cols):
            cv2.line(grid, (c * THUMB_W, 0), (c * THUMB_W, grid.shape[0]),
                     _C_DIVIDER, 1)
        self._canvas = grid

    # ─────────────────────────────────────────────────────────────────────────
    # Compositing: apply zoom/pan + status bar → final display image
    # ─────────────────────────────────────────────────────────────────────────

    def _compose(self) -> np.ndarray:
        ch, cw = self._canvas.shape[:2]
        ww, wh_grid = self._win_w, self._win_h - 28

        # Destination viewport (window grid area)
        display = np.zeros((wh_grid, ww, 3), dtype=np.uint8)

        # Source rect in canvas coordinates
        cx0 = -self._ox / self._scale
        cy0 = -self._oy / self._scale
        cx1 = cx0 + ww  / self._scale
        cy1 = cy0 + wh_grid / self._scale

        # Clip to canvas bounds
        src_x0 = max(0, int(cx0));  src_y0 = max(0, int(cy0))
        src_x1 = min(cw, int(cx1)+1); src_y1 = min(ch, int(cy1)+1)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            pass   # fully outside — black frame
        else:
            patch = self._canvas[src_y0:src_y1, src_x0:src_x1]
            # Where does this land in the display?
            dst_x0 = int(src_x0 * self._scale + self._ox)
            dst_y0 = int(src_y0 * self._scale + self._oy)
            dst_x1 = dst_x0 + int(patch.shape[1] * self._scale)
            dst_y1 = dst_y0 + int(patch.shape[0] * self._scale)

            dst_x0c = max(0, dst_x0); dst_y0c = max(0, dst_y0)
            dst_x1c = min(ww, dst_x1); dst_y1c = min(wh_grid, dst_y1)
            if dst_x1c > dst_x0c and dst_y1c > dst_y0c:
                pw = dst_x1c - dst_x0c
                ph = dst_y1c - dst_y0c
                scaled_patch = cv2.resize(patch, (int(patch.shape[1] * self._scale),
                                                   int(patch.shape[0] * self._scale)),
                                           interpolation=cv2.INTER_LINEAR)
                sp_y0 = dst_y0c - dst_y0
                sp_x0 = dst_x0c - dst_x0
                sp_y0 = max(0, sp_y0); sp_x0 = max(0, sp_x0)
                sp_patch = scaled_patch[sp_y0:sp_y0+ph, sp_x0:sp_x0+pw]
                display[dst_y0c:dst_y0c+sp_patch.shape[0],
                        dst_x0c:dst_x0c+sp_patch.shape[1]] = sp_patch

        # Markers — drawn in screen space, pixel-perfect at any zoom
        self._draw_markers_screen(display)

        # Status bar
        bar = np.zeros((28, ww, 3), dtype=np.uint8)
        msg = self._status + ("  [UNSAVED]" if self._dirty else "")
        zoom_pct = int(self._scale * 100)
        info = f"  |  zoom {zoom_pct}%  marker r={self._r}"
        cv2.putText(bar, msg + info, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (150, 215, 150), 1, cv2.LINE_AA)
        return np.vstack([display, bar])

    # ─────────────────────────────────────────────────────────────────────────
    # View helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_view(self) -> None:
        """Fit the full canvas into the window."""
        ch, cw = self._canvas.shape[:2]
        sx = self._win_w / cw
        sy = (self._win_h - 28) / ch
        self._scale = min(sx, sy)
        self._ox    = (self._win_w - cw * self._scale) / 2
        self._oy    = ((self._win_h - 28) - ch * self._scale) / 2

    def _win_to_canvas(self, wx: int, wy: int) -> tuple[float, float]:
        return ((wx - self._ox) / self._scale,
                (wy - self._oy) / self._scale)

    # ─────────────────────────────────────────────────────────────────────────
    # Mouse callback
    # ─────────────────────────────────────────────────────────────────────────

    def _on_mouse(self, event, wx: int, wy: int, flags, param) -> None:
        ctrl_held = bool(flags & cv2.EVENT_FLAG_CTRLKEY)

        # ── Zoom via scroll wheel ─────────────────────────────────────────────
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1

            if ctrl_held:
                # Ctrl+scroll → marker size
                self._r = max(self._r_min,
                              min(_MARKER_R_MAX, self._r + delta * 2))
                self._rebuild_canvas()
            else:
                # Normal scroll → zoom centred on cursor
                factor = 1 + _ZOOM_STEP * delta
                new_scale = max(_ZOOM_MIN, min(_ZOOM_MAX,
                                               self._scale * factor))
                # Keep the canvas point under the cursor fixed
                cx, cy = self._win_to_canvas(wx, wy)
                self._scale = new_scale
                self._ox    = wx - cx * self._scale
                self._oy    = wy - cy * self._scale
            return

        # ── Middle-mouse pan ──────────────────────────────────────────────────
        if event == cv2.EVENT_MBUTTONDOWN:
            self._panning     = True
            self._pan_start_w = (wx, wy)
            self._pan_start_o = (self._ox, self._oy)
            return

        if event == cv2.EVENT_MOUSEMOVE and self._panning:
            dx = wx - self._pan_start_w[0]
            dy = wy - self._pan_start_w[1]
            self._ox = self._pan_start_o[0] + dx
            self._oy = self._pan_start_o[1] + dy
            return

        if event == cv2.EVENT_MBUTTONUP:
            self._panning = False
            return

        # ── Left-click → place point ──────────────────────────────────────────
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Map window → canvas → grid cell
        cx, cy = self._win_to_canvas(wx, wy)
        if cx < 0 or cy < 0:
            return

        col = int(cx // THUMB_W)
        row = int(cy // THUMB_H)
        idx = row * self._n_cols + col
        if idx >= len(self.frames) or col >= self._n_cols:
            return

        frame  = self.frames[idx]
        tx, ty = cx % THUMB_W, cy % THUMB_H

        # Map thumbnail coords → full-image coords
        h, w = frame.undistorted.shape[:2]
        px   = tx * w / THUMB_W
        py   = ty * h / THUMB_H

        if self._score_mode:
            best_idx, best_dist = None, float("inf")
            for i, corr in enumerate(self._saved):
                if frame not in corr:
                    continue
                fx, fy = corr[frame]
                d = math.hypot((fx * THUMB_W / w) - tx,
                               (fy * THUMB_H / h) - ty)
                if d < best_dist:
                    best_dist, best_idx = d, i
            if best_idx is not None and best_dist < 30:
                self._score_select(best_idx)
            else:
                self._sel_corr = None
                self._status   = "Click near a marker to select it."
            return

        ft = _FEATURE_TYPES[self._type_idx if not self._edit_mode
                            else self._saved_types[self._edit_idx]]
        if ft[_F_PAIR]:
            # Pair mode: accumulate up to 2 points per frame
            pts = list(self._current.get(frame, []))
            if len(pts) >= 2:
                pts = [(px, py)]   # 3rd click resets this frame
                self._status = (f"Frame {frame.stem[-8:]} reset.  "
                                f"Point A set — click again for B.")
            elif len(pts) == 1:
                pts.append((px, py))
                complete = sum(1 for p in self._current.values() if len(p)==2)
                self._status = (f"Point B set — {complete} frame(s) complete.  "
                                f"Click more frames or Enter to save.")
            else:
                pts.append((px, py))
                self._status = (f"Point A set in {frame.stem[-8:]}.  "
                                f"Click same frame again for Point B.")
            self._current[frame] = pts
            if self._edit_mode:
                self._edit_last_frame = frame
        else:
            self._current[frame] = (px, py)
            if self._edit_mode:
                self._edit_last_frame = frame
                total = len(self._saved)
                self._status = (f"Editing #{self._edit_idx}/{total-1}: "
                                f"{len(self._current)} frame(s).  "
                                f"d=del this frame  Enter=save  c=cancel")
            else:
                n = len(self._current)
                self._status = (f"Building correspondence: {n} frame(s) marked.  "
                                f"Click more or press Enter/n to save.")
        self._dirty  = True
        self._rebuild_canvas()

    # ─────────────────────────────────────────────────────────────────────────
    # Keyboard actions
    # ─────────────────────────────────────────────────────────────────────────

    def _score_select(self, idx: int) -> None:
        """Set selected correspondence by index (wraps). Updates status bar."""
        if not self._saved:
            self._sel_corr = None
            self._status   = "No correspondences loaded."
            return
        self._sel_corr = idx % len(self._saved)
        sc  = self._scores.get(self._sel_corr)
        sc_str = f"{sc:.3f}" if sc is not None else "N/A"
        total  = len(self._saved)
        n_frames = len(self._saved[self._sel_corr])
        self._status = (f"Selected #{self._sel_corr}/{total-1}  "
                        f"score={sc_str}  ({n_frames} frame(s))  "
                        f"[ / ] to navigate.")

    def _action_finalise(self) -> None:
        ft      = _FEATURE_TYPES[self._type_idx]
        is_pair = ft[_F_PAIR]
        if is_pair:
            # For pair types, filter to frames with exactly 2 points
            complete = {f: pts for f, pts in self._current.items()
                        if len(pts) == 2}
            if len(complete) < 1:
                self._status = ("Need ≥ 1 complete frame (click A then B in "
                                "same frame).  Partial frames are ignored.")
                return
            corr = complete
        else:
            if len(self._current) < 2:
                self._status = (f"Need ≥ 2 frames "
                                f"(have {len(self._current)}).  Keep clicking.")
                return
            corr = dict(self._current)
        if self._edit_mode:
            self._saved[self._edit_idx] = corr
            self._current         = {}
            self._edit_mode       = False
            self._edit_last_frame = None
            self._dirty     = True
            self._status    = f"Correspondence #{self._edit_idx} updated.  Click any frame to start a new one."
        else:
            self._saved.append(corr)
            self._saved_types.append(self._type_idx)
            n = len(self._saved)
            self._current = {}
            self._dirty   = True
            self._status  = f"Correspondence #{n-1} saved ({len(corr)} frame(s)).  Click any frame for the next one."
        self._rebuild_canvas()

    def _type_filtered_indices(self) -> list[int]:
        """Absolute indices into _saved that match the current type selector."""
        return [i for i, t in enumerate(self._saved_types)
                if t == self._type_idx]

    def _load_edit_idx(self, abs_idx: int) -> None:
        """Enter / move edit mode to the given absolute index in _saved."""
        self._edit_idx        = abs_idx
        self._edit_backup     = dict(self._saved[abs_idx])
        self._current         = dict(self._saved[abs_idx])
        self._edit_last_frame = None
        # Type stays as-is (caller already ensured it matches)

    def _edit_status(self) -> str:
        idxs  = self._type_filtered_indices()
        pos   = idxs.index(self._edit_idx) + 1 if self._edit_idx in idxs else "?"
        total = len(idxs)
        tname = _FEATURE_TYPES[self._type_idx][_F_NAME].upper()
        return (f"Editing {tname} {pos}/{total}  "
                f"(abs #{self._edit_idx}, {len(self._current)} frame(s))  "
                f"[=prev  ]=next  Enter=save  c=cancel")

    def _action_edit_last(self) -> None:
        # [: enter edit mode on last of this type, or step to previous of this type
        if not self._saved:
            self._status = "Nothing to edit."; return
        idxs = self._type_filtered_indices()
        if not idxs:
            tname = _FEATURE_TYPES[self._type_idx][_F_NAME].upper()
            self._status = f"No {tname} correspondences saved yet."; return

        if not self._edit_mode:
            if self._current:
                self._status = "Finish or clear the current correspondence first (c)."; return
            self._load_edit_idx(idxs[-1])
            self._edit_mode = True
        else:
            # Auto-save current edits before moving
            updated = dict(self._current)
            if updated != self._saved[self._edit_idx]:
                self._dirty = True
            self._saved[self._edit_idx] = updated
            # Find previous in filtered list
            try:
                pos = idxs.index(self._edit_idx)
            except ValueError:
                pos = len(idxs)   # current idx no longer in filter — jump to end
            if pos == 0:
                self._status = (f"Already at oldest {_FEATURE_TYPES[self._type_idx][_F_NAME].upper()}.  "
                                f"] to go forward.")
                self._rebuild_canvas(); return
            self._load_edit_idx(idxs[pos - 1])

        self._status = self._edit_status()
        self._rebuild_canvas()

    def _action_edit_next(self) -> None:
        # ]: move to next of this type while in edit mode
        if not self._edit_mode:
            self._status = "Press [ first to enter edit mode."; return
        idxs = self._type_filtered_indices()
        # Auto-save current edits before moving
        updated = dict(self._current)
        if updated != self._saved[self._edit_idx]:
            self._dirty = True
        self._saved[self._edit_idx] = updated
        # Find next in filtered list
        try:
            pos = idxs.index(self._edit_idx)
        except ValueError:
            pos = -1   # current idx no longer in filter — jump to start
        if pos >= len(idxs) - 1:
            self._current         = {}
            self._edit_mode       = False
            self._edit_last_frame = None
            tname = _FEATURE_TYPES[self._type_idx][_F_NAME].upper()
            self._status = f"Reached newest {tname} — edit mode exited."
            self._rebuild_canvas(); return
        self._load_edit_idx(idxs[pos + 1])
        self._status = self._edit_status()
        self._rebuild_canvas()

    def _action_delete_last(self) -> None:
        if not self._saved:
            self._status = "Nothing to delete."; return

        if self._edit_mode:
            idx = self._edit_idx
            if self._edit_last_frame is not None:
                # User left-clicked a frame while browsing → delete just that
                # frame's mark from the correspondence, leave the rest intact.
                frame = self._edit_last_frame
                if frame in self._current:
                    del self._current[frame]
                    self._saved[idx] = dict(self._current)
                self._edit_last_frame = None
                self._dirty  = True
                total = len(self._saved)
                self._status = (f"Removed {frame.stem[-10:]} from #{idx}.  "
                                f"{len(self._current)} frame(s) remaining.  "
                                f"d=del entire  Enter=save  c=cancel")
            else:
                # No frame was moved → delete the entire correspondence.
                # Before deleting, compute the filtered list to find a landing target.
                idxs = self._type_filtered_indices()
                pos  = idxs.index(idx) if idx in idxs else None

                del self._saved[idx]
                del self._saved_types[idx]
                self._edit_last_frame = None
                self._dirty = True

                # Rebuild filtered list after deletion (indices ≥ idx shifted by -1)
                idxs_after = self._type_filtered_indices()

                if not idxs_after:
                    # Nothing of this type left
                    self._edit_mode = False
                    self._current   = {}
                    tname = _FEATURE_TYPES[self._type_idx][_F_NAME].upper()
                    self._status = f"Deleted. No more {tname} correspondences."
                else:
                    # Try previous (same position - 1 in the pre-delete list maps
                    # to position - 1 in the post-delete list since we deleted idx).
                    # A simpler rule: land on the item just before the deleted slot
                    # if one exists, otherwise the first remaining item.
                    candidate = pos - 1 if (pos is not None and pos > 0) else 0
                    candidate = min(candidate, len(idxs_after) - 1)
                    self._load_edit_idx(idxs_after[candidate])
                    self._status = f"Deleted #{idx}.  " + self._edit_status()
        else:
            # Outside edit mode: delete the last one (quick cleanup shortcut).
            self._saved.pop()
            self._saved_types.pop()
            self._dirty  = True
            self._status = f"Deleted last.  {len(self._saved)} remaining."

        self._rebuild_canvas()

    def _action_clear_current(self) -> None:
        if self._edit_mode:
            self._saved[self._edit_idx] = dict(self._edit_backup)
            self._current         = {}
            self._edit_mode       = False
            self._edit_last_frame = None
            self._status    = f"Correspondence #{self._edit_idx} restored.  Click any frame to start a new one."
        else:
            self._current = {}
            self._status  = "Cleared.  Click any frame to start a new correspondence."
        self._rebuild_canvas()

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self._win_w, self._win_h)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)

        _warn_quit = False

        while True:
            # Update window size in case user resized
            try:
                rect = cv2.getWindowImageRect(self.WINDOW)
                if rect[2] > 0 and rect[3] > 0:
                    if (rect[2] != self._win_w or
                            rect[3] != self._win_h):
                        self._win_w = rect[2]
                        self._win_h = rect[3]
            except Exception:
                pass

            cv2.imshow(self.WINDOW, self._compose())
            key = cv2.waitKey(30) & 0xFF

            if key in (13, ord('n')):
                _warn_quit = False
                self._action_finalise()

            elif key == ord('['):
                _warn_quit = False
                if self._score_mode:
                    cur = self._sel_corr if self._sel_corr is not None else 0
                    self._score_select(cur - 1)
                else:
                    self._action_edit_last()

            elif key == ord(']'):
                _warn_quit = False
                if self._score_mode:
                    cur = self._sel_corr if self._sel_corr is not None else -1
                    self._score_select(cur + 1)
                else:
                    self._action_edit_next()

            elif key == ord('d'):
                _warn_quit = False
                self._action_delete_last()

            elif key == ord('c'):
                _warn_quit = False
                self._action_clear_current()

            elif key == ord('s'):
                _warn_quit = False
                self._write()
                self._rebuild_canvas()

            elif key in (ord('r'), ord('R')):
                _warn_quit = False
                self._reset_view()

            elif key in (ord('T'), ord('t')) and not self._score_mode and not self._edit_mode:
                _warn_quit = False
                self._type_idx = (self._type_idx + 1) % len(_FEATURE_TYPES)
                _ft   = _FEATURE_TYPES[self._type_idx]
                import config as _ct
                z_val = getattr(_ct, _ft[_F_Z], 0.0)
                pair_str = "  [PAIR — click A then B]" if _ft[_F_PAIR] else ""
                self._status = (f"Type → {_ft[_F_NAME].upper()}  "
                                f"(z={z_val:.3f} m){pair_str}")

            elif key in (ord('q'), 27):
                if self._dirty and not _warn_quit and not self._score_mode:
                    self._status = ("Unsaved changes!  Press s to save, "
                                    "or q again to discard and quit.")
                    _warn_quit = True
                    self._rebuild_canvas()
                else:
                    break
            elif key != 0xFF:  # real keypress — reset quit-warn; 0xFF = no key (waitKey timeout)
                _warn_quit = False

        cv2.destroyAllWindows()
