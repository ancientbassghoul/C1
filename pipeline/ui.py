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
  click       – pick a pixel in any frame
  right-click – mark the TRUE location of the pick in a target frame (scoring)
  u           – undo the last truth click
  Enter / n   – commit the current pick's scores to score.csv
  ← / →       – review saved picks from score.csv (read-only; reprojected
                with the currently loaded solve)
  c           – clear all markers / leave review (a live pick with truth
                clicks is committed first)
  s           – save the current annotated grid as a proof-sheet PNG
  r / R       – reset view (first press); double-r resets markers too
  q / Esc     – commit pending scores, print summary, quit

Scoring
───────
A pick is written to score.csv (one row per target frame) only if it has at
least one truth click; otherwise it is discarded.  Frames left unclicked are
logged as 'skipped', so unjudgeable (blurry) frames show up as missing
coverage rather than as error.

Three error measures are recorded side by side:
  err_px      – image pixels.  Not comparable across frames: a far, zoomed-out
                frame shows the same miss as far fewer pixels.
  err_m_view  – err_px × gsd (metres per pixel at the point's depth): the miss
                in metres, measured facing the camera.  Removes the zoom /
                distance effect.
  err_m       – ground distance: the truth click is cast through the target
                camera onto the terrain and compared with the pick's ground
                point.  At shallow view angles this is stretched by
                ~1 / sin(view_elev_deg) along the line of sight.
"""

from __future__ import annotations

import csv
import logging
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pipeline.frame import Frame
from pipeline.geometry import reproject_pick, pixel_to_ground
from pipeline.scoring import (_SCORE_TARGET_PX, _fmt, _fmt_metric, _metric_error,
                              _stats_block)
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

_TRUTH_COLOR = (0, 230, 255)   # BGR yellow

# cv2.waitKeyEx arrow codes: Windows, then GTK / Qt
_KEYS_LEFT  = (2424832, 65361)
_KEYS_RIGHT = (2555904, 65363)

SCORE_FIELDS = [
    "timestamp", "pick_id", "solve_file", "source_frame", "src_x", "src_y",
    "target_frame", "proj_x", "proj_y", "gt_x", "gt_y", "dx", "dy", "err_px",
    "err_m", "err_m_view", "gsd_m_per_px", "range_m", "view_elev_deg", "status",
]


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

    WINDOW = ("Raycast  [click=pick | right-click=truth | u=undo | Enter=commit | "
              "arrows=review saved | c=clear | scroll=zoom | mid-drag=pan | "
              "ctrl+scroll=marker | R=reset | s=save | q=quit]")

    def __init__(self, frames: list[Frame], surface=None,
                 score_path: str | Path | None = None,
                 solve_file: str = "") -> None:
        self.frames = [f for f in frames if f.ready]
        if not self.frames:
            raise RuntimeError("No ready frames to display.")

        # Optional GroundSurface for terrain-aware reprojection (else flat Z=0).
        self.surface = surface

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

        # Scoring state: current pick + human truth clicks in target frames
        self._score_path = Path(score_path) if score_path else None
        self._solve_file = solve_file
        self._pick_src   : tuple[Frame, float, float] | None = None
        self._pick_world : np.ndarray | None = None
        self._pick_proj  : dict[Frame, tuple[float, float]] = {}
        self._truth_map  : dict[Frame, tuple[float, float]] = {}
        self._truth_metric: dict[Frame, dict] = {}    # frame → _metric_error output
        self._truth_history: list[tuple[Frame, tuple[float, float] | None]] = []
        self._session_errs: dict[str, list[float]] = {"err_m_view": [], "err_m": []}

        # Review mode: pick_id of the saved pick on display (read-only), else None
        self._review_pid: str | None = None

        if self._score_path is not None:
            backfill_metric_scores(self._score_path, self.frames, self.surface,
                                   self._solve_file)
        self._pick_id = _next_pick_id(self._score_path)

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

        # Truth clicks: yellow cross, line to reprojection, error label
        for frame, (gx, gy) in self._truth_map.items():
            tx, ty = self._img_to_screen(frame, gx, gy)
            arm = r + 5
            cv2.line(display, (tx-arm, ty-arm), (tx+arm, ty+arm), (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(display, (tx-arm, ty+arm), (tx+arm, ty-arm), (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(display, (tx-arm, ty-arm), (tx+arm, ty+arm), _TRUTH_COLOR, 2, cv2.LINE_AA)
            cv2.line(display, (tx-arm, ty+arm), (tx+arm, ty-arm), _TRUTH_COLOR, 2, cv2.LINE_AA)
            proj = self._pick_proj.get(frame)
            if proj is not None:
                px, py = self._img_to_screen(frame, *proj)
                cv2.line(display, (px, py), (tx, ty), _TRUTH_COLOR, 1, cv2.LINE_AA)
                m = self._truth_metric.get(frame, {})
                parts = [f"{math.hypot(gx - proj[0], gy - proj[1]):.1f}px"]
                if m.get("err_m_view") is not None:
                    parts.append(f"{m['err_m_view']:.2f} m")
                if m.get("err_m") is not None:
                    elev = m.get("view_elev_deg")
                    parts.append(f"gnd {m['err_m']:.2f} m"
                                 + (f" @{elev:.0f}deg" if elev is not None else ""))
                label = " | ".join(parts)
            else:
                label = "MISSED"
            org = (tx + arm + 4, ty - arm)
            cv2.putText(display, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(display, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        _TRUTH_COLOR, 1, cv2.LINE_AA)

    # ── Hit test ──────────────────────────────────────────────────────────────

    def _hit_test(self, wx: int, wy: int):
        """Window pixel → (frame, px, py) in full-res undistorted image, or None."""
        cx, cy = self._win_to_canvas(wx, wy)
        if cx < 0 or cy < 0:
            return None
        col = int(cx // THUMB_W)
        row = int(cy // THUMB_H)
        idx = row * self._n_cols + col
        if idx >= len(self.frames) or col >= self._n_cols:
            return None
        frame  = self.frames[idx]
        tx, ty = cx % THUMB_W, cy % THUMB_H
        h, w   = frame.undistorted.shape[:2]
        return frame, tx * w / THUMB_W, ty * h / THUMB_H

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

        if event == cv2.EVENT_RBUTTONDOWN:
            hit = self._hit_test(wx, wy)
            if hit is None or self._pick_src is None:
                return
            if self._review_pid is not None:
                self._status = ("Reviewing a saved pick (read-only) - press c to clear "
                                "and pick a new point.")
                return
            frame, gx, gy = hit
            if frame is self._pick_src[0]:
                return
            self._truth_history.append((frame, self._truth_map.get(frame)))
            self._set_truth(frame, (gx, gy))
            m = self._truth_metric[frame]
            logger.info("Truth click on [%s] at (%.1f, %.1f)  view=%s m  ground=%s m  "
                        "gsd=%s m/px  range=%s m  elev=%s deg",
                        frame.stem, gx, gy,
                        *(_fmt(m[k], nd) or "n/a" for k, nd in
                          (("err_m_view", 3), ("err_m", 3), ("gsd_m_per_px", 4),
                           ("range_m", 1), ("view_elev_deg", 1))))
            self._update_score_status()
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        hit = self._hit_test(wx, wy)
        if hit is None:
            return
        source, px, py = hit

        logger.info("Click on frame %d [%s] at undist pixel (%.1f, %.1f)",
                    self._frame_idx[source], source.stem, px, py)

        self._commit_pick()
        self._review_pid = None          # a left-click always starts a live pick
        n_ok = self._start_pick(source, px, py)
        self._status = (f"Pick #{self._pick_id} ({px:.0f}, {py:.0f}) in '{source.display_name}'  "
                        f"???  reprojected into {n_ok}/{len(self.frames)-1} frame(s).  "
                        f"Right-click true location in target frames.")
        logger.info(self._status.replace("???", "→"))
        self._status = self._status.replace("???", "->")

    def _start_pick(self, source: Frame, px: float, py: float) -> int:
        """Show a pick: source marker + reprojections; returns #frames reprojected."""
        self.annotations = {f: {} for f in self.frames}
        self.annotations[source]["src"] = (px, py)
        results = reproject_pick(px, py, source, self.frames, surface=self.surface)
        for tf, proj in results.items():
            self.annotations[tf]["dst"] = proj

        self._pick_src   = (source, px, py)
        self._pick_proj  = dict(results)
        self._pick_world = pixel_to_ground(px, py, source, self.surface)
        return len(results)

    # ── Review saved picks / clear ────────────────────────────────────────────

    def _load_saved_picks(self) -> dict[str, dict]:
        """score.csv → {pick_id: {source, src, truths{stem: (x,y)}, solve_file}}."""
        picks: dict[str, dict] = {}
        for r in _read_score_rows(self._score_path):
            p = picks.setdefault(r["pick_id"], {
                "source": r["source_frame"],
                "src": (float(r["src_x"]), float(r["src_y"])),
                "truths": {},
                "solve_file": r.get("solve_file", ""),
            })
            if r.get("gt_x") and r.get("gt_y"):
                p["truths"][r["target_frame"]] = (float(r["gt_x"]), float(r["gt_y"]))
        return dict(sorted(picks.items(), key=lambda kv: int(kv[0])))

    def _show_saved_pick(self, step: int) -> None:
        """Display the next (+1) / previous (-1) saved pick, read-only."""
        self._commit_pick()
        picks = self._load_saved_picks()
        if not picks:
            self._status = f"No saved picks in {self._score_path}."
            return
        ids = list(picks)
        if self._review_pid in ids:
            i = (ids.index(self._review_pid) + step) % len(ids)
        else:
            i = 0 if step > 0 else len(ids) - 1
        pid, p = ids[i], picks[ids[i]]

        by_stem = {f.stem: f for f in self.frames}
        source = by_stem.get(p["source"])
        if source is None:
            logger.warning("Saved pick #%s: source frame %s not loaded.", pid, p["source"])
            self._clear_view()
            self._review_pid = pid           # keep position so arrows continue from here
            self._status = f"Review pick #{pid}: source frame {p['source']} not loaded."
            return

        self._clear_truth()
        self._review_pid = pid
        self._start_pick(source, *p["src"])
        missing = 0
        for stem, xy in p["truths"].items():
            tf = by_stem.get(stem)
            if tf is None or tf is source:
                missing += tf is None
                continue
            self._set_truth(tf, xy)
        if missing:
            logger.warning("Saved pick #%s: %d truth frame(s) not loaded.", pid, missing)

        n_judged = sum(f in self._pick_proj for f in self._truth_map)
        cur = {k: self._current_errs(k) for k in ("err_m_view", "err_m")}
        other = (f"  (clicked under {p['solve_file']})"
                 if p["solve_file"] and p["solve_file"] != self._solve_file else "")
        self._status = (f"Review pick {i + 1}/{len(ids)} (#{pid}) from "
                        f"{source.stem.rsplit('_', 1)[-1]}: judged {n_judged}, "
                        f"median {self._med_m(cur['err_m_view'])} "
                        f"(gnd {self._med_m(cur['err_m'])}){other}   "
                        f"[<-/-> browse | c clear]")
        logger.info("Review saved pick #%s (source %s, %d truth click(s))",
                    pid, source.stem, len(self._truth_map))

    def _clear_view(self) -> None:
        self.annotations = {f: {} for f in self.frames}
        self._pick_src   = None
        self._pick_world = None
        self._pick_proj  = {}
        self._clear_truth()
        self._review_pid = None

    def _clear_all(self) -> None:
        """Commit any live pick (if it has truth clicks), then clear everything."""
        self._commit_pick()
        self._clear_view()
        self._status = "Cleared. Click any frame to pick a new point (<-/-> to review saved picks)."

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _set_truth(self, frame: Frame, truth: tuple[float, float] | None) -> None:
        if truth is None:
            self._truth_map.pop(frame, None)
            self._truth_metric.pop(frame, None)
            return
        self._truth_map[frame] = truth
        proj = self._pick_proj.get(frame)
        err_px = math.hypot(truth[0] - proj[0], truth[1] - proj[1]) if proj else None
        self._truth_metric[frame] = _metric_error(self._pick_world, frame, *truth,
                                                  err_px=err_px, surface=self.surface)

    def _clear_truth(self) -> None:
        self._truth_map.clear()
        self._truth_metric.clear()
        self._truth_history.clear()

    def _current_errs(self, key: str) -> list[float]:
        return [self._truth_metric[f][key] for f in self._truth_map
                if f in self._pick_proj and self._truth_metric[f][key] is not None]

    @staticmethod
    def _med_m(v: list[float]) -> str:
        return f"{float(np.median(v)):.2f} m" if v else "-"

    def _update_score_status(self) -> None:
        n_judged = sum(f in self._pick_proj for f in self._truth_map)
        n_targets = len(self.frames) - 1
        cur  = {k: self._current_errs(k) for k in ("err_m_view", "err_m")}
        sess = {k: self._session_errs[k] + cur[k] for k in cur}
        self._status = (f"Pick #{self._pick_id}: judged {n_judged}/{n_targets}, "
                        f"median {self._med_m(cur['err_m_view'])} "
                        f"(gnd {self._med_m(cur['err_m'])})   |  "
                        f"session: {len(sess['err_m_view'])} judged, "
                        f"median {self._med_m(sess['err_m_view'])} "
                        f"(gnd {self._med_m(sess['err_m'])})")

    def _undo_truth(self) -> None:
        if self._review_pid is not None or not self._truth_history:
            return
        frame, prev = self._truth_history.pop()
        self._set_truth(frame, prev)    # prev: earlier click that was replaced, or None
        logger.info("Undo truth click on [%s]", frame.stem)
        self._update_score_status()

    def _commit_pick(self) -> None:
        """Append the current pick to score.csv (only if it has truth clicks)."""
        if (self._review_pid is not None           # reviewed picks are already saved
                or self._pick_src is None or not self._truth_map
                or self._score_path is None):
            self._clear_truth()
            return

        source, sx, sy = self._pick_src
        ts   = datetime.now().isoformat(timespec="seconds")
        rows = []
        for tf in self.frames:
            if tf is source:
                continue
            proj  = self._pick_proj.get(tf)
            truth = self._truth_map.get(tf)
            row = {k: "" for k in SCORE_FIELDS}
            row.update(timestamp=ts, pick_id=self._pick_id, solve_file=self._solve_file,
                       source_frame=source.stem, src_x=f"{sx:.2f}", src_y=f"{sy:.2f}",
                       target_frame=tf.stem)
            if proj is not None:
                row.update(proj_x=f"{proj[0]:.2f}", proj_y=f"{proj[1]:.2f}")
            if truth is not None:
                row.update(gt_x=f"{truth[0]:.2f}", gt_y=f"{truth[1]:.2f}")
            if proj is not None and truth is not None:
                dx, dy = truth[0] - proj[0], truth[1] - proj[1]
                err = math.hypot(dx, dy)
                m = self._truth_metric[tf]
                row.update(dx=f"{dx:.2f}", dy=f"{dy:.2f}", err_px=f"{err:.2f}",
                           **_fmt_metric(m), status="judged")
                for k in self._session_errs:
                    if m[k] is not None:
                        self._session_errs[k].append(m[k])
            elif proj is not None:
                row.update(**_fmt_metric(_metric_error(self._pick_world, tf)),
                           status="skipped")
            elif truth is not None:
                row["status"] = "missed"
            else:
                row["status"] = "not_visible"
            rows.append(row)

        self._score_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._score_path.exists() or self._score_path.stat().st_size == 0
        with open(self._score_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=SCORE_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

        n_judged = sum(r["status"] == "judged" for r in rows)
        logger.info("Committed pick #%d (%d judged) → %s",
                    self._pick_id, n_judged, self._score_path)
        self._pick_id += 1
        self._clear_truth()
        sess = self._session_errs
        self._status = (f"Committed. Session: {len(sess['err_m_view'])} judged, "
                        f"median {self._med_m(sess['err_m_view'])} "
                        f"(gnd {self._med_m(sess['err_m'])})")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self._win_w, self._win_h)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)

        # try/finally: the pending pick is committed however the viewer ends —
        # q/Esc, the window's X button, or Ctrl+C in the terminal.
        try:
            while True:
                try:
                    rect = cv2.getWindowImageRect(self.WINDOW)
                    if rect[2] > 0 and rect[3] > 0:
                        self._win_w = rect[2]
                        self._win_h = rect[3]
                except Exception:
                    pass

                cv2.imshow(self.WINDOW, self._compose())
                # waitKeyEx: plain waitKey() & 0xFF drops the arrow-key codes
                key_ex = cv2.waitKeyEx(30)
                key = key_ex & 0xFF if key_ex != -1 else 255

                # Window closed via its X button (imshow would otherwise re-create it)
                if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Viewer window closed.")
                    break

                if key_ex in _KEYS_RIGHT:
                    self._show_saved_pick(+1)
                elif key_ex in _KEYS_LEFT:
                    self._show_saved_pick(-1)
                elif key in (ord('q'), 27):
                    break
                elif key in (ord('c'), ord('C')):
                    self._clear_all()
                elif key in (ord('r'), ord('R')):
                    self._reset_view()
                elif key == ord('s'):
                    self._save_proof_sheet()
                elif key == ord('u'):
                    self._undo_truth()
                elif key in (13, 10, ord('n')) and self._review_pid is None:
                    self._commit_pick()
        finally:
            self._commit_pick()
            cv2.destroyAllWindows()

        if self._score_path is not None and self._score_path.exists():
            summary = summarize_scores(self._score_path)
            print("\n" + summary)
            out = self._score_path.with_name("score_summary.txt")
            out.write_text(summary, encoding="utf-8")
            logger.info("Score summary written: %s", out)

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
# Reprojection scoring (score.csv)
# ─────────────────────────────────────────────────────────────────────────────

def _read_score_rows(csv_path: Path) -> list[dict]:
    if csv_path is None or not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _next_pick_id(csv_path: Path | None) -> int:
    """Continue pick numbering from an existing score.csv (1 if none)."""
    ids = []
    for row in _read_score_rows(csv_path):
        try:
            ids.append(int(row["pick_id"]))
        except (KeyError, ValueError):
            pass
    return max(ids) + 1 if ids else 1


def backfill_metric_scores(csv_path: Path, frames: list, surface,
                           solve_file: str) -> None:
    """Fill the metric columns for existing score.csv rows (upgrades old files).

    Rows scored under a different solve file are left untouched — their pixels
    only make sense against the cameras they were clicked with.
    """
    if not csv_path.exists():
        return
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows   = list(reader)
    if not rows:
        return

    header_old = header != SCORE_FIELDS
    todo = [r for r in rows
            if r.get("status") in ("judged", "skipped") and not r.get("view_elev_deg")]
    if not header_old and not todo:
        return

    by_stem = {f.stem: f for f in frames}
    world_cache: dict[str, np.ndarray | None] = {}
    n_filled = n_foreign = n_missing = 0
    for r in todo:
        if r.get("solve_file", "") != solve_file:
            n_foreign += 1
            continue
        src, tf = by_stem.get(r["source_frame"]), by_stem.get(r["target_frame"])
        if src is None or tf is None:
            n_missing += 1
            continue
        pid = r["pick_id"]
        if pid not in world_cache:
            world_cache[pid] = pixel_to_ground(float(r["src_x"]), float(r["src_y"]),
                                               src, surface)
        world = world_cache[pid]
        if world is None:
            continue
        if r["status"] == "judged":
            m = _metric_error(world, tf, float(r["gt_x"]), float(r["gt_y"]),
                              err_px=float(r["err_px"]), surface=surface)
        else:
            m = _metric_error(world, tf)
        r.update(_fmt_metric(m))
        n_filled += 1

    if n_foreign:
        logger.warning("score.csv backfill: %d row(s) from a different solve file "
                       "(active: '%s') left without metric error.", n_foreign, solve_file)
    if n_missing:
        logger.warning("score.csv backfill: %d row(s) reference frames not loaded.", n_missing)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCORE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") or "" for k in SCORE_FIELDS})
    logger.info("score.csv backfill: filled metric columns for %d row(s)%s → %s",
                n_filled, " (header upgraded)" if header_old else "", csv_path)


def summarize_scores(csv_path: str | Path) -> str:
    """Human-readable accuracy summary of every row in score.csv.

    Reports the three error measures side by side (see module docstring),
    plus a per-target-frame table of medians.
    """
    rows = _read_score_rows(Path(csv_path))
    counts = {s: sum(r["status"] == s for r in rows)
              for s in ("judged", "skipped", "missed", "not_visible")}
    n_picks = len({r["pick_id"] for r in rows})

    def _f(r, k):
        v = r.get(k, "")
        return float(v) if v not in ("", None) else None

    judged = [r for r in rows if r["status"] == "judged"]
    cols = ("err_px", "err_m_view", "err_m", "gsd_m_per_px", "range_m", "view_elev_deg")

    def _vals(rs, col):
        return np.asarray([v for v in (_f(r, col) for r in rs) if v is not None])

    lines = [
        f"Reprojection accuracy - {csv_path}",
        f"  picks: {n_picks}   judged: {counts['judged']}   skipped: {counts['skipped']}   "
        f"missed: {counts['missed']}   not_visible: {counts['not_visible']}",
    ]
    if not judged:
        lines.append("  no judged rows yet.")
        return "\n".join(lines) + "\n"

    tgt_m = float(config.SCORE_TARGET_M)
    n = len(judged)
    lines += _stats_block("image error (pixels; favours far / zoomed-out frames)",
                          _vals(judged, "err_px"), "px", 1, _SCORE_TARGET_PX, n)
    lines += _stats_block("view error (px x m/px: miss in metres facing the camera)",
                          _vals(judged, "err_m_view"), "m", 2, tgt_m, n)
    lines += _stats_block("ground error (distance on the terrain; stretched at shallow views)",
                          _vals(judged, "err_m"), "m", 2, tgt_m, n)

    by_frame: dict[str, list[dict]] = {}
    for r in judged:
        by_frame.setdefault(r["target_frame"], []).append(r)

    def _med(rs, col, scale=1.0, nd=1):
        v = _vals(rs, col)
        return f"{np.median(v) * scale:.{nd}f}" if len(v) else "-"

    lines += [
        "  per target frame (medians over judged clicks):",
        f"    {'frame':<34} {'view deg':>8} {'range m':>8} {'px':>6} {'cm/px':>6} "
        f"{'view m':>7} {'ground m':>9} {'n':>3}",
    ]
    for stem in sorted(by_frame):
        rs = by_frame[stem]
        lines.append(
            f"    {stem:<34} {_med(rs, 'view_elev_deg'):>8} {_med(rs, 'range_m'):>8} "
            f"{_med(rs, 'err_px'):>6} {_med(rs, 'gsd_m_per_px', 100, 2):>6} "
            f"{_med(rs, 'err_m_view', 1, 2):>7} {_med(rs, 'err_m', 1, 2):>9} {len(rs):>3}")
    lines.append("    view m = px x cm/px;  ground m ~ view m / sin(view deg) "
                 "for misses along the line of sight.")
    return "\n".join(lines) + "\n"


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
