"""
pipeline/feature_matcher_debug_ui.py – Interactive SuperPoint / LightGlue explorer.

Shows exactly the keypoints and matches that the orientation solver receives:
  • Ground-region filter   (lower 55 % of content area, same as _match())
  • RANSAC homography      (same threshold as _match())
  • Spatial grid subsample (same MATCH_GRID_COLS × MATCH_GRID_ROWS as _match())
  • Van-bbox exclusion     skipped (detection not run in debug mode), but all other
                           filters are identical to the normal pipeline.
  • Image enhancement      respects the same --enhance flag as the normal pipeline.

Interaction flow
────────────────
1. IDLE
   All frames greyed-out.  Click any frame to make it the source.

2. SOURCE SELECTED
   • Source frame shown in full colour; every surviving filtered keypoint is a
     cyan dot.  Keypoints removed by ground / RANSAC / grid filters are not
     shown — only what the solver actually sees.
   • Frames that share ≥1 filtered match with the source become full colour.
   • Frames with no surviving matches remain greyed out.
   • A [Clear] button appears top-right of the source cell.
   • Clicking a coloured paired frame makes it the new source.

3. KEYPOINT SELECTED
   Click near any cyan dot in the source:
   • Dot highlighted in bright yellow-cyan in the source.
   • Every paired frame shows the corresponding matched dot in orange.
   Click [Clear] to return to IDLE.

Key bindings
────────────
  q / Esc  – quit
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Optional

import cv2
import numpy as np

from pipeline.frame import Frame
import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_COLS = 4
THUMB_W  = 640
THUMB_H  = 360

# ─────────────────────────────────────────────────────────────────────────────
# Colours  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
_GREY_ALPHA   = 0.22

_C_KP         = (  0, 210, 210)   # cyan      – filtered kp in source
_C_KP_SEL     = (  0, 245, 245)   # bright    – selected kp in source
_C_MATCH      = ( 30,  90, 255)   # orange    – correspondence in paired frame
_C_BTN_BG     = ( 30,  30, 140)
_C_BTN_BORDER = (180, 180, 180)
_C_BTN_TEXT   = (255, 255, 255)
_C_BANNER_BG  = (  0,   0,   0)
_C_BANNER_TXT = (165, 165, 165)
_C_PAIRED_TXT = (120, 210, 120)

# ─────────────────────────────────────────────────────────────────────────────
# Sizes
# ─────────────────────────────────────────────────────────────────────────────
_R_KP   = 4
_R_SEL  = 12
_SNAP_R = 15   # thumbnail-pixel snap radius for kp clicks

_BTN = (_BTN_X1, _BTN_Y1, _BTN_X2, _BTN_Y2) = (THUMB_W - 78, 5, THUMB_W - 5, 25)

# Tolerance (full-image pixels) for matching a selected point back into a pair
_PT_TOL = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _greyed(img: np.ndarray) -> np.ndarray:
    g   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(g,   cv2.COLOR_GRAY2BGR)
    return (bgr.astype(np.float32) * _GREY_ALPHA).astype(np.uint8)


def _banner(cell: np.ndarray, text: str, color=_C_BANNER_TXT) -> None:
    cv2.rectangle(cell, (0, 0), (cell.shape[1], 20), _C_BANNER_BG, cv2.FILLED)
    cv2.putText(cell, text, (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)


def _clear_btn(cell: np.ndarray) -> None:
    cv2.rectangle(cell, (_BTN_X1, _BTN_Y1), (_BTN_X2, _BTN_Y2), _C_BTN_BG,    cv2.FILLED)
    cv2.rectangle(cell, (_BTN_X1, _BTN_Y1), (_BTN_X2, _BTN_Y2), _C_BTN_BORDER, 1)
    cv2.putText  (cell, "Clear", (_BTN_X1 + 7, _BTN_Y2 - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.44, _C_BTN_TEXT, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main viewer
# ─────────────────────────────────────────────────────────────────────────────

class FeatureMatcherDebugViewer:
    """
    Interactive SuperPoint / LightGlue feature-match explorer.

    Shows the exact set of keypoints and matches that reach the Ceres solver
    (ground filter + RANSAC + grid subsampling applied identically to _match()).

    Parameters
    ----------
    frames  : list of Frame objects with .undistorted populated
    enhance : pass True to enable CLAHE+unsharp preprocessing, matching
              the --enhance flag in the normal pipeline (default False)
    """

    WINDOW = ("Feature Matcher Debug"
              "   [click frame → click keypoint | Clear to reset | q = quit]")

    def __init__(self, frames: list[Frame], enhance: bool = False) -> None:
        self.frames = [f for f in frames if f.undistorted is not None]
        if not self.frames:
            raise RuntimeError("No undistorted frames to display.")

        self._n_cols = min(MAX_COLS, len(self.frames))

        # ── Viewer state ──────────────────────────────────────────────────────
        self._source : Optional[Frame] = None
        self._sel_kp : Optional[int]   = None   # index into self._kps[source]
        self._status : str             = "Click any frame to select a source."

        # ── Per-frame filtered keypoints ──────────────────────────────────────
        # _kps[f] -> (N, 2) float32 pixels that survived all filters and appear
        #            in at least one matched pair involving f.
        self._kps: dict[Frame, np.ndarray] = {
            f: np.empty((0, 2), dtype=np.float32) for f in self.frames
        }

        # ── Per-pair filtered match coordinates ───────────────────────────────
        # _pair_pts[(fa, fb)] -> (pts_a, pts_b), both (M, 2) float32
        #   pts_a[i] corresponds to pts_b[i].
        #   Stored in both orderings so lookup is always (source, other).
        self._pair_pts: dict[tuple[Frame, Frame],
                             tuple[np.ndarray, np.ndarray]] = {}

        # ── Ground masks (GroundedSAM) ────────────────────────────────────────────────
        # _ground_masks[f] -> (H, W) bool — True = ground pixel
        self._ground_masks: dict[Frame, np.ndarray] = {}

        logger.info("Feature matcher debug UI: computing ground masks…")
        from pipeline.ground_mask import get_ground_masks
        self._ground_masks = get_ground_masks(self.frames)

        logger.info("Feature matcher debug UI: extracting + matching (enhance=%s)…",
                    enhance)
        self._extract_and_match(enhance)
        logger.info("Feature matcher debug UI ready (%d frames).", len(self.frames))

        self._grid = self._rebuild_grid()

    # ── Feature extraction + matching ─────────────────────────────────────────

    def _extract_and_match(self, enhance: bool) -> None:
        """
        Run SuperPoint on every frame, LightGlue on every pair, then apply
        the same post-processing filters as _match() in feature_matcher.py:
          1. Ground-region row filter
          2. RANSAC homography (same threshold)
          3. Spatial grid subsampling (same grid dimensions from config)
        Van-bbox exclusion is skipped (detection is not run in debug mode).
        """
        import pipeline.feature_matcher as fm
        import torch
        from lightglue.utils import rbd

        # Honour the --enhance flag exactly as the normal pipeline does.
        fm.set_enhance(enhance)
        fm._load_models()

        device      = fm._device
        device_type = device.type

        # ── Extract keypoints (one pass per frame, tensors reused for matching) ──
        feats_raw: dict[Frame, dict]       = {}
        raw_kps  : dict[Frame, np.ndarray] = {}   # all SuperPoint kps, (N, 2)

        for f in self.frames:
            t = fm._to_tensor(f.undistorted, device)
            with torch.autocast(device_type=device_type,
                                enabled=(device_type == "cuda")):
                with torch.no_grad():
                    feats = fm._extractor.extract(t)
            feats_raw[f] = feats
            raw_kps[f]   = rbd(feats)["keypoints"].cpu().numpy()
            logger.debug("  %s: %d raw kps", f.stem[-12:], len(raw_kps[f]))

        # ── Match every pair and apply all filters ────────────────────────────
        pairs = list(combinations(self.frames, 2))
        logger.info("  Matching %d pairs…", len(pairs))

        # Accumulate filtered points per frame before deduplication
        accumulated: dict[Frame, list[tuple[float, float]]] = {
            f: [] for f in self.frames
        }

        # ── Pass 1: per-pair ground mask + RANSAC ────────────────────────────
        raw_pair_matches: dict = {}

        for fa, fb in pairs:
            with torch.autocast(device_type=device_type,
                                enabled=(device_type == "cuda")):
                with torch.no_grad():
                    result = fm._matcher({
                        "image0": feats_raw[fa],
                        "image1": feats_raw[fb],
                    })
            result = rbd(result)
            m = result["matches"].cpu().numpy()
            if len(m) == 0:
                continue

            kp_a = raw_kps[fa][m[:, 0]].astype(np.float32)
            kp_b = raw_kps[fb][m[:, 1]].astype(np.float32)

            # Ground mask filter
            gm_a = self._ground_masks[fa]
            gm_b = self._ground_masks[fb]
            ha, wa = gm_a.shape
            hb, wb = gm_b.shape
            xa = np.clip(kp_a[:, 0].astype(int), 0, wa - 1)
            ya = np.clip(kp_a[:, 1].astype(int), 0, ha - 1)
            xb = np.clip(kp_b[:, 0].astype(int), 0, wb - 1)
            yb = np.clip(kp_b[:, 1].astype(int), 0, hb - 1)
            pts_a = kp_a[gm_a[ya, xa] & gm_b[yb, xb]]
            pts_b = kp_b[gm_a[ya, xa] & gm_b[yb, xb]]

            if len(pts_a) < fm.MIN_GROUND_MATCHES:
                continue

            # RANSAC
            if len(pts_a) >= 8:
                _, rmask = cv2.findHomography(pts_a, pts_b,
                                              cv2.RANSAC, fm.RANSAC_THRESHOLD)
                if rmask is not None:
                    keep  = rmask.ravel().astype(bool)
                    pts_a = pts_a[keep]
                    pts_b = pts_b[keep]

            if len(pts_a) >= fm.MIN_GROUND_MATCHES:
                raw_pair_matches[(fa, fb)] = (pts_a, pts_b)

        # ── Pass 2: global multi-frame-aware filter ───────────────────────────
        from pipeline.feature_matcher import _global_match_filter
        filtered = _global_match_filter(raw_pair_matches,
                                        config.MATCH_SPATIAL_DEDUP_THRESH)

        for (fa, fb), (pts_a, pts_b) in filtered.items():
            self._pair_pts[(fa, fb)] = (pts_a, pts_b)
            self._pair_pts[(fb, fa)] = (pts_b, pts_a)

            for pt in pts_a:
                accumulated[fa].append((float(pt[0]), float(pt[1])))
            for pt in pts_b:
                accumulated[fb].append((float(pt[0]), float(pt[1])))

            logger.info("    %s <-> %s : %d matches (global filter)",
                        fa.stem[-8:], fb.stem[-8:], len(pts_a))

        # ── Build deduplicated per-frame keypoint arrays ──────────────────────
        for f in self.frames:
            pts_list: list[tuple[float, float]] = []
            seen    : set[tuple[int, int]]       = set()
            for x, y in accumulated[f]:
                key = (round(x * 2), round(y * 2))   # 0.5-px dedup grid
                if key not in seen:
                    seen.add(key)
                    pts_list.append((x, y))
            self._kps[f] = (np.array(pts_list, dtype=np.float32)
                            if pts_list else np.empty((0, 2), dtype=np.float32))
            logger.debug("  %s: %d unique filtered kps",
                         f.stem[-12:], len(self._kps[f]))

    # ── Correspondence lookup ─────────────────────────────────────────────────

    def _find_correspondence(self, src_pt: np.ndarray,
                             src: Frame, dst: Frame) -> Optional[np.ndarray]:
        """
        Return the dst-frame pixel for src_pt, or None if this point has no
        surviving match in the (src, dst) pair.
        """
        key = (src, dst)
        if key not in self._pair_pts:
            return None
        pts_a, pts_b = self._pair_pts[key]
        dists = np.linalg.norm(pts_a - src_pt, axis=1)
        idx   = int(np.argmin(dists))
        return pts_b[idx] if dists[idx] < _PT_TOL else None

    # ── Cell rendering ────────────────────────────────────────────────────────

    def _make_cell(self, frame: Frame) -> np.ndarray:
        full   = frame.undistorted
        h, w   = full.shape[:2]
        sx, sy = THUMB_W / w, THUMB_H / h
        src    = self._source

        # IDLE
        if src is None:
            cell = cv2.resize(_greyed(full), (THUMB_W, THUMB_H),
                              interpolation=cv2.INTER_AREA)
            _banner(cell, frame.display_name)
            return cell

        # SOURCE
        if frame is src:
            cell = cv2.resize(full.copy(), (THUMB_W, THUMB_H),
                              interpolation=cv2.INTER_AREA)

            # Semi-transparent green overlay showing the SAM ground mask
            gm   = self._ground_masks.get(frame)
            if gm is not None:
                gm_small = cv2.resize(gm.astype(np.uint8) * 255,
                                      (THUMB_W, THUMB_H),
                                      interpolation=cv2.INTER_NEAREST)
                overlay = cell.copy()
                overlay[gm_small > 0] = (overlay[gm_small > 0] * 0.6 +
                                         np.array([255, 0, 255]) * 0.4).astype(np.uint8)
                cell = overlay

            kps  = self._kps[frame]

            for kp in kps:
                tx, ty = int(kp[0] * sx), int(kp[1] * sy)
                cv2.circle(cell, (tx, ty), _R_KP, _C_KP, cv2.FILLED, cv2.LINE_AA)

            if self._sel_kp is not None and self._sel_kp < len(kps):
                kp = kps[self._sel_kp]
                tx, ty = int(kp[0] * sx), int(kp[1] * sy)
                cv2.circle(cell, (tx, ty), _R_SEL, _C_KP_SEL, 2,          cv2.LINE_AA)
                cv2.circle(cell, (tx, ty), _R_KP,  _C_KP_SEL, cv2.FILLED, cv2.LINE_AA)

            n_kps    = len(kps)
            n_paired = sum(1 for f in self.frames
                           if f is not src and (src, f) in self._pair_pts)
            _banner(cell,
                    f"SOURCE: {frame.display_name}   "
                    f"{n_kps} kps  |  {n_paired} paired frame(s)",
                    color=(120, 230, 120))
            _clear_btn(cell)
            return cell

        # PAIRED
        if (src, frame) in self._pair_pts:
            cell = cv2.resize(full.copy(), (THUMB_W, THUMB_H),
                              interpolation=cv2.INTER_AREA)

            # Semi-transparent ground mask overlay on paired frames too
            gm = self._ground_masks.get(frame)
            if gm is not None:
                gm_small = cv2.resize(gm.astype(np.uint8) * 255,
                                      (THUMB_W, THUMB_H),
                                      interpolation=cv2.INTER_NEAREST)
                overlay = cell.copy()
                overlay[gm_small > 0] = (overlay[gm_small > 0] * 0.6 +
                                         np.array([255, 0, 255]) * 0.4).astype(np.uint8)
                cell = overlay

            # Draw all filtered keypoints in this frame as small cyan dots
            # (non-clickable; gives spatial context when tracing a source point)
            for kp in self._kps[frame]:
                tx, ty = int(kp[0] * sx), int(kp[1] * sy)
                cv2.circle(cell, (tx, ty), _R_KP, _C_KP, cv2.FILLED, cv2.LINE_AA)

            # Overlay the selected correspondence in orange on top
            if self._sel_kp is not None and self._sel_kp < len(self._kps[src]):
                sel_pt = self._kps[src][self._sel_kp]
                corr   = self._find_correspondence(sel_pt, src, frame)
                if corr is not None:
                    tx, ty = int(corr[0] * sx), int(corr[1] * sy)
                    cv2.circle(cell, (tx, ty), _R_SEL, _C_MATCH, 2,          cv2.LINE_AA)
                    cv2.circle(cell, (tx, ty), _R_KP,  _C_MATCH, cv2.FILLED, cv2.LINE_AA)

            n_matches = len(self._pair_pts[(src, frame)][0])
            _banner(cell,
                    f"{frame.display_name}   {n_matches} matches with source",
                    color=_C_PAIRED_TXT)
            return cell

        # UNMATCHED
        cell = cv2.resize(_greyed(full), (THUMB_W, THUMB_H),
                          interpolation=cv2.INTER_AREA)
        _banner(cell, f"{frame.display_name}   (no matches with source)")
        return cell

    # ── Grid ──────────────────────────────────────────────────────────────────

    def _rebuild_grid(self) -> np.ndarray:
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
                     (55, 55, 55), 1)
        for c in range(1, self._n_cols):
            cv2.line(grid, (c * THUMB_W, 0), (c * THUMB_W, grid.shape[0]),
                     (55, 55, 55), 1)
        return grid

    def _with_status_bar(self, grid: np.ndarray) -> np.ndarray:
        bar = np.zeros((28, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, self._status, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 215, 150), 1, cv2.LINE_AA)
        return np.vstack([grid, bar])

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _on_click(self, event, gx: int, gy: int, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        col = gx // THUMB_W
        row = gy // THUMB_H
        idx = row * self._n_cols + col
        if idx >= len(self.frames):
            return

        clicked = self.frames[idx]
        tx = gx % THUMB_W
        ty = gy % THUMB_H

        # [Clear] button
        if (clicked is self._source
                and _BTN_X1 <= tx <= _BTN_X2
                and _BTN_Y1 <= ty <= _BTN_Y2):
            self._source = None
            self._sel_kp = None
            self._status = "Cleared.  Click any frame to select a new source."
            self._grid   = self._rebuild_grid()
            return

        # IDLE -> select source
        if self._source is None:
            self._source = clicked
            self._sel_kp = None
            self._grid   = self._rebuild_grid()
            self._status = self._source_status()
            return

        # Click in source -> snap to nearest filtered keypoint
        if clicked is self._source:
            kps = self._kps[self._source]
            if len(kps) == 0:
                self._status = "No filtered keypoints in this frame."
                return

            h, w = self._source.undistorted.shape[:2]
            px = tx * w / THUMB_W
            py = ty * h / THUMB_H

            dists   = np.linalg.norm(kps - [px, py], axis=1)
            nearest = int(np.argmin(dists))

            if dists[nearest] > _SNAP_R * w / THUMB_W:
                self._status = "Click closer to a cyan dot to select a keypoint."
                return

            self._sel_kp = nearest
            sel_pt       = kps[nearest]
            n_frames = sum(
                1 for f in self.frames
                if f is not self._source
                and self._find_correspondence(sel_pt, self._source, f) is not None
            )
            self._status = (
                f"Keypoint ({sel_pt[0]:.0f}, {sel_pt[1]:.0f}) px  ->  "
                f"matched in {n_frames} other frame(s).   "
                f"Click another dot or [Clear]."
            )
            self._grid = self._rebuild_grid()
            return

        # Click on a paired frame -> re-source it
        if (self._source, clicked) in self._pair_pts:
            self._source = clicked
            self._sel_kp = None
            self._grid   = self._rebuild_grid()
            self._status = self._source_status()
            return

        # Click on a greyed (unmatched) frame while source active: ignore

    def _source_status(self) -> str:
        src      = self._source
        n_kps    = len(self._kps[src])
        n_paired = sum(1 for f in self.frames
                       if f is not src and (src, f) in self._pair_pts)
        return (
            f"Source: '{src.display_name}'   "
            f"{n_kps} filtered kps  |  {n_paired} paired frame(s).   "
            f"Click a cyan dot to trace it."
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW,
                         self._grid.shape[1],
                         self._grid.shape[0] + 28)
        cv2.setMouseCallback(self.WINDOW, self._on_click)

        while True:
            cv2.imshow(self.WINDOW, self._with_status_bar(self._grid))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break

        cv2.destroyAllWindows()
