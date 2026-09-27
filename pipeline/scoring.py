"""
pipeline/scoring.py – Reprojection accuracy measures (no GUI).

Shared by the interactive viewer (pipeline/ui.py, right-click truth scoring →
score.csv) and the headless track cross-scoring (raycast.py --score-tracks).

Error measures
──────────────
  err_px      – image pixels.  Not comparable across frames: a far, zoomed-out
                frame shows the same miss as far fewer pixels.
  err_m_view  – err_px × gsd (metres per pixel at the point's depth): the miss
                in metres, measured facing the camera.
  err_m       – ground distance between the truth click's terrain hit and the
                reprojected ground point.  Stretched by ~1 / sin(view_elev_deg)
                at shallow view angles.

Track cross-scoring
───────────────────
Every pick in score.csv is one physical feature clicked in several frames
(left-click source + right-click truths).  All clicks are human observations,
so a pick is a *track*: for every ordered pair A≠B we cast A's click onto the
ground, reproject into B and compare with B's click.  Clicks are image pixels,
independent of the solve, so the same tracks can score any solve file.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import numpy as np

from pipeline.geometry import ground_sample_distance, pixel_to_ground, project_to_frame
import config

logger = logging.getLogger(__name__)

_SCORE_TARGET_PX = 10.0

# CSV column → decimals for the metric fields produced by _metric_error
_METRIC_DECIMALS = {"err_m": 4, "err_m_view": 4, "gsd_m_per_px": 5,
                    "range_m": 2, "view_elev_deg": 1}

PAIR_FIELDS = [
    "pick_id", "solve_file", "source_frame", "src_x", "src_y", "target_frame",
    "tgt_x", "tgt_y", "proj_x", "proj_y", "err_px", "err_m", "err_m_view",
    "gsd_m_per_px", "range_m", "view_elev_deg", "status",
]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metric_error(world_pt, frame, gx: float | None = None, gy: float | None = None,
                  err_px: float | None = None, surface=None) -> dict:
    """Metric error measures for one target frame (any value may be None).

    Always (given *world_pt*): gsd_m_per_px, range_m, view_elev_deg of the pick
    point as seen by *frame*.  With a truth click (gx, gy): err_m, the ground
    distance between the truth ray's terrain hit and *world_pt*.  With
    *err_px*: err_m_view = err_px × gsd.
    """
    out = dict.fromkeys(_METRIC_DECIMALS)
    if world_pt is None:
        return out
    v = world_pt - frame.position_enu
    rng = float(np.linalg.norm(v))
    out["range_m"] = rng
    out["view_elev_deg"] = math.degrees(math.asin(-v[2] / rng)) if rng > 0 else None
    gsd = ground_sample_distance(world_pt, frame)
    out["gsd_m_per_px"] = gsd
    if err_px is not None and gsd is not None:
        out["err_m_view"] = err_px * gsd
    if gx is not None and gy is not None:
        gt_world = pixel_to_ground(gx, gy, frame, surface)
        if gt_world is not None:
            out["err_m"] = float(np.linalg.norm(gt_world - world_pt))
    return out


def _fmt(v, nd: int = 4) -> str:
    return "" if v is None else f"{v:.{nd}f}"


def _fmt_metric(m: dict) -> dict:
    """_metric_error output → CSV strings."""
    return {k: _fmt(m.get(k), nd) for k, nd in _METRIC_DECIMALS.items()}


def _stats_block(title: str, a: np.ndarray, unit: str, nd: int, target: float,
                 n_judged: int) -> list[str]:
    if not len(a):
        return [f"  {title}: no values"]
    ok = a <= target
    lines = [
        f"  {title}:",
        f"    median {np.median(a):.{nd}f}   mean {a.mean():.{nd}f}   "
        f"RMS {np.sqrt((a ** 2).mean()):.{nd}f}   P90 {np.percentile(a, 90):.{nd}f}   "
        f"max {a.max():.{nd}f}  ({unit})",
        f"    within {target:g} {unit}: {100.0 * ok.mean():.0f}%  ({int(ok.sum())}/{len(a)})",
    ]
    if len(a) < n_judged:
        lines.append(f"    ({n_judged - len(a)} row(s) without a value - ray missed, "
                     f"point behind camera, or scored under another solve file)")
    return lines


def _unbounded_projection(world_pt, frame):
    """project_to_frame without the image-bounds check (None if behind camera)."""
    p_cam = frame.R @ (world_pt - frame.position_enu)
    if p_cam[2] <= 0:
        return None
    K = frame.K_undist
    return (float(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]),
            float(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]))


def _short(stem: str) -> str:
    return stem.rsplit("_", 1)[-1]


def _num(r: dict, k: str):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else None


# ─────────────────────────────────────────────────────────────────────────────
# Track cross-scoring
# ─────────────────────────────────────────────────────────────────────────────

def load_tracks(score_csv: str | Path) -> dict[str, dict[str, tuple[float, float]]]:
    """score.csv → {pick_id: {frame_stem: (x, y)}} (source click + truth clicks).

    Tracks with fewer than 2 clicked frames are dropped.
    """
    tracks: dict[str, dict[str, tuple[float, float]]] = {}
    with open(score_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t = tracks.setdefault(r["pick_id"], {})
            t[r["source_frame"]] = (float(r["src_x"]), float(r["src_y"]))
            if r.get("gt_x") and r.get("gt_y"):
                t[r["target_frame"]] = (float(r["gt_x"]), float(r["gt_y"]))
    kept = {pid: t for pid, t in tracks.items() if len(t) >= 2}
    logger.info("Loaded %d track(s) from %s (%d dropped with <2 points).",
                len(kept), score_csv, len(tracks) - len(kept))
    return kept


def score_tracks(tracks: dict, frames: list, surface=None, solve_file: str = "") -> list[dict]:
    """Every ordered pair A≠B of every track → one row per PAIR_FIELDS."""
    by_stem = {f.stem: f for f in frames if f.ready}
    rows: list[dict] = []
    n_unknown = 0
    for pid, track in tracks.items():
        stems = [s for s in track if s in by_stem]
        n_unknown += len(track) - len(stems)
        for sa in stems:
            A, a_xy = by_stem[sa], track[sa]
            world = pixel_to_ground(*a_xy, A, surface)
            for sb in stems:
                if sb == sa:
                    continue
                B, b_xy = by_stem[sb], track[sb]
                row = {k: "" for k in PAIR_FIELDS}
                row.update(pick_id=pid, solve_file=solve_file,
                           source_frame=sa, src_x=f"{a_xy[0]:.2f}", src_y=f"{a_xy[1]:.2f}",
                           target_frame=sb, tgt_x=f"{b_xy[0]:.2f}", tgt_y=f"{b_xy[1]:.2f}")
                if world is None:
                    row["status"] = "ray_miss"
                    rows.append(row)
                    continue
                proj = project_to_frame(world, B.K_undist, B.R, B.position_enu,
                                        B.undistorted.shape[:2])
                if proj is not None:
                    err_px = math.hypot(b_xy[0] - proj[0], b_xy[1] - proj[1])
                    row.update(proj_x=f"{proj[0]:.2f}", proj_y=f"{proj[1]:.2f}",
                               err_px=f"{err_px:.2f}", status="ok")
                else:
                    unb = _unbounded_projection(world, B)
                    err_px = (math.hypot(b_xy[0] - unb[0], b_xy[1] - unb[1])
                              if unb is not None else None)
                    row["status"] = "out_of_frame"
                m = _metric_error(world, B, *b_xy, err_px=err_px, surface=surface)
                row.update(_fmt_metric(m))
                rows.append(row)
    if n_unknown:
        logger.warning("score_tracks: %d click(s) reference frames not loaded/ready.", n_unknown)
    logger.info("score_tracks: %d pair(s) from %d track(s).", len(rows), len(tracks))
    return rows


def write_pairs_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _group_table(rows: list[dict], key: str, title: str, tgt_m: float) -> list[str]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)

    def _vals(rs, col):
        return [v for v in (_num(r, col) for r in rs) if v is not None]

    def _med(rs, col, nd=2):
        v = _vals(rs, col)
        return f"{np.median(v):.{nd}f}" if v else "-"

    def _sort_key(item):
        v = _vals(item[1], "err_m_view")
        return np.median(v) if v else np.inf

    lines = [
        f"  {title} (medians; sorted best -> worst by view m):",
        f"    {'frame':<34} {'view deg':>8} {'px':>7} {'view m':>7} {'ground m':>9} "
        f"{'%view<=' + format(tgt_m, 'g'):>10} {'pairs':>6}",
    ]
    for stem, rs in sorted(groups.items(), key=_sort_key):
        v = np.asarray(_vals(rs, "err_m_view"))
        pct = f"{100.0 * (v <= tgt_m).mean():.0f}%" if len(v) else "-"
        lines.append(f"    {stem:<34} {_med(rs, 'view_elev_deg', 1):>8} "
                     f"{_med(rs, 'err_px', 1):>7} {_med(rs, 'err_m_view'):>7} "
                     f"{_med(rs, 'err_m'):>9} {pct:>10} {len(rs):>6}")
    return lines


def summarize_pairs(rows: list[dict], title: str = "") -> str:
    """Text report: status counts, overall stats, per-source / per-target tables."""
    counts = {s: sum(r["status"] == s for r in rows) for s in ("ok", "out_of_frame", "ray_miss")}
    n_tracks = len({r["pick_id"] for r in rows})
    tgt_m = float(config.SCORE_TARGET_M)
    scored = [r for r in rows if r["status"] != "ray_miss"]

    def _arr(col, rs=scored):
        return np.asarray([v for v in (_num(r, col) for r in rs) if v is not None])

    lines = [
        f"Track cross-reprojection accuracy{(' - ' + title) if title else ''}",
        f"  tracks: {n_tracks}   pairs: {len(rows)}   ok: {counts['ok']}   "
        f"out_of_frame: {counts['out_of_frame']}   ray_miss: {counts['ray_miss']}",
    ]
    if not scored:
        lines.append("  no scorable pairs.")
        return "\n".join(lines) + "\n"

    view = _arr("err_m_view")
    per_src = {}
    for r in scored:
        v = _num(r, "err_m_view")
        if v is not None:
            per_src.setdefault(r["source_frame"], []).append(v)
    src_meds = [np.median(v) for v in per_src.values()]
    lines += [
        "  OVERALL SCORE (view error, m):",
        f"    median over all pairs: {np.median(view):.2f} m   "
        f"mean of per-source medians: {np.mean(src_meds):.2f} m   "
        f"({len(per_src)} source frames)",
    ]
    n = len(scored)
    lines += _stats_block("image error (px; ok pairs only - favours far / zoomed-out frames)",
                          _arr("err_px"), "px", 1, _SCORE_TARGET_PX, counts["ok"])
    lines += _stats_block("view error (px x m/px: miss in metres facing the camera)",
                          view, "m", 2, tgt_m, n)
    lines += _stats_block("ground error (distance on the terrain; stretched at shallow views)",
                          _arr("err_m"), "m", 2, tgt_m, n)
    lines += _group_table(scored, "source_frame", "per SOURCE frame (where you pick)", tgt_m)
    lines += _group_table(scored, "target_frame", "per TARGET frame (where it lands)", tgt_m)
    lines.append("    out_of_frame pairs are included: the feature was clicked in the target "
                 "but the solve projects it outside the image.")
    return "\n".join(lines) + "\n"


def plot_pair_matrix(rows: list[dict], out_png: str | Path, title: str = "") -> Path | None:
    """Heatmap: source frame (rows) × target frame (cols), median view error (m)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        v = _num(r, "err_m_view")
        if v is not None:
            cells.setdefault((r["source_frame"], r["target_frame"]), []).append(v)
    if not cells:
        logger.warning("plot_pair_matrix: nothing to plot.")
        return None

    stems = sorted({s for s, _ in cells} | {t for _, t in cells})
    idx = {s: i for i, s in enumerate(stems)}
    n = len(stems)
    M = np.full((n + 1, n + 1), np.nan)          # last row/col = medians (margins)
    for (s, t), v in cells.items():
        M[idx[s], idx[t]] = np.median(v)
    row_all: dict[str, list[float]] = {}
    col_all: dict[str, list[float]] = {}
    for (s, t), v in cells.items():
        row_all.setdefault(s, []).extend(v)
        col_all.setdefault(t, []).extend(v)
    for s, v in row_all.items():
        M[idx[s], n] = np.median(v)
    for t, v in col_all.items():
        M[n, idx[t]] = np.median(v)
    M[n, n] = np.median([x for v in cells.values() for x in v])

    labels = [_short(s) for s in stems] + ["median"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * (n + 1), 0.8 + 0.55 * (n + 1)))
    vmax = np.nanpercentile(M, 95)
    im = ax.imshow(np.ma.masked_invalid(M), cmap="RdYlGn_r", vmin=0, vmax=vmax)
    ax.set_xticks(range(n + 1), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n + 1), labels, fontsize=8)
    ax.set_xlabel("target frame (where it lands)")
    ax.set_ylabel("source frame (where you pick)")
    ax.axhline(n - 0.5, color="black", lw=1.5)
    ax.axvline(n - 0.5, color="black", lw=1.5)
    for i in range(n + 1):
        for j in range(n + 1):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        fontweight="bold" if (i == n or j == n) else "normal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="median view error (m) = px x m/px")
    ax.set_title(f"Reprojection error by source/target frame{(' - ' + title) if title else ''}",
                 fontsize=10)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    logger.info("Pair matrix written: %s", out_png)
    return out_png
