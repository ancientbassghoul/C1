"""Reprojection scoring: scale-independent error measures + track cross-scoring.

Run:  venv\\Scripts\\python -m pytest tests/test_scoring.py
  or  venv\\Scripts\\python tests/test_scoring.py      (no pytest needed)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.geometry import ground_sample_distance, pixel_to_ground, project_to_frame
from pipeline.scoring import _metric_error, load_tracks, score_tracks

# Nadir camera (OpenCV: X right, Y down, Z forward): cam X = East,
# cam Y = South, cam Z = Down.
_R_NADIR = np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0]])
_K = np.array([[900.0, 0.0, 640.0],
               [0.0, 900.0, 360.0],
               [0.0, 0.0, 1.0]])
_SHAPE = (720, 1280)


def _stub(stem: str, R, position_enu):
    return SimpleNamespace(stem=stem, K_undist=_K, R=R, position_enu=position_enu,
                           ready=True, undistorted=SimpleNamespace(shape=(*_SHAPE, 3)))


def _nadir_frame(altitude_m: float, stem: str):
    return _stub(stem, _R_NADIR, np.array([10.0, -20.0, altitude_m]))


def _oblique_frame(elev_deg: float, height_m: float, target, stem: str):
    """Camera looking north and *elev_deg* down at *target* from *height_m*."""
    elev = np.radians(elev_deg)
    fwd = np.array([0.0, np.cos(elev), -np.sin(elev)])          # cam Z
    right = np.array([1.0, 0.0, 0.0])                            # cam X
    down = np.cross(fwd, right)                                  # cam Y
    return _stub(stem, np.vstack([right, down, fwd]),
                 np.asarray(target) - fwd * (height_m / np.sin(elev)))


def _project(frame, pt):
    return project_to_frame(pt, frame.K_undist, frame.R, frame.position_enu, _SHAPE)


def test_scale_independent_ground_error():
    pick_world = np.array([10.0, -20.0, 0.0])
    truth_world = pick_world + np.array([0.5, 0.0, 0.0])    # 0.5 m ground miss

    err_px, err_m, gsd = {}, {}, {}
    for alt in (20.0, 80.0):
        f = _nadir_frame(alt, f"alt{alt:.0f}")
        proj = _project(f, pick_world)
        gt = _project(f, truth_world)
        assert proj is not None and gt is not None

        # pixel_to_ground round-trips the projected pixel back to the ground point
        np.testing.assert_allclose(pixel_to_ground(*proj, f), pick_world, atol=1e-6)

        err_px[alt] = float(np.hypot(gt[0] - proj[0], gt[1] - proj[1]))
        m = _metric_error(pick_world, f, *gt, err_px=err_px[alt])
        err_m[alt], gsd[alt] = m["err_m"], m["gsd_m_per_px"]

        # GSD × pixel error == ground error (nadir, flat ground)
        assert np.isclose(gsd[alt], ground_sample_distance(pick_world, f))
        assert np.isclose(m["err_m_view"], 0.5, atol=1e-6)
        assert np.isclose(m["view_elev_deg"], 90.0)
        assert np.isclose(m["range_m"], alt)

    # Same 0.5 m miss: pixel error differs 4×, ground error doesn't.
    assert np.isclose(err_px[20.0] / err_px[80.0], 4.0, rtol=1e-6)
    assert np.isclose(err_m[20.0], 0.5, atol=1e-6)
    assert np.isclose(err_m[80.0], 0.5, atol=1e-6)


def test_oblique_view_stretches_ground_error():
    # Camera 20 m up, looking north and 30° down at the pick point.
    elev = np.radians(30.0)
    pick_world = np.array([0.0, 0.0, 0.0])
    f = _oblique_frame(30.0, 20.0, pick_world, "oblique")

    proj = _project(f, pick_world)
    assert np.allclose(proj, (640.0, 360.0))
    gt = (proj[0], proj[1] - 3.0)          # 3 px straight up = further along the ground
    err_px = 3.0
    m = _metric_error(pick_world, f, *gt, err_px=err_px)

    assert np.isclose(m["view_elev_deg"], 30.0)
    # For a small vertical miss, ground distance ≈ view error / sin(elevation).
    assert np.isclose(m["err_m"], m["err_m_view"] / np.sin(elev), rtol=0.02)


def test_metric_error_without_pick_world():
    f = _nadir_frame(20.0, "alt20")
    assert all(v is None for v in _metric_error(None, f, 640.0, 360.0).values())


def _track_frames(point):
    return [_nadir_frame(20.0, "f_a"), _nadir_frame(80.0, "f_b"),
            _oblique_frame(30.0, 20.0, point, "f_c")]


def test_track_pairs_perfect_and_offset():
    P = np.array([10.0, -20.0, 0.0])
    frames = _track_frames(P)
    track = {f.stem: _project(f, P) for f in frames}
    assert all(v is not None for v in track.values())

    rows = score_tracks({"1": track}, frames)
    assert len(rows) == 6
    assert all(r["status"] == "ok" for r in rows)
    assert all(float(r["err_px"]) < 1e-2 and float(r["err_m"]) < 1e-3 for r in rows)

    # Offset f_b's click by 5 px → only pairs involving f_b become non-zero.
    track["f_b"] = (track["f_b"][0] + 5.0, track["f_b"][1])
    rows = score_tracks({"1": track}, frames)
    for r in rows:
        involves_b = "f_b" in (r["source_frame"], r["target_frame"])
        assert (float(r["err_m"]) > 0.05) == involves_b, r
        if r["target_frame"] == "f_b":
            assert abs(float(r["err_px"]) - 5.0) < 1e-2


def test_track_out_of_frame():
    P = np.array([10.0, -20.0, 0.0])
    fa = _nadir_frame(20.0, "f_a")
    fb = _nadir_frame(20.0, "f_b")
    fb.position_enu = np.array([40.0, -20.0, 20.0])      # P is 30 m west: off-image
    track = {"f_a": _project(fa, P), "f_b": (100.0, 360.0)}   # human saw it at the left edge
    rows = score_tracks({"1": track}, [fa, fb])
    r = next(r for r in rows if r["target_frame"] == "f_b")
    assert r["status"] == "out_of_frame"
    assert r["err_px"] == "" and r["proj_x"] == ""
    assert r["err_m"] != "" and r["err_m_view"] != ""


def test_load_tracks(tmp_path=None):
    import csv, tempfile
    d = Path(tmp_path or tempfile.mkdtemp())
    p = d / "score.csv"
    fields = ["pick_id", "source_frame", "src_x", "src_y", "target_frame",
              "gt_x", "gt_y", "status"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(dict(pick_id=1, source_frame="a", src_x=1, src_y=2, target_frame="b",
                        gt_x=3, gt_y=4, status="judged"))
        w.writerow(dict(pick_id=1, source_frame="a", src_x=1, src_y=2, target_frame="c",
                        gt_x="", gt_y="", status="skipped"))
        w.writerow(dict(pick_id=1, source_frame="a", src_x=1, src_y=2, target_frame="d",
                        gt_x=5, gt_y=6, status="missed"))
        w.writerow(dict(pick_id=2, source_frame="a", src_x=1, src_y=2, target_frame="b",
                        gt_x="", gt_y="", status="skipped"))
    tracks = load_tracks(p)
    assert tracks == {"1": {"a": (1.0, 2.0), "b": (3.0, 4.0), "d": (5.0, 6.0)}}


if __name__ == "__main__":
    test_scale_independent_ground_error()
    test_oblique_view_stretches_ground_error()
    test_metric_error_without_pick_world()
    test_track_pairs_perfect_and_offset()
    test_track_out_of_frame()
    test_load_tracks()
    print("OK")
