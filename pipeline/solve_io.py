"""
pipeline/solve_io.py – Serialize / deserialize solved camera state.

Used for the RunPod headless workflow:
  Pod  : --export-solve  →  solved_cameras.json
  Local: --import-solve  →  skip pipeline, open GUI with loaded poses
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def export_solve(frames: list, path: str | Path) -> None:
    """Write solved camera state for all ready frames to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ready = [f for f in frames if f.ready]
    if not ready:
        logger.warning("export_solve: no ready frames — JSON will be empty.")

    cameras = {}
    for f in ready:
        cameras[f.stem] = {
            "position_enu":    f.position_enu.tolist(),
            "heading_deg":     float(f.heading_deg),
            "gimbal_pitch_deg": float(f.gimbal_pitch_deg),
            "camera_roll_deg": float(f.camera_roll_deg),
            "K_undist":        f.K_undist.tolist(),
        }

    payload = {
        "metadata": {
            "created":     datetime.now(timezone.utc).isoformat(),
            "frame_count": len(cameras),
        },
        "cameras": cameras,
    }

    path.write_text(json.dumps(payload, indent=2))
    logger.info("Solved cameras written to %s  (%d frame(s))", path, len(cameras))


def import_solve(frames: list, path: str | Path) -> int:
    """
    Load solved camera state from JSON and apply it to matching frames (by stem).

    Requires frames to already have `undistorted` and `K_undist` set
    (i.e. load_frames + undistort_all must have run first).

    Returns the number of frames successfully updated.
    """
    from pipeline.pose import build_rotation

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Solve file not found: {path}")

    data = json.loads(path.read_text())
    cameras = data.get("cameras", {})

    stem_to_frame = {f.stem: f for f in frames}
    n_matched = 0

    for stem, cam in cameras.items():
        f = stem_to_frame.get(stem)
        if f is None:
            logger.warning("import_solve: stem %r in JSON but not in loaded frames — skipped", stem)
            continue

        f.position_enu    = np.array(cam["position_enu"],  dtype=np.float64)
        f.heading_deg     = float(cam["heading_deg"])
        f.gimbal_pitch_deg = float(cam["gimbal_pitch_deg"])
        f.camera_roll_deg = float(cam["camera_roll_deg"])
        f.K_undist        = np.array(cam["K_undist"],      dtype=np.float64)
        f.R               = build_rotation(f.heading_deg, f.gimbal_pitch_deg, f.camera_roll_deg)
        n_matched += 1

    unmatched_local = [f.stem for f in frames if f.stem not in cameras]
    if unmatched_local:
        logger.warning(
            "import_solve: %d local frame(s) have no entry in JSON: %s",
            len(unmatched_local), unmatched_local,
        )

    return n_matched
