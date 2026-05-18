"""
pipeline/pitch_from_geocalib.py – Batch pitch + roll estimation using GeoCalib.

GeoCalib (https://github.com/cvg/GeoCalib, ECCV 2024) estimates camera tilt
from a single image using learned line and vanishing-point detection.

We run it in batch across all frames with shared_intrinsics=True, which
constrains all frames to share the same focal length — correct since all
frames come from the same physical camera.  Each frame gets its own
independent gravity estimate (and therefore its own pitch and roll).

The results seed the Ceres orientation solver: pitch is used as the starting
value and to set a tight per-frame search window; roll is used as a fallback
for frames where HUD bracket detection fails.

Install:
  venv\\Scripts\\pip install geocalib
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

_model        = None
_model_device = None


def _get_model():
    """Lazy-load GeoCalib model (weights downloaded on first use, ~111 MB)."""
    global _model, _model_device
    if _model is not None:
        return _model, _model_device
    try:
        import torch
        from geocalib import GeoCalib
    except ImportError:
        raise ImportError(
            "\nGeoCalib not installed. Run:\n"
            "  venv\\Scripts\\pip install geocalib\n"
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Loading GeoCalib model on %s…", device)
    model = GeoCalib().to(device).eval()
    _model, _model_device = model, device
    logger.info("GeoCalib ready.")
    return model, device


def _tensor_to_float(val) -> float:
    """Safely convert a possibly-batched tensor or float to a Python float."""
    import torch
    if torch.is_tensor(val):
        return float(val.detach().cpu().squeeze())
    return float(val)


def calibrate_all_frames(frames: list) -> dict:
    """
    Run GeoCalib on all frames in one batch call.

    Uses shared_intrinsics=True so all frames share a single focal length
    estimate — correct since all frames come from the same camera body.
    Each frame receives its own independent gravity estimate.

    The gravity object returned by GeoCalib exposes .pitch and .roll
    directly in radians; no manual trigonometry needed.

    Returns a dict mapping Frame → (pitch_deg, roll_deg).
    Frames without an undistorted image are silently skipped.
    """
    eligible = [f for f in frames if f.undistorted is not None]
    if not eligible:
        return {}

    try:
        import torch

        model, device = _get_model()

        # Build [C, H, W] float tensors in [0, 1] — replicating model.load_image()
        # Our images are BGR (OpenCV); convert to RGB before making the tensor.
        tensors = []
        for f in eligible:
            rgb    = f.undistorted[:, :, ::-1].copy().astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
            tensors.append(tensor)

        # Stack list of [C, H, W] tensors → single (N, C, H, W) batch tensor
        batch = torch.stack(tensors, dim=0)

        with torch.no_grad():
            result = model.calibrate(batch, shared_intrinsics=True)

        # result["gravity"] is indexable — gravities[i] is the per-frame
        # gravity object with .pitch and .roll attributes (radians).
        gravities = result['gravity']

        results: dict = {}
        for i, f in enumerate(eligible):
            g         = gravities[i]
            pitch_deg = math.degrees(_tensor_to_float(g.pitch))
            roll_deg  = math.degrees(_tensor_to_float(g.roll))
            results[f] = (pitch_deg, roll_deg)
            logger.info(
                "[%s] GeoCalib: pitch=%.1f°  roll=%.1f°",
                f.stem, pitch_deg, roll_deg,
            )

        return results

    except Exception as exc:
        logger.warning(
            "GeoCalib batch failed: %s — all frames use 0° pitch seed.", exc
        )
        return {}
