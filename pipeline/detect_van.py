"""
pipeline/detect_van.py – Van detection for keypoint-masking during pitch refinement.

Provides VanDetector, which locates the white transit van in each undistorted
frame and returns its bounding box.  The bbox is passed to pipeline/refine.py
so that SuperPoint keypoints falling on the van are excluded from ground-scatter
matching — the van sits above the ground plane and would otherwise corrupt the
pitch estimate.

Detection strategy
──────────────────
1. GroundingDINO (primary) — open-vocabulary detector prompted with
   "white van . delivery van . cargo van".  Robust to viewpoint and scale
   changes across frames.  Weights (~700 MB) are downloaded on first use.

2. White-blob fallback — if GroundingDINO returns nothing above the confidence
   threshold, the frame is searched for the largest pure-white connected
   component with a plausible aspect ratio in the lower 60 % of the image.
   Fast and parameter-free; works reliably because the van is always the
   brightest large object in the scene.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from pipeline.frame import Frame
import config

logger = logging.getLogger(__name__)

DINO_PROMPT = "white van . delivery van . cargo van"


# ─────────────────────────────────────────────────────────────────────────────
# GroundingDINO loader (lazy)
# ─────────────────────────────────────────────────────────────────────────────

_dino_model = None


def _load_dino():
    global _dino_model
    if _dino_model is not None:
        return _dino_model
    try:
        from groundingdino.util.inference import load_model
        import groundingdino
        import os
        pkg_dir = os.path.dirname(groundingdino.__file__)
        config_path = os.path.join(pkg_dir, "config", "GroundingDINO_SwinT_OGC.py")
        weights = "groundingdino_swint_ogc.pth"
        if not os.path.exists(weights):
            logger.info("Downloading GroundingDINO weights (~700 MB)…")
            import urllib.request
            url = (
                "https://github.com/IDEA-Research/GroundingDINO/"
                "releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            )
            urllib.request.urlretrieve(url, weights)
        _dino_model = load_model(config_path, weights)
        return _dino_model
    except ImportError:
        raise ImportError(
            "\nGroundingDINO not installed. See Setup in README.md.\n"
        )


def _detect_dino(
    img_bgr: np.ndarray,
    conf_thresh: float,
) -> Optional[tuple[float, float, float, float]]:
    """Run GroundingDINO on an OpenCV BGR image; return best bbox or None."""
    import torch
    from PIL import Image
    from groundingdino.util.inference import predict
    import groundingdino.datasets.transforms as T

    model = _load_dino()

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    image_tensor, _ = transform(pil_img, None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    boxes, logits, _ = predict(
        model=model,
        image=image_tensor,
        caption=DINO_PROMPT,
        box_threshold=conf_thresh,
        text_threshold=0.25,
        device=device,
    )

    if len(boxes) == 0:
        return None

    h, w = img_bgr.shape[:2]
    best_conf, best_bbox = 0.0, None

    for box, logit in zip(boxes, logits):
        conf = float(logit)
        if conf <= best_conf:
            continue
        cx, cy, bw, bh = box.tolist()
        # Skip detections that cover more than half the frame — likely false positives
        if bw * bh > 0.50:
            continue
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        best_conf, best_bbox = conf, (x1, y1, x2, y2)

    return best_bbox


# ─────────────────────────────────────────────────────────────────────────────
# White-blob fallback
# ─────────────────────────────────────────────────────────────────────────────

def _detect_white_blob(
    img: np.ndarray,
) -> Optional[tuple[float, float, float, float]]:
    """
    Fallback van detector based on colour: the van is the largest pure-white
    object in the scene outside the HUD zones.

    Strategy
    ────────
    1. Convert to HSV and threshold for high-value, low-saturation pixels.
    2. Mask out the sky (top 35 % of image) and HUD strips (top/bottom 12 %).
    3. Find connected components; return the bounding box of the largest blob
       with a plausible van aspect ratio (width/height between 0.4 and 3.0)
       whose centre is in the lower 60 % of the image.

    Returns (x1, y1, x2, y2) or None.
    """
    h, w = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,
                       np.array([0,   0, 200], dtype=np.uint8),
                       np.array([180, 60, 255], dtype=np.uint8))

    mask[:int(h * 0.35), :] = 0
    mask[:int(h * 0.12), :] = 0
    mask[int(h * 0.88):, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_area = config.VAN_BLOB_MIN_AREA
    best_bbox = None

    for i in range(1, n):
        x    = int(stats[i, cv2.CC_STAT_LEFT])
        y    = int(stats[i, cv2.CC_STAT_TOP])
        bw_  = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_  = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < best_area:
            continue
        if bw_ > w * 0.25:
            continue
        aspect = bw_ / bh_ if bh_ > 0 else 0
        if not (0.4 <= aspect <= 3.0):
            continue
        if (y + bh_ / 2) < h * 0.40:
            continue

        best_area = area
        best_bbox = (float(x), float(y), float(x + bw_), float(y + bh_))

    return best_bbox


# ─────────────────────────────────────────────────────────────────────────────
# VanDetector
# ─────────────────────────────────────────────────────────────────────────────

class VanDetector:
    """
    Detects the white transit van in undistorted drone frames.

    Primary:  GroundingDINO (open-vocabulary, text-prompted).
    Fallback: white-blob colour segmentation.
    """

    def __init__(self, conf_thresh: float = config.YOLO_CONF_THRESH) -> None:
        self._conf_thresh = conf_thresh

    def detect(
        self, frame: Frame,
    ) -> Optional[tuple[float, float, float, float]]:
        """
        Detect the van in frame.undistorted.

        Tries GroundingDINO first; falls back to white-blob if nothing is found
        above the confidence threshold.  Returns (x1, y1, x2, y2) or None.
        """
        if frame.undistorted is None:
            return None

        # ── GroundingDINO ──────────────────────────────────────────────────────
        try:
            bbox = _detect_dino(frame.undistorted, self._conf_thresh)
            if bbox is not None:
                logger.info(
                    "[%s] Van detected via GroundingDINO  bbox=(%.0f,%.0f → %.0f,%.0f)",
                    frame.stem, *bbox,
                )
                return bbox
        except Exception as exc:
            logger.warning("[%s] GroundingDINO failed (%s) — trying fallback.", frame.stem, exc)

        # ── White-blob fallback ────────────────────────────────────────────────
        blob = _detect_white_blob(frame.undistorted)
        if blob is not None:
            logger.info(
                "[%s] Van detected via white-blob  bbox=(%.0f,%.0f → %.0f,%.0f)",
                frame.stem, *blob,
            )
        else:
            logger.warning("[%s] Van not detected by either method.", frame.stem)
        return blob

    def detect_all(
        self, frames: list[Frame],
    ) -> dict[Frame, tuple[float, float, float, float]]:
        """Run detect() on every frame. Returns a Frame → bbox dict for hits only."""
        detections: dict = {}
        for i, frame in enumerate(frames):
            logger.info("Van detection %d/%d: %s", i + 1, len(frames), frame.stem)
            bbox = self.detect(frame)
            if bbox is not None:
                detections[frame] = bbox
        logger.info("Van detected in %d/%d frame(s).", len(detections), len(frames))
        return detections
