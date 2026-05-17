"""
mask_van.py  -  Undistort frames, detect van, paint black rectangle, save.

Detection: YOLOv8s (vehicle classes) with a white-blob fallback.
When we're happy with results, this logic moves into the main pipeline.

Usage:
  venv\Scripts\python mask_van.py --frames_dir ..\C1_IMGS --out_dir ..\C1_MASKED

Optional:
  --conf     YOLO confidence threshold (default 0.15)
  --model    YOLO model name (default yolov8s.pt)
  --padding  Extra pixels around the bbox (default 10)
  --no-blob  Disable white-blob fallback
"""

import argparse, sys, os, cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.van import _detect_white_blob


VEHICLE_CLASSES = {2, 5, 7}   # COCO: car, bus, truck
DINO_PROMPT     = "white van . delivery van . cargo van"


def load_dino():
    """Load GroundingDINO model. Weights download automatically on first run."""
    try:
        from groundingdino.util.inference import load_model, load_image, predict
        import groundingdino
        pkg_dir    = os.path.dirname(groundingdino.__file__)
        config     = os.path.join(pkg_dir, "config", "GroundingDINO_SwinT_OGC.py")
        weights    = "groundingdino_swint_ogc.pth"
        if not os.path.exists(weights):
            print("Downloading GroundingDINO weights (~700MB)...")
            import urllib.request
            url = ("https://github.com/IDEA-Research/GroundingDINO/"
                   "releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
            urllib.request.urlretrieve(url, weights)
        model = load_model(config, weights)
        return model
    except ImportError:
        raise ImportError(
            "GroundingDINO not installed. Run:\n"
            r"  venv\Scripts\pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
        )


def detect_dino(img_bgr, model, conf_thresh):
    """Run GroundingDINO on an OpenCV BGR image, return best bbox or None."""
    import torch
    from PIL import Image
    from groundingdino.util.inference import predict
    import groundingdino.datasets.transforms as T

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

    # boxes are normalised [cx, cy, w, h] — convert to pixel x1,y1,x2,y2
    h, w = img_bgr.shape[:2]
    best_conf, best_bbox = 0.0, None
    for box, logit in zip(boxes, logits):
        conf = float(logit)
        if conf <= best_conf:
            continue
        cx, cy, bw, bh = box.tolist()

        # If the box covers more than 50% of the image area, skip it
        if (bw * bh) > 0.50:
            continue

        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h
        best_conf, best_bbox = conf, (x1, y1, x2, y2)
    return best_bbox


def detect_yolo(img, model, conf_thresh):
    """Run YOLO, return best vehicle bbox (x1,y1,x2,y2) or None."""
    results  = model(img, verbose=False)
    best_bbox, best_conf = None, conf_thresh
    for result in results:
        for box in result.boxes:
            cls  = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls in VEHICLE_CLASSES and conf > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best_bbox, best_conf = (x1, y1, x2, y2), conf
    return best_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', required=True)
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--conf',       type=float, default=0.15)
    parser.add_argument('--model',      default='yolov8s.pt')
    parser.add_argument('--padding',    type=int,   default=10)
    parser.add_argument('--use_yolo',   action='store_true',
                        help='Use YOLOv8 instead of GroundingDINO (default: GroundingDINO).')
    parser.add_argument('--no-blob',    action='store_true', dest='no_blob',
                        help='Disable white-blob fallback when primary detector fails.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load pipeline for undistortion only
    from pipeline.frame     import load_frames
    from pipeline.ocr       import extract_telemetry_all
    from pipeline.undistort import undistort_all

    print("Loading frames and running OCR + undistort...")
    frames = load_frames(args.frames_dir)
    extract_telemetry_all(frames)
    undistort_all(frames)

    # Load detector
    if args.use_yolo:
        print(f"Loading YOLO model '{args.model}'...")
        from ultralytics import YOLO
        detector = YOLO(args.model)
        det_method = 'YOLO'
    else:
        print("Loading GroundingDINO...")
        detector = load_dino()
        det_method = 'DINO' 

    results_summary = []

    for frame in frames:
        if frame.undistorted is None:
            print(f"  {frame.stem}: no undistorted image — skip.")
            continue

        img = frame.undistorted.copy()
        h, w = img.shape[:2]

        # Blur the HUD horizon/bracket graphics so DINO stops recognising
        # them as a van, while preserving surrounding content for matching.
        cx, cy = w // 2, h // 2
        hud_x1, hud_y1 = max(0, cx - 80), max(0, cy - 32)
        hud_x2, hud_y2 = min(w, cx + 80), min(h, cy + 16)
        roi = img[hud_y1:hud_y2, hud_x1:hud_x2]
        img[hud_y1:hud_y2, hud_x1:hud_x2] = cv2.GaussianBlur(roi, (21, 21), 0)

        if args.use_yolo:
            bbox   = detect_yolo(img, detector, args.conf)
        else:
            bbox   = detect_dino(img, detector, args.conf)
        method = det_method

        # 2. Fallback to white blob
        if bbox is None and not args.no_blob:
            bbox   = _detect_white_blob(img)
            method = 'blob'

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Apply padding, clamp to image
            x1 = max(0,   int(x1) - args.padding)
            y1 = max(0,   int(y1) - args.padding)
            x2 = min(w-1, int(x2) + args.padding)
            y2 = min(h-1, int(y2) + args.padding)

            # Draw filled black rectangle + red border so it's visible
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255),  2)

            label = f"{method} [{x1},{y1}→{x2},{y2}]"
        else:
            label = "NOT DETECTED"

        frame_num = frame.stem.split('_')[-1]
        out_name  = frame_num + '_masked.png'
        out_path  = os.path.join(args.out_dir, out_name)
        cv2.imwrite(out_path, img)

        results_summary.append((frame_num, label))
        print(f"  {frame_num}: {label}")

    print(f"\nSaved {len(results_summary)} image(s) to {args.out_dir}")
    not_detected = [r[0] for r in results_summary if 'NOT DETECTED' in r[1]]
    if not_detected:
        print(f"WARNING: van not detected in: {not_detected}")


if __name__ == '__main__':
    main()
