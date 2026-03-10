"""Orchestrate evaluation on images or video sequences."""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from solodet.eval.metrics import compute_map, print_results
from solodet.inference.detector import DroneDetector
from solodet.inference.video import VideoPipeline
from solodet.utils.config import load_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def evaluate_images(
    weights: str | Path,
    data_dir: str | Path,
    split: str = "test",
    sahi_config: str | Path | None = None,
    scale_bins_config: str | Path | None = None,
    device: str = "cuda:0",
    conf: float = 0.001,
) -> dict:
    """Evaluate detector on an image dataset split.

    Args:
        weights: Path to model weights.
        data_dir: Dataset root with images/ and labels/ subdirs.
        split: Dataset split to evaluate.
        sahi_config: SAHI config path (None to disable).
        scale_bins_config: Scale bins config path.
        device: Torch device.
        conf: Confidence threshold.

    Returns:
        Evaluation results dict.
    """
    data_dir = Path(data_dir)
    img_dir = data_dir / "images" / split
    lbl_dir = data_dir / "labels" / split

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    detector = DroneDetector(weights, sahi_config, device, conf)

    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])

    predictions = []
    ground_truths = []
    image_sizes = {}
    ann_id = 0

    for img_id, img_path in enumerate(tqdm(images, desc="Evaluating")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        image_sizes[img_id] = (img_w, img_h)

        # Get predictions
        dets = detector.predict(img)
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            predictions.append({
                "image_id": img_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format: x,y,w,h
                "score": det["confidence"],
                "category_id": 0,
            })

        # Load ground truth
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.is_file():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Convert normalised YOLO to pixel COCO format
                bw = w * img_w
                bh = h * img_h
                bx = xc * img_w - bw / 2
                by = yc * img_h - bh / 2
                ground_truths.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "bbox": [bx, by, bw, bh],
                    "category_id": 0,
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

    results = compute_map(predictions, ground_truths, image_sizes, scale_bins_config)
    print_results(results)
    return results


def evaluate_video(
    weights: str | Path,
    video_path: str | Path,
    gt_labels_dir: str | Path | None = None,
    sahi_config: str | Path | None = None,
    tracker_config: str | Path | None = None,
    scale_bins_config: str | Path | None = None,
    device: str = "cuda:0",
) -> dict:
    """Evaluate full pipeline on a video sequence.

    Args:
        weights: Path to model weights.
        video_path: Input video path.
        gt_labels_dir: Directory with per-frame YOLO-format ground truth labels.
        sahi_config: SAHI config path.
        tracker_config: Tracker config path.
        scale_bins_config: Scale bins config path.
        device: Torch device.

    Returns:
        Evaluation results dict.
    """
    pipeline = VideoPipeline(weights, sahi_config, tracker_config, device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    all_dets = pipeline.process_video(video_path)

    if gt_labels_dir is None:
        print(f"Processed {len(all_dets)} frames, no GT available for mAP.")
        total_dets = sum(len(d) for d in all_dets)
        return {"frames": len(all_dets), "total_detections": total_dets}

    gt_dir = Path(gt_labels_dir)
    predictions = []
    ground_truths = []
    image_sizes = {}
    ann_id = 0

    for frame_idx, frame_dets in enumerate(all_dets):
        image_sizes[frame_idx] = (width, height)

        for det in frame_dets:
            x1, y1, x2, y2 = det["bbox"]
            predictions.append({
                "image_id": frame_idx,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": det["confidence"],
                "category_id": 0,
            })

        lbl_path = gt_dir / f"{frame_idx:06d}.txt"
        if lbl_path.is_file():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                bw = w * width
                bh = h * height
                bx = xc * width - bw / 2
                by = yc * height - bh / 2
                ground_truths.append({
                    "id": ann_id,
                    "image_id": frame_idx,
                    "bbox": [bx, by, bw, bh],
                    "category_id": 0,
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

    results = compute_map(predictions, ground_truths, image_sizes, scale_bins_config)
    print_results(results)
    return results
