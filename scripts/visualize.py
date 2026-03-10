"""CLI: Generate demo videos or visualise dataset samples.

Usage:
    # Visualise dataset labels
    python scripts/visualize.py --data-dir data/merged --split train --num 20

    # Visualise inference results on video
    python scripts/visualize.py --video output_det.mp4
"""

import argparse
import random
from pathlib import Path

import cv2
from solodet.utils.viz import draw_detections

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description="Visualise SoloDet data and results")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset root to visualise ground truth labels")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num", type=int, default=10,
                        help="Number of samples to visualise")
    parser.add_argument("--output", type=Path, default=Path("viz_output"),
                        help="Output directory for visualisations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.data_dir:
        _visualise_labels(args)
    else:
        parser.error("Must specify --data-dir")


def _visualise_labels(args):
    """Draw ground truth bboxes on dataset images."""
    img_dir = args.data_dir / "images" / args.split
    lbl_dir = args.data_dir / "labels" / args.split

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])

    # Sample images with labels (positive frames)
    positive = []
    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.is_file() and lbl_path.read_text().strip():
            positive.append(img_path)

    random.seed(args.seed)
    samples = random.sample(positive, min(args.num, len(positive)))

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        dets = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            px_w = w * img_w
            px_h = h * img_h
            dets.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": max(px_w, px_h),  # Show pixel size as "confidence"
            })

        annotated = draw_detections(img, dets)
        cv2.imwrite(str(output_dir / img_path.name), annotated)

    print(f"Saved {len(samples)} visualisations to {output_dir}")


if __name__ == "__main__":
    main()
