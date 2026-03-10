"""CLI: Run inference on video or images.

Usage:
    # Video with SAHI + tracking
    python scripts/infer.py --weights runs/phase1/weights/best.pt \
        --source video.mp4 --output output.mp4 \
        --sahi-config configs/inference/sahi.yaml \
        --tracker-config configs/inference/tracker.yaml

    # Image directory (no tracking)
    python scripts/infer.py --weights runs/phase1/weights/best.pt \
        --source data/test/images/ --output results/
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from solodet.inference.detector import DroneDetector
from solodet.inference.video import VideoPipeline
from solodet.utils.viz import draw_detections

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def main():
    parser = argparse.ArgumentParser(description="SoloDet inference")
    parser.add_argument("--weights", type=Path, required=True, help="Model weights")
    parser.add_argument("--source", type=Path, required=True,
                        help="Input video file or image directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output video path or directory for annotated images")
    parser.add_argument("--sahi-config", type=Path, default=None,
                        help="SAHI config (enables tiled inference)")
    parser.add_argument("--tracker-config", type=Path, default=None,
                        help="Tracker config")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold override")
    args = parser.parse_args()

    source = args.source

    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        _infer_video(args)
    elif source.is_dir():
        _infer_images(args)
    else:
        raise ValueError(f"Source must be a video file or image directory: {source}")


def _infer_video(args):
    """Run full pipeline on a video."""
    pipeline = VideoPipeline(
        weights=args.weights,
        sahi_config=args.sahi_config,
        tracker_config=args.tracker_config,
        device=args.device,
        conf=args.conf,
    )

    output = args.output or args.source.with_name(f"{args.source.stem}_det.mp4")
    print(f"Processing {args.source} -> {output}")

    all_dets = pipeline.process_video(args.source, output)

    total_dets = sum(len(d) for d in all_dets)
    frames_with_dets = sum(1 for d in all_dets if d)
    print(f"\nDone: {len(all_dets)} frames, {total_dets} detections "
          f"({frames_with_dets} frames with detections)")


def _infer_images(args):
    """Run detector on image directory."""
    detector = DroneDetector(
        weights=args.weights,
        sahi_config=args.sahi_config,
        device=args.device,
        conf=args.conf,
    )

    output_dir = args.output or args.source.parent / f"{args.source.name}_det"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in args.source.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    print(f"Processing {len(images)} images...")

    total_dets = 0
    for img_path in tqdm(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        dets = detector.predict(img)
        total_dets += len(dets)

        annotated = draw_detections(img, dets)
        cv2.imwrite(str(output_dir / img_path.name), annotated)

    print(f"\nDone: {len(images)} images, {total_dets} total detections")
    print(f"Annotated images saved to {output_dir}")


if __name__ == "__main__":
    main()
