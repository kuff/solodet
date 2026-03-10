"""CLI: Run evaluation with scale-bin breakdown.

Usage:
    # Image evaluation
    python scripts/evaluate.py --weights runs/phase1/weights/best.pt \
        --data-dir data/merged --split test \
        --scale-bins configs/eval/scale_bins.yaml

    # Video evaluation
    python scripts/evaluate.py --weights runs/phase1/weights/best.pt \
        --video test_video.mp4 --gt-labels data/test_gt/ \
        --sahi-config configs/inference/sahi.yaml
"""

import argparse
from pathlib import Path

from solodet.eval.evaluate import evaluate_images, evaluate_video


def main():
    parser = argparse.ArgumentParser(description="Evaluate SoloDet model")
    parser.add_argument("--weights", type=Path, required=True, help="Model weights")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for evaluation")

    # Image evaluation
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset root for image evaluation")
    parser.add_argument("--split", type=str, default="test")

    # Video evaluation
    parser.add_argument("--video", type=Path, default=None,
                        help="Video file for video evaluation")
    parser.add_argument("--gt-labels", type=Path, default=None,
                        help="GT labels directory for video evaluation")

    # Configs
    parser.add_argument("--sahi-config", type=Path, default=None)
    parser.add_argument("--tracker-config", type=Path, default=None)
    parser.add_argument("--scale-bins", type=Path,
                        default=Path("configs/eval/scale_bins.yaml"))

    args = parser.parse_args()

    if args.video:
        evaluate_video(
            weights=args.weights,
            video_path=args.video,
            gt_labels_dir=args.gt_labels,
            sahi_config=args.sahi_config,
            tracker_config=args.tracker_config,
            scale_bins_config=args.scale_bins,
            device=args.device,
        )
    elif args.data_dir:
        evaluate_images(
            weights=args.weights,
            data_dir=args.data_dir,
            split=args.split,
            sahi_config=args.sahi_config,
            scale_bins_config=args.scale_bins,
            device=args.device,
            conf=args.conf,
        )
    else:
        parser.error("Must specify either --data-dir (images) or --video")


if __name__ == "__main__":
    main()
