"""CLI: Launch YOLOv8 training.

Usage:
    python scripts/train.py --model configs/model/yolov8m-p2.yaml \
        --data configs/data/merged.yaml \
        --config configs/train/baseline.yaml \
        --name phase1_baseline

    python scripts/train.py --model configs/model/yolov8m-p2-cbam.yaml \
        --data configs/data/merged.yaml \
        --config configs/train/tiny_focus.yaml \
        --pretrained runs/phase1_baseline/weights/best.pt \
        --name phase2_cbam
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from solodet.model.register import register_custom_modules
from solodet.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train SoloDet model")
    parser.add_argument("--model", type=Path, required=True,
                        help="Model YAML config (e.g. configs/model/yolov8m-p2.yaml)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Data YAML config (e.g. configs/data/merged.yaml)")
    parser.add_argument("--config", type=Path, required=True,
                        help="Training config (e.g. configs/train/baseline.yaml)")
    parser.add_argument("--pretrained", type=Path, default=None,
                        help="Pretrained weights to initialise from (e.g. yolov8m.pt)")
    parser.add_argument("--name", type=str, default="solodet",
                        help="Experiment name")
    parser.add_argument("--project", type=str, default="runs",
                        help="Project directory for outputs")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device(s) or 'cpu'")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    # Register custom modules (needed for CBAM/ECA model YAMLs)
    register_custom_modules()

    train_cfg = load_config(args.config)

    # Load model
    if args.pretrained and args.pretrained.suffix == ".pt":
        # Load pretrained weights, then apply model YAML architecture
        model = YOLO(str(args.model))
        # Transfer matching layers from pretrained
        print(f"Loading pretrained weights from {args.pretrained}")
        pretrained = YOLO(str(args.pretrained))
        # Use Ultralytics built-in transfer
        model = YOLO(str(args.model)).load(str(args.pretrained))
    elif args.resume:
        model = YOLO(str(args.pretrained or args.model))
    else:
        model = YOLO(str(args.model))

    # Train
    model.train(
        data=str(args.data),
        project=args.project,
        name=args.name,
        device=args.device,
        exist_ok=True,
        **train_cfg,
    )

    print(f"\nTraining complete. Results saved to {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
