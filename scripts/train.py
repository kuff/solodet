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

    # Resume from crash:
    python scripts/train.py --resume \
        --model runs/phase1_validation/weights/last.pt \
        --data configs/data/anti_uav.yaml \
        --config configs/train/baseline.yaml \
        --name phase1_validation --device 0
"""

import argparse
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from solodet.model.register import register_custom_modules
from solodet.utils.config import load_config


def setup_logging(project: str, name: str) -> logging.Logger:
    """Set up file + console logging to the run directory."""
    log_dir = Path(project) / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger("solodet.train")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


def gpu_monitor(logger: logging.Logger, interval: int = 300, stop_event: threading.Event = None):
    """Log GPU stats periodically. Runs in a background thread."""
    try:
        import torch
    except ImportError:
        return

    while not stop_event.is_set():
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_mem / 1024**3
            logger.info(
                f"GPU memory: {mem_alloc:.2f}G allocated, "
                f"{mem_reserved:.2f}G reserved, {mem_total:.2f}G total"
            )
        stop_event.wait(interval)


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

    # Resolve run directory for logging (match Ultralytics' nested structure)
    logger = setup_logging(args.project, args.name)
    logger.info(f"Arguments: {vars(args)}")

    # Start GPU monitoring thread
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=gpu_monitor, args=(logger, 300, stop_event), daemon=True
    )
    monitor.start()

    train_cfg = load_config(args.config)
    logger.info(f"Training config: {train_cfg}")

    # Load model
    if args.resume:
        logger.info(f"Resuming training from {args.model}")
        model = YOLO(str(args.model))
    elif args.pretrained and args.pretrained.suffix == ".pt":
        logger.info(f"Loading pretrained weights from {args.pretrained}")
        model = YOLO(str(args.model)).load(str(args.pretrained))
    else:
        logger.info(f"Building model from {args.model}")
        model = YOLO(str(args.model))

    logger.info("Starting training...")
    start_time = time.time()

    try:
        model.train(
            data=str(args.data),
            project=args.project,
            name=args.name,
            device=args.device,
            exist_ok=True,
            resume=args.resume,
            **train_cfg,
        )
        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed/3600:.1f}h. "
                     f"Results saved to {args.project}/{args.name}/")
    except Exception:
        elapsed = time.time() - start_time
        logger.exception(f"Training failed after {elapsed/3600:.1f}h")
        raise
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
