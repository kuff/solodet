# SoloDet Project Guide

## Overview
Ground-to-air RGB drone detection system for video sequences. Single-class (drone) detection using YOLOv8m with P2 small-object head, SAHI tiled inference, and ByteTrack temporal tracking.

## Environment
- **Conda env**: `solodet` (Python 3.11)
- **Activate**: `conda activate solodet`
- **Python binary**: `/home/peter/miniconda3/envs/solodet/bin/python`
- **Install**: `pip install -e ".[dev]"` (editable mode from pyproject.toml)
- **Key deps**: torch 2.10+cu128, ultralytics 8.4, sahi 0.11, supervision 0.27, opencv 4.11
- **CUDA**: Available (GPU-enabled)

## Project Structure
```
solodet/
├── CLAUDE.md              # this file
├── plan.md                # full implementation plan
├── research.md            # research notes & references
├── pyproject.toml         # deps & project metadata
├── configs/
│   ├── data/anti_uav.yaml       # dataset config for Ultralytics
│   ├── model/yolov8m-p2.yaml    # YOLOv8m + P2 head (stride 4), nc=1
│   ├── model/yolov8m-p2-cbam.yaml
│   ├── train/baseline.yaml      # imgsz=1280, 150 epochs, batch=8
│   ├── train/tiny_focus.yaml
│   ├── inference/sahi.yaml
│   ├── inference/tracker.yaml
│   └── eval/scale_bins.yaml
├── solodet/               # main Python package
│   ├── data/
│   │   ├── adapters/      # dataset format converters
│   │   │   ├── base.py    # DatasetAdapter ABC
│   │   │   └── anti_uav.py
│   │   ├── merge.py       # symlink-based dataset merger
│   │   └── stats.py       # scale distribution stats
│   ├── model/             # attention modules, custom module registration
│   ├── inference/         # detector, tracker, multiframe, video pipeline
│   ├── eval/              # per-scale-bin mAP metrics
│   └── utils/             # config, viz, io helpers
├── scripts/               # CLI entry points
│   ├── prepare_data.py    # convert/merge/stats
│   ├── train.py
│   ├── infer.py
│   ├── evaluate.py
│   └── visualize.py
├── data/                  # gitignored
│   ├── raw/anti_uav/      # original Anti-UAV RGBT dataset (6.3GB)
│   └── processed/anti_uav/  # YOLO format (35GB)
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
└── runs/                  # gitignored, training outputs
```

## Datasets

### Anti-UAV RGBT (converted)
- **Raw**: `data/raw/anti_uav/` — 318 video sequences (train/val/test splits)
- **Processed**: `data/processed/anti_uav/` — YOLO format
- **Config**: `configs/data/anti_uav.yaml`
- **Stats**: 296,901 total frames (280,067 positive, 16,834 negative)
  - train: 149,528 (142,106 pos, 7,422 neg)
  - val: 61,999 (58,262 pos, 3,737 neg)
  - test: 85,374 (79,699 pos, 5,675 neg)
- **Format**: Each frame extracted as JPG, one `.txt` label per image (`class_id cx cy w h` normalized)
- **Conversion**: `python scripts/prepare_data.py --dataset anti_uav --raw-dir data/raw/anti_uav`

### Planned (not yet downloaded)
- LRDDv2, Drone-vs-Bird, GA-Fly, DroneDetect

## Pretrained Model
- `yolov8m.pt` (50MB) in project root — COCO pretrained, used as starting weights for training

## Key Commands
```bash
# Activate environment
conda activate solodet

# Convert a dataset
python scripts/prepare_data.py --dataset anti_uav --raw-dir data/raw/anti_uav

# Merge datasets
python scripts/prepare_data.py --merge --datasets anti_uav

# Dataset stats
python scripts/prepare_data.py --stats --data-dir data/processed/anti_uav
```

## Design Decisions
- **imgsz: 1280** for training (tiny drones stay ~24px instead of ~12px at 640)
- **P2 head** (stride 4) for 320x320 feature maps at 1280 input
- **Low confidence thresholds** (0.15) — ByteTrack + multi-frame confirmation handles filtering
- **Symlinks** for merged dataset to avoid duplicating images
- **Single class** (drone=0) — birds/insects are background negatives
- **Dual-mode inference**: real-time (single-pass P2 + TensorRT FP16, >=25 FPS) and offline (SAHI + full-image, max accuracy)

## GPU / Training Constraints
- **GPU**: RTX 2070 (8GB VRAM)
- **Max batch size**: 2 at imgsz=1280 with AMP (uses ~5.5GB, ~2.6GB headroom)
  - AutoBatch forward-only test overestimates (showed 12GB for batch=2); actual training with AMP is much lower
- **Epoch time**: ~77 min at batch=1, ~40-45 min at batch=2 (Anti-UAV, fraction=0.1, 15k images)
- **Full training estimate**: ~30-35h for 50 epochs at 10% data; ~4-5 days for full 150-epoch run

## Training

### Current Config (baseline.yaml)
- `batch: 2`, `imgsz: 1280`, `workers: 4`, `save_period: 5`
- Validation run: `fraction: 0.1`, `epochs: 50` (restore to `fraction: 1.0`, `epochs: 150` for full run)

### Launching Training
```bash
# Fresh training
/home/peter/miniconda3/envs/solodet/bin/python scripts/train.py \
    --model configs/model/yolov8m-p2.yaml \
    --data configs/data/anti_uav.yaml \
    --config configs/train/baseline.yaml \
    --pretrained yolov8m.pt \
    --name phase1_validation --device 0

# Resume from crash
/home/peter/miniconda3/envs/solodet/bin/python scripts/train.py --resume \
    --model runs/detect/runs/phase1_validation/weights/last.pt \
    --data configs/data/anti_uav.yaml \
    --config configs/train/baseline.yaml \
    --name phase1_validation --device 0
```

### Monitoring
- **Log file**: `runs/phase1_validation/train_*.log` (persistent, survives crashes)
- **Results CSV**: `runs/detect/runs/phase1_validation/results.csv`
- **GPU**: `nvidia-smi` or check log (GPU memory logged every 5 min)
- **Checkpoints**: `last.pt` saved every epoch, periodic saves every 5 epochs

### Run Directory Note
Ultralytics nests output under `runs/detect/runs/<name>/` (it adds `detect/` automatically). Log files are written to `runs/<name>/` by the script.

## Bugs Found & Fixed
- `solodet/model/register.py`: `ultralytics.nn.modules.__all__` is a tuple, not a list — fixed to convert to list, append, convert back
- `scripts/train.py`: originally created 3 YOLO model instances, only last used — fixed to single `.load()` call
- `configs/data/anti_uav.yaml`: relative path didn't resolve — changed to absolute path
