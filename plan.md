# SoloDet Implementation Plan

## Context

We're building a ground-to-air RGB drone detection system for video sequences. The core challenge is multi-scale detection: the same drone can appear as a 5-pixel dot at long range or a clearly visible 500px object up close. The research phase (documented in `research.md`) identified datasets, architectures, and a phased approach. This plan turns that research into an implementable system.

A copy of this plan is committed to the project at `/home/peter/Desktop/solodet/plan.md`.

## Project Structure

```
solodet/
├── research.md                     # existing
├── plan.md                         # this plan
├── pyproject.toml                  # dependencies & project metadata
├── .gitignore
│
├── configs/
│   ├── model/
│   │   ├── yolov8m-p2.yaml        # YOLOv8m + P2 head (stride 4), nc=1
│   │   └── yolov8m-p2-cbam.yaml   # Phase 2: + CBAM attention in neck
│   ├── data/
│   │   ├── anti_uav.yaml          # Anti-UAV 300 RGB
│   │   ├── lrddv2.yaml            # LRDDv2
│   │   ├── drone_vs_bird.yaml     # Drone-vs-Bird
│   │   ├── gafly.yaml             # GA-Fly
│   │   ├── dronedetect.yaml       # DroneDetect
│   │   └── merged.yaml            # unified merged dataset
│   ├── train/
│   │   ├── baseline.yaml          # Phase 1: imgsz=1280, 150 epochs
│   │   └── tiny_focus.yaml        # Phase 2: imgsz=1920, more augmentation
│   ├── inference/
│   │   ├── sahi.yaml              # SAHI tile sizes, overlap, thresholds
│   │   └── tracker.yaml           # ByteTrack parameters
│   └── eval/
│       └── scale_bins.yaml        # tiny/small/medium/large px thresholds
│
├── solodet/                        # main Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # ABC: convert(), get_splits()
│   │   │   ├── anti_uav.py        # video frames + JSON -> YOLO
│   │   │   ├── lrddv2.py          # images + annotations -> YOLO
│   │   │   ├── drone_vs_bird.py   # video frames + challenge CSV -> YOLO
│   │   │   ├── gafly.py           # images + annotations -> YOLO
│   │   │   └── dronedetect.py     # already YOLO, just organise
│   │   ├── merge.py               # symlink datasets into unified split
│   │   └── stats.py               # scale distribution & dataset stats
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py           # CBAM, ECA nn.Module implementations
│   │   └── register.py            # register custom modules with Ultralytics
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── detector.py            # DroneDetector: standard + SAHI inference
│   │   ├── tracker.py             # ByteTrack bridge for SAHI outputs
│   │   ├── multiframe.py          # multi-frame confirmation (min_hits/window)
│   │   └── video.py               # VideoPipeline: detect -> track -> confirm
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py             # per-scale-bin mAP via pycocotools
│   │   └── evaluate.py            # orchestrate evaluation on images/video
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # YAML loading with overrides
│       ├── viz.py                 # draw boxes, tracks on frames
│       └── io.py                  # video I/O, frame extraction
│
├── scripts/                        # CLI entry points
│   ├── prepare_data.py            # download/convert/merge datasets
│   ├── train.py                   # launch training
│   ├── infer.py                   # run inference on video/images
│   ├── evaluate.py                # run evaluation with scale-bin breakdown
│   └── visualize.py               # generate demo videos
│
├── data/                           # gitignored
│   ├── raw/{anti_uav,lrddv2,...}  # downloaded originals
│   ├── processed/{dataset}/images+labels/{train,val,test}/
│   └── merged/images+labels/{train,val,test}/  # symlinks
│
└── runs/                           # gitignored, training outputs
```

## Implementation Steps

### Step 1: Project scaffolding

Create the directory tree, `pyproject.toml`, `.gitignore`, and initialise git.

**`pyproject.toml` dependencies:**
- `torch>=2.1`, `torchvision`
- `ultralytics>=8.3` — model defs, training, built-in tracking
- `sahi>=0.11` — tiled inference
- `opencv-python-headless>=4.8` — video I/O
- `supervision>=0.19` — detection/tracking viz helpers
- `pycocotools` — COCO-style mAP
- `pyyaml`, `pandas`, `tqdm`
- `gdown`, `huggingface_hub` — dataset downloading
- Dev: `jupyterlab`, `matplotlib`

**`.gitignore`:** `data/`, `runs/`, `*.pt`, `__pycache__/`, `.conda/`

### Step 2: Configuration files

**`configs/model/yolov8m-p2.yaml`** — Ultralytics-format YAML:
- Copy the upstream `yolov8-p2.yaml` architecture (backbone + head with P2 level)
- Set `nc: 1` (single "drone" class)
- Use `m` scale: `[0.67, 0.75, 768]`
- The P2 head adds a 4th detection output at stride 4 (feature map 4x larger than P3)

**`configs/train/baseline.yaml`:**
- `imgsz: 1280` — critical for tiny targets (a 20px drone stays ~24px after resize instead of ~12px at 640)
- `batch: 8` (P2 head is memory-heavy)
- `epochs: 150`, `optimizer: AdamW`, `lr0: 0.001`
- `copy_paste: 0.15`, `scale: 0.9` (aggressive scale jitter)
- `close_mosaic: 15`

**`configs/inference/sahi.yaml`:**
- `slice_height/width: 640`, `overlap: 0.25`
- `confidence_threshold: 0.15` (low — let tracker filter)
- `perform_standard_pred: true` (also full-image detection for large targets)

**`configs/inference/tracker.yaml`:**
- ByteTrack with `track_high_thresh: 0.15`, `track_low_thresh: 0.05`
- `track_buffer: 60` (2s at 30fps)

**`configs/eval/scale_bins.yaml`:**
- tiny: 0–30px, small: 30–50px, medium: 50–100px, large: 100+px

### Step 3: Data pipeline

**`solodet/data/adapters/base.py`** — Abstract base:
```python
class DatasetAdapter(ABC):
    def convert(self, raw_dir, output_dir, split) -> None: ...
    def get_splits(self, raw_dir) -> dict[str, list[Path]]: ...
```

All adapters convert to YOLO format: one `.txt` per image, `class_id x_center y_center w h` (normalised), single class `0 = drone`.

**Per-adapter strategy:**
- `anti_uav.py`: Extract video frames, parse per-frame JSON (bbox + exist flag), skip exist=False frames, convert pixel coords to normalised
- `lrddv2.py`: Parse annotation format (likely VOC XML or JSON), convert bboxes
- `drone_vs_bird.py`: Extract video frames, parse challenge-format annotations, map drone classes to 0, birds become hard negatives
- `gafly.py`: Parse annotations (likely VOC XML), convert
- `dronedetect.py`: Already YOLO format — validate and organise

**`solodet/data/merge.py`**: Create `data/merged/` with symlinks, prefix filenames with dataset name to avoid collisions. Generate `configs/data/merged.yaml`.

**`solodet/data/stats.py`**: Print scale distribution histogram (tiny/small/medium/large), positive vs negative frame counts, per-dataset breakdown.

**`scripts/prepare_data.py`**: CLI that orchestrates convert -> merge -> stats.

### Step 4: Model code

**`solodet/model/attention.py`** — CBAM implementation:
- Channel attention: global avg+max pool -> shared MLP -> sigmoid
- Spatial attention: channel avg+max -> 7x7 conv -> sigmoid

**`solodet/model/register.py`**:
- Injects CBAM/ECA into `ultralytics.nn.modules` namespace so YAML parser can resolve them
- Must be called before loading a model YAML that references custom modules

**`configs/model/yolov8m-p2-cbam.yaml`** — Insert CBAM after C2f blocks in the neck after P2/P3 feature fusion stages.

### Step 5: Training pipeline

**`solodet/train/trainer.py`** + **`scripts/train.py`**:
1. Parse CLI args (model YAML, data YAML, train config, name, pretrained weights)
2. Call `register_custom_modules()` if needed
3. `model = YOLO(model_yaml)` or load from pretrained
4. `model.train(data=..., **train_config)`

**Training strategy:**
- Phase 1: `yolov8m-p2.yaml` + `baseline.yaml` on merged dataset. Start from `yolov8m.pt` COCO pretrained weights (matching layers loaded, P2 head initialised randomly).
- Phase 2: Fine-tune Phase 1 best checkpoint with `yolov8m-p2-cbam.yaml` + `tiny_focus.yaml` (imgsz=1920, batch=4, more copy-paste).

### Step 6: Inference pipeline

**`solodet/inference/detector.py`** — `DroneDetector` class:
- Wraps both standard `YOLO.predict()` and SAHI `AutoDetectionModel` + `get_sliced_prediction()`
- SAHI runs 640x640 tiles with 25% overlap + full-image pass, merges via NMS
- Low confidence threshold (0.15) — tracker handles filtering

**`solodet/inference/tracker.py`**:
- Bridges SAHI output (ObjectPrediction format) to ByteTrack input (bbox + confidence arrays)
- Uses `supervision.ByteTrack` or standalone bytetrack for tracking when SAHI is active
- Falls back to Ultralytics built-in `model.track()` when SAHI is disabled

**`solodet/inference/multiframe.py`** — `MultiFrameConfirmer`:
- Maintains per-track detection history: `{track_id: [(frame_idx, confidence), ...]}`
- A track is "confirmed" only if detected in >= `min_hits` of last `window` frames
- Default: min_hits=3, window=5
- Critical for tiny-target regime: suppresses transient false positives from noise

**`solodet/inference/video.py`** — `VideoPipeline`:
- Per-frame loop: detect (SAHI) -> track (ByteTrack) -> confirm (multi-frame) -> yield
- Outputs: per-frame confirmed detections with track IDs
- Optional: write annotated output video

**`scripts/infer.py`**: CLI for video/image inference with config paths.

### Step 7: Evaluation pipeline

**`solodet/eval/metrics.py`**:
- Convert predictions + ground truth to COCO format
- Overall mAP@0.5 and mAP@0.5:0.95 via `pycocotools.COCOeval`
- **Per-scale-bin mAP**: filter GT+preds by max(w,h) in pixels into each bin, run COCOeval per bin
- This is the key metric — overall mAP can hide poor tiny-target performance

**`solodet/eval/evaluate.py`** + **`scripts/evaluate.py`**:
- Image mode: run detector on test split, compute per-scale-bin mAP
- Video mode: run full pipeline on video sequences, compute MOTA/IDF1 via `motmetrics`

### Step 8: Utility modules

- `solodet/utils/config.py`: `load_config(path, overrides)` — YAML loading
- `solodet/utils/viz.py`: Draw bboxes, track IDs, confidence on frames
- `solodet/utils/io.py`: Video reader/writer wrappers, frame extraction

## Key Design Decisions

1. **`imgsz: 1280` for training**: At 640, a 20px drone becomes ~12px — nearly undetectable. At 1280 it stays ~24px. The P2 head (stride 4) then produces 320x320 feature maps.

2. **SAHI + full-image dual inference**: Tiles catch tiny targets; full-image pass catches large targets spanning multiple tiles. NMS merges results.

3. **Low confidence thresholds -> tracker filters**: For tiny targets, single-frame confidence is inherently low. ByteTrack's 2-pass association + multi-frame confirmation handles filtering downstream.

4. **Symlinks for merged dataset**: Avoids duplicating gigabytes of images. Each processed dataset stays in one place.

5. **Per-scale-bin evaluation**: Without this, overall mAP can be dominated by easy large targets. Separate tiny/small/medium/large mAP reveals whether tiny-target detection actually improves.

6. **Single class (drone)**: Birds/insects are hard negatives from the background. A separate drone-vs-bird classifier can be added as Phase 3 post-processing if needed.

## Verification

1. **Data pipeline**: Run `scripts/prepare_data.py --stats` on a single dataset, verify YOLO-format labels are correct by visualising a few samples with `scripts/visualize.py`
2. **Training smoke test**: Train for 5 epochs on a small subset, verify loss decreases and checkpoints save
3. **Inference smoke test**: Run `scripts/infer.py` on a test video with SAHI+tracking, verify annotated output video shows detections with track IDs
4. **Evaluation**: Run `scripts/evaluate.py` with scale bins, verify per-bin mAP is computed and printed in a table
5. **Multi-scale correctness**: Manually inspect predictions on frames containing both tiny and large drones to confirm both are detected
