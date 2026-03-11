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
6. **Real-time benchmark**: Export to TensorRT FP16, run `scripts/benchmark.py` on test video, verify >=25 FPS sustained with tracking + multi-frame confirmation enabled

---

## Addendum: Real-Time Inference Requirement

### New Requirement

The system must be capable of **real-time inference (>=25 FPS)** on an RTX 2070 (8GB) while maintaining multi-scale drone detection across the tiny (<30px), medium (30-100px), and large (>100px) regimes. This addendum revises the inference pipeline design to meet this constraint. Training, data, and evaluation pipelines are unaffected.

### Original Design (Offline-First)

The original inference pipeline centred on **SAHI tiled inference** as the primary detection strategy:

```
Frame -> SAHI tiling (640x640, 25% overlap) -> YOLOv8m-P2 on each tile + full image
      -> NMS merge -> ByteTrack -> Multi-frame confirmation -> Output
```

- **5-9 forward passes per frame** (4-8 tiles + 1 full-image pass at 1280)
- Each tile pass: ~15ms, full-image pass: ~60ms
- **Total: ~120-200ms/frame -> 5-8 FPS on RTX 2070**
- Maximises small-object recall — SAHI preserves full resolution for every region of the image
- No consideration of inference latency as a constraint

This design treated SAHI as mandatory (described in research.md as "the single biggest win for tiny targets with minimal effort"). The implicit assumption was that accuracy on the tiniest targets outweighed throughput.

### Revised Design (Dual-Mode: Real-Time + Offline)

The key insight is that the **P2 detection head already provides much of what SAHI offers**, but architecturally rather than at inference time:

| Property | SAHI (640 tiles through P3) | P2 head at 1280 (single pass) |
|----------|---------------------------|-------------------------------|
| Effective feature resolution for small objects | 80x80 per tile | 320x320 global |
| Number of anchor positions covering a 20px drone | ~4-6 (within one tile) | ~25 (P2 feature map) |
| Forward passes per frame | 5-9 | **1** |
| Handles objects at tile boundaries | Via overlap + NMS merge | Natively (no tiling) |
| Latency | ~120-200ms | **~40-80ms (PyTorch), ~20-35ms (TensorRT FP16)** |

SAHI's advantage over single-pass P2 is strongest in the **extreme tiny regime (5-20px)**, where even the P2 feature map cell gets very little signal. For the 20px+ regime, P2 at 1280 is comparable or better (no tile-boundary artefacts, global context preserved).

#### Real-time mode

```
Frame -> YOLOv8m-P2 single pass at 1280 (TensorRT FP16)
      -> ByteTrack (low thresholds) -> Multi-frame confirmation -> Output
```

- **1 forward pass per frame**, ~20-35ms on RTX 2070 with TensorRT FP16
- ByteTrack + multi-frame confirmation compensate for lower single-frame confidence on tiny targets
- Target: **25-35 FPS sustained**

#### Offline mode (unchanged from original plan)

```
Frame -> SAHI tiling + full-image pass -> NMS merge
      -> ByteTrack -> Multi-frame confirmation -> Output
```

- Maximum accuracy, ~5-8 FPS
- Used for batch processing, evaluation, and as an accuracy ceiling benchmark

### Arguments For This Decision

1. **P2 head is architecturally equivalent to SAHI for most target sizes.** The P2 head at stride 4 with 1280 input produces 320x320 feature maps. SAHI with 640x640 tiles through a standard P3 head (stride 8) produces 80x80 per tile — with 4-8 tiles, the total anchor coverage is similar, but P2 achieves it in a single pass without tile-boundary artefacts or NMS merging.

2. **Temporal methods compensate for the accuracy gap on the tiniest targets.** Research.md documents that multi-frame motion analysis "dramatically reduces false negatives for dot-like targets at range." ByteTrack's two-pass association (high-confidence first, then low-confidence) rescues weak single-frame detections by associating them with existing tracks. A drone that produces a 0.10-confidence detection on individual frames can be reliably tracked across frames and confirmed by the multi-frame confirmer. This temporal pipeline costs <1ms per frame — effectively free.

3. **SAHI remains available as a plug-and-play upgrade.** Dropping SAHI from the real-time path doesn't remove it from the system. The offline mode preserves the original pipeline for batch evaluation, post-hoc analysis, or deployment scenarios where latency is not a constraint. This means we sacrifice nothing — we gain a real-time mode alongside the existing offline mode.

4. **TensorRT FP16 is a near-free 2x speedup.** Ultralytics natively supports `model.export(format='engine', half=True)`. FP16 quantisation has negligible accuracy impact on detection tasks and roughly halves inference latency. This is standard practice for deployment.

5. **A speed variant (YOLOv8s-P2) provides a further fallback.** If YOLOv8m-P2 at TensorRT FP16 does not sustain 25 FPS on the target hardware, YOLOv8s-P2 (~2.5x fewer FLOPs) should reach 40-50 FPS. This can be trained with the same pipeline and configs, only changing the model YAML.

### What Is Lost

The **5-20px extreme-tiny regime** takes the biggest accuracy hit in real-time mode. A 10px drone at 1280 input occupies ~0.006% of the image area. The P2 feature map cell covering it receives minimal signal — SAHI would zoom into that region via a 640x640 tile, making the drone ~1.6% of the tile and much more detectable. Multi-frame confirmation partially compensates (by accumulating weak evidence over time), but some very faint, slow-moving long-range targets that SAHI would catch in a single frame may require several frames of tracking before confirmation.

This is an acceptable tradeoff: the 5-20px regime is the hardest and rarest case, and the temporal pipeline provides a different (time-domain rather than spatial-domain) mechanism for recovering those detections.

### Changes to Existing Plan Sections

#### Project Structure (additions)

```
configs/
│   ├── inference/
│   │   ├── sahi.yaml              # SAHI config (offline mode)
│   │   ├── tracker.yaml           # ByteTrack parameters (shared)
│   │   ├── realtime.yaml          # real-time mode config (NEW)
│   │   └── offline.yaml           # offline mode config (NEW)
│   ...
scripts/
│   ├── export.py                  # TensorRT/ONNX export (NEW)
│   ├── benchmark.py               # FPS benchmarking (NEW)
│   ...
```

#### Step 2: Configuration files (additions)

**`configs/inference/realtime.yaml`:**
- `mode: realtime`
- `imgsz: 1280`
- `confidence_threshold: 0.15`
- `use_sahi: false`
- `engine: tensorrt` (FP16)
- `multiframe_min_hits: 3`, `multiframe_window: 5`

**`configs/inference/offline.yaml`:**
- `mode: offline`
- `imgsz: 1280`
- `confidence_threshold: 0.15`
- `use_sahi: true`
- `sahi_slice_size: 640`, `sahi_overlap: 0.25`
- `perform_standard_pred: true`
- `multiframe_min_hits: 3`, `multiframe_window: 5`

#### Step 5: Training pipeline (addition)

**New step 5b — TensorRT export:**

After training completes, export the best checkpoint for real-time deployment:

```bash
python scripts/export.py \
    --weights runs/phase1_baseline/weights/best.pt \
    --format engine --half --imgsz 1280
```

This produces a `.engine` file optimised for the target GPU. The export is hardware-specific (an engine built on RTX 2070 only runs on RTX 2070).

**`scripts/export.py`:**
1. Load trained weights via `YOLO(weights_path)`
2. Call `model.export(format=format, half=half, imgsz=imgsz)`
3. Print output path and run a single warmup inference to verify

#### Step 6: Inference pipeline (revised)

**`solodet/inference/detector.py`** — `DroneDetector` class:
- `mode` parameter: `"realtime"` or `"offline"`
- **Real-time mode**: Loads TensorRT `.engine` (or falls back to PyTorch), single-pass `model.predict()` at 1280
- **Offline mode**: SAHI `AutoDetectionModel` + `get_sliced_prediction()` with full-image pass, as in original plan
- Both modes output the same detection format (list of `[x1, y1, x2, y2, confidence, class_id]`)

**`solodet/inference/tracker.py`** — unchanged. ByteTrack operates on detection arrays regardless of how they were produced.

**`solodet/inference/multiframe.py`** — unchanged, but **elevated in importance**. In real-time mode, multi-frame confirmation is the primary mechanism for recovering weak tiny-target detections that SAHI would have caught spatially. The confirmer's role shifts from "nice-to-have temporal smoothing" to "critical accuracy recovery for the tiny regime."

**`solodet/inference/video.py`** — `VideoPipeline`:
- Accepts `mode` parameter to select real-time or offline detection
- Per-frame loop structure is identical in both modes: detect -> track -> confirm -> yield
- Real-time mode additionally tracks FPS and warns if sustained throughput drops below target

**`scripts/infer.py`**: CLI adds `--mode {realtime,offline}` flag, defaulting to `realtime`.

**`scripts/benchmark.py`** (new): Runs inference on a video without writing output, reports:
- Mean/P50/P95 per-frame latency (ms)
- Sustained FPS over the full video
- GPU memory usage
- Breakdown: detection time vs tracking time vs confirmation time

#### Key Design Decisions (revision)

Decision #2 is revised from:

> **SAHI + full-image dual inference**: Tiles catch tiny targets; full-image pass catches large targets spanning multiple tiles. NMS merges results.

To:

> **Dual-mode inference — real-time (single-pass P2) and offline (SAHI + full-image)**: The P2 head at 1280 provides high-resolution feature maps (320x320) that cover most of the accuracy benefit of SAHI tiling, in a single forward pass. Real-time mode uses this with TensorRT FP16 for >=25 FPS. Offline mode retains SAHI for maximum accuracy when latency is unconstrained. The temporal pipeline (ByteTrack + multi-frame confirmation) bridges the accuracy gap in real-time mode for the tiniest targets.

#### Verification (addition)

Added as item 6 (above):

> **Real-time benchmark**: Export to TensorRT FP16, run `scripts/benchmark.py` on test video, verify >=25 FPS sustained with tracking + multi-frame confirmation enabled.

### Expected Performance Summary

| Mode | Pipeline | Est. FPS (RTX 2070) | Tiny (5-20px) | Small (20-50px) | Medium+ (50px+) |
|------|----------|---------------------|---------------|-----------------|------------------|
| Real-time (m, TRT FP16) | P2 single-pass + ByteTrack | 25-35 | Reduced (temporal recovery) | Good | Excellent |
| Real-time (s, TRT FP16) | P2 single-pass + ByteTrack | 40-50 | Reduced | Moderate | Good |
| Offline (m, PyTorch) | SAHI + P2 + ByteTrack | 5-8 | Best | Best | Excellent |

### Phase 2 Interaction

The Phase 2 CBAM attention modules remain valuable and are arguably **more important** with the real-time constraint. In offline mode, SAHI spatially zooms into every region — attention is less critical. In real-time single-pass mode, the model must decide where to focus within one forward pass. Channel and spatial attention (CBAM) in the neck helps the P2 head allocate capacity to salient sky regions where tiny drones appear, partially compensating for the loss of SAHI's explicit spatial focus. CBAM adds negligible latency (~1-2ms).
