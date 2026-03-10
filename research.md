# Ground-to-Air Drone Detection on RGB Video Sequences

## Scope

Visual-only (RGB), ground-based camera looking upward, detecting and tracking
drones in video sequences. No thermal/IR, no radar, no RF.

**Key challenge**: The detector must handle the full range of apparent drone
sizes — from tiny targets at long range (as few as 5–50 pixels in 1080p) to
clearly visible drones at close range. This is fundamentally a **multi-scale
detection** problem where the long-range / tiny-target regime is the hardest
and most important case.

---

## 1. Datasets

### 1.1 Primary — Ground-to-Air RGB Video

| Dataset | Scale | Format | Key Features | Link |
|---------|-------|--------|--------------|------|
| **Anti-UAV 300 (RGB subset)** | 300 paired video sequences, 580k+ bounding boxes | Video (RGB + IR pairs; RGB usable standalone) | Ground-based cameras, diverse UAV models, occlusion/scale variation, target-absent annotations | [GitHub](https://github.com/ZhaoJ9014/Anti-UAV) |
| **Drone-vs-Bird Challenge** | 77 video sequences, 104,760 frames | Video (RGB) | Average object size **34x23 px (0.1% of image)**. Drones filmed at considerable distances with birds/insects as distractors; moving cameras, cluttered backgrounds | [OBSS](https://obss.tech/en/drone-vs-bird-detection-challenge/), [Paper](https://www.mdpi.com/1424-8220/21/8/2824) |
| **GA-Fly (2025)** | 10,800 high-res images | Images (RGB) | DJI Mini 4 Pro at varied angles, distances, lighting. Benchmarked with 8 DL models (Faster R-CNN, YOLOv8/v10, etc.) | [Paper](https://link.springer.com/article/10.1007/s11760-025-04616-4) |
| **LRDDv2 (2025)** | 39,516 annotated images | Images (RGB) | **Majority of targets are ≤50 pixels in 1080p**. Includes target range information for 8,000+ images (enables range estimation). Diverse weather, time-of-day, backgrounds | [arXiv](https://arxiv.org/html/2508.03331v1) |
| **UETT4K Anti-UAV (2025)** | 4K resolution frames | Images (RGB, extracted from 4K video) | High-resolution, multiple drone types, diverse environments | [Paper](https://ui.adsabs.harvard.edu/abs/2025IEEEA..1373553S/abstract) |
| **DroneDetect** | Images (size TBC) | Images (RGB), YOLO format | Ground-based + aerial images, YOLO-compatible bounding boxes | [IEEE DataPort](https://ieee-dataport.org/documents/dronedetect-benchmark-uav-dataset-deep-learning-based-drone-detection) |
| **UAV Detection & Tracking Benchmark** | Video sequences | Video (RGB) | Bounding box + attribute annotations (weather, occlusion, scale) for detection & tracking | [GitHub](https://github.com/KostadinovShalon/UAVDetectionTrackingBenchmark) |
| **Maciullo Drone Detection Dataset** | Images | Images (RGB) | Community dataset with drone annotations | [GitHub](https://github.com/Maciullo/DroneDetectionDataset) |

### 1.2 Supplementary / Synthetic

| Dataset | Description | Link |
|---------|-------------|------|
| **SynDroneVision** | Synthetic rendered drone images to augment real-world training data | [arXiv](https://arxiv.org/html/2411.05633v1) |
| **YOLO-Drone (synthetic mix)** | Game-engine synthetic data mixed with small real sets; +4–8 mAP for camouflaged/low-contrast targets | [EmergentMind](https://www.emergentmind.com/topics/yolo-drone) |

### 1.3 Excluded (not ground-to-air or not RGB)

| Dataset | Reason Excluded |
|---------|----------------|
| **Anti-UAV410 / Anti-UAV600** | Thermal IR only — no RGB |
| **VisDrone** | Drone-*mounted* camera (aerial perspective), not ground-to-air |
| **UAVDT** | Drone-*mounted* camera (aerial perspective) |
| **SeaDronesSee** | Drone-mounted, maritime SAR context |
| **RFUAV** | Radio-frequency based, not visual |

---

## 2. Methods & Techniques

### 2.1 The Multi-Scale Challenge

Standard detectors struggle with drones because the same object class spans a
huge range of apparent sizes:

| Regime | Apparent size (1080p) | Characteristics | Primary difficulty |
|--------|----------------------|-----------------|-------------------|
| **Long range** | 5–20 px | Dot-like, no shape features, easily confused with noise/birds | Extremely low signal; detector must learn from ~dozens of pixels |
| **Medium range** | 20–80 px | Silhouette visible, some shape cues | Balancing recall with false positives from birds/debris |
| **Close range** | 80–500+ px | Full shape, colour, rotors visible | Standard detection; main risk is motion blur |

A viable system must perform well across **all three regimes simultaneously**.
This drives both architecture and inference-strategy choices.

### 2.2 Inference Strategies for Tiny Targets

These are applied at inference time (and optionally training) to boost
small-object recall without retraining the base model:

| Strategy | How it works | Benefit | Reference |
|----------|-------------|---------|-----------|
| **SAHI (Slicing Aided Hyper Inference)** | Slices high-res input into overlapping tiles, runs detector on each tile, merges results with NMS | Preserves full resolution for tiny targets; plug-and-play with any detector. Critical for 4K input | [Pysource tutorial](https://pysource.com/2025/04/23/how-to-accurately-detect-small-objects-with-yolo-and-sahi/) |
| **Super-resolution preprocessing (NSSRD)** | Noise suppression + SRCNN upsampling before detection | Enhances shape detail for sub-50px targets | [Paper](https://www.mdpi.com/2076-3417/15/6/3076) |
| **Multi-frame motion analysis** | Aggregate detections across N consecutive frames; exploit temporal consistency to confirm tiny targets | Reduces false negatives for dot-like targets that a single frame misses | [Paper](https://arxiv.org/html/2411.02582v1) |
| **Test-time augmentation (TTA)** | Run inference at multiple scales, flip/rotate, merge | Marginal gains at all scales; most useful at boundary sizes | — |

### 2.3 Architectural Techniques for Multi-Scale Detection

These are modifications to the detector architecture itself:

**Extra small-object detection head (P2 level)**
Standard YOLO uses P3–P5 feature maps (stride 8–32). Adding a P2 head
(stride 4) doubles the spatial resolution available for tiny targets.
This is the single most impactful architectural change for long-range drones.
Trade-off: higher memory and compute.

**Advanced Feature Pyramid Networks**

| Method | Key Innovation | Why it helps for multi-scale | Reference |
|--------|---------------|------------------------------|-----------|
| **SMA-YOLO (2025)** | Non-Semantic Sparse Attention (NSSA) + Bidirectional Multi-Branch Auxiliary FPN (BIMA-FPN) | Integrates high-level semantics with low-level spatial detail; +13% mAP on VisDrone | [Paper](https://www.nature.com/articles/s41598-025-92344-7) |
| **RPS-YOLO (2025)** | Recursive Feature Pyramid (RFP) on YOLOv8 | Recursive refinement lets small-object features benefit from multiple top-down passes | [Paper](https://www.mdpi.com/2076-3417/15/4/2039) |
| **YOLO-MARS (2025)** | Shallow Guided Cross-Scale FPN (SGCS-FPN) | Shallow feature guidance branches create cross-scale semantic associations for tiny targets | [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12031147/) |
| **CF-YOLO (2025)** | Cross-Scale FPN + Feature Recalibration + Sandwich Fusion | Prevents small-target information loss across scales; +12.7% mAP50 on VisDrone | [Paper](https://www.nature.com/articles/s41598-025-99634-0) |
| **CSSDet (2024)** | Extra small-object head + bidirectional weighted FPN | Cross-scale feature enhancement specifically for small objects | [Paper](https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2414848) |
| **High-Resolution FPN (HR-FPN)** | Maintains high-res feature maps throughout the pyramid | Avoids downsampling loss that destroys tiny-target features | [Paper](https://ieeexplore.ieee.org/document/10154030/) |

**Attention mechanisms** — Focus the detector on salient regions:
- **CBAM / ECA / SE-Net** — channel and spatial attention, lightweight
- **SegNext Attention** — used in LRDS-YOLO for small-object focus
- **Non-Semantic Sparse Attention** — SMA-YOLO's approach, avoids semantic suppression of small targets

### 2.4 Single-Stage Detectors (YOLO Family)

The YOLO family dominates the drone detection literature. Key variants
addressing multi-scale / tiny-target detection:

| Method | Key Innovation | Performance | Reference |
|--------|---------------|-------------|-----------|
| **YOLOv8 / YOLOv11** | Current baselines; good speed/accuracy tradeoff | Baseline | [Ultralytics](https://docs.ultralytics.com/) |
| **EDGS-YOLOv8** | Lightweight anti-drone model | 0.971 AP on DUT Anti-UAV, 4.23 MB | — |
| **LAMS-YOLO (2025)** | Linear attention + adaptive downsampling (preserves small-object features during downsampling) | Lightweight, improved small-object recall | [Paper](https://www.mdpi.com/2072-4292/17/4/705) |
| **LRDS-YOLO (2025)** | Light Adaptive-weight Downsampling + SegNext Attention + dynamic detection head | Specifically targets small-object detection | [Paper](https://www.nature.com/articles/s41598-025-07021-6) |
| **MFA-YOLO (2025)** | Multi-Feature Aggregation | Small-object-focused feature fusion | [Paper](https://www.nature.com/articles/s41598-025-32247-9) |
| **Improved YOLO for long-range drones (2025)** | Optimised specifically for long-distance small drone detection | Improved recall at range | [Paper](https://www.nature.com/articles/s41598-025-95580-z) |
| **SSD / RTMDet** | Alternative single-stage architectures benchmarked in GA-Fly | See GA-Fly paper | — |

### 2.5 Two-Stage Detectors

- **Faster R-CNN / Faster R-CNN-FPN** — higher accuracy, slower; region proposals can capture tiny objects better than anchor grids
- **Cascade R-CNN** — multi-stage refinement, progressively tightens IoU thresholds; benefits small objects with imprecise initial proposals

### 2.6 Transformer-Based Approaches

- **C3TR module** in detection neck — global context via self-attention
- **Sparse transformers** — specifically designed for small-object detection in remote sensing
- **Swin Transformer** backbones as drop-in replacements for CNN feature extractors
- Transformers excel in sky-background scenes where global context helps distinguish tiny drones from clutter at long range

### 2.7 Temporal / Video-Specific Methods

These are **especially important for the tiny-target regime**, where a single
frame may provide insufficient evidence for confident detection.

| Technique | Description | Benefit |
|-----------|-------------|---------|
| **Multi-frame motion analysis** | Combine YOLO detections across N frames; confirm persistent tiny detections | Dramatically reduces false negatives for dot-like targets at range. [Paper](https://arxiv.org/html/2411.02582v1) |
| **Kalman filter tracking + confidence boosting** | Track detections across frames; propagate max confidence along tracks | Smooths per-frame noise, rescues low-confidence tiny-target detections |
| **LSTM / Transformer temporal fusion necks** | Sequence model on top of per-frame features | Up to +73% on drone-vs-bird classification, +35% overall |
| **ByteTrack / BoT-SORT** | Modern multi-object trackers; associate detections across frames | Real-time, works well with YOLO; can maintain tracks through momentary misses |
| **Bio-inspired magnocellular motion detection (2025)** | Models biological motion-sensitive neurons to detect tiny moving objects against complex backgrounds | Specifically designed for tiny drone detection in video. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1568494624006665) |
| **GM-YOLO Multi-Tracker** | YOLO detection + Kalman motion + appearance similarity metric | Robust multi-object tracking in video |
| **3D convolution / video backbones** | SlowFast, X3D for spatio-temporal features | Captures motion patterns directly; helps distinguish drones from static noise |

### 2.8 Data Augmentation Strategies

Augmentation is especially important for the tiny-target regime where real
training examples are scarce:

- **Copy-paste augmentation at multiple scales** — paste drone crops at various sizes onto sky backgrounds, with emphasis on tiny placements
- **Synthetic data mixing** — game-engine renders + real data yields +4–8 mAP, especially for small/camouflaged targets
- **Progressive resizing** — train at increasing resolutions to learn scale-invariant features
- **Mosaic / MixUp** — standard YOLO augmentations
- **Domain randomization** — vary lighting, haze, background to improve generalization
- **Small-object oversampling** — repeat images containing tiny targets during training to balance scale distribution

---

## 3. Recommended Approach

### Phase 1 — Multi-Scale Baseline
1. **Dataset**: Anti-UAV 300 (RGB) + **LRDDv2** (critical for tiny-target training) + Drone-vs-Bird
2. **Model**: YOLOv8m or YOLOv11m with an **added P2 small-object detection head** (stride 4)
3. **Inference**: **SAHI tiling** for high-resolution input — this is the single biggest win for tiny targets with minimal effort
4. **Tracking**: ByteTrack for frame-to-frame association; configure low detection threshold to catch tentative tiny-target detections that tracking can confirm over time

### Phase 2 — Tiny-Target Focus
5. **Multi-frame confirmation**: Aggregate detections over N frames; require temporal persistence before reporting a tiny target (reduces false positives from noise)
6. **Attention modules** (CBAM or ECA) in the neck to focus on salient sky regions
7. **Advanced FPN**: Replace standard FPN with a cross-scale variant (e.g. SGCS-FPN from YOLO-MARS, or BIMA-FPN from SMA-YOLO) to improve small/large feature flow
8. **Super-resolution preprocessing**: Apply lightweight SR (e.g. SRCNN) to ROIs or full frame for the ≤50px regime
9. **Copy-paste augmentation**: Paste drone crops at tiny sizes onto sky backgrounds; oversample small-target training images

### Phase 3 — Robustness Across All Scales
10. **Temporal fusion**: LSTM or lightweight Transformer neck for sequence-level confidence (up to +73% on drone-vs-bird)
11. **Drone-vs-Bird training**: Minimise bird/insect false positives, especially at long range where silhouettes are ambiguous
12. **Synthetic data** (SynDroneVision) for rare conditions and underrepresented scales
13. **Evaluate across scale bins**: Report mAP separately for tiny (<30px), medium (30–100px), and large (>100px) targets to ensure no regime is neglected
14. **Evaluate on UETT4K** for high-res performance and GA-Fly for diverse conditions

---

## 4. Key Surveys & References

### Surveys
- [ML for Drone Detection from Images: Review of Techniques (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0925231225004953)
- [Deep Learning for UAV Detection: A Review](https://www.sciencedirect.com/science/article/abs/pii/S1574013723000813)
- [Securing the Skies: Anti-UAV Survey (CVPR 2025 Workshop)](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/papers/Dong_Securing_the_Skies_A_Comprehensive_Survey_on_Anti-UAV_Methods_Benchmarking_CVPRW_2025_paper.pdf)
- [Small Object Detection in Aerial Images: Survey (2025)](https://link.springer.com/article/10.1007/s10462-025-11150-9)
- [Real-Time Aerial Object Detection: Survey (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12736610/)
- [Vision-Based Anti-UAV: Opportunities and Challenges (2025)](https://arxiv.org/html/2507.10006v1)

### Multi-Scale & Tiny-Target Detection
- [NSSRD: Noise Suppression Super-Resolution for Small UAS Detection (2025)](https://www.mdpi.com/2076-3417/15/6/3076)
- [SMA-YOLO: Multi-Scale Small Object Detection (2025)](https://www.nature.com/articles/s41598-025-92344-7)
- [RPS-YOLO: Recursive Pyramid Structure for Small Objects (2025)](https://www.mdpi.com/2076-3417/15/4/2039)
- [YOLO-MARS: Shallow Guided Cross-Scale FPN (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12031147/)
- [CSSDet: Cross-Scale Feature Enhancement (2024)](https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2414848)
- [HR-FPN: High-Resolution Feature Pyramid Network](https://ieeexplore.ieee.org/document/10154030/)
- [Improved YOLO for Long-Range Drone Detection (2025)](https://www.nature.com/articles/s41598-025-95580-z)
- [Real-Time Detection for Small UAVs: YOLO + Multi-Frame Motion (2024)](https://arxiv.org/html/2411.02582v1)
- [Early Detection of Small- and Medium-Sized Drones in Complex Environments (2025)](https://cdnsciencepub.com/doi/10.1139/dsa-2025-0018)
- [Does Deep Super-Resolution Enhance UAV Detection?](https://www.researchgate.net/publication/337537832_Does_Deep_Super-Resolution_Enhance_UAV_Detection)

### Temporal & Video Methods
- [Drone vs. Bird: Grand Challenge Results](https://www.mdpi.com/1424-8220/21/8/2824)
- [Sequence Models for Drone vs Bird Classification](https://ar5iv.labs.arxiv.org/html/2207.10409)
- [Bio-inspired Magnocellular Computation for Tiny Drone Detection in Video (2024)](https://www.sciencedirect.com/science/article/abs/pii/S1568494624006665)
- [Bio-inspired Motion Detection for UAV vs Bird (2025)](https://www.nature.com/articles/s41598-025-99951-4)

### Datasets
- [LRDDv2: Long-Range Drone Detection with Range Information (2025)](https://arxiv.org/abs/2508.03331)
- [GA-Fly: Ground-to-Air Drone Visual Detection (2025)](https://link.springer.com/article/10.1007/s11760-025-04616-4)
- [Anti-UAV Official Repository](https://github.com/ZhaoJ9014/Anti-UAV)
- [SAHI: How to Accurately Detect Small Objects with YOLO](https://pysource.com/2025/04/23/how-to-accurately-detect-small-objects-with-yolo-and-sahi/)
