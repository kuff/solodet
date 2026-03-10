"""Comprehensive end-to-end validation tests for SoloDet.

Tests use only stdlib + numpy + opencv + pytest (no ultralytics/sahi/torch
unless guarded with importorskip).  All file I/O uses the ``tmp_path``
fixture so nothing touches real data.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_W, IMG_H = 320, 240


def _make_image(path: Path, width: int = IMG_W, height: int = IMG_H) -> Path:
    """Write a solid-colour synthetic PNG and return its path."""
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _make_yolo_label(path: Path, lines: list[str]) -> Path:
    """Write YOLO-format label lines to *path* and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


# ===================================================================
# 1. Config loading
# ===================================================================


class TestConfigLoading:
    """Validate solodet.utils.config.load_config."""

    def test_load_none_returns_empty(self):
        from solodet.utils.config import load_config

        assert load_config(None) == {}

    def test_load_none_with_overrides(self):
        from solodet.utils.config import load_config

        cfg = load_config(None, overrides={"a": 1})
        assert cfg == {"a": 1}

    def test_load_missing_file(self, tmp_path):
        from solodet.utils.config import load_config

        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg == {}

    def test_load_yaml_file(self, tmp_path):
        from solodet.utils.config import load_config

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("imgsz: 640\nbatch: 16\n")
        cfg = load_config(yaml_path)
        assert cfg["imgsz"] == 640
        assert cfg["batch"] == 16

    def test_load_with_overrides(self, tmp_path):
        from solodet.utils.config import load_config

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("imgsz: 640\nbatch: 16\n")
        cfg = load_config(yaml_path, overrides={"batch": 32, "new_key": True})
        assert cfg["imgsz"] == 640
        assert cfg["batch"] == 32
        assert cfg["new_key"] is True

    def test_load_real_baseline_config(self):
        """Load the real configs/train/baseline.yaml from the project."""
        from solodet.utils.config import load_config

        real_path = Path(__file__).resolve().parent.parent / "configs" / "train" / "baseline.yaml"
        if not real_path.is_file():
            pytest.skip("baseline.yaml not found in configs/train/")
        cfg = load_config(real_path)
        assert "imgsz" in cfg
        assert "epochs" in cfg
        assert isinstance(cfg["imgsz"], int)

    def test_save_config_roundtrip(self, tmp_path):
        from solodet.utils.config import load_config, save_config

        original = {"lr0": 0.001, "batch": 8, "names": {0: "drone"}}
        out_path = tmp_path / "saved.yaml"
        save_config(original, out_path)
        loaded = load_config(out_path)
        assert loaded["lr0"] == pytest.approx(0.001)
        assert loaded["batch"] == 8
        assert loaded["names"] == {0: "drone"}


# ===================================================================
# 2. DatasetAdapter base helpers
# ===================================================================


class TestDatasetAdapterBase:
    """Test static helpers on the abstract base class."""

    def _make_concrete(self):
        """Return a minimal concrete subclass of DatasetAdapter."""
        from solodet.data.adapters.base import DatasetAdapter

        class _Stub(DatasetAdapter):
            def convert(self, raw_dir, output_dir, split):
                pass

            def get_splits(self, raw_dir):
                return {}

        return _Stub()

    def test_pixel_to_yolo_basic(self):
        from solodet.data.adapters.base import DatasetAdapter

        xc, yc, w, h = DatasetAdapter._pixel_to_yolo(100, 50, 200, 150, 400, 300)
        assert xc == pytest.approx(0.375)
        assert yc == pytest.approx(1.0 / 3.0)
        assert w == pytest.approx(0.25)
        assert h == pytest.approx(1.0 / 3.0)

    def test_pixel_to_yolo_clamped(self):
        from solodet.data.adapters.base import DatasetAdapter

        # Box extends beyond image boundaries
        xc, yc, w, h = DatasetAdapter._pixel_to_yolo(-10, -10, 500, 400, 400, 300)
        assert 0.0 <= xc <= 1.0
        assert 0.0 <= yc <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0

    def test_xywh_to_yolo(self):
        from solodet.data.adapters.base import DatasetAdapter

        # top-left (100,50), width=100, height=100, image 400x300
        xc, yc, w, h = DatasetAdapter._xywh_to_yolo(100, 50, 100, 100, 400, 300)
        assert xc == pytest.approx(0.375)
        assert yc == pytest.approx(1.0 / 3.0)
        assert w == pytest.approx(0.25)
        assert h == pytest.approx(1.0 / 3.0)

    def test_write_label(self, tmp_path):
        stub = self._make_concrete()
        lbl_path = tmp_path / "labels" / "test.txt"
        bboxes = [(0.5, 0.5, 0.1, 0.2), (0.3, 0.4, 0.05, 0.06)]
        stub._write_label(lbl_path, bboxes)

        assert lbl_path.is_file()
        lines = lbl_path.read_text().strip().splitlines()
        assert len(lines) == 2
        parts0 = lines[0].split()
        assert parts0[0] == "0"  # CLASS_ID
        assert float(parts0[1]) == pytest.approx(0.5)

    def test_write_label_empty(self, tmp_path):
        stub = self._make_concrete()
        lbl_path = tmp_path / "labels" / "empty.txt"
        stub._write_label(lbl_path, [])
        assert lbl_path.is_file()
        assert lbl_path.read_text().strip() == ""


# ===================================================================
# 3. LRDDv2 adapter
# ===================================================================


class TestLRDDv2Adapter:
    """Create a fake raw dataset and run convert()."""

    def _setup_raw(self, raw_dir: Path):
        """Create raw_dir/images/train/*.png + raw_dir/labels/train/*.txt"""
        img_dir = raw_dir / "images" / "train"
        lbl_dir = raw_dir / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(3):
            _make_image(img_dir / f"frame_{i:04d}.png")
            # YOLO label with class_id=5 (should be remapped to 0)
            _make_yolo_label(lbl_dir / f"frame_{i:04d}.txt", [f"5 0.5 0.5 0.1 0.{i+1}"])

    def test_get_splits(self, tmp_path):
        from solodet.data.adapters.lrddv2 import LRDDv2Adapter

        raw_dir = tmp_path / "raw"
        self._setup_raw(raw_dir)

        adapter = LRDDv2Adapter()
        splits = adapter.get_splits(raw_dir)
        assert "train" in splits
        assert len(splits["train"]) == 3

    def test_convert_creates_output(self, tmp_path):
        from solodet.data.adapters.lrddv2 import LRDDv2Adapter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        self._setup_raw(raw_dir)

        adapter = LRDDv2Adapter()
        adapter.convert(raw_dir, out_dir, "train")

        img_out = out_dir / "images" / "train"
        lbl_out = out_dir / "labels" / "train"
        assert img_out.is_dir()
        assert lbl_out.is_dir()

        images = sorted(img_out.iterdir())
        labels = sorted(lbl_out.iterdir())
        assert len(images) == 3
        assert len(labels) == 3

    def test_convert_label_content(self, tmp_path):
        from solodet.data.adapters.lrddv2 import LRDDv2Adapter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        self._setup_raw(raw_dir)

        adapter = LRDDv2Adapter()
        adapter.convert(raw_dir, out_dir, "train")

        lbl_out = out_dir / "labels" / "train"
        any_label = sorted(lbl_out.iterdir())[0]
        lines = any_label.read_text().strip().splitlines()
        assert len(lines) == 1
        parts = lines[0].split()
        # Class remapped to 0
        assert parts[0] == "0"
        # 5 fields total
        assert len(parts) == 5

    def test_convert_prefixes_filenames(self, tmp_path):
        from solodet.data.adapters.lrddv2 import LRDDv2Adapter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        self._setup_raw(raw_dir)

        adapter = LRDDv2Adapter()
        adapter.convert(raw_dir, out_dir, "train")

        img_out = out_dir / "images" / "train"
        names = [f.stem for f in sorted(img_out.iterdir())]
        assert all(n.startswith("lrddv2_") for n in names)

    def test_convert_invalid_split_raises(self, tmp_path):
        from solodet.data.adapters.lrddv2 import LRDDv2Adapter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        self._setup_raw(raw_dir)

        adapter = LRDDv2Adapter()
        with pytest.raises(ValueError, match="Split 'val' not found"):
            adapter.convert(raw_dir, out_dir, "val")


# ===================================================================
# 4. DroneDetect adapter
# ===================================================================


class TestDroneDetectAdapter:
    """Fake YOLO dataset -> convert -> verify class remapping to 0."""

    def _setup_raw(self, raw_dir: Path):
        img_dir = raw_dir / "images" / "train"
        lbl_dir = raw_dir / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(4):
            _make_image(img_dir / f"img{i}.jpg")
            # Multiple classes (2, 7) -- should all be remapped to 0
            _make_yolo_label(
                lbl_dir / f"img{i}.txt",
                [f"2 0.5 0.5 0.1 0.1", f"7 0.3 0.3 0.05 0.05"],
            )

    def test_convert_creates_structure(self, tmp_path):
        from solodet.data.adapters.dronedetect import DroneDetectAdapter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        self._setup_raw(raw_dir)

        adapter = DroneDetectAdapter()
        adapter.convert(raw_dir, out_dir, "train")

        assert (out_dir / "images" / "train").is_dir()
        assert (out_dir / "labels" / "train").is_dir()
        assert len(list((out_dir / "images" / "train").iterdir())) == 4
        assert len(list((out_dir / "labels" / "train").iterdir())) == 4

    def test_class_remapped_to_zero(self, tmp_path):
        from solodet.data.adapters.dronedetect import DroneDetectAdapter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        self._setup_raw(raw_dir)

        adapter = DroneDetectAdapter()
        adapter.convert(raw_dir, out_dir, "train")

        lbl_dir = out_dir / "labels" / "train"
        for lbl_path in lbl_dir.iterdir():
            for line in lbl_path.read_text().strip().splitlines():
                cls_id = int(line.split()[0])
                assert cls_id == 0, f"Expected class 0, got {cls_id} in {lbl_path.name}"

    def test_prefixed_filenames(self, tmp_path):
        from solodet.data.adapters.dronedetect import DroneDetectAdapter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        self._setup_raw(raw_dir)

        adapter = DroneDetectAdapter()
        adapter.convert(raw_dir, out_dir, "train")

        names = [f.stem for f in (out_dir / "images" / "train").iterdir()]
        assert all(n.startswith("dronedetect_") for n in names)

    def test_missing_label_creates_empty(self, tmp_path):
        """Image with no matching label file gets an empty label."""
        from solodet.data.adapters.dronedetect import DroneDetectAdapter

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"

        img_dir = raw_dir / "images" / "train"
        lbl_dir = raw_dir / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        _make_image(img_dir / "orphan.png")
        # No label file created

        adapter = DroneDetectAdapter()
        adapter.convert(raw_dir, out_dir, "train")

        out_lbl = out_dir / "labels" / "train" / "dronedetect_orphan.txt"
        assert out_lbl.is_file()
        assert out_lbl.read_text().strip() == ""


# ===================================================================
# 5. Merge
# ===================================================================


class TestMergeDatasets:
    """Create two fake processed datasets, merge, verify symlinks."""

    def _setup_processed(self, processed_dir: Path):
        """Create processed_dir/datasetA and processed_dir/datasetB."""
        for ds_name in ("datasetA", "datasetB"):
            for split in ("train", "val"):
                img_dir = processed_dir / ds_name / "images" / split
                lbl_dir = processed_dir / ds_name / "labels" / split
                img_dir.mkdir(parents=True)
                lbl_dir.mkdir(parents=True)

                for i in range(3):
                    fname = f"{ds_name}_{i:03d}"
                    _make_image(img_dir / f"{fname}.png")
                    _make_yolo_label(lbl_dir / f"{fname}.txt", ["0 0.5 0.5 0.1 0.1"])

    def test_merge_creates_symlinks(self, tmp_path):
        from solodet.data.merge import merge_datasets

        processed = tmp_path / "processed"
        merged = tmp_path / "merged"
        self._setup_processed(processed)

        merge_datasets(processed, merged)

        for split in ("train", "val"):
            imgs = list((merged / "images" / split).iterdir())
            lbls = list((merged / "labels" / split).iterdir())
            # 3 from A + 3 from B = 6
            assert len(imgs) == 6, f"Expected 6 images in {split}, got {len(imgs)}"
            assert len(lbls) == 6
            # Every entry is a symlink
            for p in imgs + lbls:
                assert p.is_symlink(), f"{p} should be a symlink"

    def test_merge_no_collisions(self, tmp_path):
        from solodet.data.merge import merge_datasets

        processed = tmp_path / "processed"
        merged = tmp_path / "merged"
        self._setup_processed(processed)

        merge_datasets(processed, merged)

        train_names = sorted(f.name for f in (merged / "images" / "train").iterdir())
        # All names should be unique (no duplicates)
        assert len(train_names) == len(set(train_names))

    def test_merge_specific_datasets(self, tmp_path):
        from solodet.data.merge import merge_datasets

        processed = tmp_path / "processed"
        merged = tmp_path / "merged"
        self._setup_processed(processed)

        merge_datasets(processed, merged, datasets=["datasetA"])

        train_imgs = list((merged / "images" / "train").iterdir())
        assert len(train_imgs) == 3  # only datasetA

    def test_merge_empty_split_still_creates_dir(self, tmp_path):
        """test split has no data but the directory is still created."""
        from solodet.data.merge import merge_datasets

        processed = tmp_path / "processed"
        merged = tmp_path / "merged"
        self._setup_processed(processed)

        merge_datasets(processed, merged)

        # Our setup only creates train and val, so test/ should be empty but exist
        assert (merged / "images" / "test").is_dir()
        assert len(list((merged / "images" / "test").iterdir())) == 0


# ===================================================================
# 6. Stats
# ===================================================================


class TestComputeStats:
    """Generate fake processed dataset with known box sizes, verify counts."""

    def _setup_dataset(self, data_dir: Path):
        """Create a dataset with deterministic box sizes.

        3 images:
          - img0: 1 box -> max_dim = 20px  (tiny: 0..30)
          - img1: 1 box -> max_dim = 40px  (small: 30..50)
          - img2: no box                   (negative frame)
        Image size: 200x200
        """
        img_dir = data_dir / "images" / "train"
        lbl_dir = data_dir / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        img_size = 200

        # img0: box 20px wide, 10px tall => max_dim = 20 => tiny
        _make_image(img_dir / "img0.png", img_size, img_size)
        w_norm = 20.0 / img_size  # 0.1
        h_norm = 10.0 / img_size  # 0.05
        _make_yolo_label(lbl_dir / "img0.txt", [f"0 0.5 0.5 {w_norm} {h_norm}"])

        # img1: box 40px wide, 30px tall => max_dim = 40 => small
        _make_image(img_dir / "img1.png", img_size, img_size)
        w_norm = 40.0 / img_size  # 0.2
        h_norm = 30.0 / img_size  # 0.15
        _make_yolo_label(lbl_dir / "img1.txt", [f"0 0.5 0.5 {w_norm} {h_norm}"])

        # img2: negative (empty label)
        _make_image(img_dir / "img2.png", img_size, img_size)
        _make_yolo_label(lbl_dir / "img2.txt", [])

    def test_compute_stats_counts(self, tmp_path):
        pd = pytest.importorskip("pandas")
        from solodet.data.stats import compute_stats

        data_dir = tmp_path / "dataset"
        self._setup_dataset(data_dir)

        df = compute_stats(data_dir, splits=["train"])

        assert len(df) == 1
        row = df.iloc[0]
        assert row["split"] == "train"
        assert row["images"] == 3
        assert row["positive"] == 2
        assert row["negative"] == 1
        assert row["boxes"] == 2

    def test_compute_stats_scale_bins(self, tmp_path):
        pd = pytest.importorskip("pandas")
        from solodet.data.stats import compute_stats

        data_dir = tmp_path / "dataset"
        self._setup_dataset(data_dir)

        df = compute_stats(data_dir, splits=["train"])
        row = df.iloc[0]

        assert row["n_tiny"] == 1   # 20px box
        assert row["n_small"] == 1  # 40px box
        assert row["n_medium"] == 0
        assert row["n_large"] == 0

    def test_compute_stats_custom_bins(self, tmp_path):
        pd = pytest.importorskip("pandas")
        from solodet.data.stats import compute_stats

        data_dir = tmp_path / "dataset"
        self._setup_dataset(data_dir)

        # Custom bins: everything under 50 is "micro"
        bins_yaml = tmp_path / "bins.yaml"
        bins_yaml.write_text(
            "bins:\n"
            "  micro:\n"
            "    min: 0\n"
            "    max: 50\n"
            "  big:\n"
            "    min: 50\n"
            "    max: 100000\n"
        )

        df = compute_stats(data_dir, scale_bins_config=bins_yaml, splits=["train"])
        row = df.iloc[0]
        assert row["n_micro"] == 2  # both 20px and 40px fall in micro
        assert row["n_big"] == 0


# ===================================================================
# 7. MultiFrameConfirmer
# ===================================================================


class TestMultiFrameConfirmer:
    """Validate multi-frame confirmation logic."""

    def test_not_confirmed_until_min_hits(self):
        from solodet.inference.multiframe import MultiFrameConfirmer

        mfc = MultiFrameConfirmer(min_hits=3, window=5)

        # First two frames: track 1 is seen but not yet confirmed
        det = {"track_id": 1, "confidence": 0.8, "bbox": [0, 0, 10, 10]}
        result = mfc.update(0, [det.copy()])
        assert len(result) == 0

        result = mfc.update(1, [det.copy()])
        assert len(result) == 0

        # Third frame: confirmed!
        result = mfc.update(2, [det.copy()])
        assert len(result) == 1
        assert result[0]["confirmed"] is True
        assert result[0]["hits"] == 3

    def test_window_pruning(self):
        from solodet.inference.multiframe import MultiFrameConfirmer

        mfc = MultiFrameConfirmer(min_hits=3, window=3)

        det = {"track_id": 1, "confidence": 0.9, "bbox": [0, 0, 10, 10]}

        # Frames 0, 1, 2: build up to 3 hits -> confirmed
        mfc.update(0, [det.copy()])
        mfc.update(1, [det.copy()])
        result = mfc.update(2, [det.copy()])
        assert len(result) == 1  # confirmed (3 hits in window [0,1,2])

        # Frame 3: track NOT seen -> still has hits from frames 1,2
        result = mfc.update(3, [])
        assert len(result) == 0

        # Frame 4: track NOT seen -> only frame 2 remains in window [2,3,4]
        result = mfc.update(4, [])
        assert len(result) == 0

        # Frame 5: track seen again but only 1 hit in window [3,4,5]
        result = mfc.update(5, [det.copy()])
        assert len(result) == 0

    def test_multiple_tracks(self):
        from solodet.inference.multiframe import MultiFrameConfirmer

        mfc = MultiFrameConfirmer(min_hits=2, window=5)

        det_a = {"track_id": 1, "confidence": 0.8, "bbox": [0, 0, 10, 10]}
        det_b = {"track_id": 2, "confidence": 0.7, "bbox": [20, 20, 30, 30]}

        # Frame 0: both tracks
        mfc.update(0, [det_a.copy(), det_b.copy()])

        # Frame 1: only track 1
        result = mfc.update(1, [det_a.copy()])
        # Track 1 has 2 hits -> confirmed; track 2 not present
        assert len(result) == 1
        assert result[0]["track_id"] == 1

        # Frame 2: only track 2
        result = mfc.update(2, [det_b.copy()])
        # Track 2 now has 2 hits -> confirmed
        assert len(result) == 1
        assert result[0]["track_id"] == 2

    def test_reset(self):
        from solodet.inference.multiframe import MultiFrameConfirmer

        mfc = MultiFrameConfirmer(min_hits=2, window=5)

        det = {"track_id": 1, "confidence": 0.8, "bbox": [0, 0, 10, 10]}
        mfc.update(0, [det.copy()])
        mfc.update(1, [det.copy()])

        mfc.reset()

        # After reset, track history is gone; need min_hits again
        result = mfc.update(2, [det.copy()])
        assert len(result) == 0


# ===================================================================
# 8. Visualization
# ===================================================================


class TestVisualization:
    """draw_detections on a blank image."""

    def test_draw_detections_returns_correct_shape(self):
        from solodet.utils.viz import draw_detections

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.85, "track_id": 0},
            {"bbox": [300, 50, 400, 150], "confidence": 0.72, "track_id": 1, "confirmed": True},
        ]

        result = draw_detections(blank, dets)
        assert result.shape == blank.shape
        assert result.dtype == blank.dtype

    def test_draw_detections_modifies_pixels(self):
        from solodet.utils.viz import draw_detections

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [{"bbox": [50, 50, 200, 200], "confidence": 0.9}]

        result = draw_detections(blank, dets)
        # Some pixels should have changed (box + label drawn)
        assert not np.array_equal(result, blank)

    def test_draw_detections_does_not_mutate_input(self):
        from solodet.utils.viz import draw_detections

        original = np.zeros((480, 640, 3), dtype=np.uint8)
        snapshot = original.copy()
        dets = [{"bbox": [10, 10, 100, 100], "confidence": 0.5}]

        draw_detections(original, dets)
        np.testing.assert_array_equal(original, snapshot)

    def test_draw_empty_detections(self):
        from solodet.utils.viz import draw_detections

        blank = np.full((100, 100, 3), 42, dtype=np.uint8)
        result = draw_detections(blank, [])
        np.testing.assert_array_equal(result, blank)

    def test_draw_unconfirmed_detection(self):
        from solodet.utils.viz import draw_detections

        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        dets = [{"bbox": [10, 10, 50, 50], "confidence": 0.4, "confirmed": False}]
        result = draw_detections(blank, dets)
        # Should draw something (gray box for unconfirmed)
        assert not np.array_equal(result, blank)


# ===================================================================
# 9. Video I/O
# ===================================================================


class TestVideoIO:
    """Write a tiny video, read it back, verify properties."""

    def _write_test_video(self, video_path: Path, n_frames: int = 5, w: int = 64, h: int = 48):
        from solodet.utils.io import VideoWriter

        writer = VideoWriter(video_path, fps=10.0, width=w, height=h)
        for i in range(n_frames):
            frame = np.full((h, w, 3), i * 40, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_write_and_read_video(self, tmp_path):
        from solodet.utils.io import VideoReader

        video_path = tmp_path / "test.mp4"
        self._write_test_video(video_path, n_frames=5, w=64, h=48)

        assert video_path.is_file()
        reader = VideoReader(video_path)
        assert reader.width == 64
        assert reader.height == 48
        assert reader.frame_count == 5

    def test_read_frames_iteration(self, tmp_path):
        from solodet.utils.io import VideoReader

        video_path = tmp_path / "test.mp4"
        self._write_test_video(video_path, n_frames=3)

        reader = VideoReader(video_path)
        frames = list(reader)
        assert len(frames) == 3
        for frame in frames:
            assert frame.shape == (48, 64, 3)

    def test_video_reader_len(self, tmp_path):
        from solodet.utils.io import VideoReader

        video_path = tmp_path / "test.mp4"
        self._write_test_video(video_path, n_frames=7)

        reader = VideoReader(video_path)
        assert len(reader) == 7

    def test_video_reader_invalid_path_raises(self, tmp_path):
        from solodet.utils.io import VideoReader

        with pytest.raises(IOError, match="Cannot open video"):
            VideoReader(tmp_path / "nonexistent.mp4")

    def test_extract_frames(self, tmp_path):
        from solodet.utils.io import extract_frames

        video_path = tmp_path / "test.mp4"
        self._write_test_video(video_path, n_frames=6, w=64, h=48)

        out_dir = tmp_path / "frames"
        saved = extract_frames(video_path, out_dir, every_n=1)

        assert len(saved) == 6
        for p in saved:
            assert p.is_file()
            img = cv2.imread(str(p))
            assert img is not None
            assert img.shape[:2] == (48, 64)

    def test_extract_frames_every_n(self, tmp_path):
        from solodet.utils.io import extract_frames

        video_path = tmp_path / "test.mp4"
        self._write_test_video(video_path, n_frames=6, w=64, h=48)

        out_dir = tmp_path / "frames_skip"
        saved = extract_frames(video_path, out_dir, every_n=2)

        # Frames 0, 2, 4 => 3 frames
        assert len(saved) == 3


# ===================================================================
# 10. Attention modules (requires torch)
# ===================================================================


class TestAttentionModules:
    """Import CBAM and ECA, verify forward pass shapes.

    Guarded with pytest.importorskip so these tests are silently skipped
    when torch is not installed.
    """

    def test_cbam_forward_shape(self):
        torch = pytest.importorskip("torch")
        from solodet.model.attention import CBAM

        channels = 16
        model = CBAM(channels=channels, reduction=4)
        x = torch.randn(2, channels, 8, 8)
        out = model(x)
        assert out.shape == x.shape

    def test_cbam_preserves_dtype(self):
        torch = pytest.importorskip("torch")
        from solodet.model.attention import CBAM

        model = CBAM(channels=32)
        x = torch.randn(1, 32, 4, 4)
        out = model(x)
        assert out.dtype == x.dtype

    def test_eca_forward_shape(self):
        torch = pytest.importorskip("torch")
        from solodet.model.attention import ECA

        channels = 64
        model = ECA(channels=channels)
        x = torch.randn(2, channels, 8, 8)
        out = model(x)
        assert out.shape == x.shape

    def test_eca_small_channels(self):
        torch = pytest.importorskip("torch")
        from solodet.model.attention import ECA

        # Edge case: very small channel count
        channels = 8
        model = ECA(channels=channels)
        x = torch.randn(1, channels, 4, 4)
        out = model(x)
        assert out.shape == x.shape

    def test_cbam_channel_attention_alone(self):
        torch = pytest.importorskip("torch")
        from solodet.model.attention import ChannelAttention

        ca = ChannelAttention(channels=16, reduction=4)
        x = torch.randn(2, 16, 6, 6)
        out = ca(x)
        assert out.shape == x.shape

    def test_cbam_spatial_attention_alone(self):
        torch = pytest.importorskip("torch")
        from solodet.model.attention import SpatialAttention

        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 16, 8, 8)
        out = sa(x)
        assert out.shape == x.shape
