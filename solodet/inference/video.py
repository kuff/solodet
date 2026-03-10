"""VideoPipeline: detect -> track -> confirm on video sequences."""

from pathlib import Path

import cv2
import numpy as np

from solodet.inference.detector import DroneDetector
from solodet.inference.tracker import DroneTracker
from solodet.inference.multiframe import MultiFrameConfirmer
from solodet.utils.config import load_config
from solodet.utils.viz import draw_detections


class VideoPipeline:
    """Full video inference pipeline: SAHI detection + ByteTrack + multi-frame confirmation.

    Args:
        weights: Path to YOLO weights.
        sahi_config: Path to SAHI config YAML (None to disable).
        tracker_config: Path to tracker config YAML (None for defaults).
        device: Torch device string.
        conf: Confidence threshold override.
    """

    def __init__(
        self,
        weights: str | Path,
        sahi_config: str | Path | None = None,
        tracker_config: str | Path | None = None,
        device: str = "cuda:0",
        conf: float | None = None,
    ):
        self.detector = DroneDetector(weights, sahi_config, device, conf)
        self.tracker = DroneTracker(tracker_config)

        # Load multi-frame params from tracker config
        tracker_cfg = load_config(tracker_config) if tracker_config else {}
        mf = tracker_cfg.get("multiframe", {})
        self.confirmer = MultiFrameConfirmer(
            min_hits=mf.get("min_hits", 3),
            window=mf.get("window", 5),
        )

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
    ) -> list[list[dict]]:
        """Run full pipeline on a video file.

        Args:
            video_path: Input video path.
            output_path: If set, write annotated output video.

        Returns:
            List of per-frame confirmed detections with track IDs.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        all_detections = []
        frame_idx = 0

        try:
            from tqdm import tqdm
            pbar = tqdm(total=total, desc="Processing video")
        except ImportError:
            pbar = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect -> Track -> Confirm
            raw_dets = self.detector.predict(frame)
            tracked_dets = self.tracker.update(raw_dets)
            confirmed_dets = self.confirmer.update(frame_idx, tracked_dets)

            all_detections.append(confirmed_dets)

            if writer:
                annotated = draw_detections(frame, confirmed_dets)
                writer.write(annotated)

            frame_idx += 1
            if pbar:
                pbar.update(1)

        cap.release()
        if writer:
            writer.release()
        if pbar:
            pbar.close()

        return all_detections

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> list[dict]:
        """Process a single frame through the full pipeline.

        Args:
            frame: BGR image.
            frame_idx: Frame index for multi-frame confirmation.

        Returns:
            Confirmed detections for this frame.
        """
        raw_dets = self.detector.predict(frame)
        tracked_dets = self.tracker.update(raw_dets)
        return self.confirmer.update(frame_idx, tracked_dets)

    def reset(self) -> None:
        """Reset tracker and confirmer state for a new video."""
        self.tracker.reset()
        self.confirmer.reset()
