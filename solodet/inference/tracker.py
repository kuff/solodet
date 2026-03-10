"""ByteTrack bridge for tracking detections across frames."""

import numpy as np
import supervision as sv

from solodet.utils.config import load_config
from pathlib import Path


class DroneTracker:
    """ByteTrack-based multi-object tracker for drone detections.

    Args:
        config: Path to tracker config YAML, or None for defaults.
    """

    def __init__(self, config: str | Path | None = None):
        cfg = load_config(config) if config else {}
        self.tracker = sv.ByteTrack(
            track_activation_threshold=cfg.get("track_high_thresh", 0.15),
            lost_track_buffer=cfg.get("track_buffer", 60),
            minimum_matching_threshold=cfg.get("match_thresh", 0.8),
            frame_rate=30,
        )

    def update(self, detections: list[dict]) -> list[dict]:
        """Update tracker with new frame detections.

        Args:
            detections: List of detection dicts with 'bbox' and 'confidence'.

        Returns:
            List of tracked detection dicts with added 'track_id' key.
        """
        if not detections:
            # Still update tracker to age out lost tracks
            empty = sv.Detections.empty()
            self.tracker.update_with_detections(empty)
            return []

        bboxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confs = np.array([d["confidence"] for d in detections], dtype=np.float32)
        class_ids = np.array([d.get("class_id", 0) for d in detections], dtype=int)

        sv_detections = sv.Detections(
            xyxy=bboxes,
            confidence=confs,
            class_id=class_ids,
        )

        tracked = self.tracker.update_with_detections(sv_detections)

        results = []
        if tracked.tracker_id is not None:
            for i in range(len(tracked)):
                results.append({
                    "bbox": tracked.xyxy[i].tolist(),
                    "confidence": float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                    "class_id": int(tracked.class_id[i]) if tracked.class_id is not None else 0,
                    "track_id": int(tracked.tracker_id[i]),
                })

        return results

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracker.reset()
