"""Multi-frame confirmation to suppress transient false positives."""

from collections import defaultdict


class MultiFrameConfirmer:
    """Confirm tracks only if detected in enough recent frames.

    Critical for tiny-target regime where single-frame confidence is low
    and noise can produce transient false positives.

    Args:
        min_hits: Minimum detections within the window to confirm a track.
        window: Sliding window size in frames.
    """

    def __init__(self, min_hits: int = 3, window: int = 5):
        self.min_hits = min_hits
        self.window = window
        # track_id -> list of (frame_idx, confidence)
        self._history: dict[int, list[tuple[int, float]]] = defaultdict(list)

    def update(self, frame_idx: int, tracked_detections: list[dict]) -> list[dict]:
        """Update with tracked detections and return only confirmed ones.

        Args:
            frame_idx: Current frame index.
            tracked_detections: Detections with 'track_id' from tracker.

        Returns:
            Only those detections whose track has been seen in >= min_hits
            of the last `window` frames.
        """
        # Record current detections
        seen_ids = set()
        for det in tracked_detections:
            tid = det["track_id"]
            seen_ids.add(tid)
            self._history[tid].append((frame_idx, det["confidence"]))

        # Prune old entries outside window
        cutoff = frame_idx - self.window + 1
        for tid in list(self._history.keys()):
            self._history[tid] = [
                (f, c) for f, c in self._history[tid] if f >= cutoff
            ]
            if not self._history[tid]:
                del self._history[tid]

        # Filter to confirmed tracks
        confirmed = []
        for det in tracked_detections:
            tid = det["track_id"]
            hits = len(self._history.get(tid, []))
            if hits >= self.min_hits:
                det["confirmed"] = True
                det["hits"] = hits
                confirmed.append(det)

        return confirmed

    def reset(self) -> None:
        """Reset confirmation history."""
        self._history.clear()
