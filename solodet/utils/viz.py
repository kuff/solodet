"""Visualization utilities: draw bboxes, track IDs, confidence on frames."""

import cv2
import numpy as np


# Color palette for track IDs
_COLORS = [
    (0, 255, 0),    # green
    (0, 255, 255),  # yellow
    (255, 0, 0),    # blue
    (255, 0, 255),  # magenta
    (0, 165, 255),  # orange
    (255, 255, 0),  # cyan
    (128, 0, 255),  # purple
    (0, 128, 255),  # light orange
]


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw bounding boxes with track IDs and confidence on a frame.

    Args:
        frame: BGR image (will be copied, not modified in-place).
        detections: List of detection dicts with 'bbox', optionally
            'track_id', 'confidence', 'confirmed'.
        thickness: Line thickness for boxes.
        font_scale: Font scale for labels.

    Returns:
        Annotated frame copy.
    """
    img = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        track_id = det.get("track_id")
        conf = det.get("confidence", 0.0)
        confirmed = det.get("confirmed", True)

        # Color by track ID
        if track_id is not None:
            color = _COLORS[track_id % len(_COLORS)]
        else:
            color = (0, 255, 0)

        # Dashed box for unconfirmed
        if not confirmed:
            color = (128, 128, 128)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label
        parts = []
        if track_id is not None:
            parts.append(f"ID:{track_id}")
        parts.append(f"{conf:.2f}")
        label = " ".join(parts)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

    return img
