"""DroneDetector: standard YOLO and SAHI tiled inference."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from solodet.utils.config import load_config


class DroneDetector:
    """Unified drone detector supporting both standard and SAHI inference.

    Args:
        weights: Path to YOLO weights file (.pt).
        sahi_config: Path to SAHI config YAML, or None to disable tiling.
        device: Torch device string (e.g. 'cuda:0', 'cpu').
        conf: Confidence threshold override (uses config default if None).
    """

    def __init__(
        self,
        weights: str | Path,
        sahi_config: str | Path | None = None,
        device: str = "cuda:0",
        conf: float | None = None,
    ):
        self.model = YOLO(str(weights))
        self.device = device
        self.sahi_enabled = sahi_config is not None
        self.sahi_cfg = load_config(sahi_config) if sahi_config else {}
        self.conf = conf or self.sahi_cfg.get("confidence_threshold", 0.15)

    def predict(self, frame: np.ndarray) -> list[dict]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of detections, each a dict with keys:
                bbox: [x1, y1, x2, y2] in pixel coords
                confidence: float
                class_id: int (always 0 for drone)
        """
        if self.sahi_enabled:
            return self._predict_sahi(frame)
        return self._predict_standard(frame)

    def _predict_standard(self, frame: np.ndarray) -> list[dict]:
        """Standard full-image YOLO inference."""
        results = self.model.predict(
            frame, conf=self.conf, device=self.device, verbose=False
        )
        return self._parse_results(results)

    def _predict_sahi(self, frame: np.ndarray) -> list[dict]:
        """SAHI tiled inference + optional full-image pass."""
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(self.model.ckpt_path),
            confidence_threshold=self.conf,
            device=self.device,
        )

        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=self.sahi_cfg.get("slice_height", 640),
            slice_width=self.sahi_cfg.get("slice_width", 640),
            overlap_height_ratio=self.sahi_cfg.get("overlap_height_ratio", 0.25),
            overlap_width_ratio=self.sahi_cfg.get("overlap_width_ratio", 0.25),
            perform_standard_pred=self.sahi_cfg.get("perform_standard_pred", True),
            postprocess_type=self.sahi_cfg.get("postprocess_type", "NMS"),
            postprocess_match_metric=self.sahi_cfg.get("postprocess_match_metric", "IOS"),
            postprocess_match_threshold=self.sahi_cfg.get("postprocess_match_threshold", 0.5),
        )

        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            detections.append({
                "bbox": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
                "confidence": pred.score.value,
                "class_id": 0,
            })
        return detections

    @staticmethod
    def _parse_results(results) -> list[dict]:
        """Parse Ultralytics results into standard detection dicts."""
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append({
                    "bbox": xyxy.tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                })
        return detections
