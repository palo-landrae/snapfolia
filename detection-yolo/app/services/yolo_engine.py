import logging
import torch
import numpy as np
from ultralytics import YOLO

from app.schema import Detection, InferenceResult, SpeedMetrics


logger = logging.getLogger(__name__)


class YoloEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.ready = False

        self._load_model()

    def _load_model(self) -> None:
        """Internal helper to load the YOLO model onto the specific device."""
        try:
            logger.info(
                f"Loading YOLO model from {self.model_path} onto {self.device}..."
            )
            self.model = YOLO(self.model_path, task="detect")

            self.ready = True
            logger.info("YOLO Engine initialized and ready for inference.")
        except Exception as e:
            self.ready = False
            logger.error(f"Critical failure loading YOLO model: {e}", exc_info=True)

    def predict(self, img: np.ndarray, confidence: float = 0.7) -> InferenceResult:
        """
        Performs object detection inference on the provided image.
        """
        if not self.ready or self.model is None:
            logger.error("Inference attempted on uninitialized or failed engine.")
            return InferenceResult(status="error", message="Engine not initialized")

        try:
            results = self.model.predict(img, conf=confidence, verbose=False)
            logging.info(
                f"Inference completed for image. Processing results... {results}"
            )

            raw_speed_metrics = getattr(
                results[0],
                "speed",
                {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0},
            )
            speed_metrics = SpeedMetrics(
                preprocess=round(raw_speed_metrics.get("preprocess", 0), 2),
                inference=round(raw_speed_metrics.get("inference", 0), 2),
                postprocess=round(raw_speed_metrics.get("postprocess", 0), 2),
                total=round(sum(raw_speed_metrics.values()), 2),
            )

            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                names = getattr(r, "names", {})

                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = names.get(class_id, "unknown")
                    detection = Detection(
                        class_name=class_name,
                        class_id=class_id,
                        confidence=round(float(box.conf[0]), 4),
                        bbox=[round(x, 2) for x in box.xyxy[0].tolist()],
                    )
                    detections.append(detection)

            return InferenceResult(
                status="success",
                message=f"Inference completed successfully. Detected {len(detections)} objects.",
                results=detections,
                speed_ms=speed_metrics,
                device=self.device,
            )

        except Exception as e:
            logger.exception(f"Inference error: {e}")
            return InferenceResult(status="error", message=str(e))

    def is_healthy(self) -> bool:
        try:
            if not self.ready or self.model is None:
                return False

            # If it's TensorRT, we just check if the AutoBackend is loaded
            # We avoid calling .parameters() entirely
            if hasattr(self.model, 'model'):
                return True
                
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False