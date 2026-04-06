import logging
import torch
import numpy as np
from typing import Dict, Any
from ultralytics import YOLO

from app.schema import Detection, DetectionResponse


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
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.ready = True
            logger.info("YOLO Engine initialized and ready for inference.")
        except Exception as e:
            self.ready = False
            logger.error(f"Critical failure loading YOLO model: {e}", exc_info=True)

    def predict(self, img: np.ndarray, confidence: float = 0.7) -> Dict[str, Any]:
        """
        Performs object detection inference on the provided image.
        """
        if not self.ready or self.model is None:
            logger.error("Inference attempted on uninitialized or failed engine.")
            return {"status": "error", "message": "Engine not initialized"}

        try:
            results = self.model.predict(img, conf=confidence, verbose=False)
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

            return DetectionResponse(status="success", result=detections).model_dump()

        except Exception as e:
            logger.exception(f"Inference error: {e}")
            return {"status": "error", "message": str(e)}

    def is_healthy(self) -> bool:
        """
        Performs a heartbeat check on the model and hardware.
        """
        try:
            # 1. Check if the wrapper itself exists
            if not self.ready or self.model is None:
                return False

            # 2. Safely check if the underlying PyTorch model is loaded
            torch_model = getattr(self.model, "model", None)

            if torch_model is None:
                logger.warning(
                    "YOLO wrapper exists but underlying torch model is None."
                )
                return False

            # 3. Verify parameters are accessible
            _ = next(torch_model.parameters())

            # 4. Hardware check
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.error("CUDA device lost or unavailable.")
                return False

            return True
        except StopIteration:
            # This happens if the model has no parameters (very unlikely for YOLO)
            return True
        except Exception as e:
            logger.error(f"Health check failed with error: {e}")
            return False
