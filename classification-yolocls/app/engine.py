import logging
import torch
import numpy as np
from typing import List
from ultralytics import YOLO
from app.schema import Classification, InferenceResult, SpeedMetrics


# Standardized logging
logger = logging.getLogger(__name__)


class YoloClsEngine:
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

    def predict(self, img: np.ndarray, top_k: int = 3) -> InferenceResult:
        """
        Performs object classification inference on the provided image.
        """
        if not self.ready or self.model is None:
            logger.error("Inference attempted on uninitialized or failed engine.")
            return InferenceResult(
                status="error",
                message="Engine not initialized",
                results=[],
                speed_ms=None,
            )

        try:
            results = self.model(img, verbose=False)

            r = results[0]

            # 1. Extract the internal YOLO speed metrics
            # These are usually in milliseconds
            speed_metrics = getattr(
                r, "speed", {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
            )

            # 2. Extract Top-K Results
            classifications: List[Classification] = []
            if r.probs is not None:
                topk_indices = r.probs.top5[:top_k]
                topk_confidences = r.probs.top5conf[:top_k].tolist()

                for class_id, confidence in zip(topk_indices, topk_confidences):
                    classifications.append(
                        Classification(
                            class_id=int(class_id),
                            class_name=r.names[int(class_id)],
                            confidence=round(float(confidence), 4),
                        )
                    )

            # 3. Return a combined response
            return InferenceResult(
                status="success",
                results=classifications,
                speed_ms=SpeedMetrics(
                    preprocess=round(speed_metrics.get("preprocess", 0), 2),
                    inference=round(speed_metrics.get("inference", 0), 2),
                    postprocess=round(speed_metrics.get("postprocess", 0), 2),
                    total=round(sum(speed_metrics.values()), 2),
                ),
            )

        except Exception as e:
            logger.exception(f"Inference error: {e}")
            return InferenceResult(
                status="error",
                message=str(e),
                results=[],
                speed_ms=None,
            )

    def is_healthy(self) -> bool:
        """
        Performs a heartbeat check on the model and hardware.
        """
        # 1. Check if the wrapper itself exists
        if not self.ready or self.model is None:
            return False

        try:
            # 2. Safely check if the underlying PyTorch model is loaded
            # Ultralytics stores the torch.nn.Module in the .model attribute
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
