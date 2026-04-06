import cv2
import logging
from typing import Any, Dict, Optional

from app.client import celery_app
from app.engine import YoloClsEngine
from app.schema import ClassificationResponse, DetectionResponse
from app.settings import settings
from app.redis_cache import cache_service

# Setup structured logging
logger = logging.getLogger(__name__)


class ClassificationTask:
    """
    Encapsulates the object classification logic to avoid global variable issues
    and improve testability.
    """

    _engine: Optional[YoloClsEngine] = None

    @classmethod
    def get_engine(cls) -> YoloClsEngine:
        """Lazy loader for the YOLO engine to conserve worker memory."""
        if cls._engine is None:
            logger.info(
                f"Initializing YOLO engine with model: {settings.CLS_MODEL_PATH}"
            )
            cls._engine = YoloClsEngine(settings.CLS_MODEL_PATH)
        return cls._engine


@celery_app.task(
    name="classify_objects_task",
    bind=True,  # Allows access to task instance (self)
    max_retries=3,  # Retries on transient failures
    default_retry_delay=5,  # Seconds between retries
)
def classify_objects(self, detection_response:  Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task to perform object classification on an image.
    Includes caching based on image content hash.
    """

    try:
        data = DetectionResponse.model_validate(detection_response)
        image_path = data.cropped_image_path
        if image_path is None:
            logger.error("Received None instead of a valid image path")
            return {"status": "error", "message": "No image path provided"}

        # 3. Load Image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"OpenCV failed to load image: {image_path}")
            raise ValueError(f"Invalid image format or corrupted file: {image_path}")

        # 4. Inference
        engine = ClassificationTask.get_engine()
        inference_results = engine.predict(img, top_k=3)

        message = f"Task completed successfully. Found {len(inference_results.results)} objects."

        response_dict = ClassificationResponse(
            status=inference_results.status,
            results=inference_results.results,
            speed_ms=inference_results.speed_ms,
            task_id=self.request.id,
            image_path=image_path,
            message=message,
        )

        logger.info(
            message + f" (task_id: {self.request.id}, image_path: {image_path})"
        )

        json_response = response_dict.model_dump()

        return json_response

    except Exception as exc:
        logger.exception(f"Unexpected error during classification task: {exc}")
        # Automatically retry the task if it's a transient error
        # (e.g., Redis connection issues or temporary hardware timeout)
        raise self.retry(exc=exc)
