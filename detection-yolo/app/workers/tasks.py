import cv2
import logging
from typing import Any, Dict, Optional

from app.workers.client import celery_app
from app.services.yolo_engine import YoloEngine
from app.schemas.detection import Detection, DetectionResponse
from app.services.redis_cache import cache_service
from app.core.settings import settings

# Setup structured logging
logger = logging.getLogger(__name__)


class DetectionTask:
    """
    Encapsulates the object detection logic to avoid global variable issues
    and improve testability.
    """

    _engine: Optional[YoloEngine] = None

    @classmethod
    def get_engine(cls) -> YoloEngine:
        """Lazy loader for the YOLO engine to conserve worker memory."""
        if cls._engine is None:
            logger.info(f"Initializing YOLO engine with model: {settings.MODEL_PATH}")
            cls._engine = YoloEngine(settings.MODEL_PATH)
        return cls._engine


@celery_app.task(
    name="detect_objects_task",
    bind=True,  # Allows access to task instance (self)
    max_retries=3,  # Retries on transient failures
    default_retry_delay=5,  # Seconds between retries
)
def detect_objects(self, image_path: str, confidence: float = 0.2) -> Dict[str, Any]:
    """
    Celery task to perform object detection on an image.
    Includes caching based on image content hash.
    """
    logger.info(
        f"Starting detection task for image: {image_path} (confidence: {confidence})"
    )

    try:
        # 1. Compute image hash for caching
        try:
            img_hash = cache_service.compute_image_hash(image_path)
        except FileNotFoundError:
            logger.error(f"Image file not found at path: {image_path}")
            return {"status": "error", "message": "File not found"}

        # 2. Check Cache
        cached_result = cache_service.get(img_hash)
        if cached_result:
            logger.info(f"Cache hit for image hash: {img_hash}")
            return cached_result

        # 3. Load Image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"OpenCV failed to load image: {image_path}")
            raise ValueError(f"Invalid image format or corrupted file: {image_path}")

        # 4. Inference
        engine = DetectionTask.get_engine()
        inference_results = engine.predict(img, confidence=confidence)

        # 5. Serialization via Pydantic
        # Using model_validate then model_dump ensures the data matches our schema
        raw_detections = inference_results.get("result", [])
        detections = [Detection.model_validate(d) for d in raw_detections]

        response = DetectionResponse(
            status="success",
            task_id=self.request.id,
            image_path=image_path,
            result=detections,
            file_hash=img_hash,
        )

        result_dict = response.model_dump()

        cache_service.set(img_hash, result_dict)

        logger.info(f"Task completed successfully. Found {len(detections)} objects.")

        return result_dict

    except Exception as exc:
        logger.exception(f"Unexpected error during detection task: {exc}")
        # Automatically retry the task if it's a transient error
        # (e.g., Redis connection issues or temporary hardware timeout)
        raise self.retry(exc=exc)
