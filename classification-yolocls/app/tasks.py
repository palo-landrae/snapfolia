import cv2
import logging
from typing import Any, Dict, Optional

from app.client import celery_app
from app.engine import YoloClsEngine
from app.schema import ClassificationResponse
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
def classify_objects(self, image_path: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Celery task to perform object classification on an image.
    Includes caching based on image content hash.
    """
    logger.info(
        f"Starting classification task for image: {image_path} (top_k: {top_k})"
    )

    try:
        # 1. Compute image hash for caching
        try:
            img_hash = cache_service.compute_image_hash(image_path)
        except FileNotFoundError:
            logger.error(f"Image file not found at path: {image_path}")
            return ClassificationResponse(
                status="error",
                message="File not found",
                results=[],
                speed_ms=None,
            ).model_dump()

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
        engine = ClassificationTask.get_engine()
        inference_results = engine.predict(img, top_k=top_k)

        message = f"Task completed successfully. Found {len(inference_results.results)} objects."

        response_dict = ClassificationResponse(
            status=inference_results.status,
            results=inference_results.results,
            speed_ms=inference_results.speed_ms,
            task_id=self.request.id,
            image_path=image_path,
            file_hash=img_hash,
            message=message,
        )

        logger.info(
            message + f" (task_id: {self.request.id}, image_path: {image_path})"
        )

        json_response = response_dict.model_dump()

        cache_service.set(img_hash, json_response)

        return json_response

    except Exception as exc:
        logger.exception(f"Unexpected error during classification task: {exc}")
        # Automatically retry the task if it's a transient error
        # (e.g., Redis connection issues or temporary hardware timeout)
        raise self.retry(exc=exc)
