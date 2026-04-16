import cv2
import logging
from typing import Any, Dict, Optional
from celery.signals import worker_process_init
from app.services.celery_client import celery_app
from app.services.cls_engine import EfficientNetEngine
from app.schema import ClassificationResponse, DetectionResponse
from app.settings import settings
from app.services.redis_client import RedisClient


# Setup structured logging
logger = logging.getLogger(__name__)


class WorkerContainer:
    """
    Encapsulates the object classification logic to avoid global variable issues
    and improve testability.
    """

    _engine: Optional[EfficientNetEngine] = None
    _cache_service: Optional[RedisClient] = None

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            cls._engine = EfficientNetEngine(
                settings.CLS_MODEL_PATH, settings.CLASS_MAP
            )
        return cls._engine

    @classmethod
    def get_cache_service(cls):
        if cls._cache_service is None:
            cls._cache_service = RedisClient(settings.REDIS_URL)
        return cls._cache_service


@worker_process_init.connect
def pre_load_model(**kwargs):
    """
    Optional: Warm up the model as soon as the worker starts.
    This prevents the very first request from being slow.
    """
    WorkerContainer.get_engine()


@celery_app.task(
    name="classify_objects_task",
    bind=True,  # Allows access to task instance (self)
    max_retries=3,  # Retries on transient failures
    default_retry_delay=5,  # Seconds between retries
)
def classify_objects(self, detection_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task to perform object classification on an image.
    Includes caching based on image content hash.
    """

    try:
        data = DetectionResponse.model_validate(detection_response)
        logging.info(
            f"Received classification task for image: {data.detections.results[0] if data.detections else 'N/A'}"
        )
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
        engine = WorkerContainer.get_engine()
        inference_results = engine.predict(img, top_k=3)

        message = f"Task completed successfully. Found {len(inference_results.results)} objects."

        response_dict = ClassificationResponse(
            status=inference_results.status,
            task_id=str(self.request.id),
            message=message,
            classifications=inference_results,
            detections=data.detections,
            original_image_path=data.original_image_path,
            cropped_image_path=data.cropped_image_path,
            file_hash=data.file_hash,
        )

        logger.info(
            message + f" (task_id: {self.request.id}, image_path: {image_path})"
        )

        json_response = response_dict.model_dump(mode="json")

        cache_service = WorkerContainer.get_cache_service()

        file_hash = data.file_hash
        if file_hash is None:
            logger.error("Received None instead of a valid file hash")
            return {"status": "error", "message": "No file hash provided"}
        cache_key = cache_service.generate_cache_key(file_hash, version="v1")
        cache_service.set_json(cache_key, json_response)
        logger.info(f"Cached results for {image_path} with key {cache_key}")

        cache_service.push_to_stream("scan_stream", json_response)

        return json_response

    except Exception as exc:
        logger.exception(f"Unexpected error during classification task: {exc}")
        # Automatically retry the task if it's a transient error
        # (e.g., Redis connection issues or temporary hardware timeout)
        raise self.retry(exc=exc)
