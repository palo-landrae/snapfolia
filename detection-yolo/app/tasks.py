import cv2
import logging
from pathlib import Path
from typing import Any, Dict
from celery.signals import worker_process_init
from app.services.redis_client import RedisClient
from app.settings import settings
from app.schema import Detection, DetectionResponse
from app.services.celery_client import celery_app
from app.services.yolo_engine import YoloEngine
from app.services.image_manager import ImageManager

logger = logging.getLogger(__name__)


class WorkerContainer:
    """Singleton container to keep models in worker memory."""

    _engine = None
    _image_manager = None
    _cache_service = None

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            cls._engine = YoloEngine(settings.YOLO_MODEL_PATH)
        return cls._engine

    @classmethod
    def get_image_manager(cls):
        if cls._image_manager is None:
            cls._image_manager = ImageManager()
        return cls._image_manager

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
    name="object_detection_task", bind=True, max_retries=3, default_retry_delay=10
)
def detect_object_and_crop(
    self, image_path_str: str, confidence: float = 0.7
) -> Dict[str, Any]:
    """
    Object detection pipeline: Detect -> Crop -> Save -> Return Schema. Uses the globally cached engine.
    """
    logger.info(
        f"Received task for image: {image_path_str} with confidence: {confidence}"
    )

    try:
        # 1. Check Cache First
        image_path = Path(image_path_str)

        cache_service = WorkerContainer.get_cache_service()
        file_hash = cache_service.compute_file_hash(str(image_path))
        cache_key = cache_service.generate_cache_key(file_hash, settings.MODEL_VERSION)

        # If we have a cached result for this exact file and model version, return it immediately
        cached_data = cache_service.get_json(cache_key)
        if cached_data:
            logger.info(f"Cache hit for {image_path_str} with key {cache_key}")
            return DetectionResponse(
                **{**cached_data, "status": "cached"},
            ).model_dump()

        # 2. Else, load the image and run inference
        img = cv2.imread(image_path)
        if img is None:
            return {
                "status": "error",
                "message": f"Could not read image at {image_path}",
            }

        engine = WorkerContainer.get_engine()
        if not engine.is_healthy():
            raise RuntimeWarning("Engine health check failed. Attempting retry.")

        raw_results = engine.predict(img, confidence=confidence)

        logging.info(f"Raw detection results for {image_path_str}: {raw_results}")

        if raw_results.status == "error":
            return raw_results.model_dump()

        # 3. Extract Detections (Turn dict back into Pydantic for logic)
        detections = raw_results.results if raw_results.results else []

        # 4. Crop the best detection (if any)
        cropped_path_str = None
        if detections:
            best_det = max(detections, key=lambda d: d.confidence)
            image_manager = WorkerContainer.get_image_manager()
            try:
                cropped_path = image_manager.crop_and_save(
                    original_path=image_path, bbox=best_det.bbox, suffix="crop"
                )
                cropped_path_str = str(cropped_path)
            except Exception as e:
                logger.error(f"Cropping failed: {e}", exc_info=True)
                cropped_path_str = None

        # 5. Wrap everything in your DetectionResponse schema
        response = DetectionResponse(
            status="success",
            task_id=self.request.id,
            detections=raw_results,
            original_image_path=str(image_path),
            cropped_image_path=cropped_path_str,
            file_hash=file_hash,
            message=f"Detected {len(detections)} objects and saved crop.",
        )

        # 6. Return as dict for JSON serialization
        response_dict = response.model_dump()

        # 7. Cache the result for future identical requests
        logger.info(f"Caching result for {image_path_str} with key {cache_key}")
        cache_service.set_json(cache_key, response_dict)

        return response_dict

    except Exception as exc:
        logger.exception("Task failed, retrying...")
        raise self.retry(exc=exc)
