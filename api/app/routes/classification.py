from fastapi import APIRouter, UploadFile, HTTPException
from app.core.schema import ClassificationResponse
from app.core.settings import settings
from app.services.celery import celery
from app.services.redis_cache_client import cache_service
from app.services.image_processor import ImageProcessor

router = APIRouter()
processor = ImageProcessor(upload_dir=settings.UPLOAD_DIR)


@router.post("/classify", response_model=ClassificationResponse)
async def classify(file: UploadFile):
    """
    Upload an image and send it to the classification worker via Celery.
    Uses cache to avoid reprocessing the same image.
    Returns a task_id or cached result.
    """

    # Compute hash for caching
    file_hash = cache_service.compute_hash_from_upload(file)

    # Check if the image is already in the cache
    cached_result = cache_service.get(file_hash)
    if cached_result is not None:
        response = ClassificationResponse.model_validate(cached_result)
        response.status = "cached"
        return response

    # Save the uploaded file
    try:
        file_path = await processor.save_uploadfile(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    # Send Celery task to classification queue

    task = celery.send_task(
        "classify_objects_task", args=[str(file_path)], queue="classification"
    )

    return ClassificationResponse(
        status="pending",
        task_id=task.id,
        image_path=str(file_path),
        results=[],
        file_hash=file_hash,
    )
