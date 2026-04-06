from fastapi import APIRouter, UploadFile, HTTPException
from celery import chain  # Import chain correctly
from app.core.schema import ClassificationResponse
from app.core.settings import settings

# Assuming 'celery_app' is your Celery instance imported from your celery config
from app.services.celery import celery
from app.services.redis_cache_client import cache_service
from app.services.image_processor import ImageProcessor

router = APIRouter()
processor = ImageProcessor(upload_dir=settings.UPLOAD_DIR)


@router.post("/pipeline", response_model=ClassificationResponse)
async def start_pipeline(file: UploadFile):
    # 1. Compute hash for caching
    # Important: Ensure compute_hash_from_upload handles the file pointer correctly
    file_hash = cache_service.compute_hash_from_upload(file)

    # 2. Check Cache
    cached_result = cache_service.get(file_hash)
    if cached_result:
        # Re-validate cached data into our Pydantic model
        response = ClassificationResponse.model_validate(cached_result)
        response.status = "cached"
        return response

    # 3. Save the uploaded file
    try:
        # Reset file pointer if the hashing consumed it
        await file.seek(0)
        file_path = await processor.save_uploadfile(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    # 4. Define the Celery Chain
    # We use .signature() or .s() to create the workflow
    # Task 1: Detect & Crop -> Task 2: Classify Crop
    workflow = chain(
        celery.signature(
            "object_detection_task", args=[str(file_path), 0.4], queue="detection"
        ),
        celery.signature("classify_objects_task", queue="classification"),
    )

    # 5. Execute the chain
    result = workflow.apply_async()
    # Guard: Ensure the result exists before accessing .id
    if result is None:
        raise HTTPException(
            status_code=500, detail="Task pipeline failed to initialize."
        )

    return ClassificationResponse(
        status="pending",
        task_id=str(result.id),  # This is the ID of the LAST task in the chain
        image_path=str(file_path),
        results=[],
        file_hash=file_hash,
    )
