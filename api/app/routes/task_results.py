import cv2
import logging
from io import BytesIO
from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from celery.result import AsyncResult

from app.services.celery import celery
from app.core.schema import DetectionResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/result/{task_id}", response_model=DetectionResponse)
def get_result(task_id: str):
    """
    Poll for the result of a detection task.
    """
    task = AsyncResult(task_id, app=celery)

    if task.state == "PENDING":
        return DetectionResponse(status="pending", task_id=task_id, result=[])

    if task.state == "FAILURE":
        return DetectionResponse(status="failure", task_id=task_id, result=[])

    if task.state == "SUCCESS":
        return DetectionResponse(**task.result)

    return DetectionResponse(status=task.state.lower(), task_id=task_id, result=[])


@router.get("/result/{task_id}/image")
def get_result_image(task_id: str, padding: int = 10):
    """
    Returns the original image with the highest-confidence bounding box drawn on it.
    """
    task = AsyncResult(task_id, app=celery)

    # 1. Validate Task State
    if task.state != "SUCCESS":
        status_code = 202 if task.state in ["PENDING", "STARTED", "RETRY"] else 500
        raise HTTPException(
            status_code=status_code, detail=f"Task is in state: {task.state}"
        )

    # 2. Extract Data from Task Result
    # task.result is the dict returned by your worker
    result_data = task.result
    detections = result_data.get("result", [])
    image_path = result_data.get("image_path")

    if not image_path or not detections:
        raise HTTPException(
            status_code=404, detail="No detection data found for this task"
        )

    # 3. Load and Process Image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Image not found on disk at: {image_path}")
        raise HTTPException(status_code=404, detail="Original image file not found")

    try:
        # Get the detection with the highest confidence
        best_det = max(detections, key=lambda d: d["confidence"])

        # Coordinates and Labeling
        x1, y1, x2, y2 = map(int, best_det["bbox"])

        # Apply Padding and Boundary Checks
        h, w = img.shape[:2]
        x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
        x2, y2 = min(x2 + padding, w - 1), min(y2 + padding, h - 1)

        # Draw Rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add Label
        label = f"{best_det['class_name']} {best_det['confidence']:.2f}"
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # 4. Stream Response
        _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

    except Exception as e:
        logger.exception(f"Error processing result image: {e}")
        raise HTTPException(
            status_code=500, detail="Internal error visualizing detection"
        )
