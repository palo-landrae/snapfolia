import cv2
import logging
from io import BytesIO
from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from celery.result import AsyncResult

from app.services.celery import celery
from app.schema import ClassificationResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/result/{task_id}", response_model=ClassificationResponse)
def get_result(task_id: str):
    """
    Poll for the result of a classification task.
    """
    task = AsyncResult(task_id, app=celery)

    if task.state == "PENDING":
        return ClassificationResponse(status="pending", task_id=task_id)

    if task.state == "FAILURE":
        return ClassificationResponse(status="failure", task_id=task_id)

    if task.state == "SUCCESS":
        return ClassificationResponse(**task.result)

    return ClassificationResponse(status=task.state.lower(), task_id=task_id)


@router.get("/result/{task_id}/image")
def get_result_image(task_id: str, padding: int = 10):

    task = AsyncResult(task_id, app=celery)

    if task.state != "SUCCESS":
        status_code = 202 if task.state in ["PENDING", "STARTED", "RETRY"] else 500
        raise HTTPException(
            status_code=status_code, detail=f"Task is in state: {task.state}"
        )

    result_data = task.result

    # ----------------------------
    # 1. Extract detections
    # ----------------------------
    detections_wrapper = result_data.get("detections", {})
    detections = detections_wrapper.get("results", [])

    # ----------------------------
    # 2. Extract classifications
    # ----------------------------
    classifications_wrapper = result_data.get("classifications", {})
    classifications = classifications_wrapper.get("results", [])

    image_path = result_data.get("original_image_path")

    if not image_path:
        raise HTTPException(status_code=404, detail="No image path found")

    if not detections:
        raise HTTPException(status_code=404, detail="No detections found")

    if not classifications:
        raise HTTPException(status_code=404, detail="No classification found")

    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=404, detail="Image file not found")

    try:
        # ----------------------------
        # 3. Get best detection (bbox source)
        # ----------------------------
        best_det = max(detections, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = map(int, best_det["bbox"])

        # ----------------------------
        # 4. Get best classification (label source)
        # ----------------------------
        best_cls = classifications[0]  # already top_k=3 sorted

        class_name = best_cls["class_name"]
        class_conf = best_cls["confidence"]

        # ----------------------------
        # 5. Padding
        # ----------------------------
        h, w = img.shape[:2]
        x1 = max(x1 - padding, 0)
        y1 = max(y1 - padding, 0)
        x2 = min(x2 + padding, w - 1)
        y2 = min(y2 + padding, h - 1)

        # ----------------------------
        # 6. Draw bbox
        # ----------------------------
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # ----------------------------
        # 7. Combined label (THIS IS THE IMPORTANT PART)
        # ----------------------------
        label = f"{class_name} ({class_conf:.2f})"

        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        # ----------------------------
        # 8. Return image
        # ----------------------------
        _, buffer = cv2.imencode(
            ".jpg",
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )

        return StreamingResponse(
            BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {e}")
