from pydantic import BaseModel, Field
from typing import Any, List, Literal, Optional, Dict


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0, le=1)
    bbox: List[float]  # [x1, y1, x2, y2]


class SpeedMetrics(BaseModel):
    preprocess: float
    inference: float
    postprocess: float
    total: float


class InferenceResult(BaseModel):
    status: Literal["success", "pending", "failure", "cached", "error"]
    message: Optional[str] = None
    results: List[Detection] = []
    speed_ms: SpeedMetrics | None = None
    device: Optional[str] = None


class DetectionResponse(BaseModel):
    status: Literal["success", "pending", "failure", "cached", "error"]
    task_id: str
    message: Optional[str] = None

    detections: InferenceResult | None = None

    original_image_path: Optional[str] = None
    cropped_image_path: Optional[str] = None
    file_hash: Optional[str] = None
