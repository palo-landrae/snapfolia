from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0, le=1)
    bbox: List[float]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    status: Literal["success", "pending", "failure", "cached", "error"]
    task_id: Optional[str] = None
    message: Optional[str] = None
    result: List[Detection] = []
    original_image_path: Optional[str] = None
    cropped_image_path: Optional[str] = None
    file_hash: Optional[str] = None


class Classification(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class SpeedMetrics(BaseModel):
    preprocess: float
    inference: float
    postprocess: float
    total: float


class ClassificationResponse(BaseModel):
    status: Literal["success", "pending", "failure", "cached"]
    task_id: str | None = None
    message: str | None = None
    results: List[Classification] = []
    speed_ms: SpeedMetrics | None = None
    detections: List[Detection] = []
    original_image_path: Optional[str] = None
    cropped_image_path: Optional[str] = None
    file_hash: Optional[str] = None

