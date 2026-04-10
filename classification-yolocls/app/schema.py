from pydantic import BaseModel, ConfigDict
from typing import List, Literal, Optional


class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


class Classification(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class SpeedMetrics(BaseModel):
    preprocess: float
    inference: float
    postprocess: float
    total: float


class InferenceResult(BaseModel):
    status: Literal["success", "pending", "failure", "cached"]
    message: str | None = None
    results: List[Classification] = []
    speed_ms: SpeedMetrics | None = None


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


class DetectionResponse(BaseModel):
    # Allows the model to handle both strings and Path objects automatically
    model_config = ConfigDict(arbitrary_types_allowed=True)
    status: Literal["success", "pending", "failure", "cached", "error"]
    task_id: Optional[str] = None
    message: Optional[str] = None
    result: List[Detection] = []
    original_image_path: Optional[str] = None
    cropped_image_path: Optional[str] = None
    file_hash: Optional[str] = None
