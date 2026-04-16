from datetime import datetime, timezone

from pydantic import BaseModel, Field
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


class Location(BaseModel):
    type: str
    coordinates: List[float]  # [latitude, longitude]


class Context(BaseModel):
    location: Optional[Location] = None
    device_model: Optional[str] = None
    app_version: Optional[str] = None

class Validation(BaseModel):
    user_confirmed: Optional[bool] = None
    correction: Optional[bool] = None


class DetectionInferenceResult(BaseModel):
    status: Literal["success", "pending", "failure", "cached"]
    message: str | None = None
    results: List[Detection] = []
    speed_ms: SpeedMetrics | None = None
    device: Optional[str] = None


class DetectionResponse(BaseModel):
    status: Literal["success", "pending", "failure", "cached", "error"]
    task_id: str
    message: Optional[str] = None

    detections: DetectionInferenceResult | None = None

    original_image_path: Optional[str] = None
    cropped_image_path: Optional[str] = None
    file_hash: Optional[str] = None


class ClassificationInferenceResult(BaseModel):
    status: Literal["success", "pending", "failure", "cached"]
    message: str | None = None
    results: List[Classification] = []
    speed_ms: SpeedMetrics | None = None
    device: Optional[str] = None


class ClassificationResponse(BaseModel):
    status: Literal["success", "pending", "failure", "cached"]
    user_id: Optional[str] = None
    task_id: str
    message: Optional[str] = None

    classifications: ClassificationInferenceResult | None = None
    detections: DetectionInferenceResult | None = None

    context: Optional[Context] = None

    original_image_path: Optional[str] = None
    cropped_image_path: Optional[str] = None
    file_hash: Optional[str] = None
    validation: Optional[Validation] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
