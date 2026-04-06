from typing import List, Literal
from pydantic import BaseModel


class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    status: Literal["success", "pending", "failure", "cached"]
    task_id: str | None = None
    message: str | None = None
    result: List[Detection] = []
    image_path: str | None = None
    file_hash: str | None = None


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
    file_hash: str | None = None
    image_path: str | None = None
