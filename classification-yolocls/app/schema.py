from pydantic import BaseModel
from typing import List


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
    status: str
    message: str | None = None
    results: List[Classification] = []
    speed_ms: SpeedMetrics | None = None


class ClassificationResponse(BaseModel):
    status: str
    task_id: str | None = None
    message: str | None = None
    results: List[Classification] = []
    speed_ms: SpeedMetrics | None = None
    file_hash: str | None = None
    image_path: str | None = None
