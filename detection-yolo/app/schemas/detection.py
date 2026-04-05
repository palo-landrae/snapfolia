from pydantic import BaseModel
from typing import List, Literal, Sequence


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]


class DetectionResponse(BaseModel):
    status: Literal["success", "pending", "failure", "cached"]
    task_id: str | None = None
    result: Sequence[Detection] = []
    image_path: str | None = None
    file_hash: str | None = None
