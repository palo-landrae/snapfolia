from pydantic import BaseModel, Field, ConfigDict
from typing import List, Literal, Optional
from pathlib import Path

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0, le=1)
    bbox: List[float] # [x1, y1, x2, y2]

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