import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple
from app.settings import settings

logger = logging.getLogger(__name__)

class ImageManager:
    """
    Handles physical disk operations for images, specifically cropping 
    and standardized storage paths.
    """
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def crop_and_save(
        self, 
        original_path: Path, 
        bbox: List[float], 
        suffix: str = "crop"
    ) -> Path:
        """
        Crops an image based on [x1, y1, x2, y2] and saves it.
        Returns the Path to the new file.
        """
        if not original_path.exists():
            raise FileNotFoundError(f"Source image missing: {original_path}")

        # 1. Efficient Loading
        # We use imread, but check for None immediately
        img = cv2.imread(str(original_path))
        if img is None:
            raise ValueError(f"Unable to decode image: {original_path}")

        # 2. Boundary Validation Logic
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self._clamp_coordinates(bbox, w, h)

        # Validate that the crop area is actually valid (not zero width/height)
        if x2 <= x1 or y2 <= y1:
            logger.error(f"Invalid crop dimensions for {original_path}: {x1, y1, x2, y2}")
            raise ValueError("Crop dimensions result in an empty image.")

        # 3. Execution
        cropped_img = img[y1:y2, x1:x2]

        # 4. Smart Path Generation
        # Uses .with_stem (Python 3.9+) to avoid messy string formatting
        crop_path = original_path.parent / f"{original_path.stem}_{suffix}{original_path.suffix}"

        # 5. Atomic-like write check
        success = cv2.imwrite(str(crop_path), cropped_img)
        if not success:
            raise IOError(f"OS failed to write image to {crop_path}")

        return crop_path

    @staticmethod
    def _clamp_coordinates(bbox: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
        """Ensures bounding box stays within the actual pixel grid."""
        x1, y1, x2, y2 = map(int, bbox)
        return (
            max(0, min(x1, width)),
            max(0, min(y1, height)),
            max(0, min(x2, width)),
            max(0, min(y2, height))
        )