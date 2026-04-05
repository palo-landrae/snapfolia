from pathlib import Path
from fastapi import UploadFile
import aiofiles
from werkzeug.utils import secure_filename
import uuid
import os
from app.core.settings import settings


class ImageProcessor:
    def __init__(self, upload_dir: str):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, file: UploadFile) -> str:
        original_name = file.filename or "upload.jpg"
        name, ext = os.path.splitext(original_name)

        if not ext:
            ext = ".jpg"

        return secure_filename(name) + ext

    async def save_uploadfile(
        self, file: UploadFile, prefix: str | None = None
    ) -> Path:
        """
        Safely save an UploadFile to disk with optional UUID prefix.
        Returns the path of the saved file.
        """
        filename = self._safe_filename(file)
        if prefix is None:
            prefix = uuid.uuid4().hex
        file_path = self.upload_dir / f"{prefix}_{filename}"

        async with aiofiles.open(file_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        return file_path
