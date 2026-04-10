import json
import hashlib
from fastapi import UploadFile
import redis
from typing import Any, Optional
from app.settings import settings


class RedisCacheClient:
    def __init__(self, url: str):
        # decode_responses=True ensures we get 'str' back, not 'bytes'
        # This resolves the "ResponseT" vs "str" type mismatch
        self.client: redis.Redis = redis.Redis.from_url(url, decode_responses=True)

    def compute_hash_from_path(self, image_path: str) -> str:
        """Compute SHA256 hash of an image file given its path."""
        hash_func = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def compute_hash_from_upload(self, file: UploadFile) -> str:
        """Compute SHA256 hash of a FastAPI UploadFile without saving to disk."""
        hash_func = hashlib.sha256()
        file.file.seek(0)
        for chunk in iter(lambda: file.file.read(4096), b""):
            hash_func.update(chunk)
        file.file.seek(0)  # Reset pointer so FastAPI can read file later
        return hash_func.hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get and deserialize JSON value from Redis."""
        value = self.client.get(key)
        if value is not None:
            # value is guaranteed to be a str here because of decode_responses=True
            return json.loads(str(value))
        return None

    def set(self, key: str, value: Any, expire_seconds: int = 3600) -> None:
        """Serialize and store value in Redis."""
        # json.dumps converts Python objects (dict, list, etc.) to a JSON string
        serialized_value = json.dumps(value)
        self.client.set(key, serialized_value, ex=expire_seconds)

    def delete(self, key: str) -> None:
        """Remove a key from the cache."""
        self.client.delete(key)


# Global instance for use across the app
cache_service = RedisCacheClient(settings.REDIS_URL)
