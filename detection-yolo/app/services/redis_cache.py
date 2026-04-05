import json
import hashlib
import redis
from typing import Any, Optional, Union
from app.core.settings import settings

class RedisClient:
    def __init__(self, url: str):
        # decode_responses=True ensures we get 'str' back, not 'bytes'
        # This resolves the "ResponseT" vs "str" type mismatch
        self.client: redis.Redis = redis.Redis.from_url(
            url, 
            decode_responses=True
        )

    def compute_image_hash(self, image_path: str) -> str:
        """Compute SHA256 hash of image file."""
        hash_func = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
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
cache_service = RedisClient(settings.REDIS_URL)