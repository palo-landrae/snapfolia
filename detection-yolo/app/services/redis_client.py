import json
import hashlib
import logging
from typing import Any, Optional
import redis
from app.settings import settings

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self, url: str):
        # Using a connection pool is more efficient for high-concurrency Celery workers
        self.pool = redis.ConnectionPool.from_url(url, decode_responses=True)
        self.client = redis.Redis(connection_pool=self.pool)

    def compute_file_hash(self, file_path: str) -> str:
        """Computes SHA256 of a file to use as a unique identity."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except FileNotFoundError:
            logger.error(f"File not found for hashing: {file_path}")
            raise

    def generate_cache_key(self, file_hash: str, model_name: str) -> str:
        """
        Creates a versioned key: e.g., 'detect:v1:abc123hash'
        This ensures that if you upgrade your model, you don't return
        old results from the previous version.
        """
        return f"detect:{model_name}:{file_hash}"

    def get_json(self, key: str) -> Optional[Any]:
        """Fetch and deserialize JSON."""
        try:
            data = self.client.get(key)
            return json.loads(str(data)) if data else None
        except (json.JSONDecodeError, redis.RedisError) as e:
            logger.warning(f"Cache read error for key {key}: {e}")
            return None

    def set_json(self, key: str, value: Any, expire: int = 86400) -> bool:
        """Serialize and store. Default expire is 24 hours."""
        try:
            serialized_value = json.dumps(value)
            result = self.client.set(key, serialized_value, ex=expire)
            return bool(result)
        except (TypeError, redis.RedisError) as e:
            logger.error(f"Cache write error for key {key}: {e}")
            return False


# Global instance
cache_service = RedisClient(settings.REDIS_URL)
