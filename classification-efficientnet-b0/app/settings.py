from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from app.class_map import CLASS_MAP

# Adjusting to ensure we find the .env at the project root
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    RABBITMQ_URL: str = Field(default="amqp://guest:guest@rabbitmq:5672//")
    UPLOAD_DIR: str = Field(default="/volumes/images")
    CLS_MODEL_PATH: str = Field(default="/app/models/best.engine")
    CLASS_MAP: dict[int, str] = CLASS_MAP


settings = Settings()
