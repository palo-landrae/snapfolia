from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

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

    APP_NAME: str = "Snapfolia 2026"
    ENV: str = Field(default="development")
    UPLOAD_DIR: str = Field(default="/volumes/images")

    # ---------------------------
    # URLS
    # ---------------------------
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    RABBITMQ_URL: str = Field(default="amqp://guest:guest@rabbitmq:5672//")

    # ---------------------------
    # MODEL PATHS
    # ---------------------------
    YOLO_MODEL_PATH: str = Field(default="/app/models/best.pt")


settings = Settings()
