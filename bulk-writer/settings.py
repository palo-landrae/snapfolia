from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    RABBITMQ_URL: str = Field(default="amqp://guest:guest@rabbitmq:5672//")
    REDIS_URL: str = Field(default="redis://redis:6379/0")  # ✅ FIXED

    MONGO_URL: str = Field(default="mongodb://mongodb:27017")  # ✅ ADD THIS
    MONGO_DB: str = Field(default="snapfolia")

    UPLOAD_DIR: str = Field(default="/volumes/images")


settings = Settings()
