from celery import Celery
from app.core.settings import settings


# app/workers/api_client.py (or wherever your API initializes celery)
celery = Celery(
    "api_client",
    broker=settings.RABBITMQ_URL,
    backend=settings.REDIS_URL,
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # CRITICAL: The API must know where to route the task!
    task_routes={"detect_objects_task": {"queue": "detection"}},
)
