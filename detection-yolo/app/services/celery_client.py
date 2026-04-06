import logging
from celery import Celery
from celery.signals import worker_process_init
from app.settings import settings

# Setup logging
logger = logging.getLogger(__name__)


def create_celery_app() -> Celery:
    """
    Factory function to initialize the Celery application
    """

    app = Celery(
        "detection_worker",
        broker=settings.RABBITMQ_URL,
        backend=settings.REDIS_URL,
        include=["app.tasks"],  # Ensure tasks are discovered
    )

    app.conf.update(
        # --- Serialization & Security ---
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # --- Timezone Settings ---
        timezone="UTC",
        enable_utc=True,
        # --- Queue Configuration ---
        task_default_queue="detection",
        task_create_missing_queues=True,
        task_routes={"app.tasks": {"queue": "detection"}},
        # --- Reliability (The "ML Essentials") ---
        # Late acknowledgment ensures if a worker crashes during a heavy YOLO
        # inference, the task is returned to the queue instead of lost.
        task_acks_late=True,
        # Prevents a single worker from "pre-claiming" multiple heavy tasks.
        # This ensures tasks are distributed evenly across multiple GPU workers.
        worker_prefetch_multiplier=1,
        # If the worker process is lost (e.g. OOM), reject the task so it retries.
        task_reject_on_worker_lost=True,
        # --- Performance & Concurrency ---
        # IMPORTANT: For GPU tasks, usually set concurrency to 1 per worker container
        # to prevent multiple processes fighting for the same VRAM.
        worker_concurrency=1,
        # Restart the worker process after 200 tasks to prevent memory leaks/fragmentation.
        worker_max_tasks_per_child=200,
        # Hard limit: Kill a task if it takes longer than 5 minutes.
        task_time_limit=300,
        task_soft_time_limit=270,
        # --- Monitoring ---
        task_track_started=True,
        result_expires=3600,  # Results cleared after 1 hour
    )
    logger.info("Celery app configured successfully with queue: detection")
    return app


# Initialize the singleton instance
celery_app = create_celery_app()
