import logging
from celery import Celery
from app.core.settings import settings

# Setup logging for the worker configuration phase
logger = logging.getLogger(__name__)

def create_celery_app() -> Celery:
    """
    Factory function to initialize the Celery application.
    Using a factory pattern makes testing and multi-environment setup easier.
    """
    
    # Define the tasks to include to ensure they are registered upon startup
    tasks_to_include = ["app.workers.tasks"]

    app = Celery(
        "detection_worker",
        broker=settings.RABBITMQ_URL,
        backend=settings.REDIS_URL,
        include=tasks_to_include
    )

    # Professional Configuration
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
        task_routes={
            "detect_objects_task": {"queue": "detection"}
        },

        # --- Reliability & Performance ---
        # Late acknowledgment means the task is only marked "done" after it finishes.
        # If the worker crashes mid-task, the message stays in the queue for another worker.
        task_acks_late=True,
        
        # Prevents one worker from hoarding tasks; vital for long-running ML jobs.
        worker_prefetch_multiplier=1,
        
        # Limits the number of times a task can be executed if it keeps failing/crashing.
        task_reject_on_worker_lost=True,

        # --- Monitoring ---
        task_track_started=True,
        result_expires=3600,  # Results cleared after 1 hour to save Redis memory
    )

    logger.info("Celery app configured successfully with queue: detection")
    return app

celery_app = create_celery_app()