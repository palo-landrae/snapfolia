from fastapi import FastAPI

from app.routes.task_results import router as result_router
from app.routes.pipeline import router as pipeline_router

app = FastAPI(title="Snapfolia API")

app.include_router(result_router)
app.include_router(pipeline_router)
