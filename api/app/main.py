from fastapi import FastAPI
from app.routes.detection import router as detect_router
from app.routes.classification import router as classify_router
from app.routes.task_results import router as result_router

app = FastAPI(title="Snapfolia API")

app.include_router(detect_router)
app.include_router(classify_router)
app.include_router(result_router)
