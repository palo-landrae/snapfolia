from fastapi import FastAPI

from app.routes.task_results import router as result_router
from app.routes.pipeline import router as pipeline_router

# In your FastAPI backend
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Snapfolia API")

app.include_router(result_router)
app.include_router(pipeline_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your phone to talk to your PC
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
