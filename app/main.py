from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.detection import router as detection_router
from app.routes.analysis import router as analysis_router
from app.services.model_inference import load_all_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm ML models on startup so the first request isn't slow
    load_all_models()
    yield


app = FastAPI(title="SkinScope API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)
app.include_router(analysis_router)
