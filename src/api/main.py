"""
src/api/main.py
────────────────────────────────────────────────────────────────
FastAPI application for the Chest X-Ray Classifier.

Start:
  uvicorn src.api.main:app --reload --port 8000
  Docs: http://localhost:8000/docs
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.predict import router as predict_router
from src.api.schemas import HealthResponse
from src.utils.config import APIConfig, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown tasks."""
    logger.info("Chest X-Ray Classifier API v%s starting", APIConfig.version)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.device = device
    logger.info("Device: %s", device)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title       = APIConfig.title,
    version     = APIConfig.version,
    description = APIConfig.description,
    docs_url    = "/docs"  if APIConfig.debug else None,
    redoc_url   = "/redoc" if APIConfig.debug else None,
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """API health check."""
    device = getattr(app.state, "device", "cpu")
    return HealthResponse(
        status  = "ok",
        version = APIConfig.version,
        model   = settings.model.backbone,
        device  = device,
    )


@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
