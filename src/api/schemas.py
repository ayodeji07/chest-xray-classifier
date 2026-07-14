"""
src/api/schemas.py
────────────────────────────────────────────────────────────────
Pydantic models for the chest X-ray classifier API.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status:    str = "ok"
    version:   str
    model:     str
    device:    str


class PathologyPrediction(BaseModel):
    class_name:   str
    display_name: str
    probability:  float = Field(ge=0.0, le=1.0)
    severity:     str


class PredictResponse(BaseModel):
    predictions:    list[PathologyPrediction]
    top_prediction: Optional[PathologyPrediction]
    processing_ms:  float
    model_backbone: str
    disclaimer:     str = (
        "This tool is for research and educational purposes only. "
        "It is not a medical device and must not be used for clinical diagnosis."
    )


class GradCAMResponse(BaseModel):
    class_name:    str
    probability:   float
    heatmap_b64:   str   # base64-encoded PNG overlay
    processing_ms: float
