"""
src/api/routes/predict.py
────────────────────────────────────────────────────────────────
Prediction and Grad-CAM endpoints.

Routes
──────
  POST /predict          — upload X-ray image, get pathology predictions
  POST /predict/gradcam  — upload X-ray image, get Grad-CAM overlay
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import io
import time
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.api.schemas import GradCAMResponse, PathologyPrediction, PredictResponse
from src.utils.config import APIConfig, PATHOLOGY_DISPLAY_NAMES, PATHOLOGY_SEVERITY
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

# Module-level singletons — loaded once on first request
_model      = None
_gradcam    = None
_transform  = None
_device     = None


def _get_model():
    """Load model on first call and cache."""
    global _model, _device
    if _model is None:
        import torch
        from src.models.model import load_model_for_inference
        from src.utils.config import Paths
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model  = load_model_for_inference(Paths.best_model, _device)
    return _model, _device


def _get_transform():
    """Load inference transform on first call and cache."""
    global _transform
    if _transform is None:
        from src.data.transforms import get_inference_transform
        _transform = get_inference_transform()
    return _transform


def _decode_upload(upload: UploadFile) -> "PIL.Image.Image":
    """Read an uploaded file and return a PIL Image."""
    from PIL import Image

    content = upload.file.read()
    if len(content) > APIConfig.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: "
                   f"{APIConfig.max_upload_bytes // 1_048_576} MB",
        )
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read image. Upload a valid PNG or JPEG.",
        )
    return img


@router.post(
    "",
    response_model = PredictResponse,
    summary        = "Classify pathologies in a chest X-ray",
)
def predict(
    file:      UploadFile = File(..., description="Chest X-ray PNG or JPEG"),
    threshold: float      = Query(0.3, ge=0.0, le=1.0),
) -> PredictResponse:
    """Upload a chest X-ray and receive per-pathology probabilities.

    Returns predictions for all 10 pathology classes above the
    given probability threshold, ranked by probability.
    """
    t_start = time.perf_counter()

    model, device = _get_model()
    transform     = _get_transform()
    image         = _decode_upload(file)

    # Run inference
    tensor = transform(image)
    probs  = model.predict_single(tensor, device)

    # Build response
    predictions = []
    for cls_name, prob in sorted(probs.items(), key=lambda x: -x[1]):
        if prob < threshold:
            continue
        predictions.append(PathologyPrediction(
            class_name   = cls_name,
            display_name = PATHOLOGY_DISPLAY_NAMES.get(cls_name, cls_name),
            probability  = round(prob, 4),
            severity     = PATHOLOGY_SEVERITY.get(cls_name, "unknown"),
        ))

    top = predictions[0] if predictions else None

    return PredictResponse(
        predictions    = predictions,
        top_prediction = top,
        processing_ms  = round((time.perf_counter() - t_start) * 1000, 1),
        model_backbone = model.backbone_name,
    )


@router.post(
    "/gradcam",
    response_model = GradCAMResponse,
    summary        = "Get a Grad-CAM explanation heatmap",
)
def predict_gradcam(
    file:       UploadFile = File(...),
    class_name: str        = Query(
        "Pneumonia",
        description="Pathology class to visualise",
    ),
) -> GradCAMResponse:
    """Upload an X-ray and receive a Grad-CAM heatmap overlay.

    The heatmap highlights which regions of the image most
    influenced the model's prediction for the specified class.
    """
    t_start = time.perf_counter()

    model, device = _get_model()
    transform     = _get_transform()
    image         = _decode_upload(file)

    from src.explainability.gradcam import GradCAM
    from src.explainability.visualise import generate_gradcam_overlay

    tensor = transform(image)

    with GradCAM(model, model.get_features_layer()) as cam:
        heatmap  = cam.generate(tensor, class_name, device)
        prob     = cam.get_probability(tensor, device)[class_name]

    overlay = generate_gradcam_overlay(image, heatmap)

    # Encode overlay as base64 PNG for JSON transport
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return GradCAMResponse(
        class_name    = class_name,
        probability   = round(prob, 4),
        heatmap_b64   = heatmap_b64,
        processing_ms = round((time.perf_counter() - t_start) * 1000, 1),
    )
