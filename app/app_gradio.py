"""
app/app_gradio.py
────────────────────────────────────────────────────────────────
Gradio demo for HuggingFace Spaces deployment.

One file — upload to HuggingFace Spaces and it runs immediately.

Deploy:
  1. Create a new Space on huggingface.co/spaces
  2. Set SDK: Gradio
  3. Upload this file as app.py + requirements.txt + checkpoints/
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Make src/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).parents[1]))

import gradio as gr
import numpy as np
from PIL import Image

from src.data.transforms import get_inference_transform, get_gradcam_transform
from src.explainability.gradcam import GradCAM
from src.explainability.visualise import generate_gradcam_overlay, get_top_predictions
from src.utils.config import (
    PATHOLOGY_CLASSES,
    PATHOLOGY_DISPLAY_NAMES,
    PATHOLOGY_SEVERITY,
    Paths,
)

# ── Model loading ─────────────────────────────────────────────────
import torch
from src.models.model import load_model_for_inference

HF_MODEL_REPO = "ayodeji21/chest-xray-classifier"

# Auto-download the checkpoint from HuggingFace Hub if missing —
# checkpoints/ is gitignored, so a fresh Space build won't have it
# unless it's uploaded manually or fetched here.
if not Paths.best_model.exists():
    import shutil
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.pt")
    Paths.best_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(downloaded, Paths.best_model)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model  = load_model_for_inference(Paths.best_model, _device)
_TRANSFORM = get_inference_transform()


def predict(image_pil: Image.Image) -> tuple:
    """Run inference and return predictions + Grad-CAM overlay.

    Args:
        image_pil: PIL Image from the Gradio upload widget.

    Returns:
        Tuple of (overlay_image, label_dict) for Gradio outputs.
    """
    if image_pil is None:
        return None, {}

    image_rgb = image_pil.convert("RGB")
    tensor    = _TRANSFORM(image_rgb)

    # Predictions
    probs    = _model.predict_single(tensor, _device)
    top_preds = get_top_predictions(probs, threshold=0.1, top_k=10)

    # Grad-CAM for top predicted class
    top_cls = max(probs, key=probs.get)
    with GradCAM(_model, _model.get_features_layer()) as cam:
        heatmap = cam.generate(tensor, top_cls, _device)

    overlay = generate_gradcam_overlay(image_rgb, heatmap)

    # Gradio label output: {label: confidence}
    label_dict = {
        PATHOLOGY_DISPLAY_NAMES.get(p["class_name"], p["class_name"]): p["probability"]
        for p in top_preds[:5]
    }

    return overlay, label_dict


# ── Gradio interface ──────────────────────────────────────────────
with gr.Blocks(
    title="Chest X-Ray Pathology Classifier",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("""
# 🫁 Chest X-Ray Pathology Classifier

Upload a posterior-anterior (PA) chest X-ray to detect 10 common thoracic pathologies.

**Model**: DenseNet121 (CheXNet architecture) trained on NIH ChestX-ray14

---
⚠️ **Research use only** — not a medical device. Do not use for clinical diagnosis.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label   = "Upload Chest X-Ray",
                type    = "pil",
                height  = 400,
            )
            submit_btn = gr.Button("Analyse", variant="primary")

        with gr.Column(scale=1):
            gradcam_output = gr.Image(
                label  = "Grad-CAM Heatmap (top prediction)",
                height = 400,
            )
            label_output = gr.Label(
                label    = "Pathology Probabilities",
                num_top_classes = 5,
            )

    submit_btn.click(
        fn      = predict,
        inputs  = [image_input],
        outputs = [gradcam_output, label_output],
    )

    gr.Markdown("""
---
### About Grad-CAM
The heatmap highlights which regions of the X-ray most influenced
the model's top prediction. Red areas had the highest activation.

### Pathology Classes
""" + " · ".join(PATHOLOGY_DISPLAY_NAMES[c] for c in PATHOLOGY_CLASSES))


if __name__ == "__main__":
    demo.launch(share=False)
