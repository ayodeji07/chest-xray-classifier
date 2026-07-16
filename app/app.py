"""
app/app.py
────────────────────────────────────────────────────────────────
Streamlit demo for the Chest X-Ray Pathology Classifier.

Upload a chest X-ray → receive pathology predictions,
probability bars, and a Grad-CAM heatmap showing which
regions the model is focusing on.

Run:
  streamlit run app/app.py
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Streamlit adds this script's own directory (app/) to sys.path, not the
# project root, so `src...` imports below would fail without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.utils.config import Paths

# ── Page config — must be first Streamlit call ────────────────────
st.set_page_config(
    page_title     = "Chest X-Ray Classifier",
    page_icon      = "🫁",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .severity-critical { color: #e74c3c; font-weight: 700; }
  .severity-high     { color: #e67e22; font-weight: 600; }
  .severity-moderate { color: #f39c12; font-weight: 500; }
  .severity-low      { color: #27ae60; font-weight: 400; }
  .disclaimer-box {
    background: #fff3cd; border: 1px solid #ffc107;
    border-radius: 6px; padding: 12px 16px;
    font-size: 0.85rem; margin-bottom: 1rem;
  }
  .metric-card {
    background: #f8f9fa; border: 1px solid #dee2e6;
    border-radius: 8px; padding: 10px 14px; text-align: center;
  }
  [data-testid="stSidebar"] { min-width: 240px; max-width: 280px; }
</style>
""", unsafe_allow_html=True)

_SEVERITY_COLOURS = {
    "critical": "#e74c3c",
    "high":     "#e67e22",
    "moderate": "#f39c12",
    "low":      "#27ae60",
}

_SAMPLE_DIR = Path(__file__).parents[1] / "data" / "sample_xrays"

# Model checkpoint and its eval results are hosted together on
# HuggingFace Hub — checkpoints/ and data/processed/ are both
# gitignored, so hosted deployments like Streamlit Cloud have
# neither unless fetched at runtime.
HF_MODEL_REPO = "ayodeji21/chest-xray-classifier"

# Reliability bands from the model's own measured per-class test AUC
# (data/processed/eval_results.json, written by evaluate.py) — this
# model's accuracy varies a lot by class, and predictions shouldn't
# all be presented with the same implied confidence.
_RELIABILITY_BANDS = [
    (0.85, "High reliability",     "#27ae60"),
    (0.75, "Moderate reliability", "#f39c12"),
    (0.0,  "Low reliability",      "#e74c3c"),
]


@st.cache_data
def _load_class_auc() -> dict:
    """Load per-class test AUC from the last evaluation run, if any.

    Downloads eval_results.json from HuggingFace Hub on demand if it
    isn't present locally (e.g. on Streamlit Cloud).
    """
    import json

    if not Paths.eval_results.exists():
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename="eval_results.json")
            Paths.eval_results.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(downloaded, Paths.eval_results)
        except Exception:
            return {}

    try:
        results = json.loads(Paths.eval_results.read_text())
        return results.get("auc_per_class", {})
    except (FileNotFoundError, ValueError):
        return {}


def _reliability_label(auc: float | None) -> tuple[str, str] | None:
    """Return (label, colour) for a class's measured AUC, or None if
    no evaluation data is available for it.
    """
    if auc is None:
        return None
    for min_auc, label, colour in _RELIABILITY_BANDS:
        if auc >= min_auc:
            return label, colour
    return None


# ── Cached model loading ──────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def _load_model():
    """Load the classifier once and cache for the session."""
    import torch
    from src.models.model import load_model_for_inference
    from src.utils.config import Paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model_for_inference(Paths.best_model, device)
    return model, device


@st.cache_data(show_spinner=False)
def _run_inference(img_bytes: bytes) -> dict:
    """Run model inference and return predictions + Grad-CAM."""
    from PIL import Image
    from src.data.transforms import get_inference_transform
    from src.explainability.gradcam import GradCAM
    from src.explainability.visualise import (
        generate_gradcam_overlay,
        get_top_predictions,
    )

    model, device = _load_model()
    transform     = get_inference_transform()
    image         = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor        = transform(image)

    # Predictions
    probs = model.predict_single(tensor, device)

    # Grad-CAM for top predicted class
    top_cls = max(probs, key=probs.get)
    with GradCAM(model, model.get_features_layer()) as cam:
        heatmap = cam.generate(tensor, top_cls, device)

    overlay = generate_gradcam_overlay(image, heatmap)
    top_preds = get_top_predictions(probs, threshold=0.1, top_k=10)

    return {
        "probs":     probs,
        "top_preds": top_preds,
        "top_cls":   top_cls,
        "image":     image,
        "overlay":   overlay,
    }


# ── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🫁 Chest X-Ray AI")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("DenseNet121 (CheXNet)")
    st.markdown("**Dataset**")
    st.markdown("NIH ChestX-ray14")
    st.markdown("**Classes**")
    st.markdown("10 pathologies")
    st.markdown("---")

    threshold = st.slider(
        "Prediction threshold",
        min_value=0.05, max_value=0.95,
        value=0.3, step=0.05,
        help=(
            "The model scores every pathology from 0-100%. This slider "
            "hides anything scored below it, so you only see findings "
            "the model is at least this confident about. Lower = see "
            "more possible findings (more false alarms). Higher = only "
            "see the findings it's more sure of (may miss weaker signals)."
        ),
    )

    show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.75rem; color:#888'>
Grad-CAM highlights the image regions that influenced each prediction.
Red = high activation, blue = low activation.
</div>
""", unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────

st.title("🫁 Chest X-Ray Pathology Classifier")
st.markdown(
    "Upload a posterior-anterior (PA) chest X-ray to detect "
    "10 common thoracic pathologies using a DenseNet121 model "
    "trained on NIH ChestX-ray14."
)

# Disclaimer
st.markdown("""
<div class='disclaimer-box'>
⚠️ <strong>Research use only.</strong> This tool is not a medical device
and must not be used for clinical diagnosis or patient management.
Always consult a qualified radiologist for medical interpretation.
</div>
""", unsafe_allow_html=True)

# ── File upload ───────────────────────────────────────────────────
col_upload, col_sample = st.columns([3, 1])

with col_sample:
    st.markdown("**Or use a sample:**")
    sample_files = list(_SAMPLE_DIR.glob("*.png")) + \
                   list(_SAMPLE_DIR.glob("*.jpg"))
    if sample_files:
        for sf in sample_files:
            if st.button(sf.stem[:20], use_container_width=True):
                st.session_state["uploaded_bytes"] = sf.read_bytes()

with col_upload:
    uploaded = st.file_uploader(
        "Upload chest X-ray (PNG or JPEG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded:
        st.session_state["uploaded_bytes"] = uploaded.read()

# ── Run inference ─────────────────────────────────────────────────
img_bytes = st.session_state.get("uploaded_bytes")

if not img_bytes:
    st.info("Upload a chest X-ray above to begin.")
    st.stop()

# Check model is available — auto-download from HuggingFace Hub if
# missing (checkpoints/ is gitignored, so hosted deployments like
# Streamlit Cloud start without it).
if not Paths.best_model.exists():
    with st.spinner("Downloading model checkpoint (first run only)..."):
        try:
            import shutil
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.pt")
            Paths.best_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(downloaded, Paths.best_model)
        except Exception as exc:
            st.error(f"Could not download model checkpoint from HuggingFace Hub: {exc}")

if not Paths.best_model.exists():
    st.error(
        "No trained model found at `checkpoints/best_model.pt`.\n\n"
        "Train the model first:\n"
        "```bash\npython -m src.training.train\n```\n"
        "Or download a pre-trained checkpoint from the project README."
    )
    st.stop()

with st.spinner("Analysing X-ray..."):
    results = _run_inference(img_bytes)

probs     = results["probs"]
top_preds = results["top_preds"]
top_cls   = results["top_cls"]
image     = results["image"]
overlay   = results["overlay"]

# Filter by threshold
visible = [p for p in top_preds if p["probability"] >= threshold]

# ── Layout: image | predictions ───────────────────────────────────
st.markdown("---")
img_col, pred_col = st.columns([1, 1], gap="large")

with img_col:
    if show_gradcam:
        tab_orig, tab_cam = st.tabs(["Original", "Grad-CAM"])
        with tab_orig:
            st.image(image, width="stretch", caption="Uploaded X-ray")
        with tab_cam:
            st.image(overlay, width="stretch",
                     caption=f"Grad-CAM — {top_cls}")
    else:
        st.image(image, width="stretch", caption="Uploaded X-ray")

with pred_col:
    st.markdown("### Pathology Predictions")

    if not visible:
        st.info(f"No pathologies detected above {threshold:.0%} threshold.")
    else:
        import plotly.graph_objects as go

        classes = [p["display_name"] for p in visible]
        probs_v = [p["probability"]  for p in visible]
        colours = [
            _SEVERITY_COLOURS.get(p["severity"], "#95a5a6")
            for p in visible
        ]

        fig = go.Figure(go.Bar(
            x            = probs_v,
            y            = classes,
            orientation  = "h",
            marker_color = colours,
            text         = [f"{v:.1%}" for v in probs_v],
            textposition = "outside",
        ))
        fig.update_layout(
            xaxis        = dict(range=[0, 1.15], tickformat=".0%"),
            yaxis        = dict(autorange="reversed"),
            height       = max(250, len(visible) * 40),
            margin       = dict(l=10, r=60, t=10, b=10),
            plot_bgcolor = "white",
            showlegend   = False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-finding detail cards
    if visible:
        st.markdown("#### Findings")
        class_auc = _load_class_auc()
        for pred in visible:
            sev    = pred["severity"]
            colour = _SEVERITY_COLOURS.get(sev, "#95a5a6")

            display_name = pred["display_name"]
            reliability  = _reliability_label(class_auc.get(pred["class_name"]))
            reliability_html = ""
            if reliability:
                rel_label, rel_colour = reliability
                tooltip = (
                    f"Based on measured model accuracy for "
                    f"{display_name} on the held-out test set"
                )
                reliability_html = (
                    f"<span style='color:{rel_colour}; font-size:0.75rem; "
                    f"border:1px solid {rel_colour}; border-radius:8px; "
                    f"padding:1px 6px; margin-left:6px' "
                    f"title='{tooltip}'>{rel_label}</span>"
                )

            st.markdown(
                f"<div style='border-left:4px solid {colour}; "
                f"padding:8px 12px; margin-bottom:8px; background:#fafafa'>"
                f"<strong>{pred['display_name']}</strong> &nbsp;"
                f"<span style='color:{colour}'>{pred['probability']:.1%}</span>"
                f"<span style='color:#888; font-size:0.8rem'> · {sev}</span>"
                f"{reliability_html}"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Reliability reflects this model's own measured AUC per class on "
            "its test set — not the confidence of this specific prediction. "
            "Low-reliability classes should be treated with extra scepticism."
        )

# ── Grad-CAM class selector ───────────────────────────────────────
if show_gradcam and visible:
    from src.utils.config import PATHOLOGY_DISPLAY_NAMES as PATHOLOGY_DISPLAY_NAMES_LOCAL

    st.markdown("---")
    st.markdown("### Grad-CAM — Select class to visualise")
    selected_cls = st.selectbox(
        "Class",
        options   = [p["class_name"]   for p in visible],
        format_func = lambda c: PATHOLOGY_DISPLAY_NAMES_LOCAL.get(c, c),
        label_visibility = "collapsed",
    )

    if selected_cls != top_cls:
        from PIL import Image as PilImage
        from src.data.transforms import get_gradcam_transform
        from src.explainability.gradcam import GradCAM
        from src.explainability.visualise import generate_gradcam_overlay

        model, device = _load_model()
        tensor  = get_gradcam_transform()(
            PilImage.open(io.BytesIO(img_bytes)).convert("RGB")
        )
        with GradCAM(model, model.get_features_layer()) as cam:
            heatmap = cam.generate(tensor, selected_cls, device)
        overlay_sel = generate_gradcam_overlay(
            PilImage.open(io.BytesIO(img_bytes)).convert("RGB"), heatmap
        )
        st.image(overlay_sel, caption=f"Grad-CAM — {selected_cls}",
                 width="stretch")
