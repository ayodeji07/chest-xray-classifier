"""
src/explainability/visualise.py
────────────────────────────────────────────────────────────────
Grad-CAM heatmap overlay and multi-class visualisation helpers.

These functions take the raw Grad-CAM output (a 2D float array)
and produce display-ready images for the Streamlit app and the
evaluation notebook.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.config import GradCAMConfig, PATHOLOGY_CLASSES, Paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_gradcam_overlay(
    original_image: "PIL.Image.Image",
    heatmap:        np.ndarray,
    alpha:          float = GradCAMConfig.alpha,
    colormap:       str   = GradCAMConfig.colormap,
) -> "PIL.Image.Image":
    """Overlay a Grad-CAM heatmap on the original X-ray image.

    Args:
        original_image: Original X-ray as a PIL Image.
        heatmap:        2D float array from GradCAM.generate(),
                        values in [0, 1].
        alpha:          Blend weight: 0 = heatmap only, 1 = original only.
        colormap:       OpenCV colormap name (e.g. "jet", "hot").

    Returns:
        Blended PIL Image (RGB).
    """
    from src.utils.image_utils import overlay_heatmap
    return overlay_heatmap(original_image, heatmap, alpha=alpha, colormap=colormap)


def generate_multi_class_grid(
    original_image: "PIL.Image.Image",
    heatmaps:       dict[str, np.ndarray],
    probs:          dict[str, float],
    top_k:          int   = 4,
    alpha:          float = GradCAMConfig.alpha,
) -> "PIL.Image.Image":
    """Create a grid showing the top-k Grad-CAM heatmaps.

    Arranges the original image alongside the top-k most
    confidently predicted pathology heatmaps in a grid layout.
    Useful for the evaluation notebook.

    Args:
        original_image: Original X-ray PIL Image.
        heatmaps:       Dict from GradCAM.generate_all_classes().
        probs:          Class probabilities from model prediction.
        top_k:          Number of top classes to show.
        alpha:          Heatmap blend alpha.

    Returns:
        Composite PIL Image grid.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.warning("matplotlib not installed — returning single overlay")
        top_class = max(probs, key=probs.get)
        return generate_gradcam_overlay(
            original_image, heatmaps[top_class], alpha=alpha
        )

    from src.utils.image_utils import overlay_heatmap, resize_for_display

    # Sort classes by probability descending
    ranked = sorted(probs.items(), key=lambda x: -x[1])[:top_k]

    n_cols = top_k + 1   # original + top_k overlays
    fig    = plt.figure(figsize=(4 * n_cols, 4))
    gs     = gridspec.GridSpec(1, n_cols, figure=fig)

    # Original image
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(original_image, cmap="gray")
    ax0.set_title("Original", fontsize=10, fontweight="bold")
    ax0.axis("off")

    # Heatmap overlays for top-k classes
    for col_idx, (cls_name, prob) in enumerate(ranked, start=1):
        heatmap = heatmaps.get(cls_name)
        if heatmap is None:
            continue

        overlay = overlay_heatmap(
            original_image, heatmap, alpha=alpha
        )
        ax = fig.add_subplot(gs[col_idx])
        ax.imshow(overlay)
        ax.set_title(
            f"{cls_name}\n{prob:.1%}",
            fontsize=9,
            color="red" if prob > 0.5 else "black",
        )
        ax.axis("off")

    plt.tight_layout()

    # Render to PIL Image
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


def save_gradcam_sample(
    original_image: "PIL.Image.Image",
    heatmap:        np.ndarray,
    class_name:     str,
    probability:    float,
    output_dir:     Optional[Path] = None,
) -> Path:
    """Save a single Grad-CAM overlay image to disk.

    Used during evaluation to generate sample visualisations for
    the model card and documentation.

    Args:
        original_image: Original X-ray PIL Image.
        heatmap:        2D float Grad-CAM array.
        class_name:     Pathology class name.
        probability:    Model's predicted probability for this class.
        output_dir:     Directory to save into.  Defaults to
                        data/processed/gradcam/.

    Returns:
        Path to the saved PNG file.
    """
    out_dir = output_dir or Paths.gradcam_output
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay  = generate_gradcam_overlay(original_image, heatmap)
    filename = f"gradcam_{class_name.lower().replace(' ', '_')}_{probability:.2f}.png"
    out_path = out_dir / filename
    overlay.save(out_path)

    logger.info(
        "Grad-CAM saved: %s (class=%s, prob=%.3f)",
        out_path.name, class_name, probability,
    )
    return out_path


def get_top_predictions(
    probs:     dict[str, float],
    threshold: float = 0.3,
    top_k:     int   = 5,
) -> list[dict]:
    """Return the top-k predictions above a probability threshold.

    Args:
        probs:     Dict of class → probability from model inference.
        threshold: Minimum probability to include in results.
        top_k:     Maximum number of results.

    Returns:
        List of dicts with keys: class_name, probability,
        severity, display_name.  Sorted by probability descending.

    Example::

        preds = get_top_predictions({"Pneumonia": 0.73, "Effusion": 0.12})
        # → [{"class_name": "Pneumonia", "probability": 0.73, ...}]
    """
    from src.utils.config import PATHOLOGY_DISPLAY_NAMES, PATHOLOGY_SEVERITY

    results = []
    for cls_name, prob in sorted(probs.items(), key=lambda x: -x[1]):
        if prob < threshold:
            continue
        results.append({
            "class_name":   cls_name,
            "probability":  round(prob, 4),
            "severity":     PATHOLOGY_SEVERITY.get(cls_name, "unknown"),
            "display_name": PATHOLOGY_DISPLAY_NAMES.get(cls_name, cls_name),
        })

    return results[:top_k]
