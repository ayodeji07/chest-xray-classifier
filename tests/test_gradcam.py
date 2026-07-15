"""
tests/test_gradcam.py — Unit tests for Grad-CAM.
"""
from __future__ import annotations
import numpy as np
import pytest
import torch

from src.utils.config import PATHOLOGY_CLASSES, NUM_CLASSES


def _make_mock_model():
    """Return a minimal mock model for Grad-CAM testing."""
    from src.models.model import ChestXrayClassifier
    model = ChestXrayClassifier(pretrained=False)
    model.eval()
    return model


class TestGradCAM:

    def test_generate_returns_2d_array(self):
        from src.explainability.gradcam import GradCAM
        model   = _make_mock_model()
        tensor  = torch.randn(3, 224, 224)
        target  = model.get_features_layer()
        cam_obj = GradCAM(model, target)
        heatmap = cam_obj.generate(tensor, "Pneumonia")
        cam_obj._remove_hooks()
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_unknown_class_raises_value_error(self):
        from src.explainability.gradcam import GradCAM
        model  = _make_mock_model()
        target = model.get_features_layer()
        cam    = GradCAM(model, target)
        with pytest.raises(ValueError, match="Unknown class"):
            cam.generate(torch.randn(3, 224, 224), "NotAClass")
        cam._remove_hooks()

    def test_context_manager_removes_hooks(self):
        from src.explainability.gradcam import GradCAM
        model  = _make_mock_model()
        target = model.get_features_layer()
        with GradCAM(model, target) as cam:
            assert len(cam._hooks) == 2
        assert len(cam._hooks) == 0

    def test_get_probability_returns_dict(self):
        from src.explainability.gradcam import GradCAM
        model  = _make_mock_model()
        target = model.get_features_layer()
        with GradCAM(model, target) as cam:
            probs = cam.get_probability(torch.randn(3, 224, 224))
        assert set(probs.keys()) == set(PATHOLOGY_CLASSES)
        assert all(0.0 <= v <= 1.0 for v in probs.values())


class TestVisualise:

    def test_get_top_predictions_filters_by_threshold(self):
        from src.explainability.visualise import get_top_predictions
        probs = {cls: 0.1 * (i + 1) for i, cls in enumerate(PATHOLOGY_CLASSES)}
        top   = get_top_predictions(probs, threshold=0.5, top_k=5)
        assert all(p["probability"] >= 0.5 for p in top)
        assert len(top) <= 5

    def test_get_top_predictions_sorted_descending(self):
        from src.explainability.visualise import get_top_predictions
        probs = {cls: float(i) / NUM_CLASSES for i, cls in
                 enumerate(PATHOLOGY_CLASSES)}
        top   = get_top_predictions(probs, threshold=0.0, top_k=10)
        probs_out = [p["probability"] for p in top]
        assert probs_out == sorted(probs_out, reverse=True)

    def test_overlay_heatmap_returns_pil_image(self):
        from PIL import Image
        from src.utils.image_utils import overlay_heatmap
        img     = Image.new("RGB", (224, 224))
        heatmap = np.random.rand(7, 7).astype(np.float32)
        result  = overlay_heatmap(img, heatmap, alpha=0.4)
        assert result.mode == "RGB"
        assert result.size == img.size
