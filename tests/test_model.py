"""
tests/test_model.py — Unit tests for model, loss, and metrics.
"""
from __future__ import annotations
import numpy as np
import pytest
import torch

from src.utils.config import NUM_CLASSES, PATHOLOGY_CLASSES


class TestChestXrayClassifier:
    """Tests for the classifier model."""

    def _build(self, backbone="densenet121"):
        from src.models.model import ChestXrayClassifier
        return ChestXrayClassifier(
            backbone=backbone, pretrained=False, dropout_rate=0.0
        )

    def test_output_shape_densenet(self):
        model  = self._build("densenet121")
        x      = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, NUM_CLASSES)

    def test_output_shape_efficientnet(self):
        model  = self._build("efficientnet_v2_s")
        x      = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, NUM_CLASSES)

    def test_predict_proba_in_zero_one(self):
        model = self._build()
        x     = torch.randn(1, 3, 224, 224)
        probs = model.predict_proba(x)
        assert probs.shape == (1, NUM_CLASSES)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_predict_single_returns_dict(self):
        model = self._build()
        tensor = torch.randn(3, 224, 224)
        result = model.predict_single(tensor)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(PATHOLOGY_CLASSES)
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_unknown_backbone_raises(self):
        from src.models.model import ChestXrayClassifier
        with pytest.raises(ValueError, match="Unknown backbone"):
            ChestXrayClassifier(backbone="resnet999", pretrained=False)

    def test_checkpoint_save_and_load(self, tmp_path):
        from src.models.model import (
            ChestXrayClassifier, save_checkpoint, load_checkpoint
        )
        model     = ChestXrayClassifier(pretrained=False)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser)
        ckpt_path = tmp_path / "test_checkpoint.pt"

        save_checkpoint(model, optimiser, scheduler,
                        epoch=1, val_auc=0.75, best_auc=0.75,
                        history=[], path=ckpt_path)
        assert ckpt_path.exists()

        model2 = ChestXrayClassifier(pretrained=False)
        ckpt   = load_checkpoint(ckpt_path, model2)
        assert ckpt["epoch"]   == 1
        assert ckpt["val_auc"] == 0.75

    def test_gradients_flow_through_model(self):
        model  = self._build()
        x      = torch.randn(2, 3, 224, 224, requires_grad=True)
        logits = model(x)
        loss   = logits.sum()
        loss.backward()
        assert x.grad is not None


class TestWeightedBCELoss:

    def test_forward_returns_scalar(self):
        from src.models.losses import WeightedBCELoss
        pos_weights = torch.ones(NUM_CLASSES)
        criterion   = WeightedBCELoss(pos_weights)
        logits  = torch.randn(4, NUM_CLASSES)
        targets = torch.randint(0, 2, (4, NUM_CLASSES)).float()
        loss    = criterion(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_no_pos_weights_still_works(self):
        from src.models.losses import WeightedBCELoss
        criterion = WeightedBCELoss(pos_weights=None)
        logits    = torch.randn(4, NUM_CLASSES)
        targets   = torch.zeros(4, NUM_CLASSES)
        loss      = criterion(logits, targets)
        assert torch.isfinite(loss)

    def test_build_criterion(self):
        from src.models.losses import build_criterion
        pos_weights = torch.ones(NUM_CLASSES) * 2.0
        criterion   = build_criterion(pos_weights)
        assert isinstance(criterion.criterion,
                          torch.nn.BCEWithLogitsLoss)


class TestMetrics:

    def _make_predictions(self, n=200):
        np.random.seed(42)
        probs  = np.random.rand(n, NUM_CLASSES).astype(np.float32)
        labels = (np.random.rand(n, NUM_CLASSES) > 0.7).astype(np.float32)
        # Ensure every class has at least one positive
        for i in range(NUM_CLASSES):
            labels[i, i] = 1.0
        return probs, labels

    def test_compute_metrics_returns_eval_metrics(self):
        from src.models.metrics import compute_metrics
        probs, labels = self._make_predictions()
        m = compute_metrics(probs, labels, loss=0.5)
        assert 0.0 <= m.mean_auc   <= 1.0
        assert 0.0 <= m.mean_auprc <= 1.0
        assert len(m.auc_per_class) == NUM_CLASSES

    def test_to_dict_serialisable(self):
        import json
        from src.models.metrics import compute_metrics
        probs, labels = self._make_predictions()
        m = compute_metrics(probs, labels)
        d = m.to_dict()
        assert json.dumps(d)   # should not raise

    def test_metrics_to_dataframe_shape(self):
        from src.models.metrics import compute_metrics, metrics_to_dataframe
        probs, labels = self._make_predictions()
        m  = compute_metrics(probs, labels)
        df = metrics_to_dataframe(m)
        assert len(df) == NUM_CLASSES + 1   # +1 for MEAN row
        assert "auc" in df.columns
