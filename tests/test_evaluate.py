"""
tests/test_evaluate.py — Unit tests for the evaluation pipeline.
"""
from __future__ import annotations

import json
import math
import random
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.utils.config import NUM_CLASSES, PATHOLOGY_CLASSES, Paths, TrainingConfig


def _make_labels_df(n=40, zero_positive_class=None):
    """Create a labels DataFrame. If zero_positive_class is given,
    that column is forced to all-0 so the class has no positive
    samples anywhere in the resulting splits — reproduces the
    scenario that used to crash evaluate_on_test_set().
    """
    random.seed(42)
    rows = []
    for i in range(n):
        row = {"image_id": f"{i:08d}_000.png"}
        for cls in PATHOLOGY_CLASSES:
            row[cls] = random.randint(0, 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    if zero_positive_class:
        df[zero_positive_class] = 0
    return df


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "processed",      tmp_path / "processed")
    monkeypatch.setattr(Paths, "eval_results",   tmp_path / "processed" / "eval_results.json")
    monkeypatch.setattr(Paths, "nih_splits_csv", tmp_path / "no_train_val_list.txt")
    monkeypatch.setattr(Paths, "nih_test_csv",   tmp_path / "no_test_list.txt")
    return tmp_path


@pytest.fixture
def fast_config(monkeypatch):
    monkeypatch.setattr(TrainingConfig, "batch_size_gpu",  4)
    monkeypatch.setattr(TrainingConfig, "batch_size_cpu",  4)
    monkeypatch.setattr(TrainingConfig, "num_workers_gpu", 0)
    monkeypatch.setattr(TrainingConfig, "num_workers_cpu", 0)


def _make_checkpoint(tmp_path):
    from src.models.model import ChestXrayClassifier, save_checkpoint

    model     = ChestXrayClassifier(pretrained=False)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser)
    ckpt_path = tmp_path / "test_model.pt"
    save_checkpoint(
        model, optimiser, scheduler,
        epoch=1, val_auc=0.5, best_auc=0.5, history=[],
        path=ckpt_path,
    )
    return ckpt_path


class TestComputeMetricsWithZeroPositiveClass:
    """Regression tests for the crash when a pathology class has no
    positive samples in the evaluated split (AUC is mathematically
    undefined -> NaN). This used to crash the per-class summary
    logging with `ValueError: cannot convert float NaN to integer`.
    """

    def test_zero_positive_class_produces_nan_not_a_crash(self):
        from src.models.metrics import compute_metrics

        np.random.seed(0)
        n = 50
        probs  = np.random.rand(n, NUM_CLASSES).astype(np.float32)
        labels = (np.random.rand(n, NUM_CLASSES) > 0.5).astype(np.float32)
        labels[:, 0] = 0.0   # first class has zero positives

        metrics = compute_metrics(probs, labels, loss=0.5)
        assert math.isnan(metrics.auc_per_class[PATHOLOGY_CLASSES[0]])
        # every other class should still have a real score
        assert not math.isnan(metrics.auc_per_class[PATHOLOGY_CLASSES[1]])

    def test_nan_auc_is_json_serialisable_via_to_dict(self):
        from src.models.metrics import compute_metrics

        np.random.seed(0)
        n = 50
        probs  = np.random.rand(n, NUM_CLASSES).astype(np.float32)
        labels = (np.random.rand(n, NUM_CLASSES) > 0.5).astype(np.float32)
        labels[:, 0] = 0.0

        metrics = compute_metrics(probs, labels)
        # json.dumps allows NaN by default (as literal NaN) - just
        # confirm serialisation doesn't raise.
        json.dumps(metrics.to_dict())


class TestEvaluateOnTestSet:

    def test_does_not_crash_when_a_class_has_no_positives(
        self, tmp_path, isolated_paths, fast_config
    ):
        from src.training.evaluate import evaluate_on_test_set

        ckpt_path = _make_checkpoint(tmp_path)
        df = _make_labels_df(60, zero_positive_class=PATHOLOGY_CLASSES[0])

        with patch("src.training.evaluate.prepare_data", return_value=(df, tmp_path)):
            metrics = evaluate_on_test_set(
                checkpoint_path=ckpt_path, split="test", save_plots=False
            )

        assert math.isnan(metrics.auc_per_class[PATHOLOGY_CLASSES[0]])

    def test_writes_results_json(self, tmp_path, isolated_paths, fast_config):
        from src.training.evaluate import evaluate_on_test_set

        ckpt_path = _make_checkpoint(tmp_path)
        df = _make_labels_df(60)

        with patch("src.training.evaluate.prepare_data", return_value=(df, tmp_path)):
            evaluate_on_test_set(
                checkpoint_path=ckpt_path, split="test", save_plots=False
            )

        assert Paths.eval_results.exists()
        results = json.loads(Paths.eval_results.read_text())
        assert results["split"] == "test"
        assert "auc_per_class" in results

    def test_save_plots_writes_roc_and_pr_pngs(
        self, tmp_path, isolated_paths, fast_config
    ):
        from src.training.evaluate import evaluate_on_test_set

        ckpt_path = _make_checkpoint(tmp_path)
        df = _make_labels_df(60, zero_positive_class=PATHOLOGY_CLASSES[0])

        with patch("src.training.evaluate.prepare_data", return_value=(df, tmp_path)):
            evaluate_on_test_set(
                checkpoint_path=ckpt_path, split="test", save_plots=True
            )

        assert (Paths.processed / "roc_curves_test.png").exists()
        assert (Paths.processed / "pr_curves_test.png").exists()
