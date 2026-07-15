"""
tests/test_train.py — Unit tests for the training loop.
"""
from __future__ import annotations

import random
from unittest.mock import patch

import pandas as pd
import pytest

from src.utils.config import PATHOLOGY_CLASSES, Paths, TrainingConfig


def _make_labels_df(n=40):
    random.seed(42)
    rows = []
    for i in range(n):
        row = {"image_id": f"{i:08d}_000.png"}
        for cls in PATHOLOGY_CLASSES:
            row[cls] = random.randint(0, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def _mock_build_model(device):
    """Build a real (untrained, non-pretrained) model — avoids
    downloading ImageNet weights over the network during tests.
    """
    from src.models.model import ChestXrayClassifier
    model = ChestXrayClassifier(pretrained=False)
    model.to(device)
    return model, device


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect all filesystem writes to tmp_path so tests never touch
    the real project's checkpoints/ or data/processed/ directories.
    """
    ckpt_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(Paths, "checkpoints",    ckpt_dir)
    monkeypatch.setattr(Paths, "best_model",     ckpt_dir / "best_model.pt")
    monkeypatch.setattr(Paths, "processed",      tmp_path / "processed")
    monkeypatch.setattr(Paths, "raw",            tmp_path / "raw")
    monkeypatch.setattr(Paths, "labels",         tmp_path / "labels")
    monkeypatch.setattr(Paths, "sample_xrays",   tmp_path / "sample_xrays")
    monkeypatch.setattr(Paths, "subset_images",  tmp_path / "subset_images")
    monkeypatch.setattr(Paths, "gradcam_output", tmp_path / "gradcam_output")
    monkeypatch.setattr(Paths, "nih_splits_csv", tmp_path / "no_train_val_list.txt")
    monkeypatch.setattr(Paths, "nih_test_csv",   tmp_path / "no_test_list.txt")
    return ckpt_dir


@pytest.fixture
def fast_training_config(monkeypatch):
    """Small batch sizes / epoch counts so tests run in seconds."""
    monkeypatch.setattr(TrainingConfig, "batch_size_gpu",  2)
    monkeypatch.setattr(TrainingConfig, "batch_size_cpu",  2)
    monkeypatch.setattr(TrainingConfig, "num_workers_gpu", 0)
    monkeypatch.setattr(TrainingConfig, "num_workers_cpu", 0)
    monkeypatch.setattr(TrainingConfig, "epochs", 2)
    monkeypatch.setattr(TrainingConfig, "early_stop_patience", 10)


class TestDryRun:
    """Regression tests for the bug where dry_run=True overwrote real
    checkpoints on disk (a 1-batch smoke test should never touch
    checkpoints/, since it isn't a real trained model).
    """

    def test_dry_run_does_not_write_any_checkpoint_files(
        self, tmp_path, isolated_paths, fast_training_config
    ):
        from src.training.train import train

        df = _make_labels_df(40)
        with patch("src.training.train.prepare_data", return_value=(df, tmp_path)), \
             patch("src.training.train.build_model", side_effect=_mock_build_model):
            train(resume=False, dry_run=True)

        assert list(isolated_paths.glob("*.pt")) == []
        assert not Paths.best_model.exists()

    def test_dry_run_does_not_overwrite_existing_best_model(
        self, tmp_path, isolated_paths, fast_training_config
    ):
        """Even if a real best_model.pt already exists on disk, a dry
        run must leave it completely untouched.
        """
        from src.training.train import train

        isolated_paths.mkdir(parents=True, exist_ok=True)
        Paths.best_model.write_bytes(b"real trained model bytes")
        original_bytes = Paths.best_model.read_bytes()

        df = _make_labels_df(40)
        with patch("src.training.train.prepare_data", return_value=(df, tmp_path)), \
             patch("src.training.train.build_model", side_effect=_mock_build_model):
            train(resume=False, dry_run=True)

        assert Paths.best_model.read_bytes() == original_bytes

    def test_dry_run_completes_in_one_epoch_regardless_of_epochs_config(
        self, tmp_path, isolated_paths, fast_training_config, monkeypatch
    ):
        from src.training.train import train

        monkeypatch.setattr(TrainingConfig, "epochs", 20)
        df = _make_labels_df(40)
        with patch("src.training.train.prepare_data", return_value=(df, tmp_path)), \
             patch("src.training.train.build_model", side_effect=_mock_build_model):
            result = train(resume=False, dry_run=True)

        assert result["epochs_trained"] == 1


class TestRealTraining:

    def test_writes_checkpoint_and_best_model(
        self, tmp_path, isolated_paths, fast_training_config
    ):
        from src.training.train import train

        df = _make_labels_df(40)
        with patch("src.training.train.prepare_data", return_value=(df, tmp_path)), \
             patch("src.training.train.build_model", side_effect=_mock_build_model):
            result = train(resume=False, dry_run=False)

        assert Paths.best_model.exists()
        assert list(isolated_paths.glob("checkpoint_epoch_*.pt"))
        assert 0.0 <= result["best_auc"] <= 1.0

    def test_resume_continues_from_saved_epoch(
        self, tmp_path, isolated_paths, fast_training_config, monkeypatch
    ):
        from src.training.train import train

        monkeypatch.setattr(TrainingConfig, "epochs", 1)
        df = _make_labels_df(40)
        with patch("src.training.train.prepare_data", return_value=(df, tmp_path)), \
             patch("src.training.train.build_model", side_effect=_mock_build_model):
            train(resume=False, dry_run=False)

        monkeypatch.setattr(TrainingConfig, "epochs", 2)
        with patch("src.training.train.prepare_data", return_value=(df, tmp_path)), \
             patch("src.training.train.build_model", side_effect=_mock_build_model):
            result = train(resume=True, dry_run=False)

        # Resumed run should add exactly one more epoch (epoch 2) on
        # top of the epoch 1 record already in the loaded history.
        assert result["epochs_trained"] == 2
        assert result["history"][-1]["epoch"] == 2
