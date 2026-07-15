"""
tests/test_dataloader.py — Unit tests for split_dataframe() and build_dataloaders().
"""
from __future__ import annotations

import random

import pandas as pd
import pytest

from src.utils.config import PATHOLOGY_CLASSES, Paths, TrainingConfig


def _make_labels_df(n=60):
    """Create a minimal labels DataFrame for testing."""
    random.seed(42)
    rows = []
    for i in range(n):
        row = {"image_id": f"{i:08d}_000.png"}
        for cls in PATHOLOGY_CLASSES:
            row[cls] = random.randint(0, 1)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _no_official_splits(monkeypatch, tmp_path):
    """By default, point split files at paths that don't exist so
    split_dataframe() takes the random-split fallback path — matches
    how this project actually runs outside of the real NIH dataset.
    """
    monkeypatch.setattr(Paths, "nih_splits_csv", tmp_path / "no_train_val_list.txt")
    monkeypatch.setattr(Paths, "nih_test_csv",   tmp_path / "no_test_list.txt")


class TestSplitDataframe:

    def test_random_fallback_produces_roughly_70_10_20_split(self):
        from src.data.dataloader import split_dataframe

        df = _make_labels_df(100)
        train_df, val_df, test_df = split_dataframe(df, val_fraction=0.1, seed=42)

        assert len(train_df) + len(val_df) + len(test_df) == 100
        assert len(test_df) == 20
        # val_fraction=0.1 of the total, carved from the 80% train_val portion
        assert 8 <= len(val_df) <= 12

    def test_official_splits_used_when_files_present(self, tmp_path, monkeypatch):
        from src.data.dataloader import split_dataframe

        df = _make_labels_df(20)
        train_ids = df["image_id"].iloc[:14].tolist()
        test_ids  = df["image_id"].iloc[14:].tolist()

        train_file = tmp_path / "train_val_list.txt"
        test_file  = tmp_path / "test_list.txt"
        train_file.write_text("\n".join(train_ids))
        test_file.write_text("\n".join(test_ids))
        monkeypatch.setattr(Paths, "nih_splits_csv", train_file)
        monkeypatch.setattr(Paths, "nih_test_csv",   test_file)

        train_df, val_df, test_df = split_dataframe(df, val_fraction=0.1, seed=42)

        assert len(test_df) == len(test_ids)
        assert set(test_df["image_id"]) == set(test_ids)
        assert len(train_df) + len(val_df) == len(train_ids)

    def test_split_is_deterministic_for_a_given_seed(self):
        from src.data.dataloader import split_dataframe

        df = _make_labels_df(50)
        r1 = split_dataframe(df, seed=7)
        r2 = split_dataframe(df, seed=7)
        for a, b in zip(r1, r2):
            assert list(a["image_id"]) == list(b["image_id"])


class TestBuildDataloaders:

    def test_returns_three_loaders_with_expected_batch_shape(self, tmp_path, monkeypatch):
        from src.data.dataloader import build_dataloaders

        monkeypatch.setattr(TrainingConfig, "batch_size_gpu", 4)
        monkeypatch.setattr(TrainingConfig, "batch_size_cpu", 4)
        monkeypatch.setattr(TrainingConfig, "num_workers_gpu", 0)
        monkeypatch.setattr(TrainingConfig, "num_workers_cpu", 0)

        df = _make_labels_df(60)
        train_loader, val_loader, test_loader = build_dataloaders(df, tmp_path)

        assert len(train_loader) > 0
        images, labels = next(iter(train_loader))
        assert images.shape[1:] == (3, 224, 224)
        assert labels.shape[1] == len(PATHOLOGY_CLASSES)

    def test_pos_weights_attached_to_train_dataset(self, tmp_path, monkeypatch):
        from src.data.dataloader import build_dataloaders

        monkeypatch.setattr(TrainingConfig, "batch_size_gpu", 4)
        monkeypatch.setattr(TrainingConfig, "batch_size_cpu", 4)

        df = _make_labels_df(60)
        train_loader, _, _ = build_dataloaders(df, tmp_path)
        assert hasattr(train_loader.dataset, "pos_weights")
        assert train_loader.dataset.pos_weights.shape == (len(PATHOLOGY_CLASSES),)

    def test_subset_size_caps_total_images_across_splits(self, tmp_path, monkeypatch):
        from src.data.dataloader import build_dataloaders

        monkeypatch.setattr(TrainingConfig, "batch_size_gpu", 1)
        monkeypatch.setattr(TrainingConfig, "batch_size_cpu", 1)

        df = _make_labels_df(100)
        train_loader, val_loader, test_loader = build_dataloaders(
            df, tmp_path, subset_size=20
        )
        total = (
            len(train_loader.dataset)
            + len(val_loader.dataset)
            + len(test_loader.dataset)
        )
        assert total <= 20
