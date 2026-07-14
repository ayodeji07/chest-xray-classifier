"""
tests/test_dataset.py — Unit tests for the data pipeline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.utils.config import PATHOLOGY_CLASSES, NUM_CLASSES


def _make_labels_df(n=50):
    """Create a minimal labels DataFrame for testing."""
    import random
    random.seed(42)
    rows = []
    for i in range(n):
        row = {"image_id": f"{i:08d}_000.png", "label_str": "No Finding"}
        for cls in PATHOLOGY_CLASSES:
            row[cls] = random.randint(0, 1)
        rows.append(row)
    return pd.DataFrame(rows)


class TestChestXrayDataset:
    """Tests for ChestXrayDataset."""

    def test_len(self, tmp_path):
        from src.data.dataset import ChestXrayDataset
        df = _make_labels_df(30)
        ds = ChestXrayDataset(df, tmp_path, transform=None, subset_size=0)
        assert len(ds) == 30

    def test_getitem_returns_black_image_when_file_missing(self, tmp_path):
        from src.data.dataset import ChestXrayDataset
        from src.data.transforms import get_val_transform
        df = _make_labels_df(5)
        ds = ChestXrayDataset(df, tmp_path, transform=get_val_transform())
        image, label = ds[0]
        assert image.shape == (3, 224, 224)
        assert label.shape == (NUM_CLASSES,)
        assert label.dtype == torch.float32

    def test_label_tensor_values_are_binary(self, tmp_path):
        from src.data.dataset import ChestXrayDataset
        df = _make_labels_df(20)
        ds = ChestXrayDataset(df, tmp_path)
        _, label = ds[0]
        assert all(v in (0.0, 1.0) for v in label.tolist())

    def test_pos_weights_shape(self, tmp_path):
        from src.data.dataset import ChestXrayDataset
        df = _make_labels_df(50)
        ds = ChestXrayDataset(df, tmp_path)
        w  = ds.get_pos_weights()
        assert w.shape == (NUM_CLASSES,)
        assert (w > 0).all()

    def test_pos_weights_capped_at_50(self, tmp_path):
        from src.data.dataset import ChestXrayDataset
        df = _make_labels_df(50)
        # Zero out all positives for first class
        df[PATHOLOGY_CLASSES[0]] = 0
        ds = ChestXrayDataset(df, tmp_path)
        w  = ds.get_pos_weights()
        assert w[0] <= 50.0

    def test_subset_size_reduces_dataset(self, tmp_path):
        from src.data.dataset import ChestXrayDataset
        df = _make_labels_df(100)
        ds = ChestXrayDataset(df, tmp_path, subset_size=20)
        assert len(ds) <= 20

    def test_label_distribution_returns_dataframe(self, tmp_path):
        from src.data.dataset import ChestXrayDataset
        df = _make_labels_df(50)
        ds = ChestXrayDataset(df, tmp_path)
        dist = ds.label_distribution()
        assert list(dist.columns) == ["class", "count", "prevalence_pct"]
        assert len(dist) == NUM_CLASSES

    def test_black_image_fallback(self, tmp_path):
        from src.data.dataset import ChestXrayDataset
        img = ChestXrayDataset._black_image()
        assert img.size == (224, 224)
        assert img.mode == "RGB"


class TestTransforms:
    """Tests for image transforms."""

    def test_train_transform_output_shape(self):
        from src.data.transforms import get_train_transform
        from PIL import Image
        transform = get_train_transform()
        img    = Image.new("RGB", (256, 256))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_is_deterministic(self):
        from src.data.transforms import get_val_transform
        from PIL import Image
        transform = get_val_transform()
        img = Image.new("RGB", (256, 256))
        t1  = transform(img)
        t2  = transform(img)
        assert torch.allclose(t1, t2)

    def test_inference_transform_equals_val_transform(self):
        from src.data.transforms import get_val_transform, get_inference_transform
        from PIL import Image
        img = Image.new("RGB", (256, 256))
        assert torch.allclose(
            get_val_transform()(img),
            get_inference_transform()(img)
        )
