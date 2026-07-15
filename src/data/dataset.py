"""
src/data/dataset.py
────────────────────────────────────────────────────────────────
PyTorch Dataset implementation for NIH ChestX-ray14.

ChestXrayDataset
────────────────
A standard PyTorch Dataset that:
  - Reads images from disk on demand (memory-efficient for 112k images)
  - Returns (image_tensor, label_tensor) pairs
  - Handles missing/corrupt images gracefully (logs warning, skips)
  - Supports all three training modes (subset / full / kaggle)

The label tensor is a multi-hot float32 vector of length NUM_CLASSES.
Multi-label classification means a patient can have multiple
pathologies simultaneously — this is common in chest X-rays.

Class imbalance
───────────────
NIH ChestX-ray14 is heavily imbalanced — "No Finding" makes up
~50% of the dataset and some pathologies appear in <1% of images.
We compute per-class positive weights for the weighted BCE loss.
The weights are exposed via ChestXrayDataset.get_pos_weights().
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.config import PATHOLOGY_CLASSES, NUM_CLASSES, Paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChestXrayDataset(Dataset):
    """PyTorch Dataset for NIH ChestX-ray14 chest X-rays.

    Args:
        labels_df:  DataFrame with 'image_id' column and one binary
                    column per pathology class.  Produced by
                    src.data.extract.load_labels_dataframe().
        image_dir:  Directory containing the PNG images.
        transform:  torchvision transform to apply to each image.
                    Use get_train_transform() for training,
                    get_val_transform() for validation/test.
        subset_size: If > 0, use only this many images (stratified).
                    Used in subset mode for local development.

    Example::

        from src.data.extract import prepare_data
        from src.data.transforms import get_train_transform

        labels_df, image_dir = prepare_data()
        dataset = ChestXrayDataset(labels_df, image_dir, get_train_transform())
        image, labels = dataset[0]
        print(image.shape)   # torch.Size([3, 224, 224])
        print(labels.shape)  # torch.Size([10])
    """

    def __init__(
        self,
        labels_df:   pd.DataFrame,
        image_dir:   Path,
        transform    = None,
        subset_size: int = 0,
    ) -> None:
        self.image_dir  = Path(image_dir)
        self.transform  = transform
        self._skipped   = 0

        # Optionally subsample for local dev
        if subset_size > 0 and len(labels_df) > subset_size:
            labels_df = self._stratified_sample(labels_df, subset_size)
            logger.info(
                "Dataset subsampled to %d images (subset_size=%d)",
                len(labels_df), subset_size,
            )

        self.df = labels_df.reset_index(drop=True)

        # Pre-extract the label matrix as a numpy array for speed.
        # Shape: (N, NUM_CLASSES) — float32 for BCEWithLogitsLoss.
        self._labels = self.df[PATHOLOGY_CLASSES].values.astype(np.float32)

        logger.info(
            "ChestXrayDataset ready: %d images, %d classes",
            len(self.df), NUM_CLASSES,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image_tensor, label_tensor) for one example.

        Args:
            idx: Integer index into the dataset.

        Returns:
            Tuple of:
              image_tensor — float32 tensor of shape (3, 224, 224)
              label_tensor — float32 tensor of shape (NUM_CLASSES,)
        """
        row       = self.df.iloc[idx]
        img_path  = self.image_dir / row["image_id"]
        label     = torch.tensor(self._labels[idx], dtype=torch.float32)

        # Load image — if the file is missing or corrupt, return a
        # black image rather than crashing the DataLoader worker.
        try:
            from src.utils.image_utils import load_xray
            image = load_xray(img_path)
        except (FileNotFoundError, OSError) as exc:
            if self._skipped < 10:   # only log the first 10 to avoid spam
                logger.warning("Skipping %s: %s", row["image_id"], exc)
            self._skipped += 1
            image = self._black_image()

        if self.transform:
            image = self.transform(image)
        else:
            from torchvision.transforms.functional import to_tensor
            image = to_tensor(image)

        return image, label

    # ── Class imbalance helpers ────────────────────────────────────

    def get_pos_weights(self) -> torch.Tensor:
        """Compute per-class positive weights for weighted BCE loss.

        Positive weight for class k = (N - n_pos_k) / n_pos_k
        where N = total samples and n_pos_k = positive samples for k.

        This is the standard approach for imbalanced multi-label
        classification — down-weights the majority class (No Finding)
        and up-weights rare pathologies like Hernia.

        Returns:
            Float tensor of shape (NUM_CLASSES,).
        """
        n_total   = len(self.df)
        n_pos     = self._labels.sum(axis=0)
        # Clamp to avoid division by zero for classes with 0 positives
        n_pos     = np.maximum(n_pos, 1.0)
        weights   = (n_total - n_pos) / n_pos
        # Cap weights at 50 to prevent extreme gradients
        weights   = np.minimum(weights, 50.0)

        logger.debug(
            "Positive weights per class: %s",
            dict(zip(PATHOLOGY_CLASSES, weights.round(2).tolist()))
        )
        return torch.tensor(weights, dtype=torch.float32)

    def label_distribution(self) -> pd.DataFrame:
        """Return per-class label counts and prevalence.

        Useful for understanding class imbalance before training.

        Returns:
            DataFrame with columns: class, count, prevalence_pct.
        """
        counts = self._labels.sum(axis=0)
        return pd.DataFrame({
            "class":          PATHOLOGY_CLASSES,
            "count":          counts.astype(int),
            "prevalence_pct": (counts / len(self.df) * 100).round(2),
        }).sort_values("count", ascending=False).reset_index(drop=True)

    # ── Private helpers ────────────────────────────────────────────

    def _stratified_sample(
        self,
        df:          pd.DataFrame,
        target_size: int,
    ) -> pd.DataFrame:
        """Sample a stratified subset of the dataset.

        Ensures each pathology class has at least some representation
        in the sample, rather than randomly sampling which could
        exclude rare classes entirely.

        Args:
            df:          Full labels DataFrame.
            target_size: Target number of images.

        Returns:
            Sampled DataFrame of size <= target_size.
        """
        from src.utils.config import PATHOLOGY_CLASSES
        import random

        random.seed(42)
        sampled_ids = set()

        # First pass: ensure at least min_per_class per pathology
        min_per_class = max(5, target_size // (len(PATHOLOGY_CLASSES) * 4))
        for cls in PATHOLOGY_CLASSES:
            pos_rows = df[df[cls] == 1]["image_id"].tolist()
            take     = min(len(pos_rows), min_per_class)
            sampled_ids.update(random.sample(pos_rows, take))

        # The per-class floor above can itself exceed target_size when
        # target_size is small relative to the number of classes — trim
        # back down so we always honour the target_size upper bound.
        if len(sampled_ids) > target_size:
            sampled_ids = set(random.sample(list(sampled_ids), target_size))

        # Second pass: fill remaining slots randomly
        remaining = target_size - len(sampled_ids)
        if remaining > 0:
            pool = df[~df["image_id"].isin(sampled_ids)]["image_id"].tolist()
            take = min(len(pool), remaining)
            sampled_ids.update(random.sample(pool, take))

        return df[df["image_id"].isin(sampled_ids)].copy()

    @staticmethod
    def _black_image() -> "PIL.Image.Image":
        """Return a black 224×224 RGB image as a fallback."""
        from PIL import Image
        from src.utils.config import ModelConfig
        size = ModelConfig.image_size
        return Image.new("RGB", (size, size), color=0)
