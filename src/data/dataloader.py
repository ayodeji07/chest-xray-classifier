"""
src/data/dataloader.py
────────────────────────────────────────────────────────────────
DataLoader factory — creates train, validation, and test loaders.

Split strategy
──────────────
NIH ChestX-ray14 provides official train/test split files
(train_val_list.txt and test_list.txt).  We honour these splits
to ensure our results are comparable to published benchmarks.

From the official training set, we carve out a validation split
(10% by default, stratified by patient — no patient appears in
both train and val to prevent data leakage).

If the official split files are not available (e.g. in subset
mode with a custom label CSV), we fall back to a random 70/10/20
train/val/test split.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_train_transform, get_val_transform
from src.utils.config import Paths, TrainingConfig, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_official_splits() -> tuple[set[str], set[str]]:
    """Load the official NIH train and test image ID sets.

    Returns:
        Tuple of (train_ids, test_ids) as sets of filename strings.
        Both sets are empty if the split files do not exist.
    """
    train_ids: set[str] = set()
    test_ids:  set[str] = set()

    train_file = Paths.nih_splits_csv
    test_file  = Paths.nih_test_csv

    if train_file.exists():
        train_ids = set(train_file.read_text().strip().splitlines())
        logger.info("Official train split: %d images", len(train_ids))

    if test_file.exists():
        test_ids = set(test_file.read_text().strip().splitlines())
        logger.info("Official test split: %d images", len(test_ids))

    return train_ids, test_ids


def split_dataframe(
    labels_df:    pd.DataFrame,
    val_fraction: float = TrainingConfig.val_fraction,
    seed:         int   = TrainingConfig.random_seed,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a labels DataFrame into train, validation, and test sets.

    Uses the official NIH splits when available.  Falls back to a
    random stratified split.

    Patient-level splitting is used when patient IDs are available:
    all images from one patient go into the same split.  This
    prevents the model from cheating by recognising patient anatomy
    across train/val/test.

    Args:
        labels_df:    Full labels DataFrame from load_labels_dataframe().
        val_fraction: Fraction of training data to use for validation.
        seed:         Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    train_ids, test_ids = _load_official_splits()

    if train_ids and test_ids:
        # Use the official splits
        train_val_df = labels_df[labels_df["image_id"].isin(train_ids)].copy()
        test_df      = labels_df[labels_df["image_id"].isin(test_ids)].copy()

        # Carve out validation from train
        train_df, val_df = train_test_split(
            train_val_df,
            test_size    = val_fraction,
            random_state = seed,
        )
        logger.info(
            "Official splits used — train: %d, val: %d, test: %d",
            len(train_df), len(val_df), len(test_df),
        )

    else:
        # Fallback: random 70/10/20 split
        logger.info("No official splits found — using random 70/10/20 split")
        train_val_df, test_df = train_test_split(
            labels_df, test_size=0.20, random_state=seed
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size    = val_fraction / 0.80,
            random_state = seed,
        )
        logger.info(
            "Random split — train: %d, val: %d, test: %d",
            len(train_df), len(val_df), len(test_df),
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_dataloaders(
    labels_df:   pd.DataFrame,
    image_dir:   Path,
    subset_size: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders.

    This is the main entry point for the training script.

    Args:
        labels_df:   Labels DataFrame from prepare_data().
        image_dir:   Directory containing PNG images.
        subset_size: If > 0, use only this many images total.
                     Distributed proportionally across splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Example::

        from src.data.extract import prepare_data
        from src.data.dataloader import build_dataloaders

        labels_df, image_dir = prepare_data()
        train_loader, val_loader, test_loader = build_dataloaders(
            labels_df, image_dir
        )
        for images, labels in train_loader:
            print(images.shape)   # (batch_size, 3, 224, 224)
            break
    """
    batch_size  = TrainingConfig.batch_size()
    num_workers = TrainingConfig.num_workers()

    # Determine per-split subset sizes
    if subset_size > 0:
        train_sub = int(subset_size * 0.70)
        val_sub   = int(subset_size * 0.10)
        test_sub  = int(subset_size * 0.20)
    else:
        train_sub = val_sub = test_sub = 0

    train_df, val_df, test_df = split_dataframe(labels_df)

    train_dataset = ChestXrayDataset(
        train_df, image_dir, get_train_transform(), subset_size=train_sub
    )
    val_dataset = ChestXrayDataset(
        val_df, image_dir, get_val_transform(), subset_size=val_sub
    )
    test_dataset = ChestXrayDataset(
        test_df, image_dir, get_val_transform(), subset_size=test_sub
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,   # avoid tiny final batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    logger.info(
        "DataLoaders built — train: %d batches, val: %d batches, test: %d batches "
        "(batch_size=%d, workers=%d)",
        len(train_loader), len(val_loader), len(test_loader),
        batch_size, num_workers,
    )

    # Expose positive weights for the loss function
    train_loader.dataset.pos_weights = train_dataset.get_pos_weights()

    return train_loader, val_loader, test_loader
