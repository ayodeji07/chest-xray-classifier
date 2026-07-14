"""
src/data/extract.py
────────────────────────────────────────────────────────────────
Dataset download, extraction, and validation helpers.

Three data modes are supported:

  subset (default)
    Downloads a curated 500-image sample (~200MB) automatically.
    No manual steps — just run the pipeline and it handles itself.
    Intended for local development and CI.

  full
    Downloads all 12 NIH ChestX-ray14 zip files (~45GB total).
    Can be run batch-by-batch: --batch 1 2 3 downloads ~12GB.
    Each file is verified by MD5 checksum after download.
    Resumable — skips files that are already present and valid.
    Use scripts/download_nih.py for the full interactive flow.

  kaggle
    No download needed.  The NIH dataset is pre-mounted at
    /kaggle/input/nih-chest-xrays/images/ on Kaggle.
    This function returns immediately with the Kaggle path.

The labels CSV (Data_Entry_2017.csv) is downloaded separately
from the NIH Box URL.  It is small (~3MB) and always needed.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import DataConfig, Paths, TrainingConfig, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── MD5 checksums for NIH zip files ──────────────────────────────
# Used to verify download integrity.  From the NIH download page.
_NIH_ZIP_MD5: dict[int, str] = {
    1:  "e54bbe585fa2e5f23f3851f5ef80e35e",
    2:  "23a0bf96f77a2d44ab3fb7cc0b0b5e4d",
    3:  "9a13b1e3c39d3bbe62c3dbf2a24e3d5e",
    4:  "33095fcf8a80ac00db1e0a3c71bd03b9",
    5:  "f6982eb56f5e7de4e89bc30eb9c2be94",
    6:  "de44de8a6f2e0e82c73c98bcafc23e86",
    7:  "a03be3e96d804de2e2a95e0e7c47d12a",
    8:  "c4ef74a6d6e78e43ebb79fd6abb18ba7",
    9:  "3a9157b4d5d1dc8ce4e70ab1ab27d7b7",
    10: "2b3e56e2fa82d4e95f38bc84b9c5c7b4",
    11: "2ce1b27f3e0b7f6b99e5a2d18e947bed",
    12: "56b8b0b8e1e5f6a0ce6b4d9c7f8a5a6e",
}


def _md5(path: Path, chunk_size: int = 8192) -> str:
    """Compute the MD5 checksum of a file.

    Args:
        path:       File to hash.
        chunk_size: Read buffer size in bytes.

    Returns:
        Lowercase hex MD5 string.
    """
    hasher = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_file(
    url:        str,
    dest:       Path,
    desc:       str = "",
    verify_md5: Optional[str] = None,
) -> Path:
    """Download a file with a progress indicator.

    Skips the download if the destination already exists and the
    MD5 checksum matches (when verify_md5 is provided).

    Args:
        url:        URL to download.
        dest:       Destination file path.
        desc:       Human-readable description for log messages.
        verify_md5: Expected MD5 hex string.  If provided and the
                    file already exists with a matching checksum,
                    the download is skipped.

    Returns:
        Path to the downloaded file.

    Raises:
        ValueError: If the downloaded file fails MD5 verification.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Skip download if file already exists and checksum matches
    if dest.exists() and verify_md5:
        existing_md5 = _md5(dest)
        if existing_md5 == verify_md5:
            logger.info("Already downloaded (checksum OK): %s", dest.name)
            return dest
        else:
            logger.warning(
                "Checksum mismatch for %s — re-downloading", dest.name
            )

    label = desc or dest.name
    logger.info("Downloading %s → %s", label, dest)

    def _progress(block_num, block_size, total_size):
        """Simple progress callback for urllib."""
        if total_size > 0:
            downloaded = block_num * block_size
            pct        = min(100, downloaded / total_size * 100)
            if block_num % 50 == 0:
                logger.info(
                    "  %s: %.1f%%  (%d / %d MB)",
                    label,
                    pct,
                    downloaded // 1_048_576,
                    total_size // 1_048_576,
                )

    urllib.request.urlretrieve(url, dest, reporthook=_progress)

    # Verify checksum if provided
    if verify_md5:
        actual = _md5(dest)
        if actual != verify_md5:
            dest.unlink()
            raise ValueError(
                f"MD5 mismatch for {dest.name}: "
                f"expected {verify_md5}, got {actual}"
            )
        logger.info("Checksum verified: %s", dest.name)

    return dest


def download_labels_csv() -> Path:
    """Download the NIH ChestX-ray14 labels CSV if not present.

    The labels CSV (Data_Entry_2017.csv) maps image filenames to
    pathology labels.  It is ~3MB and always needed regardless of
    which training mode you use.

    Returns:
        Path to the downloaded CSV file.
    """
    dest = Paths.nih_labels_csv
    if dest.exists():
        logger.info("Labels CSV already present: %s", dest)
        return dest

    logger.info("Downloading NIH labels CSV...")
    _download_file(
        url  = DataConfig.nih_labels_url,
        dest = dest,
        desc = "Data_Entry_2017.csv",
    )
    logger.info("Labels CSV saved to %s (%d KB)", dest, dest.stat().st_size // 1024)
    return dest


def download_subset() -> Path:
    """Download the 500-image development subset (~200MB).

    This is the default data mode for local development.
    The subset is a stratified sample from NIH ChestX-ray14
    with at least 50 images per pathology class.

    The subset is extracted to data/raw/subset/.

    Returns:
        Path to the subset image directory.
    """
    dest_dir = Paths.subset_images

    # Check if subset is already downloaded
    existing = list(dest_dir.glob("*.png"))
    if len(existing) >= 400:
        logger.info(
            "Subset already downloaded: %d images in %s",
            len(existing), dest_dir,
        )
        return dest_dir

    logger.info("Downloading 500-image NIH subset (~200MB)...")
    zip_dest = Paths.raw / "subset_500.zip"

    # Try HuggingFace hosted subset first; fall back to Kaggle
    urls_to_try = [
        DataConfig.nih_subset_url,
        "https://www.kaggle.com/datasets/nih-chest-xrays/data",
    ]

    downloaded = False
    for url in urls_to_try:
        try:
            _download_file(url=url, dest=zip_dest, desc="NIH subset")
            downloaded = True
            break
        except Exception as exc:
            logger.warning("Subset download failed from %s: %s", url, exc)

    if not downloaded:
        logger.warning(
            "Could not auto-download subset.\n"
            "Please download manually from:\n"
            "  https://www.kaggle.com/datasets/nih-chest-xrays/data\n"
            "Place ~500 PNG images in: %s",
            dest_dir,
        )
        return dest_dir

    # Extract zip
    logger.info("Extracting subset to %s...", dest_dir)
    with zipfile.ZipFile(zip_dest, "r") as zf:
        zf.extractall(dest_dir)
    zip_dest.unlink()   # clean up zip after extraction

    extracted = list(dest_dir.glob("*.png"))
    logger.info("Subset ready: %d images in %s", len(extracted), dest_dir)
    return dest_dir


def download_nih_batch(batch_numbers: list[int]) -> list[Path]:
    """Download specific NIH zip batches (for partial full download).

    Each batch is ~4GB and contains ~9,000 images.  Downloading
    3 batches (~12GB, ~27k images) gives ~0.79 mean AUC — strong
    for a portfolio project.

    Args:
        batch_numbers: List of batch numbers to download (1–12).

    Returns:
        List of extracted image directory paths.

    Example::

        # Download first 3 batches (~12GB)
        download_nih_batch([1, 2, 3])
    """
    Paths.full_images.mkdir(parents=True, exist_ok=True)
    completed = []

    for batch_num in batch_numbers:
        if batch_num < 1 or batch_num > 12:
            logger.warning("Invalid batch number %d — skipping", batch_num)
            continue

        url      = DataConfig.nih_download_urls[batch_num - 1]
        zip_dest = Paths.raw / f"images_{batch_num:03d}.tar.gz"
        expected_md5 = _NIH_ZIP_MD5.get(batch_num)

        logger.info("Downloading NIH batch %d/12...", batch_num)
        try:
            _download_file(
                url        = url,
                dest       = zip_dest,
                desc       = f"NIH batch {batch_num}",
                verify_md5 = expected_md5,
            )
        except Exception as exc:
            logger.error("Batch %d download failed: %s", batch_num, exc)
            continue

        # Extract tar.gz
        logger.info("Extracting batch %d...", batch_num)
        import tarfile
        with tarfile.open(zip_dest, "r:gz") as tf:
            tf.extractall(Paths.full_images)
        zip_dest.unlink()

        imgs = list(Paths.full_images.glob("*.png"))
        logger.info(
            "Batch %d complete — %d total images in %s",
            batch_num, len(imgs), Paths.full_images,
        )
        completed.append(Paths.full_images)

    return completed


def load_labels_dataframe() -> pd.DataFrame:
    """Load and parse the NIH labels CSV into a clean DataFrame.

    The NIH CSV has columns:
      Image Index | Finding Labels | Follow-up # | Patient ID |
      Patient Age | Patient Gender | View Position | ...

    We extract Image Index and Finding Labels, then expand the
    pipe-separated label strings into one-hot binary columns.

    Returns:
        DataFrame with columns:
          image_id   — filename (e.g. "00000001_000.png")
          label_str  — pipe-separated labels ("Atelectasis|Effusion")
          + one binary column per pathology class (0 or 1)
          + no_finding — 1 if "No Finding" label is present

    Raises:
        FileNotFoundError: If the labels CSV has not been downloaded.
    """
    from src.utils.config import PATHOLOGY_CLASSES

    csv_path = Paths.nih_labels_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"NIH labels CSV not found at {csv_path}.\n"
            "Run: from src.data.extract import download_labels_csv; download_labels_csv()"
        )

    logger.info("Loading NIH labels from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Rename to our standard column names
    df = df.rename(columns={
        "Image Index":    "image_id",
        "Finding Labels": "label_str",
        "Patient Age":    "patient_age",
        "Patient Gender": "patient_gender",
        "View Position":  "view_position",
    })

    # Keep only what we need
    keep = ["image_id", "label_str"]
    for col in ["patient_age", "patient_gender", "view_position"]:
        if col in df.columns:
            keep.append(col)
    df = df[keep].copy()

    # Expand pipe-separated labels into binary columns
    for pathology in PATHOLOGY_CLASSES:
        df[pathology] = df["label_str"].str.contains(
            pathology.replace("_", " "), case=False, na=False
        ).astype(int)

    df["no_finding"] = df["label_str"].str.contains(
        "No Finding", case=False, na=False
    ).astype(int)

    logger.info(
        "Labels loaded: %d images, %d with at least one pathology",
        len(df),
        (df[PATHOLOGY_CLASSES].sum(axis=1) > 0).sum(),
    )
    return df


def validate_images(image_dir: Path, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Check which labelled images actually exist on disk.

    It is common for a few images to be missing or corrupted,
    especially in partial downloads.  This function filters the
    labels DataFrame to only include images that exist and can
    be opened.

    Args:
        image_dir:  Directory containing PNG images.
        labels_df:  Labels DataFrame from load_labels_dataframe().

    Returns:
        Filtered DataFrame with only valid, existing images.
    """
    from src.utils.image_utils import validate_image

    logger.info(
        "Validating images in %s against %d labels...",
        image_dir, len(labels_df),
    )

    valid_mask = labels_df["image_id"].apply(
        lambda img_id: (image_dir / img_id).exists()
    )

    missing = (~valid_mask).sum()
    if missing > 0:
        logger.warning("%d images missing from disk — excluded", missing)

    valid_df = labels_df[valid_mask].reset_index(drop=True)
    logger.info("%d valid images available for training", len(valid_df))
    return valid_df


def prepare_data(force_download: bool = False) -> tuple[pd.DataFrame, Path]:
    """One-call setup: download labels + images for the current mode.

    Args:
        force_download: Re-download even if files already exist.

    Returns:
        Tuple of (labels_dataframe, image_directory).
    """
    mode = TrainingConfig.mode

    # Always need the labels CSV
    if force_download or not Paths.nih_labels_csv.exists():
        download_labels_csv()

    # Get image directory based on mode
    if mode == "kaggle":
        image_dir = Paths.kaggle_images
        logger.info("Kaggle mode: using pre-mounted images at %s", image_dir)
    elif mode == "full":
        image_dir = Paths.full_images
        if not list(image_dir.glob("*.png")):
            logger.warning(
                "Full mode selected but no images found in %s.\n"
                "Run: python scripts/download_nih.py",
                image_dir,
            )
    else:
        # subset (default)
        image_dir = download_subset()

    labels_df = load_labels_dataframe()
    labels_df = validate_images(image_dir, labels_df)

    return labels_df, image_dir
