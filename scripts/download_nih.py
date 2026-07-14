"""
scripts/download_nih.py
────────────────────────────────────────────────────────────────
NIH ChestX-ray14 dataset downloader.

Downloads the NIH dataset in 12 batches (~2GB each, ~24GB total).
Supports partial downloads — specify individual batches to control
how much disk space you use.

Usage
─────
  # Download all 12 batches (~24GB)
  python scripts/download_nih.py

  # Download first 3 batches only (~6GB, ~15k images)
  python scripts/download_nih.py --batches 1 2 3

  # Download just the labels CSV (~3MB)
  python scripts/download_nih.py --labels-only

  # Check what is already downloaded
  python scripts/download_nih.py --status

Performance by batch count (approximate — actual counts vary by batch)
──────────────────────────
  Batches  Images   Disk    Expected mean AUC
  1        ~5,000   ~2GB    ~0.71
  3        ~15,000  ~6GB    ~0.79
  6        ~30,000  ~12GB   ~0.81
  12       ~60,000  ~24GB   ~0.83+
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.extract import download_labels_csv, download_nih_batch
from src.utils.config import DataConfig, Paths
from src.utils.logger import get_logger, set_log_level

logger = get_logger(__name__)


def show_status() -> None:
    """Print what is currently downloaded."""
    labels_ok = Paths.nih_labels_csv.exists()
    full_imgs  = list(Paths.full_images.glob("*.png"))
    subset_imgs = list(Paths.subset_images.glob("*.png"))

    print("\nNIH ChestX-ray14 Download Status")
    print("─" * 40)
    print(f"  Labels CSV      : {'✅ present' if labels_ok else '❌ missing'}")
    print(f"  Full images     : {len(full_imgs):,} PNG files in data/raw/images/")
    print(f"  Subset images   : {len(subset_imgs):,} PNG files in data/raw/subset/")

    if full_imgs:
        size_mb = sum(p.stat().st_size for p in full_imgs) / 1_048_576
        print(f"  Total size      : {size_mb:.0f} MB ({size_mb/1024:.1f} GB)")

    estimated_batches = len(full_imgs) // 9000
    print(f"  Est. batches    : ~{estimated_batches} of 12")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NIH ChestX-ray14 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batches", nargs="+", type=int, default=list(range(1, 13)),
        metavar="N",
        help="Batch numbers to download (1-12). Default: all.",
    )
    parser.add_argument(
        "--labels-only", action="store_true",
        help="Download only the labels CSV, no images.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show download status and exit.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    set_log_level(args.log_level)
    Paths.ensure_all()

    if args.status:
        show_status()
        return

    # Always download labels
    logger.info("Step 1: Download labels CSV")
    download_labels_csv()

    if args.labels_only:
        logger.info("Labels-only mode — done.")
        return

    # Validate batch numbers
    invalid = [b for b in args.batches if b < 1 or b > 12]
    if invalid:
        logger.error("Invalid batch numbers: %s (must be 1-12)", invalid)
        sys.exit(1)

    total_batches = len(args.batches)
    logger.info(
        "Step 2: Download %d image batch(es): %s",
        total_batches, args.batches,
    )
    logger.info(
        "Estimated disk usage: ~%d GB  (~%d minutes on 50Mbps)",
        total_batches * 4,
        total_batches * 8,
    )

    completed = download_nih_batch(args.batches)
    logger.info(
        "Download complete: %d batches, images at %s",
        len(completed), Paths.full_images,
    )

    show_status()


if __name__ == "__main__":
    main()
