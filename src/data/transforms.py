"""
src/data/transforms.py
────────────────────────────────────────────────────────────────
Image augmentation pipelines for each data split.

Medical image augmentation requires care:
  - Horizontal flip is safe for chest X-rays (anatomically valid)
  - Vertical flip is NOT safe (lungs should be at top)
  - Rotation should be small (±10°) — large rotations are clinically
    invalid and confuse the model
  - Colour jitter is appropriate since X-rays are greyscale-converted
    to RGB — the actual 'colour' carries no information
  - Random crops at inference time would hurt performance — use
    centre crop only for val/test/inference

Three pipelines:
  get_train_transform()  — augmented; used during training
  get_val_transform()    — no augmentation; used for val + test
  get_inference_transform() — no augmentation; used in the app

All three resize to 224×224 and normalise with ImageNet stats.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from src.utils.config import DataConfig, ModelConfig


def get_train_transform():
    """Return the training augmentation pipeline.

    Augmentations applied:
      - Resize to 256×256 (slightly larger than the 224 crop)
      - Random horizontal flip (p=0.5) — valid for chest X-rays
      - Small random rotation (±10°) — preserves clinical validity
      - Random brightness/contrast jitter — robust to scan variation
      - Random crop to 224×224 — effective regularisation
      - Normalise with ImageNet mean/std

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(DataConfig.resize_to),          # 256×256
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomCrop(ModelConfig.image_size),    # 224×224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DataConfig.imagenet_mean,
            std=DataConfig.imagenet_std,
        ),
    ])


def get_val_transform():
    """Return the validation/test transform pipeline (no augmentation).

    Applies only:
      - Resize to 256×256
      - Centre crop to 224×224
      - Normalise with ImageNet mean/std

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(DataConfig.resize_to),
        transforms.CenterCrop(ModelConfig.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DataConfig.imagenet_mean,
            std=DataConfig.imagenet_std,
        ),
    ])


def get_inference_transform():
    """Return the inference transform for single-image prediction.

    Identical to the val transform — no augmentation, deterministic.
    Used by the Streamlit app and the API.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return get_val_transform()


def get_gradcam_transform():
    """Return the Grad-CAM transform — same as inference.

    Kept as a separate function so it is explicit at the call site
    that the transform is for Grad-CAM (not just prediction).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return get_val_transform()
