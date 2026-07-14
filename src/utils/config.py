"""
src/utils/config.py
────────────────────────────────────────────────────────────────
Central configuration for the Chest X-Ray Pathology Classifier.

All paths, model names, training hyperparameters, and runtime
settings live here.  Nothing else in the codebase hardcodes a
path, model name, or magic number — import from this module.

Training modes
──────────────
  subset  — 500-image auto-downloaded sample for local development.
            Tests the full pipeline in minutes with no manual setup.
  full    — Complete NIH ChestX-ray14 dataset (~45GB, 112k images).
            Requires running scripts/download_nih.py first.
  kaggle  — Reads images from /kaggle/input/nih-chest-xrays/.
            Use this when running the training notebook on Kaggle.

Set TRAINING_MODE in your .env file.  The pipeline adapts
batch sizes, worker counts, and data paths accordingly.

Backbone agnosticism
────────────────────
Set BACKBONE to swap the CNN architecture without touching any
other code.  Both choices are pre-trained on ImageNet.

  densenet121       — CheXNet architecture (Stanford 2017 paper).
                      Best name recognition for medical imaging roles.
  efficientnet_v2_s — Faster training, marginally higher AUC.
                      Good alternative when compute is limited.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from pathlib import Path


# ── Project root ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]


class Paths:
    """All filesystem paths, relative to the project root."""

    # ── Data ──────────────────────────────────────────────────────
    data:           Path = ROOT / "data"
    raw:            Path = ROOT / "data" / "raw"
    labels:         Path = ROOT / "data" / "labels"
    processed:      Path = ROOT / "data" / "processed"
    sample_xrays:   Path = ROOT / "data" / "sample_xrays"

    # Image directories per training mode
    subset_images:  Path = ROOT / "data" / "raw" / "subset"
    full_images:    Path = ROOT / "data" / "raw" / "images"
    kaggle_images:  Path = Path("/kaggle/input/nih-chest-xrays/images")

    # NIH label CSV
    nih_labels_csv: Path = ROOT / "data" / "labels" / "Data_Entry_2017.csv"
    nih_splits_csv: Path = ROOT / "data" / "labels" / "train_val_list.txt"
    nih_test_csv:   Path = ROOT / "data" / "labels" / "test_list.txt"

    # Checkpoints
    checkpoints:    Path = ROOT / "checkpoints"
    best_model:     Path = ROOT / "checkpoints" / "best_model.pt"

    # Evaluation outputs
    eval_results:   Path = ROOT / "data" / "processed" / "eval_results.json"
    roc_curves:     Path = ROOT / "data" / "processed" / "roc_curves.png"
    gradcam_output: Path = ROOT / "data" / "processed" / "gradcam"

    @classmethod
    def ensure_all(cls) -> None:
        """Create all output directories if they do not exist."""
        for attr in (
            "raw", "labels", "processed", "sample_xrays",
            "subset_images", "checkpoints", "gradcam_output",
        ):
            getattr(cls, attr).mkdir(parents=True, exist_ok=True)

    @classmethod
    def images_dir(cls) -> Path:
        """Return the image directory for the current training mode."""
        mode = TrainingConfig.mode
        if mode == "kaggle":
            return cls.kaggle_images
        if mode == "full":
            return cls.full_images
        return cls.subset_images     # "subset" is the default


# ── Pathology classes ─────────────────────────────────────────────
# 10 clinically meaningful classes from NIH ChestX-ray14.
# Order matters — it must match the model's output layer ordering.
PATHOLOGY_CLASSES: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

NUM_CLASSES: int = len(PATHOLOGY_CLASSES)

# Human-readable display names for the app
PATHOLOGY_DISPLAY_NAMES: dict[str, str] = {
    "Atelectasis":       "Atelectasis",
    "Cardiomegaly":      "Cardiomegaly",
    "Consolidation":     "Consolidation",
    "Edema":             "Edema",
    "Effusion":          "Pleural Effusion",
    "Mass":              "Mass",
    "Nodule":            "Nodule",
    "Pleural_Thickening": "Pleural Thickening",
    "Pneumonia":         "Pneumonia",
    "Pneumothorax":      "Pneumothorax",
}

# Clinical severity hints shown in the app alongside predictions
PATHOLOGY_SEVERITY: dict[str, str] = {
    "Atelectasis":       "moderate",
    "Cardiomegaly":      "moderate",
    "Consolidation":     "moderate",
    "Edema":             "high",
    "Effusion":          "moderate",
    "Mass":              "high",
    "Nodule":            "moderate",
    "Pleural_Thickening": "low",
    "Pneumonia":         "high",
    "Pneumothorax":      "critical",
}


class ModelConfig:
    """Model architecture settings."""

    # CNN backbone — change this to swap architectures.
    # Valid options: "densenet121", "efficientnet_v2_s"
    backbone: str = os.getenv("BACKBONE", "densenet121")

    # ImageNet pre-training — always True; training from scratch
    # on medical images without pre-training gives much worse results.
    pretrained: bool = os.getenv("PRETRAINED", "true").lower() == "true"

    # Input image size — 224×224 is standard for ImageNet pre-trained models.
    # DenseNet121 and EfficientNetV2-S both expect this.
    image_size: int = int(os.getenv("IMAGE_SIZE", "224"))

    # Dropout rate in the classifier head
    dropout_rate: float = float(os.getenv("DROPOUT_RATE", "0.5"))

    # Where to save / load model weights
    checkpoint_dir: Path = Paths.checkpoints


class TrainingConfig:
    """Training hyperparameters and runtime settings."""

    # ── Mode ──────────────────────────────────────────────────────
    # "subset" → 500-image local dev sample
    # "full"   → complete NIH dataset (requires ~45GB)
    # "kaggle" → Kaggle platform (/kaggle/input/nih-chest-xrays/)
    mode: str = os.getenv("TRAINING_MODE", "subset")

    # Number of images to use in subset mode.
    # Increase to 2000–5000 for better local model quality.
    subset_size: int = int(os.getenv("SUBSET_SIZE", "500"))

    # ── Hyperparameters ───────────────────────────────────────────
    epochs:        int   = int(os.getenv("TRAIN_EPOCHS",    "10"))
    learning_rate: float = float(os.getenv("TRAIN_LR",      "1e-4"))
    weight_decay:  float = float(os.getenv("TRAIN_WD",      "1e-5"))
    random_seed:   int   = int(os.getenv("TRAIN_SEED",      "42"))

    # ── Batch sizes ───────────────────────────────────────────────
    # GPU (T4): 32 fits comfortably in 16GB VRAM for DenseNet121
    # CPU: 8 keeps memory usage reasonable
    batch_size_gpu: int = int(os.getenv("BATCH_SIZE_GPU", "32"))
    batch_size_cpu: int = int(os.getenv("BATCH_SIZE_CPU", "8"))

    # ── DataLoader ────────────────────────────────────────────────
    num_workers_gpu: int = int(os.getenv("NUM_WORKERS_GPU", "4"))
    num_workers_cpu: int = int(os.getenv("NUM_WORKERS_CPU", "0"))

    # ── Train / val / test split ratios ───────────────────────────
    # NIH provides official train/test split files.
    # We further split train into train + val.
    val_fraction:  float = float(os.getenv("VAL_FRACTION",  "0.1"))

    # ── Checkpointing ─────────────────────────────────────────────
    save_every_epoch: bool  = os.getenv("SAVE_EVERY_EPOCH", "true").lower() == "true"
    keep_last_n:      int   = int(os.getenv("KEEP_LAST_N_CHECKPOINTS", "3"))

    # ── Learning rate scheduler ───────────────────────────────────
    # ReduceLROnPlateau: halve LR if val AUC stagnates for patience epochs
    lr_patience:  int   = int(os.getenv("LR_PATIENCE",  "2"))
    lr_factor:    float = float(os.getenv("LR_FACTOR",  "0.5"))
    min_lr:       float = float(os.getenv("MIN_LR",     "1e-7"))

    # ── Early stopping ────────────────────────────────────────────
    early_stop_patience: int = int(os.getenv("EARLY_STOP_PATIENCE", "5"))

    @classmethod
    def batch_size(cls) -> int:
        """Return the appropriate batch size for the current device."""
        import torch
        return cls.batch_size_gpu if torch.cuda.is_available() else cls.batch_size_cpu

    @classmethod
    def num_workers(cls) -> int:
        """Return the appropriate DataLoader worker count."""
        import torch
        return cls.num_workers_gpu if torch.cuda.is_available() else cls.num_workers_cpu

    @classmethod
    def is_kaggle(cls) -> bool:
        """True when running on the Kaggle platform."""
        return cls.mode == "kaggle"


class DataConfig:
    """Dataset-specific settings."""

    # ── NIH ChestX-ray14 ─────────────────────────────────────────
    # 12 zip files, each ~4GB.  URLs from the NIH download page.
    nih_download_urls: list[str] = [
        "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5t.gz",
        "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
        "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
        "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
        "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
        "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
        "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
        "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
        "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
        "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
        "https://nihcc.box.com/shared/static/hhq8fkdgvcari67d4t1efqavlfaf1nlx.gz",
        "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
    ]

    # NIH label CSV download URL
    # The original nihcc.box.com static links expire periodically;
    # this GitHub mirror serves the same Data_Entry_2017.csv content.
    nih_labels_url: str = (
        "https://raw.githubusercontent.com/gregwchase/nih-chest-xray"
        "/master/data/Data_Entry_2017.csv"
    )

    # NIH subset: the first N images (see TrainingConfig.subset_size)
    # extracted from NIH batch 1, mirrored on HuggingFace with original
    # filenames intact so they match nih_labels_url's Image Index column.
    nih_subset_url: str = (
        "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset"
        "/resolve/main/data/images/images_001.zip"
    )

    # Image normalisation statistics from ImageNet.
    # These are used for pre-trained models trained on ImageNet.
    # NIH-specific statistics exist but ImageNet values give good results
    # because we use ImageNet pre-trained weights.
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std:  tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Minimum image size before augmentation crops to 224×224
    resize_to: int = 256


class APIConfig:
    """FastAPI settings."""

    host:        str  = os.getenv("API_HOST",  "0.0.0.0")
    port:        int  = int(os.getenv("API_PORT", "8000"))
    title:       str  = "Chest X-Ray Classifier API"
    version:     str  = "1.0.0"
    debug:       bool = os.getenv("API_DEBUG", "true").lower() == "true"
    description: str  = (
        "Multi-label chest X-ray pathology classification using DenseNet121. "
        "Returns per-class probabilities and Grad-CAM heatmaps."
    )

    # Maximum uploaded image size (bytes) — 10MB is generous for X-rays
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


class GradCAMConfig:
    """Grad-CAM visualisation settings."""

    # Target layer for Grad-CAM.
    # DenseNet121: last dense block features
    # EfficientNetV2: last convolutional block
    target_layers: dict[str, str] = {
        "densenet121":      "features.denseblock4",
        "efficientnet_v2_s": "features[-1]",
    }

    # Heatmap colour map — 'jet' is standard for medical imaging papers
    colormap: str = os.getenv("GRADCAM_COLORMAP", "jet")

    # Blend alpha: 0.0 = only heatmap, 1.0 = only original image
    alpha: float = float(os.getenv("GRADCAM_ALPHA", "0.4"))


# ── Convenience re-export ─────────────────────────────────────────
class Settings:
    """Single import point for all config sections."""

    paths:    Paths          = Paths
    model:    ModelConfig    = ModelConfig
    training: TrainingConfig = TrainingConfig
    data:     DataConfig     = DataConfig
    api:      APIConfig      = APIConfig
    gradcam:  GradCAMConfig  = GradCAMConfig

    pathologies: list[str] = PATHOLOGY_CLASSES
    num_classes: int       = NUM_CLASSES


settings = Settings()
