"""
src/models/model.py
────────────────────────────────────────────────────────────────
Chest X-Ray pathology classifier model.

Architecture
────────────
A standard transfer-learning setup:
  1. Pre-trained CNN backbone (ImageNet weights)
  2. Global average pooling
  3. Dropout for regularisation
  4. Linear classifier head → NUM_CLASSES outputs

The backbone is swappable via config — changing BACKBONE in
.env switches between DenseNet121 and EfficientNetV2-S without
touching training, evaluation, or app code.

Why no sigmoid in the forward pass?
────────────────────────────────────
The model outputs raw logits.  BCEWithLogitsLoss (used in
training) is numerically more stable than BCE + sigmoid because
it combines them in a single operation.  At inference time we
apply sigmoid explicitly to get probabilities in [0, 1].

DenseNet121 feature dimensions
───────────────────────────────
  DenseNet121 final features: 1024 channels
  After GlobalAveragePool:    (batch, 1024)
  After Dropout:              (batch, 1024)
  After Linear:               (batch, NUM_CLASSES)
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.utils.config import ModelConfig, NUM_CLASSES, PATHOLOGY_CLASSES
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Feature dimension lookup ──────────────────────────────────────
# Number of feature channels output by each backbone before the
# classifier head.  Used to size the linear layer correctly.
_FEATURE_DIMS: dict[str, int] = {
    "densenet121":       1024,
    "efficientnet_v2_s": 1280,
}


class ChestXrayClassifier(nn.Module):
    """Multi-label chest X-ray pathology classifier.

    Wraps a pre-trained CNN backbone with a custom classification
    head.  Outputs raw logits for NUM_CLASSES pathologies.

    Args:
        backbone:     CNN backbone name.  One of:
                      "densenet121", "efficientnet_v2_s".
                      Defaults to the value in ModelConfig.
        num_classes:  Number of output classes.  Defaults to NUM_CLASSES.
        pretrained:   Load ImageNet pre-trained weights.  Always True
                      in practice — training from scratch on X-rays
                      gives significantly worse results.
        dropout_rate: Dropout probability in the classifier head.

    Example::

        model = ChestXrayClassifier()
        x     = torch.randn(4, 3, 224, 224)
        logits = model(x)
        print(logits.shape)   # torch.Size([4, 10])

        probs = torch.sigmoid(logits)   # at inference time
    """

    def __init__(
        self,
        backbone:     str   = ModelConfig.backbone,
        num_classes:  int   = NUM_CLASSES,
        pretrained:   bool  = ModelConfig.pretrained,
        dropout_rate: float = ModelConfig.dropout_rate,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self.num_classes   = num_classes

        # Load the backbone and replace its classifier head
        self.features, feature_dim = _build_backbone(backbone, pretrained)

        # Classifier head: Dropout → Linear
        # Global average pooling is handled in forward() rather than
        # here because DenseNet and EfficientNet expose features
        # differently before their own pooling/classifier layers.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes),
        )

        logger.info(
            "ChestXrayClassifier: backbone=%s, classes=%d, pretrained=%s",
            backbone, num_classes, pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Logit tensor of shape (batch, num_classes).
            Apply sigmoid to get probabilities.
        """
        features = self.features(x)

        # Global average pool: (batch, C, H, W) → (batch, C)
        # F.adaptive_avg_pool2d handles any spatial size, making the
        # model robust to images slightly different from 224×224.
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        pooled = pooled.view(pooled.size(0), -1)   # flatten

        return self.classifier(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference and return class probabilities.

        Applies sigmoid to convert logits to [0, 1] probabilities.
        Use this at inference time (not during training — the loss
        function handles the sigmoid internally).

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Probability tensor of shape (batch, num_classes).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict_single(
        self,
        image_tensor: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> dict[str, float]:
        """Classify a single pre-processed image.

        Args:
            image_tensor: Float tensor of shape (3, 224, 224).
            device:       Device to run on.  Defaults to the model's
                          current device.

        Returns:
            Dict mapping pathology name → probability (0.0–1.0).

        Example::

            from src.data.transforms import get_inference_transform
            from src.utils.image_utils import load_xray

            img    = load_xray("xray.png")
            tensor = get_inference_transform()(img)
            probs  = model.predict_single(tensor)
            print(probs)
            # {"Pneumonia": 0.73, "Effusion": 0.12, ...}
        """
        if device is None:
            device = next(self.parameters()).device

        # Add batch dimension: (3, H, W) → (1, 3, H, W)
        x = image_tensor.unsqueeze(0).to(device)

        probs = self.predict_proba(x).squeeze(0).cpu().tolist()
        return {cls: float(p) for cls, p in zip(PATHOLOGY_CLASSES, probs)}

    def get_features_layer(self) -> nn.Module:
        """Return the backbone's last convolutional layer.

        Used by the Grad-CAM implementation to register the
        activation hook on the correct layer.

        Returns:
            The target convolutional module for Grad-CAM.
        """
        if self.backbone_name == "densenet121":
            # Last dense block in DenseNet121
            return self.features.denseblock4
        elif self.backbone_name == "efficientnet_v2_s":
            # Last convolutional block in EfficientNetV2-S
            return self.features[-1]
        else:
            # Generic fallback: last module in features
            return list(self.features.children())[-1]


# ── Backbone factory ──────────────────────────────────────────────

def _build_backbone(
    name:      str,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    """Load a pre-trained CNN backbone and return its feature extractor.

    Removes the backbone's original classifier so we can attach
    our own multi-label head.

    Args:
        name:       Backbone name ("densenet121" or "efficientnet_v2_s").
        pretrained: Whether to load ImageNet pre-trained weights.

    Returns:
        Tuple of (feature_extractor, feature_dim).

    Raises:
        ValueError: If the backbone name is not recognised.
    """
    import torchvision.models as models

    weights_arg = "IMAGENET1K_V1" if pretrained else None

    if name == "densenet121":
        net      = models.densenet121(weights=weights_arg)
        features = net.features     # the convolutional part
        dim      = _FEATURE_DIMS["densenet121"]

    elif name == "efficientnet_v2_s":
        net      = models.efficientnet_v2_s(weights=weights_arg)
        features = net.features
        dim      = _FEATURE_DIMS["efficientnet_v2_s"]

    else:
        raise ValueError(
            f"Unknown backbone: {name!r}. "
            "Choose 'densenet121' or 'efficientnet_v2_s'."
        )

    logger.info(
        "Backbone: %s | feature_dim: %d | pretrained: %s",
        name, dim, pretrained,
    )
    return features, dim


# ── Checkpoint helpers ────────────────────────────────────────────

def save_checkpoint(
    model:      ChestXrayClassifier,
    optimiser:  torch.optim.Optimizer,
    scheduler,
    epoch:      int,
    val_auc:    float,
    best_auc:   float,
    history:    list[dict],
    path:       Path,
) -> None:
    """Save a full training checkpoint to disk.

    Saves everything needed to resume training exactly where it
    left off: model weights, optimiser state, scheduler state,
    epoch counter, and training history.

    Args:
        model:      The classifier model.
        optimiser:  Adam/SGD optimiser.
        scheduler:  LR scheduler.
        epoch:      Current epoch number (1-indexed).
        val_auc:    Validation mean AUC at this epoch.
        best_auc:   Best validation mean AUC seen so far.
        history:    List of per-epoch metric dicts.
        path:       File path to save the checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optimiser_state":  optimiser.state_dict(),
        "scheduler_state":  scheduler.state_dict() if scheduler else None,
        "val_auc":          val_auc,
        "best_auc":         best_auc,
        "history":          history,
        "backbone":         model.backbone_name,
        "num_classes":      model.num_classes,
        "pathology_classes": PATHOLOGY_CLASSES,
    }, path)
    logger.info("Checkpoint saved: %s (epoch=%d, val_auc=%.4f)", path.name, epoch, val_auc)


def load_checkpoint(
    path:   Path,
    model:  ChestXrayClassifier,
    optimiser: Optional[torch.optim.Optimizer] = None,
    scheduler = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load a training checkpoint from disk.

    Args:
        path:      Path to the .pt checkpoint file.
        model:     Model to load weights into.
        optimiser: Optional optimiser to restore state.
        scheduler: Optional scheduler to restore state.
        device:    Device to map tensors to.

    Returns:
        The full checkpoint dict (epoch, val_auc, best_auc, history).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading checkpoint: %s", path)
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])

    if optimiser and "optimiser_state" in checkpoint:
        optimiser.load_state_dict(checkpoint["optimiser_state"])

    if scheduler and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    logger.info(
        "Checkpoint loaded: epoch=%d, val_auc=%.4f, best_auc=%.4f",
        checkpoint["epoch"],
        checkpoint["val_auc"],
        checkpoint["best_auc"],
    )
    return checkpoint


def load_model_for_inference(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> ChestXrayClassifier:
    """Load a trained model ready for inference.

    Convenience function for the app and API — loads weights
    from a checkpoint and puts the model in eval mode.

    Args:
        checkpoint_path: Path to best_model.pt or any checkpoint.
        device:          Device to load the model onto.

    Returns:
        ChestXrayClassifier in eval mode on the specified device.

    Example::

        model = load_model_for_inference(Path("checkpoints/best_model.pt"))
        probs = model.predict_single(image_tensor)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            "Train the model first:\n"
            "  python -m src.training.train\n"
            "Or download a pre-trained checkpoint from the project README."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model from checkpoint metadata
    backbone    = checkpoint.get("backbone",    ModelConfig.backbone)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)

    model = ChestXrayClassifier(
        backbone     = backbone,
        num_classes  = num_classes,
        pretrained   = False,   # we're loading our own weights
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    logger.info(
        "Model loaded for inference: %s on %s (val_auc=%.4f)",
        backbone, device,
        checkpoint.get("best_auc", 0.0),
    )
    return model


def build_model(
    device: Optional[torch.device] = None,
) -> tuple[ChestXrayClassifier, torch.device]:
    """Build a new model and move it to the appropriate device.

    Convenience factory for the training script.

    Args:
        device: Target device.  Auto-detected if None.

    Returns:
        Tuple of (model, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChestXrayClassifier()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "Model on %s — total params: %s, trainable: %s",
        device,
        f"{n_params:,}",
        f"{n_train:,}",
    )
    return model, device
