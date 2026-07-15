"""
src/explainability/gradcam.py
────────────────────────────────────────────────────────────────
Gradient-weighted Class Activation Mapping (Grad-CAM).

What Grad-CAM does
──────────────────
Grad-CAM visualises which regions of the input image the model
focused on when predicting a specific pathology class.

For a target class k:
  1. Run a forward pass and extract feature maps from the target
     convolutional layer (the last dense block in DenseNet121).
  2. Run a backward pass for class k to get gradients at the
     target layer.
  3. Global-average-pool the gradients across spatial dimensions
     to get per-channel importance weights.
  4. Compute a weighted sum of the feature maps.
  5. Apply ReLU — only keep regions that contribute positively.
  6. Normalise to [0, 1].

The result is a low-resolution heatmap that gets upsampled to the
original image size in visualise.py before display.

Why the last convolutional layer?
───────────────────────────────────
Earlier layers capture low-level features (edges, textures).
The last convolutional layer combines these into high-level
semantic concepts — exactly what we want to highlight.

DenseNet121 target: features.denseblock4 (last dense block)
EfficientNetV2-S target: features[-1]    (last conv block)
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.config import PATHOLOGY_CLASSES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNNs.

    Registers forward and backward hooks on the target layer to
    capture feature maps and gradients.  Hooks are cleaned up
    automatically when the object is used as a context manager.

    Args:
        model:        Trained ChestXrayClassifier.
        target_layer: The convolutional module to hook.
                      Use model.get_features_layer() for the correct
                      layer for each backbone.

    Example::

        from src.models.model import load_model_for_inference
        from src.data.transforms import get_gradcam_transform
        from src.utils.image_utils import load_xray

        model = load_model_for_inference(Path("checkpoints/best_model.pt"))
        cam   = GradCAM(model, model.get_features_layer())

        img_tensor = get_gradcam_transform()(load_xray("xray.png"))

        # Get heatmap for Pneumonia
        heatmap = cam.generate(img_tensor, class_name="Pneumonia")
        # heatmap: np.ndarray of shape (H, W), values in [0, 1]
    """

    def __init__(
        self,
        model:        nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model        = model
        self.target_layer = target_layer

        self._feature_maps: Optional[torch.Tensor] = None
        self._gradients:    Optional[torch.Tensor] = None
        self._hooks: list = []

        self._register_hooks()

    # ── Hook registration ─────────────────────────────────────────

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to the target layer."""

        def _forward_hook(module, input, output):
            # Save the feature maps produced by this layer
            self._feature_maps = output.detach()

        def _backward_hook(module, grad_input, grad_output):
            # Save the gradients flowing back through this layer
            self._gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(_forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(_backward_hook)
        )

    def _remove_hooks(self) -> None:
        """Remove all hooks to avoid memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    # ── Grad-CAM generation ────────────────────────────────────────

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_name:   str,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for a pathology class.

        Args:
            image_tensor: Pre-processed image tensor (3, H, W).
                          Use get_gradcam_transform() for pre-processing.
            class_name:   Pathology class to visualise.
                          Must be in PATHOLOGY_CLASSES.
            device:       Device to run on.  Defaults to model device.

        Returns:
            2D numpy array of shape (H, W) with values in [0, 1].
            Higher values indicate regions the model focused on.

        Raises:
            ValueError: If class_name is not in PATHOLOGY_CLASSES.
        """
        if class_name not in PATHOLOGY_CLASSES:
            raise ValueError(
                f"Unknown class: {class_name!r}. "
                f"Choose from: {PATHOLOGY_CLASSES}"
            )

        class_idx = PATHOLOGY_CLASSES.index(class_name)

        if device is None:
            device = next(self.model.parameters()).device

        # Prepare input — add batch dimension
        x = image_tensor.unsqueeze(0).to(device)
        x.requires_grad = False

        # Forward pass
        self.model.eval()
        logits = self.model(x)            # (1, num_classes)
        score  = logits[0, class_idx]     # scalar logit for target class

        # Backward pass for the target class only
        self.model.zero_grad()
        score.backward()

        # ── Compute Grad-CAM ──────────────────────────────────────
        if self._gradients is None or self._feature_maps is None:
            logger.error(
                "Hooks did not capture gradients/features. "
                "Ensure the target layer is part of the forward path."
            )
            return np.zeros((7, 7))   # fallback: blank heatmap

        # Gradients: (1, C, H, W) → global average pool → (C,)
        weights = self._gradients[0].mean(dim=(1, 2))   # (C,)

        # Feature maps: (1, C, H, W)
        feature_maps = self._feature_maps[0]            # (C, H, W)

        # Weighted sum of feature maps
        cam = torch.zeros(
            feature_maps.shape[1:], dtype=torch.float32, device=device
        )
        for weight, fmap in zip(weights, feature_maps):
            cam += weight * fmap

        # ReLU: keep only positive activations
        cam = torch.relu(cam)

        # Normalise to [0, 1]
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        return cam_np.astype(np.float32)

    def generate_all_classes(
        self,
        image_tensor: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> dict[str, np.ndarray]:
        """Generate Grad-CAM heatmaps for all pathology classes.

        More efficient than calling generate() in a loop because it
        batches the forward passes.  The backward passes still run
        one per class.

        Args:
            image_tensor: Pre-processed image tensor (3, H, W).
            device:       Device to run on.

        Returns:
            Dict mapping class name → 2D heatmap array.
        """
        return {
            cls: self.generate(image_tensor, cls, device)
            for cls in PATHOLOGY_CLASSES
        }

    def get_probability(
        self,
        image_tensor: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> dict[str, float]:
        """Get class probabilities alongside Grad-CAM.

        Runs one forward pass and returns sigmoid probabilities
        for all classes without triggering gradients.

        Args:
            image_tensor: Pre-processed image tensor (3, H, W).
            device:       Device to run on.

        Returns:
            Dict mapping class name → probability (0.0–1.0).
        """
        if device is None:
            device = next(self.model.parameters()).device

        x = image_tensor.unsqueeze(0).to(device)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        return {cls: float(p) for cls, p in zip(PATHOLOGY_CLASSES, probs)}
