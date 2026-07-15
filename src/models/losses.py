"""
src/models/losses.py
────────────────────────────────────────────────────────────────
Loss functions for multi-label chest X-ray classification.

Why weighted BCE?
────────────────
NIH ChestX-ray14 is heavily imbalanced:
  - "No Finding" makes up ~50% of images
  - Pneumonia appears in only ~1.2% of images
  - Hernia appears in only ~0.2% of images

Standard BCE loss would be dominated by the majority class and
the model would learn to predict "No Finding" for everything.

BCEWithLogitsLoss with pos_weight addresses this:
  - For each class k, positive samples are up-weighted by
    pos_weight[k] = (N - n_pos_k) / n_pos_k
  - This balances the gradient contribution between positive
    and negative examples for each class independently.

This is the exact approach used in the CheXNet paper.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy loss for multi-label classification.

    Wraps nn.BCEWithLogitsLoss with per-class positive weights to
    handle class imbalance.  The model output should be raw logits
    (no sigmoid) — this loss applies sigmoid internally for
    numerical stability.

    Args:
        pos_weights: Float tensor of shape (num_classes,).
                     Obtained from ChestXrayDataset.get_pos_weights().
                     If None, unweighted BCE is used.

    Example::

        pos_weights = train_dataset.get_pos_weights()
        criterion   = WeightedBCELoss(pos_weights)

        logits = model(images)            # (batch, 10)
        loss   = criterion(logits, labels)  # scalar
        loss.backward()
    """

    def __init__(self, pos_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()

        self._pos_weights = pos_weights

        # BCEWithLogitsLoss is numerically more stable than
        # BCELoss + sigmoid because it combines them into a single
        # log-sum-exp operation.
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight = pos_weights,
            reduction  = "mean",
        )

        if pos_weights is not None:
            logger.info(
                "WeightedBCELoss: pos_weights range [%.2f, %.2f]",
                pos_weights.min().item(),
                pos_weights.max().item(),
            )
        else:
            logger.info("WeightedBCELoss: no pos_weights (unweighted)")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted BCE loss.

        Args:
            logits:  Raw model outputs of shape (batch, num_classes).
                     Must be logits, not probabilities.
            targets: Binary label tensor of shape (batch, num_classes).
                     Values must be 0.0 or 1.0 (float).

        Returns:
            Scalar loss tensor.
        """
        # Move pos_weights to the same device as logits if needed.
        # This handles the case where we build the loss on CPU but
        # move data to GPU during training.
        if (self._pos_weights is not None
                and self._pos_weights.device != logits.device):
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight = self._pos_weights.to(logits.device),
                reduction  = "mean",
            )

        return self.criterion(logits, targets)

    def to(self, device):
        """Move pos_weights to device alongside the module."""
        super().to(device)
        if self._pos_weights is not None:
            self._pos_weights = self._pos_weights.to(device)
            self.criterion    = nn.BCEWithLogitsLoss(
                pos_weight = self._pos_weights,
                reduction  = "mean",
            )
        return self


def build_criterion(
    pos_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> WeightedBCELoss:
    """Build and return the training loss function.

    Args:
        pos_weights: Per-class positive weights from the dataset.
                     Pass None for unweighted BCE.
        device:      Device to place the loss on.

    Returns:
        WeightedBCELoss criterion.

    Example::

        from src.data.dataloader import build_dataloaders
        from src.models.losses import build_criterion

        train_loader, val_loader, test_loader = build_dataloaders(...)
        criterion = build_criterion(
            pos_weights = train_loader.dataset.pos_weights,
            device      = device,
        )
    """
    criterion = WeightedBCELoss(pos_weights)
    if device is not None:
        criterion = criterion.to(device)
    return criterion
