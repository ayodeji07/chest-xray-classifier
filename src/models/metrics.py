"""
src/models/metrics.py
────────────────────────────────────────────────────────────────
Evaluation metrics for multi-label chest X-ray classification.

The primary metric is AUC-ROC per class — the same metric used
in the CheXNet paper and all subsequent benchmarks on NIH
ChestX-ray14.  Mean AUC across all 10 classes is the headline
number reported in the model card.

Why AUC-ROC rather than accuracy?
───────────────────────────────────
Accuracy is misleading for imbalanced datasets.  A model that
always predicts "No Pneumonia" achieves 98.8% accuracy on NIH
but has zero clinical value.  AUC-ROC measures the model's
ability to rank positive examples above negative ones regardless
of the class prior — it is threshold-independent and robust to
class imbalance.

Secondary metrics:
  - Precision-Recall AUC (AUPRC) — more informative than ROC for
    very rare classes where nearly all examples are negative.
  - Per-class precision, recall, F1 at threshold 0.5.
  - Confusion matrix per class.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from src.utils.config import PATHOLOGY_CLASSES
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalMetrics:
    """Container for all evaluation metrics from one epoch.

    Attributes:
        auc_per_class:  Per-class AUC-ROC scores.
        mean_auc:       Mean AUC-ROC across all classes.
        auprc_per_class: Per-class area under precision-recall curve.
        mean_auprc:     Mean AUPRC.
        precision:      Per-class precision at threshold 0.5.
        recall:         Per-class recall at threshold 0.5.
        f1:             Per-class F1 score at threshold 0.5.
        loss:           Mean validation loss.
    """

    auc_per_class:   dict[str, float] = field(default_factory=dict)
    mean_auc:        float            = 0.0
    auprc_per_class: dict[str, float] = field(default_factory=dict)
    mean_auprc:      float            = 0.0
    precision:       dict[str, float] = field(default_factory=dict)
    recall:          dict[str, float] = field(default_factory=dict)
    f1:              dict[str, float] = field(default_factory=dict)
    loss:            float            = 0.0

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "mean_auc":       round(self.mean_auc,   4),
            "mean_auprc":     round(self.mean_auprc, 4),
            "loss":           round(self.loss,        4),
            "auc_per_class":  {k: round(v, 4) for k, v in self.auc_per_class.items()},
            "auprc_per_class":{k: round(v, 4) for k, v in self.auprc_per_class.items()},
            "precision":      {k: round(v, 4) for k, v in self.precision.items()},
            "recall":         {k: round(v, 4) for k, v in self.recall.items()},
            "f1":             {k: round(v, 4) for k, v in self.f1.items()},
        }

    def summary_string(self) -> str:
        """One-line summary for logging."""
        return (
            f"mean_auc={self.mean_auc:.4f}  "
            f"mean_auprc={self.mean_auprc:.4f}  "
            f"loss={self.loss:.4f}"
        )


def compute_metrics(
    all_probs:  np.ndarray,
    all_labels: np.ndarray,
    loss:       float = 0.0,
) -> EvalMetrics:
    """Compute all evaluation metrics from predictions and labels.

    Args:
        all_probs:  Predicted probabilities, shape (N, num_classes).
                    Values in [0, 1].
        all_labels: Ground-truth binary labels, shape (N, num_classes).
                    Values in {0, 1}.
        loss:       Mean validation loss for this epoch.

    Returns:
        :class:`EvalMetrics` with all metrics computed.
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_score,
        recall_score,
        f1_score,
    )

    metrics = EvalMetrics(loss=loss)
    auc_scores   = []
    auprc_scores = []

    for i, cls_name in enumerate(PATHOLOGY_CLASSES):
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]
        y_pred = (y_prob >= 0.5).astype(int)

        # Skip classes with no positive examples — AUC is undefined.
        if y_true.sum() == 0:
            logger.debug("No positives for %s — AUC skipped", cls_name)
            metrics.auc_per_class[cls_name]   = float("nan")
            metrics.auprc_per_class[cls_name] = float("nan")
        else:
            auc  = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            metrics.auc_per_class[cls_name]   = float(auc)
            metrics.auprc_per_class[cls_name] = float(auprc)
            auc_scores.append(auc)
            auprc_scores.append(auprc)

        metrics.precision[cls_name] = float(
            precision_score(y_true, y_pred, zero_division=0)
        )
        metrics.recall[cls_name] = float(
            recall_score(y_true, y_pred, zero_division=0)
        )
        metrics.f1[cls_name] = float(
            f1_score(y_true, y_pred, zero_division=0)
        )

    metrics.mean_auc   = float(np.mean(auc_scores))   if auc_scores   else 0.0
    metrics.mean_auprc = float(np.mean(auprc_scores)) if auprc_scores else 0.0

    return metrics


def collect_predictions(
    model:       "ChestXrayClassifier",
    data_loader: "DataLoader",
    device:      torch.device,
    criterion    = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run inference on a DataLoader and collect predictions.

    Args:
        model:       ChestXrayClassifier in eval mode.
        data_loader: DataLoader yielding (images, labels) batches.
        device:      Device to run inference on.
        criterion:   Optional loss function for computing val loss.

    Returns:
        Tuple of:
          all_probs   — shape (N, num_classes), sigmoid probabilities
          all_labels  — shape (N, num_classes), ground-truth labels
          mean_loss   — average loss across all batches (0.0 if no criterion)
    """
    model.eval()
    all_probs:  list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs  = torch.sigmoid(logits)

            if criterion is not None:
                loss        = criterion(logits, labels)
                total_loss += loss.item()
                n_batches  += 1

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs_arr  = np.concatenate(all_probs,  axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    mean_loss  = total_loss / n_batches if n_batches > 0 else 0.0

    return probs_arr, labels_arr, mean_loss


def evaluate(
    model:       "ChestXrayClassifier",
    data_loader: "DataLoader",
    device:      torch.device,
    criterion    = None,
) -> EvalMetrics:
    """Full evaluation: collect predictions and compute all metrics.

    Convenience wrapper around collect_predictions + compute_metrics.

    Args:
        model:       ChestXrayClassifier.
        data_loader: Validation or test DataLoader.
        device:      Device to run on.
        criterion:   Optional loss function.

    Returns:
        :class:`EvalMetrics` with all metrics.

    Example::

        val_metrics = evaluate(model, val_loader, device, criterion)
        print(val_metrics.summary_string())
        # mean_auc=0.8234  mean_auprc=0.3821  loss=0.1847
    """
    probs, labels, loss = collect_predictions(
        model, data_loader, device, criterion
    )
    metrics = compute_metrics(probs, labels, loss)

    logger.info("Evaluation: %s", metrics.summary_string())
    logger.debug(
        "Per-class AUC: %s",
        {k: f"{v:.4f}" for k, v in metrics.auc_per_class.items()},
    )
    return metrics


def metrics_to_dataframe(metrics: EvalMetrics) -> pd.DataFrame:
    """Convert EvalMetrics to a per-class summary DataFrame.

    Useful for displaying results in notebooks and the dashboard.

    Args:
        metrics: EvalMetrics from evaluate().

    Returns:
        DataFrame with columns: class, auc, auprc, precision,
        recall, f1.  Sorted by AUC descending.
    """
    rows = []
    for cls in PATHOLOGY_CLASSES:
        rows.append({
            "class":     cls,
            "auc":       metrics.auc_per_class.get(cls, float("nan")),
            "auprc":     metrics.auprc_per_class.get(cls, float("nan")),
            "precision": metrics.precision.get(cls, 0.0),
            "recall":    metrics.recall.get(cls, 0.0),
            "f1":        metrics.f1.get(cls, 0.0),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("auc", ascending=False).reset_index(drop=True)

    # Add a footer row with mean values
    mean_row = pd.DataFrame([{
        "class":     "MEAN",
        "auc":       metrics.mean_auc,
        "auprc":     metrics.mean_auprc,
        "precision": df["precision"].mean(),
        "recall":    df["recall"].mean(),
        "f1":        df["f1"].mean(),
    }])

    return pd.concat([df, mean_row], ignore_index=True)
