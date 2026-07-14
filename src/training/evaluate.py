"""
src/training/evaluate.py
────────────────────────────────────────────────────────────────
Full test-set evaluation pipeline.

Loads the best saved model and runs a complete evaluation on
the held-out test set, producing:
  - Per-class AUC-ROC and AUPRC scores
  - Precision / Recall / F1 at threshold 0.5
  - ROC curves saved as PNG
  - Precision-Recall curves saved as PNG
  - Full results JSON (consumed by the dashboard metrics page)
  - Human-readable summary table

Usage
─────
  python -m src.training.evaluate
  python -m src.training.evaluate --checkpoint checkpoints/best_model.pt
  python -m src.training.evaluate --split val   # evaluate on val set
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.data.dataloader import build_dataloaders
from src.data.extract import prepare_data
from src.models.metrics import (
    EvalMetrics,
    collect_predictions,
    compute_metrics,
    metrics_to_dataframe,
)
from src.models.model import load_model_for_inference
from src.utils.config import PATHOLOGY_CLASSES, Paths, TrainingConfig
from src.utils.logger import get_logger, set_log_level

logger = get_logger(__name__)


def evaluate_on_test_set(
    checkpoint_path: Optional[Path] = None,
    split:           str             = "test",
    save_plots:      bool            = True,
) -> EvalMetrics:
    """Evaluate the best model on the test (or val) set.

    Args:
        checkpoint_path: Path to model checkpoint.
                         Defaults to checkpoints/best_model.pt.
        split:           Which split to evaluate: "test" or "val".
        save_plots:      Whether to save ROC and PR curve plots.

    Returns:
        :class:`EvalMetrics` with all evaluation results.
    """
    ckpt = checkpoint_path or Paths.best_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("  EVALUATION — split=%s  checkpoint=%s", split, ckpt.name)
    logger.info("=" * 60)

    # ── Load model ────────────────────────────────────────────────
    model = load_model_for_inference(ckpt, device)

    # ── Load data ─────────────────────────────────────────────────
    logger.info("Loading data...")
    labels_df, image_dir = prepare_data()
    subset_size = TrainingConfig.subset_size if TrainingConfig.mode == "subset" else 0
    train_loader, val_loader, test_loader = build_dataloaders(
        labels_df, image_dir, subset_size=subset_size
    )

    loader = val_loader if split == "val" else test_loader
    logger.info(
        "Evaluating on %s set (%d batches)...",
        split, len(loader),
    )

    # ── Collect predictions ───────────────────────────────────────
    all_probs, all_labels, mean_loss = collect_predictions(
        model, loader, device
    )

    # ── Compute metrics ───────────────────────────────────────────
    metrics = compute_metrics(all_probs, all_labels, mean_loss)

    # ── Log results ───────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("  RESULTS (%s set)", split.upper())
    logger.info("─" * 50)
    logger.info("  Mean AUC-ROC : %.4f", metrics.mean_auc)
    logger.info("  Mean AUPRC   : %.4f", metrics.mean_auprc)
    logger.info("")
    logger.info("  Per-class AUC-ROC:")
    for cls, auc in sorted(
        metrics.auc_per_class.items(),
        key=lambda x: (-x[1] if not math.isnan(x[1]) else 1),
    ):
        if math.isnan(auc):
            logger.info("    %-22s   N/A  (no positive samples in split)", cls)
        else:
            bar = "█" * int(auc * 20)
            logger.info("    %-22s %.4f  %s", cls, auc, bar)

    # ── Save results JSON ─────────────────────────────────────────
    results = metrics.to_dict()
    results["split"]            = split
    results["checkpoint"]       = str(ckpt)
    results["n_samples"]        = len(all_labels)
    results["pathology_classes"] = PATHOLOGY_CLASSES

    results_path = Paths.eval_results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", results_path)

    # ── Print summary table ───────────────────────────────────────
    df = metrics_to_dataframe(metrics)
    print("\n" + df.to_string(index=False, float_format="%.4f") + "\n")

    # ── Save plots ────────────────────────────────────────────────
    if save_plots:
        _save_roc_curves(all_probs, all_labels, split)
        _save_pr_curves(all_probs, all_labels, split)

    return metrics


def _save_roc_curves(
    probs:  np.ndarray,
    labels: np.ndarray,
    split:  str,
) -> None:
    """Plot and save ROC curves for all pathology classes."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score
    except ImportError:
        logger.warning("matplotlib/sklearn not installed — skipping ROC plot")
        return

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes      = axes.flatten()

    for i, (cls, ax) in enumerate(zip(PATHOLOGY_CLASSES, axes)):
        y_true = labels[:, i]
        y_prob = probs[:, i]

        if y_true.sum() == 0:
            ax.text(0.5, 0.5, "No positives", ha="center", va="center")
            ax.set_title(cls)
            continue

        fpr, tpr, _  = roc_curve(y_true, y_prob)
        auc          = roc_auc_score(y_true, y_prob)

        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(cls)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"ROC Curves — {split.upper()} set", fontsize=14)
    plt.tight_layout()

    out_path = Paths.processed / f"roc_curves_{split}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("ROC curves saved to %s", out_path)


def _save_pr_curves(
    probs:  np.ndarray,
    labels: np.ndarray,
    split:  str,
) -> None:
    """Plot and save Precision-Recall curves for all classes."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        return

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes      = axes.flatten()

    for i, (cls, ax) in enumerate(zip(PATHOLOGY_CLASSES, axes)):
        y_true = labels[:, i]
        y_prob = probs[:, i]

        if y_true.sum() == 0:
            ax.text(0.5, 0.5, "No positives", ha="center", va="center")
            ax.set_title(cls)
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap                   = average_precision_score(y_true, y_prob)

        ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(cls)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"Precision-Recall Curves — {split.upper()} set", fontsize=14)
    plt.tight_layout()

    out_path = Paths.processed / f"pr_curves_{split}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("PR curves saved to %s", out_path)


# ── CLI ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the chest X-ray classifier")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    set_log_level(args.log_level)
    evaluate_on_test_set(
        checkpoint_path = args.checkpoint,
        split           = args.split,
        save_plots      = not args.no_plots,
    )
