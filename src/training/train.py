"""
src/training/train.py
────────────────────────────────────────────────────────────────
Training loop for the chest X-ray pathology classifier.

Features
────────
  - Auto-detects CPU vs GPU; adjusts batch size and workers
  - Full checkpoint/resume: saves every epoch, resumes automatically
  - Keeps only the last N checkpoints to save disk space
  - Saves best_model.pt whenever validation AUC improves
  - ReduceLROnPlateau scheduler: halves LR on stagnation
  - Early stopping: halts when val AUC stops improving
  - Exports training history to JSON for the notebook/dashboard

Usage
─────
  # Start fresh
  python -m src.training.train

  # Resume from last checkpoint
  python -m src.training.train --resume

  # Resume from specific checkpoint
  python -m src.training.train --resume --checkpoint checkpoints/checkpoint_epoch_3.pt

  # Dry run (1 batch per split, smoke-test the pipeline)
  python -m src.training.train --dry-run
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

from src.data.dataloader import build_dataloaders
from src.data.extract import prepare_data
from src.models.losses import build_criterion
from src.models.metrics import evaluate
from src.models.model import (
    build_model,
    load_checkpoint,
    save_checkpoint,
)
from src.utils.config import Paths, TrainingConfig, settings
from src.utils.logger import get_logger, set_log_level

logger = get_logger(__name__)


def _latest_checkpoint() -> Optional[Path]:
    """Return the most recently saved epoch checkpoint, if any."""
    ckpts = sorted(
        Paths.checkpoints.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return ckpts[-1] if ckpts else None


def _prune_old_checkpoints(keep_last: int) -> None:
    """Delete all but the most recent `keep_last` epoch checkpoints."""
    ckpts = sorted(
        Paths.checkpoints.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for old in ckpts[:-keep_last]:
        old.unlink()
        logger.debug("Pruned old checkpoint: %s", old.name)


def train(
    resume:          bool          = False,
    checkpoint_path: Optional[Path] = None,
    dry_run:         bool          = False,
) -> dict:
    """Run the full training pipeline.

    Args:
        resume:          If True, resume from the latest checkpoint
                         (or from checkpoint_path if given).
        checkpoint_path: Explicit checkpoint to resume from.
        dry_run:         If True, run only 1 batch per split to
                         verify the pipeline without a full epoch.

    Returns:
        Dict with final training metrics and history.
    """
    cfg = TrainingConfig
    Paths.ensure_all()

    logger.info("=" * 60)
    logger.info("  CHEST X-RAY CLASSIFIER — TRAINING")
    logger.info("  mode=%s  backbone=%s  epochs=%d",
                cfg.mode, settings.model.backbone, cfg.epochs)
    logger.info("=" * 60)

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s (%.1f GB VRAM)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # ── Data ──────────────────────────────────────────────────────
    logger.info("Loading data (mode=%s)...", cfg.mode)
    labels_df, image_dir = prepare_data()

    subset_size = cfg.subset_size if cfg.mode == "subset" else 0
    train_loader, val_loader, test_loader = build_dataloaders(
        labels_df, image_dir, subset_size=subset_size
    )

    # ── Model, loss, optimiser ────────────────────────────────────
    model, device = build_model(device)

    pos_weights = getattr(train_loader.dataset, "pos_weights", None)
    criterion   = build_criterion(pos_weights, device)

    optimiser = optim.Adam(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode     = "max",       # maximise val AUC
        factor   = cfg.lr_factor,
        patience = cfg.lr_patience,
        min_lr   = cfg.min_lr,
        verbose  = True,
    )

    # ── Resume ────────────────────────────────────────────────────
    start_epoch = 1
    best_auc    = 0.0
    history: list[dict] = []
    no_improve  = 0

    if resume:
        ckpt_path = checkpoint_path or _latest_checkpoint()
        if ckpt_path and ckpt_path.exists():
            ckpt = load_checkpoint(
                ckpt_path, model, optimiser, scheduler, device
            )
            start_epoch = ckpt["epoch"] + 1
            best_auc    = ckpt["best_auc"]
            history     = ckpt.get("history", [])
            logger.info(
                "Resuming from epoch %d (best_auc=%.4f)",
                start_epoch, best_auc,
            )
        else:
            logger.warning("No checkpoint found — starting from scratch")

    # ── Training loop ─────────────────────────────────────────────
    total_t = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_t = time.perf_counter()
        logger.info("─" * 55)
        logger.info("Epoch %d / %d", epoch, cfg.epochs)

        # Train
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if dry_run and batch_idx >= 1:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimiser.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            train_loss += loss.item()
            n_batches  += 1

            if batch_idx % 50 == 0:
                logger.info(
                    "  Batch %d/%d  loss=%.4f",
                    batch_idx + 1, len(train_loader),
                    loss.item(),
                )

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validate
        if dry_run:
            # Only run 1 val batch in dry-run mode
            val_metrics = _dry_run_val(model, val_loader, device, criterion)
        else:
            val_metrics = evaluate(model, val_loader, device, criterion)

        val_auc  = val_metrics.mean_auc
        val_loss = val_metrics.loss

        # LR scheduler step
        scheduler.step(val_auc)
        current_lr = optimiser.param_groups[0]["lr"]

        epoch_time = time.perf_counter() - epoch_t
        logger.info(
            "Epoch %d complete — train_loss=%.4f  val_loss=%.4f  "
            "val_auc=%.4f  lr=%.2e  time=%.0fs",
            epoch, avg_train_loss, val_loss, val_auc,
            current_lr, epoch_time,
        )

        # Record history
        epoch_record = {
            "epoch":       epoch,
            "train_loss":  round(avg_train_loss, 4),
            "val_loss":    round(val_loss,        4),
            "val_auc":     round(val_auc,         4),
            "learning_rate": current_lr,
            "time_s":      round(epoch_time, 1),
            "auc_per_class": {
                k: round(v, 4)
                for k, v in val_metrics.auc_per_class.items()
            },
        }
        history.append(epoch_record)

        # Save checkpoint every epoch
        if cfg.save_every_epoch:
            ckpt_path = Paths.checkpoints / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(
                model, optimiser, scheduler,
                epoch, val_auc, best_auc, history,
                ckpt_path,
            )
            _prune_old_checkpoints(cfg.keep_last_n)

        # Save best model
        if val_auc > best_auc:
            best_auc    = val_auc
            no_improve  = 0
            save_checkpoint(
                model, optimiser, scheduler,
                epoch, val_auc, best_auc, history,
                Paths.best_model,
            )
            logger.info("  ★ New best model (val_auc=%.4f)", best_auc)
        else:
            no_improve += 1
            logger.info(
                "  No improvement for %d epoch(s) (best=%.4f)",
                no_improve, best_auc,
            )

        # Early stopping
        if no_improve >= cfg.early_stop_patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement",
                no_improve,
            )
            break

        if dry_run:
            logger.info("Dry run complete — pipeline verified")
            break

    total_time = time.perf_counter() - total_t
    logger.info("=" * 55)
    logger.info(
        "Training complete — best_val_auc=%.4f  total_time=%.0fs",
        best_auc, total_time,
    )

    # Save training history to JSON
    history_path = Paths.processed / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "best_auc": best_auc,
        "epochs_trained": len(history),
        "total_time_s": round(total_time, 1),
        "history": history,
    }, indent=2))
    logger.info("Training history saved to %s", history_path)

    return {
        "best_auc":      best_auc,
        "epochs_trained": len(history),
        "history":        history,
    }


def _dry_run_val(model, val_loader, device, criterion):
    """Run validation on a single batch — for dry-run mode only."""
    from src.models.metrics import EvalMetrics, compute_metrics
    import numpy as np

    model.eval()
    with torch.no_grad():
        images, labels = next(iter(val_loader))
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        probs  = torch.sigmoid(logits).cpu().numpy()
        labs   = labels.cpu().numpy()

    # compute_metrics needs at least some positives — fake it for dry-run
    if labs.sum() == 0:
        return EvalMetrics(loss=loss.item(), mean_auc=0.0)

    return compute_metrics(probs, labs, loss.item())


# ── CLI ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the chest X-ray classifier")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from the latest checkpoint")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Explicit checkpoint path to resume from")
    p.add_argument("--dry-run",   action="store_true",
                   help="Run 1 batch per split — pipeline smoke test")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    set_log_level(args.log_level)
    train(
        resume          = args.resume,
        checkpoint_path = args.checkpoint,
        dry_run         = args.dry_run,
    )
