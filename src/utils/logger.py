"""
src/utils/logger.py
────────────────────────────────────────────────────────────────
Logging configuration for the Chest X-Ray Classifier.

Same pattern as P1/P2: every module gets its own named logger
via get_logger(__name__).  Root logger configured once at import.

Log format:
  2024-01-15 09:32:11 | INFO     | src.training.train  | Epoch 3/10 — AUC 0.823
  2024-01-15 09:32:14 | WARNING  | src.data.dataset    | 3 images not found, skipped
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


_LOG_LEVEL: str       = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FILE:  str | None = os.getenv("LOG_FILE")
_DATE_FORMAT           = "%Y-%m-%d %H:%M:%S"
_FORMAT                = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"


def _configure_root_logger() -> None:
    """Set up the root logger.  Safe to call multiple times."""
    root = logging.getLogger()
    if root.handlers:
        return

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if _LOG_FILE:
        log_path = Path(_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root on first call.

    Args:
        name: Typically ``__name__`` from the calling module.

    Returns:
        Standard :class:`logging.Logger` instance.

    Example::

        logger = get_logger(__name__)
        logger.info("Epoch %d/%d — mean AUC %.4f", epoch, total, auc)
    """
    _configure_root_logger()
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """Change the global log level at runtime.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    numeric = getattr(logging, level.upper(), None)
    if numeric is None:
        raise ValueError(f"Invalid log level: {level!r}")
    logging.getLogger().setLevel(numeric)
