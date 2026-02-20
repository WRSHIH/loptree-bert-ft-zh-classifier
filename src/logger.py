"""
src/logger.py
─────────────────────────────────────────────────────────────────
Centralised logging configuration for the entire project.

Usage
-----
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Build and return a named Logger with a consistent format.

    Parameters
    ----------
    name : str
        Logger name, conventionally ``__name__`` of the calling module.
    log_file : Optional[str]
        Optional path to a file where log records should also be written.
        The parent directory is created automatically if it does not exist.
    level : int
        Root log level (default: ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.  Multiple calls with the same ``name``
        return the *same* logger object (Python's logging registry).
    """
    logger = logging.getLogger(name)

    # Guard: do not add duplicate handlers if the logger already has some
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Formatter ──────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Stream handler (stdout) ─────────────────────────────────────────────
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    # ── File handler (optional) ─────────────────────────────────────────────
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    # Prevent log records from propagating to the root logger to avoid
    # duplicate output when other libraries configure the root logger.
    logger.propagate = False

    return logger
