"""Experiment logging helpers (file + stderr)."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_run_logging(
    run_dir: str | Path,
    *,
    level: int = logging.INFO,
    logger_name: str = "magic_vlm",
) -> logging.Logger:
    """Attach a run-directory file handler; safe to call once per process/run."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    log_path = run_path / "run.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Replace prior magic_vlm handlers so repeated inits in tests do not duplicate lines.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    logger.info("Logging initialized at %s", log_path)
    return logger
