"""
logger
======

File-based logging utilities for long-running experiments.
"""

__version__ = "1.0.0"

import logging
from pathlib import Path

__all__ = ["setup_logger"]


def setup_logger(
    model_name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures a file-based logger, avoiding duplicate handlers.

    Args:
        model_name (str): Name of the model or component (used for the filename).
        log_dir (str, optional): Directory for the log file. Defaults to "logs".
        level (int, optional): Logging level. Defaults to ``logging.INFO``.

    Returns:
        logging.Logger: A configured logger instance.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(model_name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path / f"{model_name}.log", mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger