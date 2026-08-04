"""
Logging configuration.

Author: Muhammad Ibrahim Gimba
"""

import logging
from pathlib import Path

from config.config import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "application.log"


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE)

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
