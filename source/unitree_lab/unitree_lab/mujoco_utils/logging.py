"""Logging configuration for MuJoCo utilities."""

import logging
import sys

logger = logging.getLogger("mujoco_utils")

if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def set_log_level(level: str | int) -> None:
    """Set logging level for mujoco_utils module."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    for handler in logger.handlers:
        handler.setLevel(level)
