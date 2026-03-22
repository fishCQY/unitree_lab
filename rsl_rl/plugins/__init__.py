"""Standalone plugins that can be attached to any RL runner."""

from .amp import AMPPlugin
from .symmetry import SymmetryClassifier

__all__ = ["AMPPlugin", "SymmetryClassifier"]
