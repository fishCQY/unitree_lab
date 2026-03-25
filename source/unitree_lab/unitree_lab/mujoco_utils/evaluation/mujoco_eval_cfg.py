"""Base configuration for MuJoCo evaluation."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _get_configclass():
    """Get configclass decorator, with fallback for lightweight mode."""
    try:
        from isaaclab.utils import configclass
        return configclass
    except ImportError:
        from dataclasses import dataclass
        return dataclass


@_get_configclass()
class BaseMuJoCoEvalCfg:
    """Base configuration for MuJoCo evaluation.

    This class should be subclassed by task-specific configurations.
    The subclass should override `get_class()` to return the appropriate
    MuJoCoEval class.
    """

    robot_model_path: str = MISSING
    """Path to the MuJoCo model file."""

    def get_class(self):
        """Return the MuJoCoEval class for this configuration."""
        from .mujoco_eval import MuJoCoEval
        return MuJoCoEval
