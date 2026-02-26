"""Base environment configurations for locomotion tasks.

Inheritance Hierarchy:
    LocomotionEnvCfg (base_env_cfg.py)
        - Defines default scene, actions, observations, rewards, events, terminations, curriculum
        - Robot-specific configs in robots/ inherit from this
"""

from .base_env_cfg import (
    LocomotionEnvCfg,
    BaseSceneCfg,
    BaseCommandsCfg,
    BaseActionsCfg,
    BaseObservationsCfg,
    BaseEventCfg,
    BaseRewardsCfg,
    BaseTerminationsCfg,
    BaseCurriculumCfg,
)

__all__ = [
    "LocomotionEnvCfg",
    "BaseSceneCfg",
    "BaseCommandsCfg",
    "BaseActionsCfg",
    "BaseObservationsCfg",
    "BaseEventCfg",
    "BaseRewardsCfg",
    "BaseTerminationsCfg",
    "BaseCurriculumCfg",
]
