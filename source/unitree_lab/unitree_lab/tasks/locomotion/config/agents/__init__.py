"""Agent configurations for locomotion tasks."""

from .rsl_rl_ppo_cfg import (
    ActorCriticDepthCfg,
    AMPPluginCfg,
    UnitreeG1RoughPluginRunnerCfg,
    UnitreeG1FlatPluginRunnerCfg,
    UnitreeG1RoughMLPRunnerCfg,
)

__all__ = [
    "ActorCriticDepthCfg",
    "AMPPluginCfg",
    "UnitreeG1RoughPluginRunnerCfg",
    "UnitreeG1FlatPluginRunnerCfg",
    "UnitreeG1RoughMLPRunnerCfg",
]
