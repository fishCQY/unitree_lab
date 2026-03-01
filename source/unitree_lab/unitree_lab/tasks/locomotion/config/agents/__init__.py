"""Agent configurations for locomotion tasks."""

from .rsl_rl_ppo_cfg import (
    ActorCriticDepthCfg,
    AMPCfg,
    RslRlPpoAlgorithmCfg,
    UnitreeG1RoughPPORunnerCfg,
    UnitreeG1FlatPPORunnerCfg,
    UnitreeG1RoughPPORunnerGRUCfg,
    UnitreeG1RoughDepthPPORunnerCfg,
)

__all__ = [
    "ActorCriticDepthCfg",
    "AMPCfg",
    "RslRlPpoAlgorithmCfg",
    "UnitreeG1RoughPPORunnerCfg",
    "UnitreeG1FlatPPORunnerCfg",
    "UnitreeG1RoughPPORunnerGRUCfg",
    "UnitreeG1RoughDepthPPORunnerCfg",
]
