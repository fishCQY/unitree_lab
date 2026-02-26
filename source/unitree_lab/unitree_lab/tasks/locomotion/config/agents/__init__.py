"""Agent configurations for locomotion tasks."""

from .rsl_rl_ppo_cfg import (
    AMPCfg,
    RslRlGuidanceCfg,
    RslRlPpoAlgorithmCfg,
    UnitreeG1RoughPPORunnerCfg,
    UnitreeG1FlatPPORunnerCfg,
    UnitreeG1RoughPPORunnerGRUCfg,
    UnitreeG1RoughDepthPPORunnerCfg,
)

__all__ = [
    "AMPCfg",
    "RslRlGuidanceCfg",
    "RslRlPpoAlgorithmCfg",
    "UnitreeG1RoughPPORunnerCfg",
    "UnitreeG1FlatPPORunnerCfg",
    "UnitreeG1RoughPPORunnerGRUCfg",
    "UnitreeG1RoughDepthPPORunnerCfg",
]
