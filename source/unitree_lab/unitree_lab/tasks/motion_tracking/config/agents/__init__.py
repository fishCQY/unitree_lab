"""Agent configurations for motion tracking tasks."""

from .rsl_rl_ppo_cfg import (
    UnitreeG1TrackingPPORunnerCfg,
    UnitreeG1TrackingPRPPORunnerCfg,
    UnitreeG1TrackingABPPORunnerCfg,
    UnitreeG1TrackingAEPPORunnerCfg,
    UnitreeG1TrackingFSQPPORunnerCfg,
    UnitreeG1TrackingRFSQPPORunnerCfg,
    UnitreeG1TrackingVQVAEPPORunnerCfg,
    UnitreeG1TrackingDistillationFSQRunnerCfg,
    UnitreeG1TrackingDistillationVQVAERunnerCfg,
)

__all__ = [
    "UnitreeG1TrackingPPORunnerCfg",
    "UnitreeG1TrackingPRPPORunnerCfg",
    "UnitreeG1TrackingABPPORunnerCfg",
    "UnitreeG1TrackingAEPPORunnerCfg",
    "UnitreeG1TrackingFSQPPORunnerCfg",
    "UnitreeG1TrackingRFSQPPORunnerCfg",
    "UnitreeG1TrackingVQVAEPPORunnerCfg",
    "UnitreeG1TrackingDistillationFSQRunnerCfg",
    "UnitreeG1TrackingDistillationVQVAERunnerCfg",
]
