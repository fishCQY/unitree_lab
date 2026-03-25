"""Utility functions for unitree_lab.

This module provides utilities for:
- AMP motion data loading and processing
- ONNX export with IsaacLab metadata
- WandB experiment tracking
- Checkpoint management
- Experiment directory management
- UnitreeOnPolicyRunner (ONNX + wandb) and optional ManagedExperimentRunner

See docs/UTILS_OVERVIEW.md for detailed documentation.
"""

# AMP data loading
from .amp_data_loader import (
    AMPMotionData,
    create_mirror_config,
    load_amp_motion_data,
    load_conditional_amp_data,
)

# ONNX utilities
from .onnx_utils import (
    attach_onnx_metadata,
    build_onnx_metadata,
    build_obs_spec,
    export_onnx_with_metadata,
)

# WandB utilities
from .wandb_utils import (
    WandbConfig,
    WandbManager,
    WandbFileSaver,
    init_wandb,
    log_training_metrics,
)

# Checkpoint utilities
from .checkpoint_utils import (
    CheckpointInfo,
    CheckpointManager,
    save_for_deployment,
    compare_checkpoints,
    find_best_checkpoint,
)

# Experiment tracking
from .experiment_tracker import (
    ExperimentConfig,
    ExperimentTracker,
    create_experiment,
)

# Training runner (rsl_rl OnPolicyRunner extension)
from .unitree_on_policy_runner import UnitreeOnPolicyRunner

# Optional experiment helper (non-rsl_rl)
from .training_runner import (
    RunnerConfig,
    ManagedExperimentRunner,
    create_runner,
)

__all__ = [
    # AMP data loading
    "AMPMotionData",
    "load_amp_motion_data",
    "load_conditional_amp_data",
    "create_mirror_config",
    # ONNX utilities for sim2sim
    "attach_onnx_metadata",
    "build_onnx_metadata",
    "build_obs_spec",
    "export_onnx_with_metadata",
    # WandB utilities
    "WandbConfig",
    "WandbManager",
    "WandbFileSaver",
    "init_wandb",
    "log_training_metrics",
    # Checkpoint utilities
    "CheckpointInfo",
    "CheckpointManager",
    "save_for_deployment",
    "compare_checkpoints",
    "find_best_checkpoint",
    # Experiment tracking
    "ExperimentConfig",
    "ExperimentTracker",
    "create_experiment",
    # Training runner
    "RunnerConfig",
    "UnitreeOnPolicyRunner",
    "ManagedExperimentRunner",
    "create_runner",
]
