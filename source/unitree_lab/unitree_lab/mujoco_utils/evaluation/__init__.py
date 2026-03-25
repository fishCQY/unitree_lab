"""Evaluation module for MuJoCo sim2sim."""

from .eval_task import (
    EVAL_TASKS,
    EVAL_TASKS_BABY,
    EVAL_TASKS_DEFAULT,
    EVAL_TASKS_FULL,
    TERRAIN_FLAT,
    TERRAIN_MIXED,
    TERRAIN_ROUGH,
    TERRAIN_ROUGH_HARD,
    TERRAIN_STAIRS_UP_HARD,
    TERRAIN_STAIRS_DOWN_HARD,
    EvalTask,
    get_eval_task,
    list_eval_tasks,
    vel_cmd_forward_slow,
    vel_cmd_forward_medium,
    vel_cmd_omnidirectional,
    vel_cmd_terrain_comprehensive,
    vel_cmd_zero,
)
from .batch_evaluator import BatchEvalConfig, BatchEvalResult, run_batch_eval
from .metrics import (
    EvalResult,
    MetricsCollector,
    MetricsConfig,
    is_fallen,
    compute_locomotion_metrics,
    LocomotionMetrics,
)
from .mujoco_eval import MuJoCoEval
from .mujoco_eval_cfg import BaseMuJoCoEvalCfg

__all__ = [
    # Eval tasks
    "EvalTask",
    "EVAL_TASKS",
    "EVAL_TASKS_DEFAULT",
    "EVAL_TASKS_FULL",
    "EVAL_TASKS_BABY",
    "get_eval_task",
    "list_eval_tasks",
    # Terrain presets
    "TERRAIN_FLAT",
    "TERRAIN_MIXED",
    "TERRAIN_ROUGH",
    "TERRAIN_ROUGH_HARD",
    "TERRAIN_STAIRS_UP_HARD",
    "TERRAIN_STAIRS_DOWN_HARD",
    # Velocity commands
    "vel_cmd_zero",
    "vel_cmd_forward_slow",
    "vel_cmd_forward_medium",
    "vel_cmd_omnidirectional",
    "vel_cmd_terrain_comprehensive",
    # Batch evaluation
    "BatchEvalConfig",
    "BatchEvalResult",
    "run_batch_eval",
    # Metrics
    "EvalResult",
    "MetricsCollector",
    "MetricsConfig",
    "is_fallen",
    "LocomotionMetrics",
    "compute_locomotion_metrics",
    # MuJoCo eval
    "MuJoCoEval",
    "BaseMuJoCoEvalCfg",
]
