# Copyright (c) 2024-2026, unitree_lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""G1-specific MuJoCo evaluation configuration."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

from unitree_lab.mujoco_utils.evaluation.mujoco_eval_cfg import BaseMuJoCoEvalCfg


def _get_configclass():
    try:
        from isaaclab.utils import configclass
        return configclass
    except ImportError:
        from dataclasses import dataclass
        return dataclass


@_get_configclass()
class G1LocomotionEvalCfg(BaseMuJoCoEvalCfg):
    """MuJoCo evaluation configuration for G1 locomotion.

    Example usage in env cfg::

        @configclass
        class G1RoughEnvCfg(UnitreeRLEnvCfg):
            mujoco_eval = G1LocomotionEvalCfg(
                robot_model_path="/path/to/g1_description/xml/g1.xml",
            )
    """

    robot_model_path: str = MISSING

    eval_task_names: list[str] | None = None

    num_worst_videos: int = 2

    save_mixed_terrain_video: bool = True

    num_workers: int = 16

    simulation_fn_path: str = (
        "unitree_lab.tasks.locomotion.mujoco_eval.simulator.run_locomotion_simulation"
    )

    def get_class(self):
        from unitree_lab.mujoco_utils.evaluation.mujoco_eval import MuJoCoEval
        return MuJoCoEval
