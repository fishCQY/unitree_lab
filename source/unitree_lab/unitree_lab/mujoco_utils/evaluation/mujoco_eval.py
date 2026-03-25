# Copyright (c) 2024-2026, unitree_lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo evaluation interface for training integration.

Provides ``MuJoCoEval`` - called from the training runner on each checkpoint
save to run batch sim2sim evaluation and log metrics + videos to WandB.

Design:
  1. ``batch_eval()``       - run all eval tasks via ProcessPoolExecutor
  2. ``batch_eval_and_log()`` - batch_eval + wandb upload
  3. Two-phase flow:
       Phase 1: all tasks headless in parallel (fast, no rendering)
       Phase 2: selected tasks re-run with video recording (serial)

Usage from a runner::

    if env.unwrapped.mujoco_eval is not None:
        env.unwrapped.mujoco_eval.batch_eval_and_log(onnx_path, iteration)
"""

from __future__ import annotations

import os
from typing import Any

from ..logging import logger
from .batch_evaluator import BatchEvalConfig, BatchEvalResult, run_batch_eval


class MuJoCoEval:
    """MuJoCo sim2sim evaluation for training integration.

    Two construction styles:

    - **Path API:** ``MuJoCoEval(robot_xml_path, ...)`` for scripts.
    - **Config API:** ``MuJoCoEval(cfg, env)`` as used by ``UnitreeRLEnv``;
      ``cfg`` must expose ``robot_model_path``.
    """

    simulation_fn_path: str | None = None

    def __init__(
        self,
        cfg_or_path: str | Any,
        env: Any | None = None,
        eval_task_names: list[str] | None = None,
        num_worst_videos: int = 2,
        save_mixed_terrain_video: bool = True,
        num_workers: int = 16,
    ):
        if env is not None:
            cfg = cfg_or_path
            path = getattr(cfg, "robot_model_path", None)
            if path is None:
                path = getattr(cfg, "robot_xml_path", None)
            if path is None:
                raise ValueError("MuJoCo eval config must set robot_model_path")
            self.robot_model_path = str(path)
            self.eval_task_names = getattr(cfg, "eval_task_names", eval_task_names)
            self.num_worst_videos = int(getattr(cfg, "num_worst_videos", num_worst_videos))
            self.save_mixed_terrain_video = bool(
                getattr(cfg, "save_mixed_terrain_video", save_mixed_terrain_video)
            )
            self.num_workers = int(getattr(cfg, "num_workers", num_workers))
            self.simulation_fn_path = getattr(cfg, "simulation_fn_path", self.simulation_fn_path)
        else:
            if not isinstance(cfg_or_path, str):
                raise TypeError("First argument must be an XML path str, or pass (cfg, env).")
            self.robot_model_path = str(cfg_or_path)
            self.eval_task_names = eval_task_names
            self.num_worst_videos = num_worst_videos
            self.save_mixed_terrain_video = save_mixed_terrain_video
            self.num_workers = num_workers

    def batch_eval(
        self,
        onnx_path: str,
        task_names: list[str] | None = None,
        num_workers: int | None = None,
        save_torque_data: bool = False,
    ) -> BatchEvalResult:
        """Run batch evaluation on multiple tasks in parallel."""
        policy_path = os.path.dirname(onnx_path)
        need_videos = self.save_mixed_terrain_video or self.num_worst_videos > 0
        video_dir = os.path.join(policy_path, "eval_videos") if need_videos else None
        torque_data_dir = os.path.join(policy_path, "torque_data") if save_torque_data else None

        config = BatchEvalConfig(
            num_workers=num_workers or self.num_workers,
            task_names=task_names or self.eval_task_names,
            save_torque_data=save_torque_data,
            save_mixed_terrain_video=self.save_mixed_terrain_video,
            num_worst_videos=self.num_worst_videos,
        )

        result = run_batch_eval(
            onnx_path=onnx_path,
            robot_model_path=self.robot_model_path,
            config=config,
            video_dir=video_dir,
            torque_data_dir=torque_data_dir,
            simulation_fn_path=self.simulation_fn_path,
        )

        print(result.summary())
        return result

    def batch_eval_and_log(
        self,
        onnx_path: str,
        iteration: int | None = None,
        num_workers: int | None = None,
        save_torque_data: bool = False,
    ) -> BatchEvalResult:
        """Run batch evaluation and log metrics/videos to wandb."""
        result = self.batch_eval(
            onnx_path=onnx_path,
            num_workers=num_workers,
            save_torque_data=save_torque_data,
        )
        self._log_to_wandb(result, iteration)
        return result

    def _log_to_wandb(
        self,
        result: BatchEvalResult,
        iteration: int | None = None,
    ) -> None:
        """Log evaluation results to wandb."""
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        step = int(iteration) if iteration is not None else None
        iter_label = f"iter_{iteration}" if iteration is not None else ""

        wandb.log(result.to_wandb_dict(), step=step, commit=False)
        logger.info("Logged batch eval metrics to wandb")

        if not result.video_paths:
            return

        mixed_path = result.video_paths.get("mixed_terrain")
        worst_videos = [(k, v) for k, v in result.video_paths.items() if k != "mixed_terrain"]

        if mixed_path and os.path.exists(mixed_path):
            caption = iter_label if iter_label else "mixed_terrain"
            wandb.log(
                {"sim2sim_video": wandb.Video(mixed_path, format="mp4", caption=caption)},
                step=step,
                commit=False,
            )
            logger.info(f"Uploaded sim2sim_video ({caption})")

        for i, (task_name, video_path) in enumerate(worst_videos):
            if os.path.exists(video_path):
                key = f"sim2sim_video_worst_{i + 1}"
                caption = f"{iter_label} {task_name}" if iter_label else task_name
                wandb.log(
                    {key: wandb.Video(video_path, format="mp4", caption=caption)},
                    step=step,
                    commit=False,
                )
                logger.info(f"Uploaded {key} ({caption})")
