"""MuJoCo evaluation interface for training integration.

Provides ``MuJoCoEval`` — called from the training runner on each checkpoint
save to run batch sim2sim evaluation and log metrics + videos to Wandb.

Design follows bfm_training's ``BaseMuJoCoEval``:
  1. ``batch_eval()``       – run all eval tasks, return metrics
  2. ``batch_eval_and_log()`` – batch_eval + wandb upload
  3. Two-phase flow:
       Phase 1: all tasks headless in parallel (fast, no rendering)
       Phase 2: selected tasks re-run with video recording (serial)

Usage from a runner::

    if env.unwrapped.mujoco_eval is not None:
        env.unwrapped.mujoco_eval.batch_eval_and_log(onnx_path, iteration)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .eval_task import get_eval_task, list_eval_tasks
from .metrics import LocomotionMetrics, compute_locomotion_metrics

logger = logging.getLogger(__name__)


@dataclass
class BatchEvalResult:
    """Result of a batch evaluation across multiple tasks."""

    task_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_metrics: dict[str, LocomotionMetrics] = field(default_factory=dict)
    video_paths: dict[str, str] = field(default_factory=dict)

    def to_wandb_dict(self, prefix: str = "sim2sim_eval") -> dict:
        """Convert results to wandb logging format."""
        log_dict: dict[str, float] = {}
        for task_name, metrics in self.task_metrics.items():
            tp = f"{prefix}/{task_name}"
            log_dict[f"{tp}/survival_rate"] = metrics.survival_rate
            log_dict[f"{tp}/mean_velocity_error"] = metrics.mean_velocity_error
            log_dict[f"{tp}/mean_forward_distance"] = metrics.mean_forward_distance
            log_dict[f"{tp}/velocity_error_x"] = metrics.velocity_error_x
            log_dict[f"{tp}/velocity_error_y"] = metrics.velocity_error_y
        return log_dict

    def summary(self) -> str:
        lines = ["=" * 70, "BATCH EVALUATION RESULTS", "=" * 70]
        for name in sorted(self.task_metrics):
            m = self.task_metrics[name]
            sr = f"{m.survival_rate:.0%}"
            ve = f"{m.mean_velocity_error:.3f}"
            lines.append(f"{name:30s} | surv: {sr:>4s} | vel_err: {ve:>6s} m/s")
        lines.append("=" * 70)
        return "\n".join(lines)


class MuJoCoEval:
    """MuJoCo sim2sim evaluation for training integration.

    Args:
        robot_xml_path: Path to the MuJoCo robot XML.
        eval_task_names: Which tasks to evaluate (None = all).
        num_worst_videos: Number of worst-performing tasks to record video for.
        save_mixed_terrain_video: Whether to always record mixed_terrain video.
    """

    def __init__(
        self,
        robot_xml_path: str,
        eval_task_names: list[str] | None = None,
        num_worst_videos: int = 2,
        save_mixed_terrain_video: bool = True,
    ):
        self.robot_xml_path = str(robot_xml_path)
        self.eval_task_names = eval_task_names
        self.num_worst_videos = num_worst_videos
        self.save_mixed_terrain_video = save_mixed_terrain_video

    def batch_eval(
        self,
        onnx_path: str,
        task_names: list[str] | None = None,
    ) -> BatchEvalResult:
        """Run batch evaluation on multiple tasks.

        Phase 1: all tasks headless, collecting metrics.
        """
        from ..simulation.base_simulator import BaseMujocoSimulator

        names = task_names or self.eval_task_names or list_eval_tasks()
        result = BatchEvalResult()

        for task_name in names:
            try:
                task = get_eval_task(task_name)
                simulator = BaseMujocoSimulator(
                    xml_path=self.robot_xml_path,
                    onnx_path=onnx_path,
                )
                episodes = []
                for _ in range(task.num_episodes):
                    ep = simulator.run_episode(
                        max_steps=task.max_episode_steps,
                        render=False,
                        velocity_command=task.velocity_command,
                    )
                    episodes.append(ep)

                metrics = compute_locomotion_metrics(
                    episodes, np.array(task.velocity_command), simulator.policy_dt,
                )
                result.task_results[task_name] = {"episodes": episodes}
                result.task_metrics[task_name] = metrics
                logger.info(
                    f"[eval] {task_name}: surv={metrics.survival_rate:.0%} "
                    f"vel_err={metrics.mean_velocity_error:.3f}"
                )
            except Exception as e:
                logger.warning(f"[eval] {task_name} failed: {e}")

        return result

    def _determine_video_tasks(self, result: BatchEvalResult) -> list[str]:
        """Select which tasks should have videos recorded."""
        keep: list[str] = []

        if self.save_mixed_terrain_video and "mixed_terrain" in result.task_metrics:
            keep.append("mixed_terrain")

        if self.num_worst_videos > 0:
            sorted_tasks = sorted(
                result.task_metrics.items(),
                key=lambda kv: (kv[1].survival_rate, -kv[1].mean_velocity_error),
            )
            for name, _ in sorted_tasks[: self.num_worst_videos]:
                if name not in keep:
                    keep.append(name)

        return keep

    def _record_task_video(
        self,
        onnx_path: str,
        task_name: str,
        video_dir: str,
        video_steps: int = 500,
    ) -> str | None:
        """Record a single task video using headless rendering."""
        try:
            import mujoco
        except ImportError:
            return None

        from ..simulation.base_simulator import BaseMujocoSimulator

        task = get_eval_task(task_name)
        simulator = BaseMujocoSimulator(
            xml_path=self.robot_xml_path,
            onnx_path=onnx_path,
        )
        simulator.set_velocity_command(*task.velocity_command)
        simulator.reset()

        width, height = 1280, 720
        if simulator.model.vis.global_.offwidth < width:
            simulator.model.vis.global_.offwidth = width
        if simulator.model.vis.global_.offheight < height:
            simulator.model.vis.global_.offheight = height

        renderer = mujoco.Renderer(simulator.model, height, width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        base_body_id = getattr(simulator, "_base_body_id", None)
        cam.trackbodyid = base_body_id if (base_body_id is not None and base_body_id > 0) else 1
        cam.distance = 3.0
        cam.azimuth = -150
        cam.elevation = -20
        cam.lookat[:] = [0, 0, 0.8]

        frames: list[np.ndarray] = []
        for _ in range(video_steps):
            renderer.update_scene(simulator.data, camera=cam)
            frames.append(renderer.render())
            simulator.step()
            if simulator._check_termination():
                break

        if not frames:
            return None

        os.makedirs(video_dir, exist_ok=True)
        video_file = os.path.join(video_dir, f"{task_name}.mp4")
        fps = max(1, int(round(1.0 / simulator.policy_dt)))

        try:
            import subprocess
            proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", str(fps),
                    "-i", "pipe:0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-preset", "veryfast", "-crf", "23",
                    video_file,
                ],
                stdin=subprocess.PIPE,
            )
            for frame in frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            proc.wait()
            if proc.returncode == 0 and os.path.exists(video_file):
                return video_file
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"ffmpeg video recording failed: {e}")

        # Fallback: cv2
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            if os.path.exists(video_file):
                return video_file
        except Exception as e:
            logger.warning(f"cv2 video recording failed: {e}")

        return None

    def batch_eval_and_log(
        self,
        onnx_path: str,
        iteration: int | None = None,
        num_workers: int = 16,
    ) -> BatchEvalResult:
        """Run batch evaluation and log metrics + videos to wandb.

        Two-phase flow:
          Phase 1: all tasks headless → metrics
          Phase 2: selected tasks → video recording → wandb upload
        """
        result = self.batch_eval(onnx_path)
        print(result.summary())

        # Phase 2: record videos for selected tasks
        video_tasks = self._determine_video_tasks(result)
        if video_tasks:
            policy_dir = os.path.dirname(onnx_path)
            video_dir = os.path.join(policy_dir, "eval_videos")
            for task_name in video_tasks:
                vpath = self._record_task_video(onnx_path, task_name, video_dir)
                if vpath:
                    result.video_paths[task_name] = vpath
                    logger.info(f"[eval] Recorded video: {task_name} -> {vpath}")

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
