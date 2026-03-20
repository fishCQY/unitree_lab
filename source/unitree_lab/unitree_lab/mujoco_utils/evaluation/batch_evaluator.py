"""Batch evaluator for sim2sim.

This module provides:
1. Parallel evaluation of multiple tasks
2. Video recording for visualization
3. Metrics aggregation and reporting
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .eval_task import EvalTask, LocomotionEvalTask, get_eval_task, list_eval_tasks
from .metrics import LocomotionMetrics, compute_locomotion_metrics, print_metrics


@dataclass
class EvalResult:
    """Result from a single evaluation."""
    task_name: str
    metrics: LocomotionMetrics
    episode_data: list[dict]
    video_path: str | None = None


class BatchEvaluator:
    """Batch evaluator for sim2sim policies.
    
    Runs evaluation across multiple tasks and collects metrics.
    """
    
    def __init__(
        self,
        simulator_class: type,
        xml_path: str | Path,
        onnx_path: str | Path,
        output_dir: str | Path = "eval_results",
        **simulator_kwargs,
    ):
        """Initialize batch evaluator.
        
        Args:
            simulator_class: Simulator class (subclass of BaseMujocoSimulator)
            xml_path: Path to MuJoCo XML
            onnx_path: Path to ONNX policy
            output_dir: Output directory for results
            **simulator_kwargs: Additional kwargs for simulator
        """
        self.simulator_class = simulator_class
        self.xml_path = Path(xml_path)
        self.onnx_path = Path(onnx_path)
        self.output_dir = Path(output_dir)
        self.simulator_kwargs = simulator_kwargs
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: list[EvalResult] = []
    
    def evaluate_task(
        self,
        task: LocomotionEvalTask | str,
        num_episodes: int | None = None,
        render: bool = False,
        save_video: bool = False,
        video_steps: int = 300,
    ) -> EvalResult:
        """Evaluate policy on a single task.
        
        Args:
            task: Evaluation task or task name
            num_episodes: Override number of episodes
            render: Whether to render during evaluation
            save_video: Whether to save video
            
        Returns:
            Evaluation result
        """
        if isinstance(task, str):
            task = get_eval_task(task)
        
        if num_episodes is None:
            num_episodes = task.num_episodes
        
        print(f"\n[Eval] Running task: {task.name}")
        print(f"       Description: {task.description}")
        print(f"       Episodes: {num_episodes}")
        
        # Create simulator with terrain
        simulator = self._create_simulator(task)
        
        # Run episodes
        episode_data = []
        for ep_idx in range(num_episodes):
            print(f"  Episode {ep_idx + 1}/{num_episodes}", end="\r")
            
            # Set velocity command
            if task.velocity_command_range and task.resample_command_interval:
                # Random command
                vx = np.random.uniform(*task.velocity_command_range[0])
                vy = np.random.uniform(*task.velocity_command_range[1])
                wz = np.random.uniform(*task.velocity_command_range[2])
                velocity_cmd = (vx, vy, wz)
            else:
                velocity_cmd = task.velocity_command
            
            # Run episode
            result = simulator.run_episode(
                max_steps=task.max_episode_steps,
                render=render,
                velocity_command=velocity_cmd,
            )
            
            episode_data.append(result)
        
        print()  # Clear line
        
        # Compute metrics
        metrics = compute_locomotion_metrics(
            episode_data,
            np.array(task.velocity_command),
            simulator.policy_dt,
        )
        
        # Print metrics
        print_metrics(metrics, task.name)
        
        # Video recording (if requested)
        video_path = None
        if save_video:
            video_path = self._record_video(simulator, task, duration_steps=int(video_steps))
        
        return EvalResult(
            task_name=task.name,
            metrics=metrics,
            episode_data=episode_data,
            video_path=video_path,
        )
    
    def evaluate_all(
        self,
        task_names: list[str] | None = None,
        num_episodes_per_task: int = 10,
        save_videos: bool = False,
        video_steps: int = 300,
    ) -> dict[str, EvalResult]:
        """Evaluate policy on multiple tasks.
        
        Args:
            task_names: List of task names (None = all tasks)
            num_episodes_per_task: Episodes per task
            save_videos: Whether to save videos
            
        Returns:
            Dictionary of task_name -> EvalResult
        """
        if task_names is None:
            task_names = list_eval_tasks()
        
        print(f"\n{'='*60}")
        print(f"Batch Evaluation: {len(task_names)} tasks")
        print(f"{'='*60}")
        
        results = {}
        for task_name in task_names:
            result = self.evaluate_task(
                task_name,
                num_episodes=num_episodes_per_task,
                save_video=save_videos,
                video_steps=int(video_steps),
            )
            results[task_name] = result
            self.results.append(result)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _create_simulator(self, task: LocomotionEvalTask) -> Any:
        """Create simulator for task.
        
        Override in subclass to add terrain setup, etc.
        """
        return self.simulator_class(
            xml_path=self.xml_path,
            onnx_path=self.onnx_path,
            **self.simulator_kwargs,
        )
    
    def _record_video(
        self,
        simulator: Any,
        task: LocomotionEvalTask,
        duration_steps: int = 300,
    ) -> str:
        """Record evaluation video.
        
        Args:
            simulator: Simulator instance
            task: Evaluation task
            duration_steps: Number of steps to record
            
        Returns:
            Path to saved video
        """
        try:
            import mujoco
            import cv2
        except ImportError:
            print("[Warning] cv2 not available for video recording")
            return None
        
        # Setup renderer
        width, height = 640, 480
        renderer = mujoco.Renderer(simulator.model, height, width)
        
        # Reset
        simulator.reset()
        simulator.set_velocity_command(*task.velocity_command)
        
        # Collect frames
        frames = []
        for step in range(duration_steps):
            # Render
            renderer.update_scene(simulator.data)
            frame = renderer.render()
            frames.append(frame)
            
            # Step
            simulator.step()
        
        # Save video
        video_path = self.output_dir / f"{task.name}_eval.mp4"
        
        # OpenCV often writes mp4v (MPEG-4 Part 2), which some browsers can't play in W&B.
        # We write with mp4v for compatibility then transcode to H.264 (best-effort) if ffmpeg exists.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        policy_dt = getattr(simulator, "policy_dt", 0.02)
        fps = max(1, int(round(1.0 / policy_dt)))
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        # Best-effort: transcode to H.264 for browser/W&B playback
        try:
            import subprocess

            tmp = video_path.with_suffix(".h264_tmp.mp4")
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(tmp),
            ]
            subprocess.run(cmd, check=True)
            tmp.replace(video_path)
        except Exception:
            try:
                tmp.unlink()  # type: ignore[name-defined]
            except Exception:
                pass
        print(f"  Video saved: {video_path}")
        
        return str(video_path)
    
    def _print_summary(self, results: dict[str, EvalResult]) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\n{'Task':<20} {'Survival':<12} {'Vel Error':<12} {'Distance':<12}")
        print("-" * 60)
        
        for name, result in results.items():
            m = result.metrics
            print(f"{name:<20} {m.survival_rate:>10.1%} {m.mean_velocity_error:>10.3f} {m.mean_forward_distance:>10.2f}m")
        
        # Overall
        all_metrics = [r.metrics for r in results.values()]
        mean_survival = np.mean([m.survival_rate for m in all_metrics])
        mean_vel_error = np.mean([m.mean_velocity_error for m in all_metrics])
        total_distance = sum(m.total_distance for m in all_metrics)
        
        print("-" * 60)
        print(f"{'OVERALL':<20} {mean_survival:>10.1%} {mean_vel_error:>10.3f} {total_distance:>10.2f}m")
        print("=" * 60)
    
    def save_results(self, filename: str = "eval_results.npz") -> str:
        """Save evaluation results to file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        # Prepare data
        data = {}
        for result in self.results:
            prefix = result.task_name
            data[f"{prefix}_survival_rate"] = result.metrics.survival_rate
            data[f"{prefix}_mean_velocity_error"] = result.metrics.mean_velocity_error
            data[f"{prefix}_mean_distance"] = result.metrics.mean_forward_distance
            data[f"{prefix}_episode_lengths"] = result.metrics.episode_lengths
        
        np.savez(output_path, **data)
        print(f"Results saved: {output_path}")
        
        return str(output_path)
