# Copyright (c) 2024-2026, unitree_lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation metrics for MuJoCo sim2sim evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MetricsConfig:
    """Configuration for metrics computation."""

    fall_orientation_threshold: float = 1.0  # rad (~57 degrees)
    fall_base_contact_threshold: float = 1.0  # N
    warmup_duration: float = 2.0


def is_fallen(
    projected_gravity: np.ndarray,
    base_contact_force: float | None = None,
    orientation_threshold: float = 1.0,
    contact_threshold: float = 1.0,
) -> bool:
    """Check if robot has fallen using IsaacLab-compatible criteria."""
    proj_grav_z = np.clip(projected_gravity[2], -1.0, 1.0)
    if np.arccos(-proj_grav_z) > orientation_threshold:
        return True
    if base_contact_force is not None and base_contact_force > contact_threshold:
        return True
    return False


@dataclass
class EvalResult:
    """Result of a single evaluation task."""

    task_name: str
    duration: float = 0.0
    actual_duration: float = 0.0
    survival_rate: float | None = None
    linear_velocity_error: float | None = None
    angular_velocity_error: float | None = None
    avg_torque_util: float | None = None
    error: str | None = None

    @staticmethod
    def from_error(task_name: str, error: str) -> EvalResult:
        return EvalResult(task_name=task_name, error=error)

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "duration": self.duration,
            "actual_duration": self.actual_duration,
            "survival_rate": self.survival_rate,
            "linear_velocity_error": self.linear_velocity_error,
            "angular_velocity_error": self.angular_velocity_error,
            "avg_torque_util": self.avg_torque_util,
            "error": self.error,
        }

    def summary(self) -> str:
        if self.error:
            return f"{self.task_name}: ERROR - {self.error}"
        sr = f"{self.survival_rate:.0%}" if self.survival_rate is not None else "N/A"
        lve = f"{self.linear_velocity_error:.3f}" if self.linear_velocity_error is not None else "N/A"
        ave = f"{self.angular_velocity_error:.3f}" if self.angular_velocity_error is not None else "N/A"
        return f"{self.task_name}: survival={sr} lin_vel_err={lve}m/s ang_vel_err={ave}rad/s"


class MetricsCollector:
    """Streaming metrics collector for sim2sim evaluation.

    Example:
        >>> collector = MetricsCollector("flat_forward", duration=30.0)
        >>> for t in sim_loop:
        ...     collector.step(t, projected_gravity, cmd_vel, actual_vel, torque, torque_util)
        >>> result = collector.compute()
    """

    def __init__(self, task_name: str, duration: float, config: MetricsConfig | None = None):
        self.task_name = task_name
        self.duration = duration
        self.config = config or MetricsConfig()
        self._is_fallen = False
        self._fall_time: float | None = None
        self._timestamps: list[float] = []
        self._cmd_velocities: list[np.ndarray] = []
        self._actual_velocities: list[np.ndarray] = []
        self._torques: list[np.ndarray] = []
        self._torque_utils: list[np.ndarray] = []

    def step(
        self,
        t: float,
        projected_gravity: np.ndarray,
        cmd_velocity: np.ndarray,
        actual_velocity: np.ndarray,
        torque: np.ndarray,
        torque_util: np.ndarray,
        base_contact_force: float | None = None,
    ) -> None:
        self._timestamps.append(t)
        self._cmd_velocities.append(cmd_velocity.copy())
        self._actual_velocities.append(actual_velocity.copy())
        self._torques.append(torque.copy())
        self._torque_utils.append(torque_util.copy())
        if not self._is_fallen:
            if is_fallen(
                projected_gravity,
                base_contact_force,
                self.config.fall_orientation_threshold,
                self.config.fall_base_contact_threshold,
            ):
                self._is_fallen = True
                self._fall_time = t

    def compute(self) -> EvalResult:
        if len(self._timestamps) == 0:
            return EvalResult(task_name=self.task_name, duration=self.duration)
        actual_duration = self._timestamps[-1]
        survival_time = self._fall_time if self._fall_time else actual_duration
        lin_err, ang_err = self._compute_velocity_errors()
        avg_torque_util = 0.0
        if self._torque_utils:
            avg_torque_util = float(np.mean(np.array(self._torque_utils)))
        return EvalResult(
            task_name=self.task_name,
            duration=self.duration,
            actual_duration=actual_duration,
            survival_rate=survival_time / self.duration if self.duration > 0 else 1.0,
            linear_velocity_error=lin_err,
            angular_velocity_error=ang_err,
            avg_torque_util=avg_torque_util,
        )

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.task_name}_torque_data.npz"
        np.savez(
            output_file,
            timestamps=np.array(self._timestamps),
            torques=np.array(self._torques) if self._torques else np.array([]),
            torque_utils=np.array(self._torque_utils) if self._torque_utils else np.array([]),
            cmd_velocities=np.array(self._cmd_velocities) if self._cmd_velocities else np.array([]),
            actual_velocities=np.array(self._actual_velocities) if self._actual_velocities else np.array([]),
        )
        return output_file

    def _compute_velocity_errors(self) -> tuple[float, float]:
        if len(self._timestamps) == 0:
            return 0.0, 0.0
        timestamps = np.array(self._timestamps)
        mask = timestamps >= self.config.warmup_duration
        if not np.any(mask):
            return 0.0, 0.0
        cmd = np.array(self._cmd_velocities)[mask]
        actual = np.array(self._actual_velocities)[mask]
        lin_errors = np.sqrt((cmd[:, 0] - actual[:, 0]) ** 2 + (cmd[:, 1] - actual[:, 1]) ** 2)
        lin_vel_error = float(np.sqrt(np.mean(lin_errors**2)))
        ang_errors = np.abs(cmd[:, 2] - actual[:, 2])
        ang_vel_error = float(np.sqrt(np.mean(ang_errors**2)))
        return lin_vel_error, ang_vel_error


# === Legacy compatibility (used by existing MuJoCoEval) ===

@dataclass
class LocomotionMetrics:
    """Legacy metrics for locomotion evaluation."""

    survival_rate: float = 0.0
    mean_episode_length: float = 0.0
    mean_velocity_error: float = 0.0
    velocity_error_x: float = 0.0
    velocity_error_y: float = 0.0
    velocity_error_yaw: float = 0.0
    total_distance: float = 0.0
    mean_forward_distance: float = 0.0
    mean_energy: float = 0.0
    mean_torque_magnitude: float = 0.0
    mean_base_height: float = 0.0
    base_height_variance: float = 0.0
    mean_orientation_error: float = 0.0
    episode_lengths: list[int] = field(default_factory=list)
    episode_distances: list[float] = field(default_factory=list)
    episode_survivals: list[bool] = field(default_factory=list)


def compute_locomotion_metrics(
    episode_data: list[dict],
    velocity_command: np.ndarray,
    policy_dt: float = 0.02,
) -> LocomotionMetrics:
    """Compute locomotion metrics from episode data (legacy API)."""
    metrics = LocomotionMetrics()
    if not episode_data:
        return metrics
    for ep in episode_data:
        data = ep.get("data", {})
        stats = ep.get("stats", {})
        num_steps = stats.get("num_steps", 0)
        metrics.episode_lengths.append(num_steps)
        survived = stats.get("survived", False)
        metrics.episode_survivals.append(survived)
        distance = stats.get("distance_traveled", 0.0)
        metrics.episode_distances.append(distance)
        if "base_lin_vel" in data and len(data["base_lin_vel"]) > 0:
            vel = np.array(data["base_lin_vel"])
            metrics.velocity_error_x += np.mean(np.abs(vel[:, 0] - velocity_command[0]))
            metrics.velocity_error_y += np.mean(np.abs(vel[:, 1] - velocity_command[1]))
        if "base_pos" in data and len(data["base_pos"]) > 0:
            heights = np.array(data["base_pos"])[:, 2]
            metrics.mean_base_height += np.mean(heights)
            metrics.base_height_variance += np.var(heights)
    num_episodes = len(episode_data)
    if num_episodes > 0:
        metrics.survival_rate = sum(metrics.episode_survivals) / num_episodes
        metrics.mean_episode_length = np.mean(metrics.episode_lengths)
        metrics.total_distance = sum(metrics.episode_distances)
        metrics.mean_forward_distance = np.mean(metrics.episode_distances)
        metrics.velocity_error_x /= num_episodes
        metrics.velocity_error_y /= num_episodes
        metrics.mean_velocity_error = (metrics.velocity_error_x + metrics.velocity_error_y) / 2
        metrics.mean_base_height /= num_episodes
        metrics.base_height_variance /= num_episodes
    return metrics


def print_metrics(metrics: LocomotionMetrics, task_name: str = "") -> None:
    """Print metrics summary (legacy API)."""
    print("\n" + "=" * 60)
    if task_name:
        print(f"Metrics: {task_name}")
    print("=" * 60)
    print(f"Survival Rate:       {metrics.survival_rate:.1%}")
    print(f"Mean Episode Length: {metrics.mean_episode_length:.1f} steps")
    print(f"Mean Distance:       {metrics.mean_forward_distance:.2f} m")
    print(f"Velocity Error (X):  {metrics.velocity_error_x:.3f} m/s")
    print(f"Velocity Error (Y):  {metrics.velocity_error_y:.3f} m/s")
    print(f"Mean Velocity Error: {metrics.mean_velocity_error:.3f} m/s")
    print(f"Mean Base Height:    {metrics.mean_base_height:.3f} m")
    print(f"Height Variance:     {metrics.base_height_variance:.4f}")
    print("=" * 60 + "\n")
