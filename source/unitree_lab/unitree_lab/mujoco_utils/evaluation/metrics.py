"""Metrics computation for sim2sim evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LocomotionMetrics:
    """Metrics for locomotion evaluation."""
    
    # Survival
    survival_rate: float = 0.0
    mean_episode_length: float = 0.0
    
    # Velocity tracking
    mean_velocity_error: float = 0.0
    velocity_error_x: float = 0.0
    velocity_error_y: float = 0.0
    velocity_error_yaw: float = 0.0
    
    # Distance
    total_distance: float = 0.0
    mean_forward_distance: float = 0.0
    
    # Energy
    mean_energy: float = 0.0
    mean_torque_magnitude: float = 0.0
    
    # Stability
    mean_base_height: float = 0.0
    base_height_variance: float = 0.0
    mean_orientation_error: float = 0.0
    
    # Raw data for analysis
    episode_lengths: list[int] = field(default_factory=list)
    episode_distances: list[float] = field(default_factory=list)
    episode_survivals: list[bool] = field(default_factory=list)


def compute_locomotion_metrics(
    episode_data: list[dict],
    velocity_command: np.ndarray,
    policy_dt: float = 0.02,
) -> LocomotionMetrics:
    """Compute locomotion metrics from episode data.
    
    Args:
        episode_data: List of episode dictionaries with 'data' and 'stats' keys
        velocity_command: Target velocity command [vx, vy, wz]
        policy_dt: Policy time step
        
    Returns:
        Computed metrics
    """
    metrics = LocomotionMetrics()
    
    if not episode_data:
        return metrics
    
    # Process each episode
    for ep in episode_data:
        data = ep.get("data", {})
        stats = ep.get("stats", {})
        
        # Episode length
        num_steps = stats.get("num_steps", 0)
        metrics.episode_lengths.append(num_steps)
        
        # Survival
        survived = stats.get("survived", False)
        metrics.episode_survivals.append(survived)
        
        # Distance
        distance = stats.get("distance_traveled", 0.0)
        metrics.episode_distances.append(distance)
        
        # Velocity tracking
        if "base_lin_vel" in data and len(data["base_lin_vel"]) > 0:
            vel = np.array(data["base_lin_vel"])
            vel_error = np.abs(vel[:, :2] - velocity_command[:2])
            metrics.velocity_error_x += np.mean(np.abs(vel[:, 0] - velocity_command[0]))
            metrics.velocity_error_y += np.mean(np.abs(vel[:, 1] - velocity_command[1]))
        
        # Base height
        if "base_pos" in data and len(data["base_pos"]) > 0:
            heights = np.array(data["base_pos"])[:, 2]
            metrics.mean_base_height += np.mean(heights)
            metrics.base_height_variance += np.var(heights)
    
    # Average metrics
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
    """Print metrics summary.
    
    Args:
        metrics: Computed metrics
        task_name: Optional task name for header
    """
    print("\n" + "=" * 60)
    if task_name:
        print(f"Metrics: {task_name}")
    print("=" * 60)
    
    print(f"Survival Rate:       {metrics.survival_rate:.1%}")
    print(f"Mean Episode Length: {metrics.mean_episode_length:.1f} steps")
    print(f"Mean Distance:       {metrics.mean_forward_distance:.2f} m")
    print(f"")
    print(f"Velocity Error (X):  {metrics.velocity_error_x:.3f} m/s")
    print(f"Velocity Error (Y):  {metrics.velocity_error_y:.3f} m/s")
    print(f"Mean Velocity Error: {metrics.mean_velocity_error:.3f} m/s")
    print(f"")
    print(f"Mean Base Height:    {metrics.mean_base_height:.3f} m")
    print(f"Height Variance:     {metrics.base_height_variance:.4f}")
    print("=" * 60 + "\n")


def aggregate_metrics(metrics_list: list[LocomotionMetrics]) -> dict[str, float]:
    """Aggregate metrics from multiple tasks.
    
    Args:
        metrics_list: List of metrics from different tasks
        
    Returns:
        Aggregated statistics
    """
    if not metrics_list:
        return {}
    
    return {
        "mean_survival_rate": np.mean([m.survival_rate for m in metrics_list]),
        "mean_velocity_error": np.mean([m.mean_velocity_error for m in metrics_list]),
        "mean_distance": np.mean([m.mean_forward_distance for m in metrics_list]),
        "total_distance": sum(m.total_distance for m in metrics_list),
        "total_episodes": sum(len(m.episode_lengths) for m in metrics_list),
    }
