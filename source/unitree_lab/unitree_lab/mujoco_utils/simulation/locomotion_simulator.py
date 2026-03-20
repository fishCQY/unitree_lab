"""Concrete locomotion simulator for sim2sim.

Thin wrapper around ``BaseMujocoSimulator`` that adds locomotion-specific
behaviour: velocity commands, locomotion termination criteria, and a
convenience ``run_episode`` method compatible with the batch evaluator.

All heavy lifting (ONNX loading, joint mapping, position-servo PD control,
observation building, physics stepping) is handled by the base class.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_simulator import BaseMujocoSimulator
from ..core.physics import quat_rotate_inverse


class LocomotionMujocoSimulator(BaseMujocoSimulator):
    """Locomotion policy simulator with velocity-command observations."""

    def __init__(
        self,
        *,
        onnx_path: str,
        xml_path: str,
        config_override: dict[str, Any] | None = None,
    ):
        super().__init__(xml_path=xml_path, onnx_path=onnx_path, config_override=config_override)

    def _projected_gravity(self) -> np.ndarray:
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return quat_rotate_inverse(
            self.base_quat.astype(np.float64), g_world
        ).astype(np.float32)

    def _check_termination(self) -> bool:
        """IsaacLab-compatible termination: fallen or too low."""
        if float(self.base_pos[2]) < 0.12:
            return True
        proj_g = self._projected_gravity()
        proj_grav_z = float(np.clip(proj_g[2], -1.0, 1.0))
        if float(np.arccos(-proj_grav_z)) > 1.0:
            return True
        return False

    def run_episode(
        self,
        *,
        max_steps: int,
        render: bool = False,
        velocity_command: tuple[float, float, float] = (0.5, 0.0, 0.0),
    ) -> dict[str, float | int | bool]:
        """Run one episode and return lightweight metrics dict.

        This signature is kept compatible with ``BatchEvaluator.evaluate_task``
        and ``run_headless`` in ``run_sim2sim_locomotion.py``.
        """
        self.set_velocity_command(*velocity_command)
        self.reset()

        start_x = float(self.base_pos[0])
        vel_err_sum = 0.0
        steps = 0
        terminated = False

        cmd = np.asarray(velocity_command, dtype=np.float64).reshape(3)
        for _ in range(int(max_steps)):
            self.step()
            steps += 1

            lin_vel = self.base_lin_vel
            ang_vel = self.base_ang_vel
            actual = np.array([
                float(lin_vel[0]), float(lin_vel[1]),
                float(ang_vel[2]) if ang_vel.shape[0] >= 3 else 0.0,
            ], dtype=np.float64)
            vel_err_sum += float(np.linalg.norm(actual - cmd))

            if self._check_termination():
                terminated = True
                break

        dist = float(self.base_pos[0] - start_x)
        mean_err = float(vel_err_sum / max(1, steps))
        return {
            "steps": int(steps),
            "max_steps": int(max_steps),
            "terminated": bool(terminated),
            "mean_velocity_error": mean_err,
            "forward_distance": dist,
        }
