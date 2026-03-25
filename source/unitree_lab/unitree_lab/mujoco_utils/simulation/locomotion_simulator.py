"""Concrete locomotion simulator for sim2sim.

Thin wrapper around ``BaseMujocoSimulator`` that adds locomotion-specific
behaviour: velocity commands, locomotion termination criteria, and a
convenience ``run_episode`` method compatible with the batch evaluator.

Uses explicit PD control (torques computed in Python, written to data.ctrl).
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .base_simulator import BaseMujocoSimulator
from .observation_builder import ObservationBuilder
from ..core.joint_mapping import mujoco_to_model
from ..core.physics import compute_projected_gravity, quat_rotate_inverse


class LocomotionMujocoSimulator(BaseMujocoSimulator):
    """Locomotion policy simulator with velocity-command observations.

    Implements the ABC interface (build_observation / reset) for
    locomotion tasks using ObservationBuilder + explicit PD control.
    """

    def __init__(
        self,
        *,
        onnx_path: str,
        mujoco_model_path: str,
    ):
        super().__init__(onnx_path=onnx_path, mujoco_model_path=mujoco_model_path)

        self._obs_builder = ObservationBuilder(
            model=self.model,
            data=self.data,
            onnx_config=self.onnx_config,
            joint_mapping=list(self.joint_mapping_index),
        )
        if not self.onnx_config.get("default_joint_pos"):
            self._obs_builder.default_joint_pos = self.default_joint_pos.copy()

        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._episode_length = 0
        self._velocity_command = np.zeros(3)

    def set_velocity_command(self, vx: float, vy: float, wz: float) -> None:
        self._velocity_command = np.array([vx, vy, wz])

    # --- ABC implementation ---

    def build_observation(self, **kwargs) -> tuple[np.ndarray, np.ndarray | None]:
        _, quat, lin_vel, ang_vel = self.get_base_state()

        if self.use_sensor_data:
            joint_pos_mj = self.data.sensordata[self.actuator_pos_sensor_indices]
            joint_vel_mj = self.data.sensordata[self.actuator_vel_sensor_indices]
        else:
            joint_pos_mj = self.data.qpos[-self.num_actions:]
            joint_vel_mj = self.data.qvel[-self.num_actions:]
        joint_pos_onnx = mujoco_to_model(joint_pos_mj, self.joint_mapping_index)
        joint_vel_onnx = mujoco_to_model(joint_vel_mj, self.joint_mapping_index)

        obs = self._obs_builder.build(
            joint_pos=joint_pos_onnx,
            joint_vel=joint_vel_onnx,
            base_quat=quat,
            base_ang_vel=ang_vel,
            base_lin_vel=lin_vel,
            last_action=self._last_action,
            velocity_command=self._velocity_command,
            episode_length=self._episode_length,
            step_dt=self.sim_dt * self.decimation,
        )
        return obs, None

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.reset_hidden_states()
        self._obs_builder.reset()
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._episode_length = 0

        for i in range(self.num_actions):
            idx = int(self.joint_mapping_index[i])
            self.data.qpos[7 + i] = float(self.default_joint_pos[idx])

        mujoco.mj_forward(self.model, self.data)

    # --- Locomotion-specific ---

    def _check_termination(self) -> bool:
        pos, quat, _, _ = self.get_base_state()
        if float(pos[2]) < 0.12:
            return True
        proj_g = compute_projected_gravity(quat)
        proj_grav_z = float(np.clip(proj_g[2], -1.0, 1.0))
        if float(np.arccos(-proj_grav_z)) > 1.0:
            return True
        return False

    def step_one(self) -> None:
        """Run one full policy step (inference + decimated physics)."""
        obs, extero = self.build_observation()
        action = self.step_inference(obs, extero)
        self._last_action = action.copy()
        self.action = action

        for _ in range(self.decimation):
            self.step_control(self.action)
            self.step_physics()

        self._episode_length += 1

    def run_episode(
        self,
        *,
        max_steps: int,
        render: bool = False,
        velocity_command: tuple[float, float, float] = (0.5, 0.0, 0.0),
    ) -> dict[str, float | int | bool]:
        """Run one episode and return lightweight metrics dict."""
        self.set_velocity_command(*velocity_command)
        self.reset()

        start_x = float(self.data.qpos[0])
        vel_err_sum = 0.0
        steps = 0
        terminated = False
        cmd = np.asarray(velocity_command, dtype=np.float64).reshape(3)

        for _ in range(int(max_steps)):
            self.step_one()
            steps += 1

            _, _, lin_vel, ang_vel = self.get_base_state()
            actual = np.array([
                float(lin_vel[0]), float(lin_vel[1]),
                float(ang_vel[2]) if ang_vel.shape[0] >= 3 else 0.0,
            ], dtype=np.float64)
            vel_err_sum += float(np.linalg.norm(actual - cmd))

            if self._check_termination():
                terminated = True
                break

        dist = float(self.data.qpos[0] - start_x)
        mean_err = float(vel_err_sum / max(1, steps))
        return {
            "steps": int(steps),
            "max_steps": int(max_steps),
            "terminated": bool(terminated),
            "mean_velocity_error": mean_err,
            "forward_distance": dist,
        }
