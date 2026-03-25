# Copyright (c) 2024-2026, unitree_lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 Locomotion simulator for sim2sim evaluation.

Uses the SimulatorWithRunLoop template from mujoco_utils,
specialized for G1 locomotion tasks with terrain, metrics, and visualization.
"""

from __future__ import annotations

import os
import shutil

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from unitree_lab.mujoco_utils.core.joint_mapping import model_to_mujoco, mujoco_to_model
from unitree_lab.mujoco_utils.core.physics import compute_projected_gravity
from unitree_lab.mujoco_utils.evaluation.eval_task import EvalTask
from unitree_lab.mujoco_utils.evaluation.metrics import EvalResult, MetricsCollector, is_fallen
from unitree_lab.mujoco_utils.simulation.base_simulator import SimulatorWithRunLoop
from unitree_lab.mujoco_utils.simulation.observation_builder import ObservationBuilder
from unitree_lab.mujoco_utils.terrain.setup import (
    get_spawn_position,
    setup_terrain_data_in_model,
    setup_terrain_env,
)
from unitree_lab.mujoco_utils.visualization.panels import create_combined_visualization


class G1LocomotionSimulator(SimulatorWithRunLoop):
    """MuJoCo simulator for G1 locomotion evaluation.

    Supports:
    - Terrain generation (stairs, slopes, rough, etc.)
    - Velocity command tracking with streaming metrics
    - Rich visualization with torque/command overlays
    """

    def __init__(
        self,
        onnx_path: str,
        mujoco_model_path: str,
        eval_task: EvalTask,
        video_file: str = "",
    ):
        self.eval_task = eval_task
        self.video_file = video_file
        self.original_model_path = mujoco_model_path

        self.final_model_path, self.terrain_manager, self.temp_dir = setup_terrain_env(
            eval_task.terrain, video_file, mujoco_model_path
        )

        super().__init__(
            onnx_path=onnx_path,
            mujoco_model_path=self.final_model_path,
        )

        self.default_duration = eval_task.duration

        setup_terrain_data_in_model(self.model, self.terrain_manager)

        self.metrics_collector = MetricsCollector(eval_task.name, eval_task.duration)

        # ObservationBuilder for constructing obs from dict config
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

    # --- ABC implementation ---

    def build_observation(self, **kwargs) -> tuple[np.ndarray, np.ndarray | None]:
        command = kwargs.get("command")
        if command is not None:
            self._velocity_command = np.array(command, dtype=np.float32)

        _, quat, _, ang_vel = self.get_base_state()

        # Joint state in ONNX order
        joint_pos_mj = (
            self.data.sensordata[self.actuator_pos_sensor_indices]
            if self.use_sensor_data
            else self.data.qpos[-self.num_actions:]
        )
        joint_vel_mj = (
            self.data.sensordata[self.actuator_vel_sensor_indices]
            if self.use_sensor_data
            else self.data.qvel[-self.num_actions:]
        )
        joint_pos_onnx = mujoco_to_model(joint_pos_mj, self.joint_mapping_index)
        joint_vel_onnx = mujoco_to_model(joint_vel_mj, self.joint_mapping_index)

        pos, _, lin_vel, _ = self.get_base_state()

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
        self._episode_length += 1
        return obs, None

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.reset_hidden_states()
        self._obs_builder.reset()
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._episode_length = 0
        self._velocity_command = np.zeros(3)

        # Set default joint positions
        for i in range(self.num_actions):
            idx = int(self.joint_mapping_index[i])
            self.data.qpos[7 + i] = float(self.default_joint_pos[idx])

        mujoco.mj_forward(self.model, self.data)

        # Terrain spawn position
        pos_x, pos_y, spawn_z = get_spawn_position(
            self.terrain_manager, self.original_model_path,
        )
        self.data.qpos[0] = pos_x
        self.data.qpos[1] = pos_y
        self.data.qpos[2] = spawn_z
        mujoco.mj_forward(self.model, self.data)

    # --- Hook overrides ---

    def apply_action(self, new_action: np.ndarray, step_i: int) -> None:
        self.action = new_action
        # Keep a copy in ONNX order for observations
        self._last_action = new_action.copy()

    def get_command(self, curr_time: float, command_step: int) -> np.ndarray | None:
        return self.eval_task.get_velocity_command(curr_time)

    def on_control_step(
        self,
        curr_time: float,
        command: np.ndarray | None,
        command_step: int,
        torque: np.ndarray,
        torque_util: np.ndarray,
    ) -> None:
        if command is None:
            return
        _, quat, lin_vel, ang_vel = self.get_base_state()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        projected_gravity = r.inv().apply(np.array([0.0, 0.0, -1.0]))
        body_lin_vel = r.inv().apply(lin_vel)
        body_ang_vel_z = r.inv().apply(ang_vel)[2]
        actual_velocity = np.array(
            [body_lin_vel[0], body_lin_vel[1], body_ang_vel_z], dtype=np.float32,
        )

        self.metrics_collector.step(
            t=curr_time,
            projected_gravity=projected_gravity,
            cmd_velocity=command,
            actual_velocity=actual_velocity,
            torque=torque,
            torque_util=torque_util,
        )

    def render_frame(
        self,
        command: np.ndarray | None,
        command_step: int,
        torque_util: np.ndarray,
        exteroception: np.ndarray | None,
    ) -> np.ndarray:
        self.cam.lookat[:] = self.data.qpos[:3]
        self.renderer.update_scene(self.data, self.cam)
        rgb_arr = self.renderer.render()

        _, quat, lin_vel, _ = self.get_base_state()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        base_lin_vel = r.inv().apply(lin_vel)

        return create_combined_visualization(
            sim_frame=rgb_arr,
            torque_util=torque_util,
            num_actions=self.num_actions,
            base_lin_vel=base_lin_vel,
            exteroception=None,
            exteroception_type=None,
            robot_z=self.data.qpos[2],
            sim_time=self.data.time,
            command=command,
            command_idx=command_step,
        )

    def compute_results(self, torque_data_dir: str = "", headless: bool = True):
        eval_result = self.metrics_collector.compute()
        if torque_data_dir:
            saved_path = self.metrics_collector.save(torque_data_dir)
            print(f"[G1LocomotionSimulator] Saved torque data to {saved_path}")
        if not headless:
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            print(eval_result.summary())
            print("=" * 60 + "\n")
        else:
            sr = eval_result.survival_rate or 0
            print(f"[G1LocomotionSimulator] {self.eval_task.name}: survival={sr:.0%}")
        return eval_result

    def cleanup_extras(self) -> None:
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def run_locomotion_simulation(
    onnx_path: str,
    mujoco_model_path: str,
    eval_task: EvalTask,
    render: bool = False,
    video_file: str = "",
    headless: bool = True,
    torque_data_dir: str = "",
) -> EvalResult | None:
    """Convenience entry point for batch evaluator."""
    if not os.path.exists(mujoco_model_path):
        return EvalResult.from_error(eval_task.name, f"MuJoCo model not found: {mujoco_model_path}")
    if not os.path.exists(onnx_path):
        return EvalResult.from_error(eval_task.name, f"ONNX file not found: {onnx_path}")

    try:
        simulator = G1LocomotionSimulator(
            onnx_path=onnx_path,
            mujoco_model_path=mujoco_model_path,
            eval_task=eval_task,
            video_file=video_file,
        )
        return simulator.run(
            render=render,
            video_file=video_file,
            headless=headless,
            torque_data_dir=torque_data_dir,
        )
    except Exception as e:
        return EvalResult.from_error(eval_task.name, str(e))
