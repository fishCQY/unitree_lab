"""Base MuJoCo simulator for Sim2Sim policy evaluation.

Architecture aligned with bfm_training: ABC base class with explicit PD control.

Key design:
- Abstract base class — subclasses implement build_observation() and reset()
- Explicit PD control: torques computed in Python, written to data.ctrl
- Supports feedforward / GRU / LSTM / Transformer / exteroception policies
- Actuator sensor support for parallel-mechanism robots
- Wheel / legged detection from XML joint names
- Contact parameter alignment with IsaacLab/PhysX
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    mujoco = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from ..core.onnx_utils import detect_policy_type, get_onnx_config, init_hidden_states
from ..core.physics import apply_onnx_physics_params, get_tau_limit, pd_control
from ..core.xml_parsing import parse_actuators_from_xml


class BaseMujocoSimulator(ABC):
    """Abstract base class for MuJoCo simulation with ONNX policy.

    Subclasses must implement:
    - ``build_observation(**kwargs)`` — return (obs, exteroception_or_None)
    - ``reset()`` — reset simulation state

    Example::

        class LocomotionSimulator(BaseMujocoSimulator):
            def build_observation(self, **kwargs):
                ...
                return obs, None

            def reset(self):
                mujoco.mj_resetData(self.model, self.data)
                ...
    """

    sim_dt: float = 0.001
    decimation: int = 20

    def __init__(self, onnx_path: str, mujoco_model_path: str):
        if mujoco is None:
            raise ImportError("mujoco not installed. Run: pip install mujoco")
        if ort is None:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")

        self.onnx_path = onnx_path
        self.mujoco_model_path = mujoco_model_path

        # Parse ONNX config (returns dict)
        self.onnx_config = get_onnx_config(onnx_path)
        self.joint_names = self.onnx_config.get("joint_names", [])
        self.num_actions = self.onnx_config.get("num_actions", 0) or len(self.joint_names)
        self.action_scale = np.array(self.onnx_config.get("action_scale", [0.25] * self.num_actions))
        self.default_joint_pos = np.array(self.onnx_config.get("default_joint_pos", [0.0] * self.num_actions))

        # Parse actuators from XML
        self.joint_xml = parse_actuators_from_xml(mujoco_model_path)
        self.is_wheel_mask = np.array(["wheel" in joint for joint in self.joint_xml])
        self.is_legged = not any("wheel" in joint for joint in self.joint_xml)

        # Create joint mapping (ONNX name -> MuJoCo actuator index)
        if self.joint_names and self.joint_xml:
            self.joint_mapping_index = np.array(
                [self.joint_names.index(joint) for joint in self.joint_xml]
            )
        else:
            self.joint_mapping_index = np.arange(self.num_actions)

        # Get PD gains in MuJoCo actuator order
        p_gains_seq = np.array(self.onnx_config.get("joint_stiffness", [100.0] * self.num_actions))
        self.p_gains = self._model_to_mujoco(p_gains_seq, self.joint_mapping_index)
        d_gains_seq = np.array(self.onnx_config.get("joint_damping", [10.0] * self.num_actions))
        self.d_gains = self._model_to_mujoco(d_gains_seq, self.joint_mapping_index)

        # Initialize ONNX runtime session
        self.ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.policy_type = detect_policy_type(self.ort_session)
        self.is_recurrent = self.policy_type == "recurrent"
        self.is_transformer = self.policy_type == "transformer"

        if self.is_recurrent:
            self.hidden_state, self.cell_state, _ = init_hidden_states(self.ort_session)
        else:
            self.hidden_state = None
            self.cell_state = None

        # Transformer state
        self._tf_obs_buffer: np.ndarray | None = None
        self._tf_valid_len: np.ndarray | None = None
        if self.is_transformer:
            for inp in self.ort_session.get_inputs():
                if inp.name == "obs_buffer":
                    self._tf_obs_buffer = np.zeros(inp.shape, dtype=np.float32)
                elif inp.name == "valid_len":
                    self._tf_valid_len = np.zeros(inp.shape, dtype=np.int64)

        # Exteroception input support
        input_names = [inp.name for inp in self.ort_session.get_inputs()]
        self.has_exteroception_input = "exteroception" in input_names

        self.expected_obs_dim = self.ort_session.get_inputs()[0].shape[1]

        # Initialize MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.model.opt.timestep = self.sim_dt

        # Solver iterations for humanoid contact stability
        self.model.opt.iterations = 50
        self.model.opt.ls_iterations = 50

        self.data = mujoco.MjData(self.model)

        # Apply physics parameters from ONNX config
        apply_onnx_physics_params(self.model, self.onnx_config)

        # Get torque limits from actuator ctrlrange
        self.tau_limit = get_tau_limit(self.model, self.num_actions)

        # Build actuator sensor indices for parallel mechanism support
        self.actuator_pos_sensor_indices: np.ndarray = np.array([], dtype=np.int64)
        self.actuator_vel_sensor_indices: np.ndarray = np.array([], dtype=np.int64)
        self._build_actuator_sensor_mapping()

        # Align ground contact parameters with IsaacLab defaults
        self._configure_contact_params()

        # Default observation scales
        self.obs_scales = {
            "dof_pos": 1.0,
            "dof_vel": 0.2,
            "ang_vel": 0.2,
        }

        # Current action state (MuJoCo actuator order)
        self.action = np.zeros(self.num_actions)

        print(f"[Sim2Sim] Initialized: {Path(mujoco_model_path).name}, "
              f"policy_type={self.policy_type}, actions={self.num_actions}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_to_mujoco(model_values: np.ndarray, mapping_index: np.ndarray) -> np.ndarray:
        """Reorder values from ONNX/model order to MuJoCo actuator order."""
        result = np.zeros(len(mapping_index))
        for i, idx in enumerate(mapping_index):
            result[i] = model_values[idx]
        return result

    def _build_actuator_sensor_mapping(self) -> None:
        """Build sensor indices for actuator position/velocity readings.

        Supports parallel mechanism robots where ``qpos[-num_actions:]``
        doesn't give the correct actuator joint states.
        """
        pos_indices = []
        vel_indices = []
        for actuator_name in self.joint_xml:
            pos_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{actuator_name}_p"
            )
            vel_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{actuator_name}_v"
            )
            if pos_id >= 0 and vel_id >= 0:
                pos_indices.append(self.model.sensor_adr[pos_id])
                vel_indices.append(self.model.sensor_adr[vel_id])

        self.actuator_pos_sensor_indices = np.array(pos_indices, dtype=np.int64)
        self.actuator_vel_sensor_indices = np.array(vel_indices, dtype=np.int64)
        self.use_sensor_data = len(self.actuator_pos_sensor_indices) == self.num_actions

        if self.use_sensor_data:
            print("[Sim2Sim] Using actuator sensors for joint state (parallel mechanism support)")

    def _configure_contact_params(self) -> None:
        """Align MuJoCo contact/solver parameters with IsaacLab/PhysX defaults."""
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.model.opt.noslip_iterations = 10

        HARD_SOLREF = np.array([0.005, 1.0])
        HARD_SOLIMP = np.array([0.95, 0.99, 0.001, 0.5, 2.0])

        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            self.model.geom_friction[floor_id] = [1.0, 0.005, 0.0001]
            self.model.geom_solref[floor_id] = HARD_SOLREF
            self.model.geom_solimp[floor_id] = HARD_SOLIMP

        for gid in range(self.model.ngeom):
            if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_HFIELD:
                self.model.geom_friction[gid] = [1.0, 0.005, 0.0001]
                self.model.geom_solref[gid] = HARD_SOLREF
                self.model.geom_solimp[gid] = HARD_SOLIMP
                break

        n_set = 0
        for gid in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[gid]
            if body_id <= 0:
                continue
            contype = int(self.model.geom_contype[gid])
            conaff = int(self.model.geom_conaffinity[gid])
            if contype == 0 and conaff == 0:
                continue
            self.model.geom_friction[gid] = [1.0, 0.005, 0.0001]
            self.model.geom_solref[gid] = HARD_SOLREF
            self.model.geom_solimp[gid] = HARD_SOLIMP
            n_set += 1

        print(f"[Sim2Sim] Contact: cone=pyramidal, noslip=10, robot_geoms={n_set}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_observation(self, **kwargs) -> tuple[np.ndarray, np.ndarray | None]:
        """Build observation vector for the policy.

        Returns:
            Tuple of (observation, exteroception_or_None).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulation state."""

    # ------------------------------------------------------------------
    # Policy inference
    # ------------------------------------------------------------------

    def step_inference(
        self,
        obs: np.ndarray,
        exteroception: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run policy inference to get action.

        Supports feedforward / GRU / LSTM / Transformer / exteroception.
        """
        if self.is_transformer:
            ort_inputs = {
                "obs": obs.reshape(1, -1).astype(np.float32),
                "obs_buffer": self._tf_obs_buffer,
                "valid_len": self._tf_valid_len,
            }
            ort_outputs = self.ort_session.run(None, ort_inputs)
            action = ort_outputs[0][0]
            self._tf_obs_buffer[:] = ort_outputs[1]
            self._tf_valid_len[:] = ort_outputs[2]
        elif self.is_recurrent:
            assert self.hidden_state is not None
            if self.cell_state is not None:  # LSTM
                ort_inputs = {
                    "obs": obs.reshape(1, -1).astype(np.float32),
                    "h_in": self.hidden_state,
                    "c_in": self.cell_state,
                }
                if self.has_exteroception_input and exteroception is not None:
                    ort_inputs["exteroception"] = exteroception.reshape(1, *exteroception.shape).astype(np.float32)
                ort_outputs = self.ort_session.run(None, ort_inputs)
                action = ort_outputs[0][0]
                self.hidden_state[:] = ort_outputs[1]
                self.cell_state[:] = ort_outputs[2]
            else:  # GRU
                ort_inputs = {
                    "obs": obs.reshape(1, -1).astype(np.float32),
                    "h_in": self.hidden_state,
                }
                if self.has_exteroception_input and exteroception is not None:
                    ort_inputs["exteroception"] = exteroception.reshape(1, *exteroception.shape).astype(np.float32)
                ort_outputs = self.ort_session.run(None, ort_inputs)
                action = ort_outputs[0][0]
                self.hidden_state[:] = ort_outputs[1]
        else:
            ort_inputs = {"obs": obs.reshape(1, -1).astype(np.float32)}
            if self.has_exteroception_input and exteroception is not None:
                ort_inputs["exteroception"] = exteroception.reshape(1, *exteroception.shape).astype(np.float32)
            action = self.ort_session.run(None, ort_inputs)[0][0]

        return np.clip(action, -100.0, 100.0)

    # ------------------------------------------------------------------
    # Control and physics
    # ------------------------------------------------------------------

    def step_control(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply action using explicit PD control.

        Computes torques and writes them to ``data.ctrl``.

        Returns:
            Tuple of (torque, torque_util).
        """
        current_action_scaled = action * self.action_scale
        target_q = current_action_scaled + self.default_joint_pos
        target_dq = np.zeros_like(current_action_scaled)

        # Wheel joints use velocity control
        target_q[self.is_wheel_mask] = 0.0
        target_dq[self.is_wheel_mask] = current_action_scaled[self.is_wheel_mask]

        # Read current joint state
        if self.use_sensor_data:
            current_q = self.data.sensordata[self.actuator_pos_sensor_indices]
            current_dq = self.data.sensordata[self.actuator_vel_sensor_indices]
        else:
            current_q = self.data.qpos[-self.num_actions:]
            current_dq = self.data.qvel[-self.num_actions:]

        torque, torque_util = pd_control(
            target_q, current_q, self.p_gains,
            target_dq, current_dq, self.d_gains,
            self.tau_limit,
        )

        self.data.ctrl[:] = torque
        return torque, torque_util

    def step_physics(self) -> None:
        """Step the MuJoCo physics simulation one timestep."""
        mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_base_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get base position, orientation, linear velocity, and angular velocity."""
        pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7].copy()
        lin_vel = self.data.qvel[:3].copy()
        ang_vel = self.data.qvel[3:6].copy()
        return pos, quat, lin_vel, ang_vel

    @property
    def current_time(self) -> float:
        return self.data.time

    # ------------------------------------------------------------------
    # Hidden state management
    # ------------------------------------------------------------------

    def reset_hidden_states(self) -> None:
        """Reset recurrent / transformer hidden states to zero."""
        if self.hidden_state is not None:
            self.hidden_state.fill(0)
        if self.cell_state is not None:
            self.cell_state.fill(0)
        if self.is_transformer:
            if self._tf_obs_buffer is not None:
                self._tf_obs_buffer.fill(0)
            if self._tf_valid_len is not None:
                self._tf_valid_len.fill(0)


class SimulatorWithRunLoop(BaseMujocoSimulator):
    """Base simulator with a template run-loop.

    Inner loop runs at **physics frequency** (``sim_dt``), with control
    steps at decimated frequency — matching bfm_training behavior where
    PD torques are recomputed every physics substep.

    Subclasses override hook methods:
    - ``get_command()`` — return velocity command
    - ``on_physics_step()`` — per-physics-step logic
    - ``on_control_step()`` — per-control-step logic with torque/torque_util
    - ``render_frame()`` — customize frame rendering
    - ``compute_results()`` — return final evaluation results
    - ``cleanup_extras()`` — clean up task-specific resources
    """

    default_duration: float = 30.0
    show_progress_bar: bool = False
    cam_distance: float | None = None
    window_title: str = "MuJoCo Simulation"

    def __init__(self, onnx_path: str, mujoco_model_path: str):
        super().__init__(onnx_path, mujoco_model_path)
        self.renderer = None
        self.cam = None
        self._used_gui = False

    def run(
        self,
        render: bool = False,
        video_file: str = "",
        headless: bool = True,
        duration: float | None = None,
        torque_data_dir: str = "",
    ):
        """Run the simulation.

        Args:
            render: Record video frames.
            video_file: Output video file path.
            headless: Run without display window.
            duration: Simulation duration in seconds.
            torque_data_dir: Directory to save torque data.

        Returns:
            Task-specific result from ``compute_results()``.
        """
        import cv2

        duration = duration if duration is not None else self.default_duration
        fps = 30.0
        frames = []
        command = None
        command_step = 0
        exteroception = None
        torque_util = np.zeros(self.num_actions)
        self._used_gui = not headless

        if render or not headless:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.cam)
            if self.cam_distance is not None:
                self.cam.distance = self.cam_distance

        try:
            self.reset()
            num_steps = int(duration / self.sim_dt)
            step_iter = range(num_steps)

            if self.show_progress_bar:
                try:
                    from tqdm import tqdm
                    step_iter = tqdm(step_iter, desc="Simulating")
                except ImportError:
                    pass

            print(f"[SimulatorWithRunLoop] Starting: {duration}s, {num_steps} physics steps")

            for i in step_iter:
                curr_time = i * self.sim_dt

                self.on_physics_step(i, curr_time)

                # Control step at decimated frequency
                if i % self.decimation == 0:
                    command = self.get_command(curr_time, command_step)
                    command_step += 1

                    obs, exteroception = self.build_observation(
                        command=command, curr_time=curr_time,
                    )
                    new_action = self.step_inference(obs, exteroception)
                    self.apply_action(new_action, i)

                # Explicit PD every physics step (recomputes torque from held action)
                torque, torque_util = self.step_control(self.action)
                self.step_physics()

                if i % self.decimation == 0:
                    self.on_control_step(
                        curr_time, command, command_step, torque, torque_util,
                    )

                # Render at video frame rate
                if self.renderer is not None and self.cam is not None:
                    render_interval = max(1, int(1.0 / (self.sim_dt * fps)))
                    if i % render_interval == 0:
                        frame = self.render_frame(
                            command, command_step, torque_util, exteroception,
                        )
                        if render:
                            frames.append(frame)
                        if not headless:
                            cv2.imshow(
                                self.window_title,
                                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                            )
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break

            if video_file and frames:
                self._save_video(video_file, frames, fps)

            return self.compute_results(
                torque_data_dir=torque_data_dir, headless=headless,
            )

        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Hook methods
    # ------------------------------------------------------------------

    def get_command(self, curr_time: float, command_step: int) -> np.ndarray | None:
        """Return command for current timestep."""
        return None

    def on_physics_step(self, step_i: int, curr_time: float) -> None:
        """Called every physics step."""

    def on_control_step(
        self,
        curr_time: float,
        command: np.ndarray | None,
        command_step: int,
        torque: np.ndarray,
        torque_util: np.ndarray,
    ) -> None:
        """Called every control step."""

    def apply_action(self, new_action: np.ndarray, step_i: int) -> None:
        """Apply action (with optional delay). Override for scheduling."""
        self.action = new_action

    def render_frame(
        self,
        command: np.ndarray | None,
        command_step: int,
        torque_util: np.ndarray,
        exteroception: np.ndarray | None,
    ) -> np.ndarray:
        """Render a single frame. Override for visualization overlays."""
        self.cam.lookat[:] = self.data.qpos[:3]
        self.renderer.update_scene(self.data, self.cam)
        rgb_arr = self.renderer.render()

        try:
            from scipy.spatial.transform import Rotation as R

            _, quat, lin_vel, _ = self.get_base_state()
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            base_lin_vel = r.inv().apply(lin_vel)

            from ..visualization import create_combined_visualization

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
        except ImportError:
            return rgb_arr

    def compute_results(self, torque_data_dir: str = "", headless: bool = True):
        """Compute and return final evaluation results."""
        return None

    def cleanup_extras(self) -> None:
        """Clean up task-specific resources."""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _save_video(video_file: str, frames: list[np.ndarray], fps: float) -> None:
        if not frames:
            return
        try:
            import mediapy as media
            media.write_video(video_file, frames, fps=fps)
            print(f"[SimulatorWithRunLoop] Video saved: {video_file}")
            return
        except ImportError:
            pass
        import cv2
        h, w = frames[0].shape[:2]
        try:
            import subprocess
            proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", str(int(fps)),
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
            if proc.returncode == 0:
                print(f"[SimulatorWithRunLoop] Video saved: {video_file}")
                return
        except FileNotFoundError:
            pass
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_file, fourcc, int(fps), (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[SimulatorWithRunLoop] Video saved: {video_file}")

    def _cleanup(self) -> None:
        if self.renderer is not None:
            del self.renderer
            self.renderer = None
        if getattr(self, "_used_gui", False):
            import cv2
            with contextlib.suppress(Exception):
                cv2.destroyAllWindows()
        self.cleanup_extras()
