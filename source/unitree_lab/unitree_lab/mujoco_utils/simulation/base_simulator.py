"""Base MuJoCo simulator for Sim2Sim policy evaluation.

This module provides the core simulation loop:
1. Load MuJoCo model and ONNX policy
2. Build observations matching IsaacLab semantics
3. Apply PD control from policy outputs
4. Step physics at correct frequency

Key alignment points:
- Joint order mapping (ONNX -> MuJoCo actuator order)
- Action semantics (scale, offset, position vs velocity)
- Observation semantics (per-term scale, clip, history)
- Timing (sim_dt, decimation, policy_dt)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    mujoco = None

from ..core.onnx_utils import OnnxConfig, OnnxInference, get_onnx_config
from ..core.physics import pd_control, pd_control_velocity, apply_onnx_physics_params
from ..core.xml_parsing import (
    build_joint_mapping,
    get_actuator_names,
    get_ctrl_ranges,
    parse_actuators_from_xml,
)
from .observation_builder import ObservationBuilder


class BaseMujocoSimulator:
    """Base simulator for MuJoCo sim2sim evaluation.
    
    This class handles:
    - Loading and configuring MuJoCo model
    - Loading ONNX policy with metadata
    - Joint mapping between ONNX and MuJoCo
    - PD control with correct gains/limits
    - Observation construction
    - Simulation stepping at correct frequency
    
    Subclass this for task-specific simulators (locomotion, manipulation, etc).
    """
    
    def __init__(
        self,
        xml_path: str | Path,
        onnx_path: str | Path,
        config_override: dict[str, Any] | None = None,
    ):
        """Initialize simulator.
        
        Args:
            xml_path: Path to MuJoCo XML model
            onnx_path: Path to ONNX policy
            config_override: Override config values
        """
        if mujoco is None:
            raise ImportError("mujoco not installed. Run: pip install mujoco")
        
        self.xml_path = Path(xml_path)
        self.onnx_path = Path(onnx_path)
        
        # Load ONNX config
        self.onnx_config = get_onnx_config(onnx_path)
        if config_override:
            self._apply_config_override(config_override)
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Optional spawn adjustment for heightfield terrains (set by sim2sim runner).
        # If >0, we will lift the floating base root z on every reset to avoid initial
        # interpenetration with raised terrains.
        self.spawn_root_z_offset: float = 0.0
        
        # Configure timing
        self.sim_dt = self.onnx_config.sim_dt
        self.decimation = self.onnx_config.decimation
        self.policy_dt = self.sim_dt * self.decimation
        self.model.opt.timestep = self.sim_dt

        # Build robust actuator <-> (joint,qpos,dof) mappings from the loaded MuJoCo model.
        # This is critical for scene/include XMLs where qpos slices are not aligned with actuator order.
        self._setup_actuator_state_mapping()
        
        # Setup joint mapping
        self._setup_joint_mapping()
        
        # Apply physics params from ONNX
        self._apply_physics_params()
        
        # Setup PD gains
        self._setup_pd_gains()
        
        # Load policy
        self.policy = OnnxInference(onnx_path)
        
        # Setup observation builder
        self.obs_builder = ObservationBuilder(
            model=self.model,
            data=self.data,
            onnx_config=self.onnx_config,
            joint_mapping=self.joint_mapping,
        )
        # If ONNX metadata didn't provide default_joint_pos, fall back to the simulator's default pose (ONNX order).
        if not self.onnx_config.default_joint_pos:
            self.obs_builder.default_joint_pos = self.default_joint_pos_onnx.copy()
        
        # State
        self.episode_length = 0
        self.global_phase = 0.0
        # last_action must be kept in ONNX joint order because it is fed back into observations.
        self._last_action = np.zeros(self.onnx_action_dim, dtype=np.float32)
        
        # Velocity command (for locomotion)
        self.velocity_command = np.zeros(3)  # [vx, vy, wz]
        
        print(f"[Sim2Sim] Initialized simulator:")
        print(f"  - Model: {self.xml_path.name}")
        print(f"  - Policy: {self.onnx_path.name}")
        print(f"  - Num joints: {self.num_actions}")
        print(f"  - Timing: sim_dt={self.sim_dt}, decimation={self.decimation}, policy_dt={self.policy_dt}")

    def _setup_actuator_state_mapping(self) -> None:
        """Build mappings to read/write actuator joint states robustly from MuJoCo model.

        We derive for each actuator i:
          - joint id: model.actuator_trnid[i,0]
          - qpos address: model.jnt_qposadr[jid]
          - dof address: model.jnt_dofadr[jid]

        This avoids assuming actuated joints are a contiguous slice in qpos/qvel.
        """
        model_nu = int(self.model.nu)
        if model_nu <= 0:
            raise RuntimeError("[Sim2Sim] MuJoCo model.nu==0 (no actuators).")

        self.num_actions = model_nu

        # Actuator names in model order (preferred; works with <include> scenes).
        self.actuator_names: list[str] = []
        self._actuator_joint_ids: list[int] = []
        self._qpos_adrs: list[int] = []
        self._dof_adrs: list[int] = []
        self.tau_limits = np.zeros((self.num_actions, 2), dtype=np.float32)

        for act_id in range(self.num_actions):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
            self.actuator_names.append(name or f"actuator_{act_id}")

            jid = int(self.model.actuator_trnid[act_id, 0])
            if jid < 0:
                raise RuntimeError(
                    f"[Sim2Sim] Actuator '{self.actuator_names[-1]}' has invalid actuator_trnid joint id: {jid}"
                )
            self._actuator_joint_ids.append(jid)

            qpos_adr = int(self.model.jnt_qposadr[jid])
            dof_adr = int(self.model.jnt_dofadr[jid])
            self._qpos_adrs.append(qpos_adr)
            self._dof_adrs.append(dof_adr)

            # Torque limits from joint actuation force range (more reliable than actuator ctrlrange).
            try:
                self.tau_limits[act_id, :] = self.model.jnt_actfrcrange[jid].astype(np.float32)
            except Exception:
                self.tau_limits[act_id, :] = np.array([-np.inf, np.inf], dtype=np.float32)

        # ONNX action dimension (policy output dim). Prefer metadata/output dim; fall back to num_actions.
        self.onnx_action_dim = int(self.onnx_config.output_dim) if int(self.onnx_config.output_dim or 0) > 0 else self.num_actions
    
    def _apply_config_override(self, override: dict[str, Any]) -> None:
        """Apply configuration overrides."""
        for key, value in override.items():
            if hasattr(self.onnx_config, key):
                setattr(self.onnx_config, key, value)
    
    def _setup_joint_mapping(self) -> None:
        """Setup mapping from MuJoCo actuators to ONNX joints."""
        # Prefer MuJoCo model actuator names (include-scene safe).
        model_actuator_names = list(self.actuator_names)

        # Also try parsing actuators from the provided XML (useful when model names are missing).
        xml_actuator_names: list[str] = []
        try:
            xml_actuator_names = get_actuator_names(self.xml_path)
        except Exception:
            xml_actuator_names = []

        # Choose the most informative actuator name list.
        # - If model provides real names, use them.
        # - If model names are generic and XML parse gives better names, use XML list (if sizes match).
        use_names = model_actuator_names
        if xml_actuator_names and len(xml_actuator_names) == len(model_actuator_names):
            model_generic = all((n.startswith("actuator_") or n == "") for n in model_actuator_names)
            if model_generic:
                use_names = xml_actuator_names
        if not use_names:
            use_names = model_actuator_names or xml_actuator_names
        
        # Get joint names from ONNX
        onnx_joint_names = self.onnx_config.joint_names
        
        if not onnx_joint_names:
            # If no joint names in metadata, assume same order (ONNX output already matches actuator order).
            print("[Warning] No joint_names in ONNX metadata, assuming same order as MuJoCo actuators")
            self.joint_mapping = list(range(len(use_names)))
        else:
            self.joint_mapping = build_joint_mapping(onnx_joint_names, use_names)

        if len(self.joint_mapping) != self.num_actions:
            raise RuntimeError(
                f"[Sim2Sim] joint_mapping length mismatch: mapping={len(self.joint_mapping)} vs model.nu={self.num_actions}"
            )

        # Inverse mapping: for each ONNX joint index -> actuator index
        self.inv_joint_mapping = [0] * len(self.joint_mapping)
        for act_i, onnx_i in enumerate(self.joint_mapping):
            if onnx_i < 0 or onnx_i >= len(self.inv_joint_mapping):
                raise RuntimeError(f"[Sim2Sim] Invalid mapping entry: joint_mapping[{act_i}]={onnx_i}")
            self.inv_joint_mapping[onnx_i] = act_i
        
        self.xml_actuator_names = use_names
        
        print(f"[Sim2Sim] Joint mapping: {self.joint_mapping}")
    
    def _apply_physics_params(self) -> None:
        """Apply physics parameters from ONNX to MuJoCo model."""
        apply_onnx_physics_params(
            self.model,
            armature=self.onnx_config.armature,
            damping=self.onnx_config.damping,
            friction=self.onnx_config.friction,
        )
    
    def _setup_pd_gains(self) -> None:
        """Setup PD gains from ONNX config or XML."""
        # Get from ONNX config
        if self.onnx_config.joint_stiffness:
            kp_onnx = np.array(self.onnx_config.joint_stiffness)
            # Map to MuJoCo actuator order
            self.kp = kp_onnx[self.joint_mapping]
        else:
            # Default gains
            self.kp = np.ones(self.num_actions) * 100.0
        
        if self.onnx_config.joint_damping:
            kd_onnx = np.array(self.onnx_config.joint_damping)
            self.kd = kd_onnx[self.joint_mapping]
        else:
            self.kd = np.ones(self.num_actions) * 10.0
        # tau_limits already set from model.jnt_actfrcrange in _setup_actuator_state_mapping()
        
        # Action scale and offset
        if self.onnx_config.action_scale:
            scale_onnx = np.array(self.onnx_config.action_scale)
            self.action_scale = scale_onnx[self.joint_mapping]
        else:
            self.action_scale = np.ones(self.num_actions) * 0.25

        if getattr(self.onnx_config, "action_offset", None):
            offset_onnx = np.array(self.onnx_config.action_offset, dtype=np.float32).reshape(-1)
            if offset_onnx.shape[0] != self.onnx_action_dim:
                # Best-effort: pad/truncate
                if offset_onnx.shape[0] < self.onnx_action_dim:
                    offset_onnx = np.concatenate(
                        [offset_onnx, np.zeros(self.onnx_action_dim - offset_onnx.shape[0], dtype=np.float32)]
                    )
                else:
                    offset_onnx = offset_onnx[: self.onnx_action_dim]
            self.action_offset = offset_onnx[self.joint_mapping]
        else:
            self.action_offset = np.zeros(self.num_actions, dtype=np.float32)
        
        if self.onnx_config.default_joint_pos:
            default_onnx = np.array(self.onnx_config.default_joint_pos)
            self.default_joint_pos = default_onnx[self.joint_mapping]
        else:
            # Best-effort fallback: use MuJoCo model's initial qpos for each actuator joint.
            self.default_joint_pos = np.array([self.data.qpos[a] for a in self._qpos_adrs], dtype=np.float32)

        # Also keep default joint pos in ONNX order for observation alignment.
        self.default_joint_pos_onnx = self.default_joint_pos[np.array(self.inv_joint_mapping, dtype=np.int64)].copy()
        
        print(f"[Sim2Sim] PD gains: Kp={self.kp[:3]}..., Kd={self.kd[:3]}...")
        print(f"[Sim2Sim] Action scale: {self.action_scale[:3]}...")
    
    @property
    def joint_pos(self) -> np.ndarray:
        """Get current joint positions in ONNX joint order."""
        q_act = np.array([self.data.qpos[a] for a in self._qpos_adrs], dtype=np.float32)
        return q_act[np.array(self.inv_joint_mapping, dtype=np.int64)].copy()

    @property
    def joint_pos_actuator(self) -> np.ndarray:
        """Get current joint positions in MuJoCo actuator order."""
        return np.array([self.data.qpos[a] for a in self._qpos_adrs], dtype=np.float32)
    
    @property
    def joint_vel(self) -> np.ndarray:
        """Get current joint velocities in ONNX joint order."""
        dq_act = np.array([self.data.qvel[a] for a in self._dof_adrs], dtype=np.float32)
        return dq_act[np.array(self.inv_joint_mapping, dtype=np.int64)].copy()

    @property
    def joint_vel_actuator(self) -> np.ndarray:
        """Get current joint velocities in MuJoCo actuator order."""
        return np.array([self.data.qvel[a] for a in self._dof_adrs], dtype=np.float32)
    
    @property
    def base_pos(self) -> np.ndarray:
        """Get base position [x, y, z]."""
        return self.data.qpos[:3].copy()
    
    @property
    def base_quat(self) -> np.ndarray:
        """Get base quaternion [w, x, y, z]."""
        return self.data.qpos[3:7].copy()
    
    @property
    def base_lin_vel(self) -> np.ndarray:
        """Get base linear velocity in world frame."""
        return self.data.qvel[:3].copy()
    
    @property
    def base_ang_vel(self) -> np.ndarray:
        """Get base angular velocity in world frame."""
        return self.data.qvel[3:6].copy()
    
    def reset(self, initial_state: dict[str, Any] | None = None) -> np.ndarray:
        """Reset simulation.
        
        Args:
            initial_state: Optional initial state dict with:
                - qpos: Full qpos
                - qvel: Full qvel
                - joint_pos: Just joint positions
                
        Returns:
            Initial observation
        """
        mujoco.mj_resetData(self.model, self.data)
        
        if initial_state:
            if "qpos" in initial_state:
                self.data.qpos[:] = initial_state["qpos"]
            if "qvel" in initial_state:
                self.data.qvel[:] = initial_state["qvel"]
            if "joint_pos" in initial_state:
                num_free_dof = 7 if self.model.nq > self.num_actions else 0
                self.data.qpos[num_free_dof:num_free_dof + self.num_actions] = initial_state["joint_pos"]
        else:
            # Default: set joints to default positions
            for i, qpos_adr in enumerate(self._qpos_adrs):
                self.data.qpos[qpos_adr] = float(self.default_joint_pos[i])

        # If we have a floating base and we're not explicitly overriding qpos, lift the root
        # to avoid starting with the feet embedded in a raised heightfield.
        try:
            has_floating_base = bool(self.model.nq > self.num_actions)
        except Exception:
            has_floating_base = False
        if has_floating_base and (not initial_state or "qpos" not in initial_state):
            dz = float(getattr(self, "spawn_root_z_offset", 0.0) or 0.0)
            if dz > 0.0:
                self.data.qpos[2] = float(self.data.qpos[2]) + dz
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Reset state
        self.episode_length = 0
        self.global_phase = 0.0
        self._last_action = np.zeros(self.onnx_action_dim, dtype=np.float32)
        
        # Reset policy hidden state
        self.policy.reset_hidden_state()
        
        # Reset observation builder
        self.obs_builder.reset()
        
        return self.build_observation()
    
    def build_observation(self) -> np.ndarray:
        """Build observation for policy.
        
        Override this in subclasses for task-specific observations.
        
        Returns:
            Observation array
        """
        return self.obs_builder.build(
            joint_pos=self.joint_pos,  # ONNX order
            joint_vel=self.joint_vel,  # ONNX order
            base_quat=self.base_quat,
            base_ang_vel=self.base_ang_vel,
            base_lin_vel=self.base_lin_vel,
            last_action=self._last_action,
            velocity_command=self.velocity_command,
            episode_length=self.episode_length,
            step_dt=self.policy_dt,
        )
    
    def step(self, action: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Step simulation with policy action.
        
        If action is None, queries policy with current observation.
        
        Args:
            action: Optional action to apply (if None, queries policy)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        # Get action from policy if not provided
        if action is None:
            obs = self.build_observation()
            action = self.policy(obs)

        action = np.array(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.onnx_action_dim:
            raise RuntimeError(
                f"[Sim2Sim] Policy action_dim mismatch: got {action.shape[0]} expected {self.onnx_action_dim}"
            )
        
        # Store action for observation (ONNX order)
        self._last_action = action.copy()

        # Reorder action to MuJoCo actuator order for control application
        action_act = action[np.array(self.joint_mapping, dtype=np.int64)]
        
        # Convert action to target positions
        target_q = self._action_to_target(action_act)
        
        # Step physics with decimation
        for _ in range(self.decimation):
            # Compute PD torques
            tau = self._compute_control(target_q)
            
            # Apply torques robustly: drive generalized forces directly at the joint dofs.
            # This avoids common Unitree XML conventions that clamp data.ctrl to [-1,1] with gear=1.
            self.data.ctrl[:] = 0.0
            self.data.qfrc_applied[:] = 0.0
            for i, dof_adr in enumerate(self._dof_adrs):
                self.data.qfrc_applied[dof_adr] = float(tau[i])
            
            # Step physics
            mujoco.mj_step(self.model, self.data)
        
        # Update state
        self.episode_length += 1
        self._update_phase()
        
        # Build observation
        obs = self.build_observation()
        
        # Info
        info = {
            "base_pos": self.base_pos.copy(),
            "base_quat": self.base_quat.copy(),
            "base_lin_vel": self.base_lin_vel.copy(),
            "joint_pos": self.joint_pos.copy(),  # ONNX order
        }
        
        return obs, info
    
    def _action_to_target(self, action: np.ndarray) -> np.ndarray:
        """Convert policy action to target joint positions.
        
        Implements IsaacLab/MJLab JointPositionAction semantics:
          processed_action = raw_action * scale + offset

        Notes:
        - In IsaacLab when `use_default_offset=True`, offset == default_joint_pos.
        - Our ONNX metadata exports BOTH `default_joint_pos` and `action_offset`, and they can be identical.
          Therefore, we must NOT add both (would double the default pose and destabilize standing).
        
        Args:
            action: Raw policy output
            
        Returns:
            Target joint positions
        """
        base = self.action_offset if np.any(self.action_offset) else self.default_joint_pos
        return base + action * self.action_scale
    
    def _compute_control(self, target_q: np.ndarray) -> np.ndarray:
        """Compute PD control torques.
        
        Args:
            target_q: Target joint positions
            
        Returns:
            Control torques
        """
        return pd_control(
            target_q=target_q,
            current_q=self.joint_pos_actuator,
            current_dq=self.joint_vel_actuator,
            kp=self.kp,
            kd=self.kd,
            tau_limits=self.tau_limits,
        )
    
    def _update_phase(self) -> None:
        """Update gait phase."""
        gait_period = 0.8  # Default, can be overridden
        delta_phase = self.policy_dt / gait_period
        self.global_phase = (self.global_phase + delta_phase) % 1.0
    
    def set_velocity_command(self, vx: float, vy: float, wz: float) -> None:
        """Set velocity command for locomotion.
        
        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s)
            wz: Yaw rate (rad/s)
        """
        self.velocity_command = np.array([vx, vy, wz])
    
    def run_episode(
        self,
        max_steps: int = 1000,
        render: bool = False,
        velocity_command: tuple[float, float, float] | None = None,
    ) -> dict:
        """Run a complete episode.
        
        Args:
            max_steps: Maximum number of policy steps
            render: Whether to render (opens viewer)
            velocity_command: Optional (vx, vy, wz) command
            
        Returns:
            Episode statistics
        """
        self.reset()
        
        if velocity_command:
            self.set_velocity_command(*velocity_command)
        
        if render:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            viewer = None
        
        episode_data = {
            "base_pos": [],
            "base_quat": [],
            "base_lin_vel": [],
            "joint_pos": [],
            "actions": [],
        }
        
        try:
            for step in range(max_steps):
                obs, info = self.step()
                
                # Store data
                episode_data["base_pos"].append(info["base_pos"])
                episode_data["base_quat"].append(info["base_quat"])
                episode_data["base_lin_vel"].append(info["base_lin_vel"])
                episode_data["joint_pos"].append(info["joint_pos"])
                episode_data["actions"].append(self._last_action.copy())
                
                # Check termination
                if self._check_termination():
                    break
                
                # Render
                if viewer is not None:
                    viewer.sync()
        finally:
            if viewer is not None:
                viewer.close()
        
        # Convert to arrays
        for key in episode_data:
            episode_data[key] = np.array(episode_data[key])
        
        # Compute statistics
        stats = {
            "num_steps": len(episode_data["base_pos"]),
            "survived": not self._check_termination(),
            "distance_traveled": np.linalg.norm(
                episode_data["base_pos"][-1, :2] - episode_data["base_pos"][0, :2]
            ) if len(episode_data["base_pos"]) > 1 else 0,
            "mean_velocity": np.mean(np.linalg.norm(episode_data["base_lin_vel"][:, :2], axis=1))
            if len(episode_data["base_lin_vel"]) > 0 else 0,
        }
        
        return {"data": episode_data, "stats": stats}

    def run_episodes_continuous(
        self,
        num_episodes: int = 10,
        max_steps_per_episode: int = 1000,
        velocity_command: tuple[float, float, float] | None = None,
        render: bool = True,
    ) -> list[dict]:
        """Run multiple episodes in a single viewer session (continuous playback).

        This is intended for visualization: the MuJoCo viewer window stays open
        while episodes reset internally upon termination or reaching max steps.
        """
        if not render:
            # If not rendering, just fall back to standard per-episode execution.
            return [
                self.run_episode(
                    max_steps=max_steps_per_episode,
                    render=False,
                    velocity_command=velocity_command,
                )
                for _ in range(num_episodes)
            ]

        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        results: list[dict] = []
        try:
            for _ep in range(num_episodes):
                self.reset()
                if velocity_command:
                    self.set_velocity_command(*velocity_command)

                episode_data = {
                    "base_pos": [],
                    "base_quat": [],
                    "base_lin_vel": [],
                    "joint_pos": [],
                    "actions": [],
                }

                for _step in range(max_steps_per_episode):
                    _obs, info = self.step()

                    episode_data["base_pos"].append(info["base_pos"])
                    episode_data["base_quat"].append(info["base_quat"])
                    episode_data["base_lin_vel"].append(info["base_lin_vel"])
                    episode_data["joint_pos"].append(info["joint_pos"])
                    episode_data["actions"].append(self._last_action.copy())

                    viewer.sync()

                    if self._check_termination():
                        break

                for key in episode_data:
                    episode_data[key] = np.array(episode_data[key])

                stats = {
                    "num_steps": len(episode_data["base_pos"]),
                    "survived": not self._check_termination(),
                    "distance_traveled": np.linalg.norm(
                        episode_data["base_pos"][-1, :2] - episode_data["base_pos"][0, :2]
                    )
                    if len(episode_data["base_pos"]) > 1
                    else 0,
                    "mean_velocity": np.mean(
                        np.linalg.norm(episode_data["base_lin_vel"][:, :2], axis=1)
                    )
                    if len(episode_data["base_lin_vel"]) > 0
                    else 0,
                }
                results.append({"data": episode_data, "stats": stats})
        finally:
            viewer.close()

        return results

    def run_forever_until_closed(
        self,
        max_steps_per_episode: int = 1000,
        velocity_command: tuple[float, float, float] | None = None,
    ) -> None:
        """Continuously run episodes in a single MuJoCo viewer until the user closes it.

        - Opens one viewer window
        - Runs policy steps and syncs the viewer every step
        - When termination happens or max steps reached, resets and continues
        - Exits cleanly when the viewer window is closed by the user
        """
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        try:
            while viewer.is_running():
                # Start a new episode
                self.reset()
                if velocity_command:
                    self.set_velocity_command(*velocity_command)

                for _step in range(max_steps_per_episode):
                    # Viewer might get closed during episode
                    if not viewer.is_running():
                        return

                    self.step()
                    viewer.sync()

                    if self._check_termination():
                        break
        finally:
            viewer.close()
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate.
        
        Override in subclasses for task-specific termination.
        
        Returns:
            True if episode should end
        """
        # Default: check for bad orientation
        quat = self.base_quat
        # Compute up vector in world frame
        R = self._quat_to_rotation_matrix(quat)
        up_world = R @ np.array([0, 0, 1])
        
        # Check angle from vertical
        cos_angle = up_world[2]
        if cos_angle < 0.5:  # > ~60 degrees tilt
            return True
        
        # Check height
        if self.base_pos[2] < 0.1:
            return True
        
        return False
    
    @staticmethod
    def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [w,x,y,z] to rotation matrix."""
        w, x, y, z = quat
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
