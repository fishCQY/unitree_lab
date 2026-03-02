"""Concrete locomotion simulator for sim2sim.

This implements the abstract `BaseMujocoSimulator` API used by the MuJoCo
sim2sim runner in `scripts/mujoco_eval/simulator.py`.

It intentionally avoids optional dependencies like SciPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import mujoco

from .base_simulator import BaseMujocoSimulator
from ..core.physics import quat_rotate_inverse


def _quat_rotate_inverse_np(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Alias matching the old math_utils API expected by this module."""
    return quat_rotate_inverse(quat, vec)


def _model_to_mujoco(arr: np.ndarray, mapping_index: np.ndarray) -> np.ndarray:
    """Reorder an ONNX-order array into MuJoCo actuator order via mapping_index."""
    return np.asarray(arr, dtype=np.float64).reshape(-1)[np.asarray(mapping_index, dtype=np.int64)]


class LocomotionMujocoSimulator(BaseMujocoSimulator):
    """Locomotion policy simulator with velocity-command observations."""

    def __init__(
        self,
        *,
        onnx_path: str,
        xml_path: str,
        config_override: dict[str, Any] | None = None,
    ):
        # Initialize with Base class (loads ONNX + MuJoCo model)
        super().__init__(onnx_path=onnx_path, mujoco_model_path=xml_path)

        # Ensure mapping index is integer (some ONNX exports store joint_names as np arrays).
        try:
            self.joint_mapping_index = np.asarray(self.joint_mapping_index, dtype=np.int64)
        except Exception:
            self.joint_mapping_index = np.array(self.joint_mapping_index, dtype=np.int64)

        # Ensure wheel mask is boolean for NumPy indexing.
        try:
            self.is_wheel_mask = np.asarray(self.is_wheel_mask, dtype=bool)
        except Exception:
            pass

        # Apply config overrides (deploy.yaml / CLI JSON)
        if config_override:
            for k, v in config_override.items():
                self.onnx_config[k] = v

            # Refresh commonly used fields if they were overridden
            if "action_scale" in config_override:
                self.action_scale = np.array(self.onnx_config["action_scale"], dtype=np.float64)
            if "default_joint_pos" in config_override:
                self.default_joint_pos = np.array(self.onnx_config["default_joint_pos"], dtype=np.float64)
            if "joint_stiffness" in config_override:
                p_gains_seq = np.array(self.onnx_config["joint_stiffness"], dtype=np.float64)
                self.p_gains = _model_to_mujoco(p_gains_seq, self.joint_mapping_index)
            if "joint_damping" in config_override:
                d_gains_seq = np.array(self.onnx_config["joint_damping"], dtype=np.float64)
                self.d_gains = _model_to_mujoco(d_gains_seq, self.joint_mapping_index)

        # Timing overrides (policy_dt = sim_dt * decimation)
        self.sim_dt = float(self.onnx_config.get("sim_dt", self.sim_dt))
        self.decimation = int(self.onnx_config.get("decimation", self.decimation))
        self.policy_dt = float(self.sim_dt * max(1, self.decimation))
        self.model.opt.timestep = float(self.sim_dt)

        # Build actuator state mapping (include-safe) and ONNX<->MuJoCo joint mapping.
        # `parse_actuators_from_xml()` is not include-aware, so the base class may
        # have an empty `joint_xml` list for scene XMLs. We always derive mapping
        # from the loaded MuJoCo model.
        self._setup_actuator_state_mapping_and_joint_mapping()

        # Action offset (ONNX order) if provided by deploy.yaml / metadata
        self.action_offset_onnx = np.asarray(self.onnx_config.get("action_offset", [0.0] * self.onnx_action_dim), dtype=np.float64).reshape(-1)
        if self.action_offset_onnx.shape[0] != self.onnx_action_dim:
            if self.action_offset_onnx.shape[0] < self.onnx_action_dim:
                self.action_offset_onnx = np.pad(self.action_offset_onnx, (0, self.onnx_action_dim - self.action_offset_onnx.shape[0]))
            else:
                self.action_offset_onnx = self.action_offset_onnx[: self.onnx_action_dim]
        self.action_offset = self.action_offset_onnx[self.joint_mapping_index]

        # Make solver/contact a bit more IsaacLab-like for stability.
        self._configure_solver_and_contacts()

        # Runner-set spawn lift for heightfields
        self.spawn_root_z_offset: float = 0.0

        # Velocity command (vx, vy, wz)
        self.velocity_command = np.zeros(3, dtype=np.float32)

        # Track last action in ONNX order (fed back into observations)
        self._last_action = np.zeros(int(self.onnx_action_dim), dtype=np.float32)

    def _configure_solver_and_contacts(self) -> None:
        """Best-effort physics tuning for standing stability on hfields."""
        try:
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        except Exception:
            pass
        try:
            self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        except Exception:
            pass
        try:
            self.model.opt.iterations = 50
            self.model.opt.ls_iterations = 20
            self.model.opt.noslip_iterations = 8
        except Exception:
            pass

        # Harden contacts a bit (floor/terrain + robot collidable geoms)
        try:
            hard_solref = np.array([0.002, 1.0], dtype=np.float64)
            hard_solimp = np.array([0.99, 0.999, 0.0001, 0.5, 2.0], dtype=np.float64)
            for gid in range(int(self.model.ngeom)):
                # Heightfield / floor / robot collision geoms
                if int(self.model.geom_bodyid[gid]) > 0 or int(self.model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_HFIELD):
                    self.model.geom_solref[gid] = hard_solref
                    self.model.geom_solimp[gid] = hard_solimp
                    self.model.geom_friction[gid] = [1.0, 0.005, 0.0001]
        except Exception:
            pass

    def _setup_actuator_state_mapping_and_joint_mapping(self) -> None:
        """Derive actuator->(qpos,qvel) addresses and map ONNX joint order."""
        def _norm(name: str) -> str:
            n = str(name).strip().lower()
            for suf in ("_joint", "joint"):
                if n.endswith(suf):
                    n = n[: -len(suf)]
            return n

        # Actuator names from loaded model (include-safe)
        nu = int(self.model.nu)
        act_names: list[str] = []
        qpos_adrs: list[int] = []
        dof_adrs: list[int] = []
        for act_id in range(nu):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
            act_names.append(n or f"actuator_{act_id}")
            jid = int(self.model.actuator_trnid[act_id, 0])
            qpos_adrs.append(int(self.model.jnt_qposadr[jid]))
            dof_adrs.append(int(self.model.jnt_dofadr[jid]))
        self.actuator_names = act_names
        self._qpos_adrs = np.asarray(qpos_adrs, dtype=np.int64)
        self._dof_adrs = np.asarray(dof_adrs, dtype=np.int64)

        # ONNX joint names
        self.joint_names = list(self.onnx_config.get("joint_names", []))
        if not self.joint_names:
            # Fallback: assume actuator order equals policy order
            self.joint_names = list(self.actuator_names)

        onnx_index = {_norm(nm): i for i, nm in enumerate(self.joint_names)}

        mapping: list[int] = []
        for act_nm in self.actuator_names:
            key = _norm(act_nm)
            if key in onnx_index:
                mapping.append(int(onnx_index[key]))
                continue
            # Common export uses *_joint suffix in ONNX
            key2 = _norm(f"{act_nm}_joint")
            if key2 in onnx_index:
                mapping.append(int(onnx_index[key2]))
                continue
            # Fallback: identity if sizes match
            if len(self.joint_names) == len(self.actuator_names):
                mapping.append(len(mapping))
            else:
                mapping.append(0)

        self.joint_mapping_index = np.asarray(mapping, dtype=np.int64)  # actuator -> onnx
        self.onnx_action_dim = int(len(self.joint_names))

        # Inverse mapping: onnx -> actuator (best-effort)
        inv = np.zeros((self.onnx_action_dim,), dtype=np.int64)
        for act_i, onnx_i in enumerate(self.joint_mapping_index.tolist()):
            if 0 <= int(onnx_i) < self.onnx_action_dim:
                inv[int(onnx_i)] = int(act_i)
        self.inv_joint_mapping = inv

        # Wheel mask (actuator order)
        self.is_wheel_mask = np.asarray(["wheel" in str(n).lower() for n in self.actuator_names], dtype=bool)
        self.is_legged = not bool(np.any(self.is_wheel_mask))

        # Remap control params into actuator order
        self.action_scale_onnx = np.asarray(
            self.onnx_config.get("action_scale", [1.0] * self.onnx_action_dim), dtype=np.float64
        ).reshape(-1)
        self.default_joint_pos_onnx = np.asarray(
            self.onnx_config.get("default_joint_pos", [0.0] * self.onnx_action_dim), dtype=np.float64
        ).reshape(-1)
        self.action_scale = self.action_scale_onnx[self.joint_mapping_index]
        self.default_joint_pos = self.default_joint_pos_onnx[self.joint_mapping_index]

        p_seq = np.asarray(
            self.onnx_config.get("joint_stiffness", [100.0] * self.onnx_action_dim), dtype=np.float64
        ).reshape(-1)
        d_seq = np.asarray(
            self.onnx_config.get("joint_damping", [5.0] * self.onnx_action_dim), dtype=np.float64
        ).reshape(-1)
        self.p_gains = p_seq[self.joint_mapping_index]
        self.d_gains = d_seq[self.joint_mapping_index]

    def set_velocity_command(self, vx: float, vy: float, wz: float) -> None:
        self.velocity_command[:] = (float(vx), float(vy), float(wz))

    @property
    def base_pos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[:3], dtype=np.float64)

    @property
    def base_quat(self) -> np.ndarray:
        return np.asarray(self.data.qpos[3:7], dtype=np.float64)

    def _projected_gravity(self) -> np.ndarray:
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return _quat_rotate_inverse_np(self.base_quat.astype(np.float64), g_world).astype(np.float32)

    def _check_termination(self) -> bool:
        """Simple IsaacLab-like termination (fallen/too low)."""
        # Height check
        if float(self.base_pos[2]) < 0.12:
            return True
        # Orientation check (same criterion as evaluation.metrics.is_fallen)
        proj_g = self._projected_gravity()
        proj_grav_z = float(np.clip(proj_g[2], -1.0, 1.0))
        if float(np.arccos(-proj_grav_z)) > 1.0:
            return True
        return False

    def reset(self) -> np.ndarray:
        """Reset MuJoCo state and return initial observation."""
        mujoco.mj_resetData(self.model, self.data)

        # Default pose: set actuated joints (actuator order) to defaults.
        try:
            self.data.qpos[self._qpos_adrs] = self.default_joint_pos
            self.data.qvel[self._dof_adrs] = 0.0
        except Exception:
            pass

        # Spawn lift (heightfield / mixed terrains)
        if self.model.nq >= 3 and float(self.spawn_root_z_offset) > 0.0:
            self.data.qpos[2] = float(self.data.qpos[2]) + float(self.spawn_root_z_offset)

        mujoco.mj_forward(self.model, self.data)

        # Reset recurrent hidden state
        self.reset_hidden_states()
        self._last_action.fill(0.0)

        obs, _ = self.build_observation(command=self.velocity_command)
        return obs

    def build_observation(self, **kwargs) -> tuple[np.ndarray, np.ndarray | None]:
        """Build observation vector expected by the policy.

        We implement the common 96-dim locomotion observation:
          base_ang_vel (3)
          projected_gravity (3)
          velocity_commands (3)
          joint_pos (num_actions)
          joint_vel (num_actions)
          actions (num_actions)
        """
        command = kwargs.get("command", None)
        if command is None:
            command = self.velocity_command
        command = np.asarray(command, dtype=np.float32).reshape(3)

        # Base quaternion in MuJoCo is [w, x, y, z]
        base_quat = np.asarray(self.data.qpos[3:7], dtype=np.float64)

        # MuJoCo freejoint qvel: [lin_vel(3), ang_vel(3)] but ang_vel is in local frame.
        base_ang_vel = np.asarray(self.data.qvel[3:6], dtype=np.float32)

        # Project gravity into body frame (body sees +Z up; training often uses -Z gravity)
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        proj_g = _quat_rotate_inverse_np(base_quat.astype(np.float64), g_world.astype(np.float64)).astype(np.float32)

        # Joint state in actuator order
        q_act = np.asarray(self.data.qpos[self._qpos_adrs], dtype=np.float32)
        dq_act = np.asarray(self.data.qvel[self._dof_adrs], dtype=np.float32)

        # Scatter into ONNX order
        q = np.zeros((self.onnx_action_dim,), dtype=np.float32)
        dq = np.zeros((self.onnx_action_dim,), dtype=np.float32)
        for act_i, onnx_i in enumerate(self.joint_mapping_index.tolist()):
            if 0 <= int(onnx_i) < self.onnx_action_dim:
                q[int(onnx_i)] = float(q_act[act_i])
                dq[int(onnx_i)] = float(dq_act[act_i])

        # Match IsaacLab scaling defaults (if provided in ONNX metadata, prefer them)
        obs_scales = self.onnx_config.get("observation_scales", {}) or {}
        s_ang = float(obs_scales.get("base_ang_vel", 0.25))
        s_dof_pos = float(obs_scales.get("joint_pos", 1.0))
        s_dof_vel = float(obs_scales.get("joint_vel", 0.05))
        s_cmd = float(obs_scales.get("velocity_commands", 1.0))

        # Default joint positions in ONNX order
        q0 = np.asarray(self.default_joint_pos_onnx, dtype=np.float32).reshape(-1)
        if q0.shape[0] != q.shape[0]:
            q0 = np.zeros_like(q)

        # Action feedback in ONNX order
        last_a = self._last_action
        if last_a.shape[0] != self.onnx_action_dim:
            last_a = np.zeros((self.onnx_action_dim,), dtype=np.float32)

        obs_parts = [
            (base_ang_vel * s_ang).reshape(-1),
            proj_g.reshape(-1),
            (command * s_cmd).reshape(-1),
            ((q - q0) * s_dof_pos).reshape(-1),
            (dq * s_dof_vel).reshape(-1),
            last_a.reshape(-1),
        ]
        obs = np.concatenate(obs_parts, axis=0).astype(np.float32)

        # Pad / truncate to expected dim
        exp = int(self.expected_obs_dim)
        if obs.shape[0] < exp:
            obs = np.concatenate([obs, np.zeros(exp - obs.shape[0], dtype=np.float32)], axis=0)
        elif obs.shape[0] > exp:
            obs = obs[:exp]

        return obs, None

    def step(self) -> tuple[np.ndarray, dict]:
        """One policy step: build obs -> infer -> apply control -> step physics."""
        obs, extero = self.build_observation(command=self.velocity_command)
        action_onnx = np.asarray(self.step_inference(obs, extero), dtype=np.float32).reshape(-1)
        if action_onnx.shape[0] != self.onnx_action_dim:
            action_onnx = action_onnx[: self.onnx_action_dim] if action_onnx.shape[0] > self.onnx_action_dim else np.pad(
                action_onnx, (0, self.onnx_action_dim - action_onnx.shape[0])
            )
        self._last_action[:] = action_onnx

        # Map ONNX action -> actuator order and apply PD control
        action_act = action_onnx[self.joint_mapping_index]
        self._step_control_actuator_order(action_act)

        for _ in range(max(1, int(self.decimation))):
            self.step_physics()

        # Return next obs (like common gym step), plus minimal info
        next_obs, _ = self.build_observation(command=self.velocity_command)
        info = {}
        return next_obs, info

    def run_episode(
        self,
        *,
        max_steps: int,
        render: bool = False,
        velocity_command: tuple[float, float, float] = (0.5, 0.0, 0.0),
    ) -> dict[str, float | int | bool]:
        """Run one episode headless and return lightweight metrics dict."""
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

            # Actual velocity (world linear + approximate yaw rate)
            vxy = np.asarray(self.data.qvel[0:2], dtype=np.float64)
            wz = float(self.data.qvel[5]) if self.data.qvel.shape[0] >= 6 else 0.0
            actual = np.array([float(vxy[0]), float(vxy[1]), wz], dtype=np.float64)
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

    def _step_control_actuator_order(self, action_act: np.ndarray) -> None:
        """Apply PD control in actuator order, using per-actuator joint state addresses."""
        action_act = np.asarray(action_act, dtype=np.float64).reshape(-1)
        if action_act.shape[0] != int(self.model.nu):
            # Best-effort resize
            action_act = action_act[: int(self.model.nu)] if action_act.size > int(self.model.nu) else np.pad(
                action_act, (0, int(self.model.nu) - action_act.size)
            )

        # JointPositionAction semantics: q_target = q_default + (scale * a + offset)
        current_action_scaled = action_act * self.action_scale + self.action_offset
        target_q = current_action_scaled + self.default_joint_pos
        target_dq = np.zeros_like(target_q)

        # Wheels: velocity control
        if self.is_wheel_mask.shape[0] == target_q.shape[0] and bool(np.any(self.is_wheel_mask)):
            target_q[self.is_wheel_mask] = 0.0
            target_dq[self.is_wheel_mask] = current_action_scaled[self.is_wheel_mask]

        q = np.asarray(self.data.qpos[self._qpos_adrs], dtype=np.float64)
        dq = np.asarray(self.data.qvel[self._dof_adrs], dtype=np.float64)

        from ..core.physics import pd_control

        torque, _ = pd_control(target_q, q, self.p_gains, target_dq, dq, self.d_gains, self.tau_limit)
        self.data.ctrl[:] = torque

