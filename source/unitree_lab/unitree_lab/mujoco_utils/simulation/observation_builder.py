"""Observation builder for MuJoCo sim2sim.

This module constructs observations matching IsaacLab semantics:
1. Parse observation structure from ONNX metadata
2. Compute each observation term with correct scale/clip
3. Maintain history buffers if needed
4. Handle exteroception (height scan)

Key alignment points:
- Term order must match training exactly
- Scale and clip per term
- History stacking (if used)
- Height scan grid alignment
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from ..core.onnx_utils import OnnxConfig
from ..core.physics import compute_projected_gravity, compute_base_ang_vel_body


class ObservationBuilder:
    """Builder for observations matching IsaacLab semantics.
    
    Constructs observations by:
    1. Computing each term from state
    2. Applying per-term scale
    3. Applying per-term clip
    4. Concatenating in correct order
    """
    
    # Standard observation term names
    SUPPORTED_TERMS = [
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_pos_rel",
        "joint_vel",
        "joint_vel_rel",
        "last_action",
        "gait_phase",
        "height_scan",
        "height_scan_safe",
        "base_lin_vel",
    ]
    
    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        onnx_config: OnnxConfig,
        joint_mapping: list[int],
        height_scanner: Any | None = None,
    ):
        """Initialize observation builder.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            onnx_config: ONNX configuration with obs structure
            joint_mapping: Mapping from XML actuator to ONNX joint order
            height_scanner: Optional height scanner for exteroception
        """
        self.model = model
        self.data = data
        self.config = onnx_config
        self.joint_mapping = joint_mapping
        self.height_scanner = height_scanner

        # Canonical ordering for sim2sim:
        # - All observations passed to the ONNX policy must be in ONNX joint order (metadata joint_names order).
        # - The MuJoCo actuator order may differ; mapping is handled by BaseMujocoSimulator at control-application time.
        # Therefore, ObservationBuilder assumes `joint_pos/joint_vel/last_action/default_joint_pos` are already in ONNX order.
        self.num_actions = int(onnx_config.output_dim) if int(onnx_config.output_dim or 0) > 0 else 0
        if self.num_actions <= 0:
            self.num_actions = len(onnx_config.joint_names) if onnx_config.joint_names else len(joint_mapping)
        
        # Parse observation structure (from ONNX metadata if available)
        self.obs_names = onnx_config.observation_names or []
        self.obs_dims = onnx_config.observation_dims or []
        self.obs_scales = onnx_config.observation_scales or {}

        # History stacking configuration
        self.history_length = onnx_config.history_length or 1
        self.single_frame_dims = onnx_config.single_frame_dims or {}
        self.history_newest_first = bool(getattr(onnx_config, "history_newest_first", True))

        # If metadata is missing, try best-effort inference from model IO dims.
        # This avoids hard-crashes like: Got 96 Expected 490.
        if not self.obs_names and onnx_config.input_dim and onnx_config.output_dim:
            inferred = self._infer_obs_structure_from_io(
                input_dim=int(onnx_config.input_dim),
                num_actions=int(onnx_config.output_dim),
            )
            if inferred is not None:
                self.obs_names = inferred["observation_names"]
                self.obs_dims = inferred["observation_dims"]
                self.history_length = inferred["history_length"]
                self.single_frame_dims = inferred["single_frame_dims"]
                # Only fill default scales if none were provided by metadata.
                if not self.obs_scales:
                    self.obs_scales = inferred.get("observation_scales", {})
                print(
                    "[ObservationBuilder] Inferred observation structure from ONNX IO dims: "
                    f"input_dim={onnx_config.input_dim}, num_actions={onnx_config.output_dim}, "
                    f"history_length={self.history_length}, terms={self.obs_names}"
                )
        
        # Default joint positions for relative observations (ONNX order)
        if onnx_config.default_joint_pos:
            default = np.array(onnx_config.default_joint_pos, dtype=np.float32).reshape(-1)
            if default.shape[0] != self.num_actions:
                # Best-effort: pad/truncate to match action dimension
                if default.shape[0] < self.num_actions:
                    default = np.concatenate([default, np.zeros(self.num_actions - default.shape[0], dtype=np.float32)])
                else:
                    default = default[: self.num_actions]
            self.default_joint_pos = default
        else:
            self.default_joint_pos = np.zeros(self.num_actions, dtype=np.float32)
        
        # History buffers for temporal stacking
        self._history_buffers: dict[str, list[np.ndarray]] = {}
        
        # Cached values
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._global_phase = 0.0
        self._gait_period = 0.8
        
        # Track if we've printed breakdown
        self._printed_breakdown = False
        
        # Initialize history buffers if history stacking is enabled
        if self.history_length > 1:
            print(f"[ObservationBuilder] History stacking enabled: {self.history_length} frames")

    @staticmethod
    def _infer_obs_structure_from_io(input_dim: int, num_actions: int) -> dict[str, Any] | None:
        """Infer common locomotion observation layouts from ONNX IO dimensions.

        This is a best-effort fallback when ONNX metadata is missing.
        Supports typical IsaacLab locomotion terms with optional temporal stacking:
          - base_ang_vel (3)
          - projected_gravity (3)
          - velocity_commands (3)
          - joint_pos_rel (num_actions)
          - joint_vel_rel (num_actions)
          - last_action (num_actions)
          - gait_phase (2) [optional]
        """
        if input_dim <= 0 or num_actions <= 0:
            return None

        # Per-frame dims
        per_frame_no_phase = 3 + 3 + 3 + num_actions + num_actions + num_actions
        per_frame_with_phase = per_frame_no_phase + 2

        def _stacked_dims(h: int, with_phase: bool) -> tuple[list[str], list[int], dict[str, int]]:
            if with_phase:
                names = [
                    "base_ang_vel",
                    "projected_gravity",
                    "velocity_commands",
                    "joint_pos_rel",
                    "joint_vel_rel",
                    "last_action",
                    "gait_phase",
                ]
                dims = [
                    3 * h,
                    3 * h,
                    3 * h,
                    num_actions * h,
                    num_actions * h,
                    num_actions * h,
                    2 * h,
                ]
                single = {
                    "base_ang_vel": 3,
                    "projected_gravity": 3,
                    "velocity_commands": 3,
                    "joint_pos_rel": num_actions,
                    "joint_vel_rel": num_actions,
                    "last_action": num_actions,
                    "gait_phase": 2,
                }
            else:
                names = [
                    "base_ang_vel",
                    "projected_gravity",
                    "velocity_commands",
                    "joint_pos_rel",
                    "joint_vel_rel",
                    "last_action",
                ]
                dims = [
                    3 * h,
                    3 * h,
                    3 * h,
                    num_actions * h,
                    num_actions * h,
                    num_actions * h,
                ]
                single = {
                    "base_ang_vel": 3,
                    "projected_gravity": 3,
                    "velocity_commands": 3,
                    "joint_pos_rel": num_actions,
                    "joint_vel_rel": num_actions,
                    "last_action": num_actions,
                }
            return names, dims, single

        # Prefer layouts that include gait_phase if divisible.
        if input_dim % per_frame_with_phase == 0:
            h = input_dim // per_frame_with_phase
            # Guardrail: avoid absurd history lengths.
            if 1 <= h <= 20:
                names, dims, single = _stacked_dims(h, with_phase=True)
                return {
                    "observation_names": names,
                    "observation_dims": dims,
                    "history_length": h,
                    "single_frame_dims": single,
                    # IMPORTANT: don't assume IsaacLab-style scaling when metadata is missing.
                    # Some exported policies (e.g. mjlab official deploy policies) include their own
                    # normalizer and expect *unscaled* raw observations.
                    "observation_scales": {},
                }

        if input_dim % per_frame_no_phase == 0:
            h = input_dim // per_frame_no_phase
            if 1 <= h <= 20:
                names, dims, single = _stacked_dims(h, with_phase=False)
                return {
                    "observation_names": names,
                    "observation_dims": dims,
                    "history_length": h,
                    "single_frame_dims": single,
                    "observation_scales": {},
                }

        return None
    
    def reset(self) -> None:
        """Reset observation history."""
        self._history_buffers.clear()
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._global_phase = 0.0
        # Do NOT prefill history with zeros.
        # IsaacLab-style frame stacking typically repeats the first valid observation at reset
        # (or uses the last observation from the previous step), not an all-zero history.
        # We lazily initialize each term's history on first `build()` with the current term value.
    
    def build(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_quat: np.ndarray,
        base_ang_vel: np.ndarray,
        base_lin_vel: np.ndarray | None = None,
        last_action: np.ndarray | None = None,
        velocity_command: np.ndarray | None = None,
        episode_length: int = 0,
        step_dt: float = 0.02,
        height_data: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build complete observation.
        
        Args:
            joint_pos: Current joint positions
            joint_vel: Current joint velocities
            base_quat: Base quaternion [w, x, y, z]
            base_ang_vel: Base angular velocity in world frame
            base_lin_vel: Base linear velocity in world frame
            last_action: Last policy action
            velocity_command: Velocity command [vx, vy, wz]
            episode_length: Current episode step
            step_dt: Policy time step
            height_data: Optional height scan data
            
        Returns:
            Concatenated observation array
        """
        if last_action is not None:
            la = np.array(last_action, dtype=np.float32).reshape(-1)
            if la.shape[0] != self.num_actions:
                # Best-effort: pad/truncate to match ONNX action dim
                if la.shape[0] < self.num_actions:
                    la = np.concatenate([la, np.zeros(self.num_actions - la.shape[0], dtype=np.float32)])
                else:
                    la = la[: self.num_actions]
            self._last_action = la
        
        if velocity_command is None:
            velocity_command = np.zeros(3)
        
        if base_lin_vel is None:
            base_lin_vel = np.zeros(3)
        
        # Update phase
        delta_phase = step_dt / self._gait_period
        self._global_phase = (self._global_phase + delta_phase) % 1.0
        
        # Build observation by term order
        if self.obs_names:
            obs_parts = []
            breakdown = []
            
            for i, term_name in enumerate(self.obs_names):
                # Compute current frame observation
                term_obs = self._compute_term(
                    term_name=term_name,
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    base_quat=base_quat,
                    base_ang_vel=base_ang_vel,
                    base_lin_vel=base_lin_vel,
                    velocity_command=velocity_command,
                    height_data=height_data,
                )
                
                # Get expected dim (with history)
                expected_dim = self.obs_dims[i] if i < len(self.obs_dims) else len(term_obs)
                
                # Handle history stacking if enabled
                if self.history_length > 1:
                    # Lazy-init: fill history with the current term value (repeat) on first use.
                    history = self._history_buffers.get(term_name)
                    if history is None or len(history) != self.history_length:
                        self._history_buffers[term_name] = [term_obs.copy() for _ in range(self.history_length)]
                        history = self._history_buffers[term_name]
                    else:
                        # Push new observation to history (FIFO)
                        history.pop(0)  # Remove oldest
                        history.append(term_obs.copy())  # Add newest

                    # Allow toggling history stack order for sim2sim debugging.
                    term_obs = (
                        np.concatenate(history[::-1]) if self.history_newest_first else np.concatenate(history)
                    )
                
                # Pad or truncate if needed
                if len(term_obs) != expected_dim:
                    if len(term_obs) < expected_dim:
                        term_obs = np.concatenate([term_obs, np.zeros(expected_dim - len(term_obs))])
                    else:
                        term_obs = term_obs[:expected_dim]
                
                obs_parts.append(term_obs)
                breakdown.append((term_name, len(term_obs)))
            
            # Print breakdown once
            if not self._printed_breakdown:
                print("[ObservationBuilder] Observation breakdown:")
                total = 0
                for name, dim in breakdown:
                    print(f"  - {name}: {dim}")
                    total += dim
                print(f"  Total: {total} (history_length={self.history_length})")
                self._printed_breakdown = True
            
            obs = np.concatenate(obs_parts)
        else:
            # Default observation structure (matches common locomotion)
            obs = self._build_default_obs(
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                base_quat=base_quat,
                base_ang_vel=base_ang_vel,
                velocity_command=velocity_command,
            )
        
        return obs.astype(np.float32)
    
    def _compute_term(
        self,
        term_name: str,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_quat: np.ndarray,
        base_ang_vel: np.ndarray,
        base_lin_vel: np.ndarray,
        velocity_command: np.ndarray,
        height_data: np.ndarray | None,
    ) -> np.ndarray:
        """Compute a single observation term.
        
        Args:
            term_name: Name of observation term
            ... : State variables
            
        Returns:
            Observation term array
        """
        # Get scale for this term
        scale = self.obs_scales.get(term_name, 1.0)
        if isinstance(scale, list):
            scale = np.array(scale)
        
        # Compute term
        if term_name == "base_ang_vel":
            # Angular velocity in body frame.
            #
            # IMPORTANT (MuJoCo convention):
            # For a free joint, MuJoCo stores the 3 angular velocity components in qvel[3:6]
            # in the *local/body frame* (not world frame). Therefore, do NOT rotate it again.
            #
            # If in some integration you provide world-frame angular velocity here, you may
            # want to switch back to: compute_base_ang_vel_body(base_quat, base_ang_vel).
            obs = base_ang_vel.copy()
            obs = obs * scale
            
        elif term_name == "projected_gravity":
            # Gravity projection in body frame
            obs = compute_projected_gravity(base_quat)
            obs = obs * scale
            
        elif term_name == "velocity_commands":
            # Velocity command [vx, vy, wz]
            obs = velocity_command.copy()
            
        elif term_name == "joint_pos":
            # Absolute joint positions
            obs = joint_pos.copy()
            obs = obs * scale
            
        elif term_name == "joint_pos_rel":
            # Relative joint positions (w.r.t. default)
            obs = joint_pos - self.default_joint_pos
            obs = obs * scale
            
        elif term_name == "joint_vel" or term_name == "joint_vel_rel":
            # Joint velocities
            obs = joint_vel.copy()
            obs = obs * scale
            
        elif term_name == "last_action":
            # Last policy action
            obs = self._last_action.copy()
            
        elif term_name == "gait_phase":
            # Gait phase as sin/cos
            obs = np.array([
                np.sin(self._global_phase * 2 * np.pi),
                np.cos(self._global_phase * 2 * np.pi),
            ])
            
        elif term_name in ["height_scan", "height_scan_safe"]:
            # Height scan data
            if height_data is not None:
                obs = height_data.copy()
            elif self.height_scanner is not None:
                obs = self.height_scanner.scan()
            else:
                # Return zeros if no height scanner
                expected_size = self.config.height_scan_size
                if expected_size:
                    # Compute expected number of points
                    resolution = self.config.height_scan_resolution
                    nx = int(expected_size[0] / resolution) + 1
                    ny = int(expected_size[1] / resolution) + 1
                    obs = np.zeros(nx * ny)
                else:
                    obs = np.zeros(1)
            obs = obs * scale
            
        elif term_name == "base_lin_vel":
            # Linear velocity in body frame
            from ..core.physics import compute_base_lin_vel_body
            obs = compute_base_lin_vel_body(base_quat, base_lin_vel)
            obs = obs * scale
            
        else:
            # Unknown term - return zeros and warn
            print(f"[Warning] Unknown observation term: {term_name}")
            obs = np.zeros(1)
        
        return obs
    
    def _build_default_obs(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_quat: np.ndarray,
        base_ang_vel: np.ndarray,
        velocity_command: np.ndarray,
    ) -> np.ndarray:
        """Build default locomotion observation.
        
        Default order (matches common IsaacLab locomotion):
        1. base_ang_vel (3)
        2. projected_gravity (3)
        3. velocity_commands (3)
        4. joint_pos_rel (num_joints)
        5. joint_vel (num_joints)
        6. last_action (num_joints)
        """
        num_joints = len(joint_pos)
        
        # Angular velocity in body frame
        ang_vel_b = compute_base_ang_vel_body(base_quat, base_ang_vel)
        
        # Projected gravity
        proj_grav = compute_projected_gravity(base_quat)
        
        # Relative joint positions
        joint_pos_rel = joint_pos - self.default_joint_pos
        
        obs = np.concatenate([
            ang_vel_b * 0.2,  # Default scale
            proj_grav,
            velocity_command,
            joint_pos_rel,
            joint_vel * 0.05,  # Default scale
            self._last_action,
        ])
        
        return obs
    
    def set_gait_period(self, period: float) -> None:
        """Set gait period for phase computation."""
        self._gait_period = period
    
    def set_height_scanner(self, scanner: Any) -> None:
        """Set height scanner for exteroception."""
        self.height_scanner = scanner
