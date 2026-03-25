"""Physics utilities for MuJoCo sim2sim.

Provides:
1. PD control implementation matching IsaacLab / bfm_training
2. Physics parameter application from ONNX metadata
3. Torque limit extraction
4. Quaternion and gravity projection utilities
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


# =============================================================================
# PD Control
# =============================================================================


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
    tau_limit: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """PD control with optional torque limiting.

    Computes torque using PD control law::

        torque = kp * (target_q - q) + kd * (target_dq - dq)

    This matches the bfm_training explicit PD implementation.

    Args:
        target_q: Target joint positions.
        q: Current joint positions.
        kp: Proportional gains (stiffness).
        target_dq: Target joint velocities.
        dq: Current joint velocities.
        kd: Derivative gains (damping).
        tau_limit: Optional torque limits (scalar per joint).

    Returns:
        Tuple of (torque, torque_util).
    """
    torque = (target_q - q) * kp + (target_dq - dq) * kd

    if tau_limit is not None:
        torque = np.clip(torque, -tau_limit, tau_limit)
        torque_util = np.abs(torque) / (tau_limit + 1e-6)
    else:
        torque_util = np.zeros_like(torque)

    return torque, torque_util


def pd_control_velocity(
    target_dq: np.ndarray,
    current_dq: np.ndarray,
    kd: np.ndarray,
    tau_limit: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity-only PD control (for wheel joints).

    Implements: torque = Kd * (dq_target - dq_current)

    Returns:
        Tuple of (torque, torque_util).
    """
    torque = kd * (target_dq - current_dq)

    if tau_limit is not None:
        torque = np.clip(torque, -tau_limit, tau_limit)
        torque_util = np.abs(torque) / (tau_limit + 1e-6)
    else:
        torque_util = np.zeros_like(torque)

    return torque, torque_util


# =============================================================================
# Torque Limits
# =============================================================================


def get_tau_limit(model: "mujoco.MjModel", num_actions: int) -> np.ndarray:
    """Get torque limits from actuator control ranges.

    Args:
        model: MuJoCo model object.
        num_actions: Number of actuators.

    Returns:
        Array of torque limits (positive) for each actuator.
    """
    tau_limit = np.zeros(num_actions, dtype=np.float64)
    assert model.nu == num_actions, "num_actions must equal model.nu"

    for i in range(model.nu):
        ctrl_range = model.actuator_ctrlrange[i]
        tau_limit[i] = ctrl_range[1]

    return tau_limit


# =============================================================================
# Physics Parameter Application
# =============================================================================


def set_joint_armature(model: "mujoco.MjModel", armature_map: dict[str, float]) -> None:
    """Set joint armature values from configuration (substring matching)."""
    for i in range(model.nv):
        joint_id = model.dof_jntid[i]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            continue
        for pattern, value in armature_map.items():
            if pattern in joint_name:
                model.dof_armature[i] = value
                break


def set_joint_damping(model: "mujoco.MjModel", damping_map: dict[str, float]) -> None:
    """Set joint damping values from configuration (substring matching)."""
    for i in range(model.nv):
        joint_id = model.dof_jntid[i]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            continue
        for pattern, value in damping_map.items():
            if pattern in joint_name:
                model.dof_damping[i] = value
                break


def set_joint_friction(model: "mujoco.MjModel", friction_map: dict[str, float]) -> None:
    """Set joint friction loss values from configuration (substring matching)."""
    for i in range(model.nv):
        joint_id = model.dof_jntid[i]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            continue
        for pattern, value in friction_map.items():
            if pattern in joint_name:
                model.dof_frictionloss[i] = value
                break


def apply_onnx_physics_params(model: "mujoco.MjModel", onnx_config: dict) -> None:
    """Apply physics parameters from ONNX config dict to MuJoCo model.

    Checks for armature, damping (viscous_friction), and friction keys.
    """
    if mujoco is None:
        raise ImportError("mujoco not installed")

    if "armature" in onnx_config and isinstance(onnx_config["armature"], dict):
        set_joint_armature(model, onnx_config["armature"])
    if "friction" in onnx_config and isinstance(onnx_config["friction"], dict):
        set_joint_friction(model, onnx_config["friction"])
    if "viscous_friction" in onnx_config and isinstance(onnx_config["viscous_friction"], dict):
        set_joint_damping(model, onnx_config["viscous_friction"])


# =============================================================================
# Quaternion / Rotation Utilities
# =============================================================================


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse of quaternion (R^T @ vec)."""
    R = quat_to_rotation_matrix(quat)
    return R.T @ vec


def compute_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Compute gravity projection in body frame (matches IsaacLab projected_gravity_b)."""
    gravity_world = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_world)


def compute_base_ang_vel_body(quat: np.ndarray, ang_vel_world: np.ndarray) -> np.ndarray:
    """Transform angular velocity from world to body frame."""
    return quat_rotate_inverse(quat, ang_vel_world)


def compute_base_lin_vel_body(quat: np.ndarray, lin_vel_world: np.ndarray) -> np.ndarray:
    """Transform linear velocity from world to body frame."""
    return quat_rotate_inverse(quat, lin_vel_world)
