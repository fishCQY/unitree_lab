"""Physics utilities for MuJoCo sim2sim.

This module provides:
1. PD control implementation matching IsaacLab
2. Physics parameter application from ONNX metadata
3. Quaternion and gravity projection utilities
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


def pd_control(
    target_q: np.ndarray,
    current_q: np.ndarray,
    current_dq: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    tau_limits: np.ndarray | None = None,
) -> np.ndarray:
    """Compute PD control torques.
    
    Implements: τ = Kp * (q_target - q_current) - Kd * dq_current
    
    This matches IsaacLab's actuator model for position control.
    
    Args:
        target_q: Target joint positions
        current_q: Current joint positions
        current_dq: Current joint velocities
        kp: Position gains (stiffness)
        kd: Velocity gains (damping)
        tau_limits: Optional torque limits (min, max) per joint
        
    Returns:
        Control torques
    """
    # PD law
    tau = kp * (target_q - current_q) - kd * current_dq
    
    # Apply torque limits
    if tau_limits is not None:
        tau = np.clip(tau, tau_limits[:, 0], tau_limits[:, 1])
    
    return tau


def pd_control_velocity(
    target_dq: np.ndarray,
    current_dq: np.ndarray,
    kd: np.ndarray,
    tau_limits: np.ndarray | None = None,
) -> np.ndarray:
    """Compute velocity control torques.
    
    Implements: τ = Kd * (dq_target - dq_current)
    
    Used for wheel joints or velocity-controlled actuators.
    
    Args:
        target_dq: Target joint velocities
        current_dq: Current joint velocities
        kd: Velocity gains
        tau_limits: Optional torque limits
        
    Returns:
        Control torques
    """
    tau = kd * (target_dq - current_dq)
    
    if tau_limits is not None:
        tau = np.clip(tau, tau_limits[:, 0], tau_limits[:, 1])
    
    return tau


def apply_onnx_physics_params(
    model: "mujoco.MjModel",
    armature: dict[str, float] | None = None,
    damping: dict[str, float] | None = None,
    friction: dict[str, float] | None = None,
) -> None:
    """Apply physics parameters from ONNX metadata to MuJoCo model.
    
    This ensures physics consistency between training and evaluation.
    
    Args:
        model: MuJoCo model to modify
        armature: Joint armature values {joint_name: value}
        damping: Joint damping values {joint_name: value}
        friction: Joint friction values {joint_name: value}
    """
    if mujoco is None:
        raise ImportError("mujoco not installed")
    
    def _apply_to_joints(param_dict: dict[str, float], attr_name: str):
        if param_dict is None:
            return
        for joint_name, value in param_dict.items():
            try:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    dof_adr = model.jnt_dofadr[joint_id]
                    dof_num = {0: 6, 1: 3, 2: 1, 3: 1}[int(model.jnt_type[joint_id])]
                    for i in range(dof_num):
                        if attr_name == "armature":
                            model.dof_armature[dof_adr + i] = value
                        elif attr_name == "damping":
                            model.dof_damping[dof_adr + i] = value
                        elif attr_name == "frictionloss":
                            model.dof_frictionloss[dof_adr + i] = value
            except Exception as e:
                print(f"Warning: Could not apply {attr_name} to joint {joint_name}: {e}")
    
    _apply_to_joints(armature, "armature")
    _apply_to_joints(damping, "damping")
    _apply_to_joints(friction, "frictionloss")


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.
    
    Args:
        quat: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse of quaternion.
    
    Equivalent to R.T @ vec where R is rotation matrix from quat.
    
    Args:
        quat: Quaternion [w, x, y, z]
        vec: Vector to rotate
        
    Returns:
        Rotated vector
    """
    R = quat_to_rotation_matrix(quat)
    return R.T @ vec


def compute_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Compute gravity projection in body frame.
    
    This matches IsaacLab's projected_gravity_b observation.
    
    Args:
        quat: Base orientation quaternion [w, x, y, z]
        
    Returns:
        Gravity vector in body frame (normalized)
    """
    gravity_world = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_world)


def compute_base_ang_vel_body(
    quat: np.ndarray,
    ang_vel_world: np.ndarray,
) -> np.ndarray:
    """Transform angular velocity from world to body frame.
    
    Args:
        quat: Base orientation quaternion [w, x, y, z]
        ang_vel_world: Angular velocity in world frame
        
    Returns:
        Angular velocity in body frame
    """
    return quat_rotate_inverse(quat, ang_vel_world)


def compute_base_lin_vel_body(
    quat: np.ndarray,
    lin_vel_world: np.ndarray,
) -> np.ndarray:
    """Transform linear velocity from world to body frame.
    
    Args:
        quat: Base orientation quaternion [w, x, y, z]
        lin_vel_world: Linear velocity in world frame
        
    Returns:
        Linear velocity in body frame
    """
    return quat_rotate_inverse(quat, lin_vel_world)
