"""Joint mapping utilities between ONNX model and MuJoCo."""

from __future__ import annotations

import mujoco
import numpy as np

from ..logging import logger


def create_joint_mapping(onnx_joint_names: list[str], model: mujoco.MjModel) -> dict:
    """Create mapping from ONNX joint names to MuJoCo joint indices.

    Args:
        onnx_joint_names: List of joint names from ONNX model metadata.
        model: MuJoCo model object.

    Returns:
        Dictionary containing:
        - 'onnx_to_mujoco': Dict mapping ONNX index to MuJoCo joint info
        - 'mujoco_to_onnx': Dict mapping MuJoCo joint name to ONNX index
    """
    mapping = {"onnx_to_mujoco": {}, "mujoco_to_onnx": {}}

    mujoco_joint_names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            mujoco_joint_names.append(name)

    for onnx_idx, name in enumerate(onnx_joint_names):
        if name in mujoco_joint_names:
            mujoco_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_addr = model.jnt_qposadr[mujoco_idx]
            dof_addr = model.jnt_dofadr[mujoco_idx]

            mapping["onnx_to_mujoco"][onnx_idx] = {
                "name": name,
                "mujoco_id": mujoco_idx,
                "qpos_addr": qpos_addr,
                "dof_addr": dof_addr,
            }
            mapping["mujoco_to_onnx"][name] = onnx_idx
        else:
            logger.warning(f"Joint {name} from ONNX not found in MuJoCo model")

    return mapping


def model_to_mujoco(model_joints: np.ndarray, joint_mapping_index: np.ndarray) -> np.ndarray:
    """Convert joint values from model (ONNX) order to MuJoCo order."""
    mujoco_joints = np.zeros(len(joint_mapping_index))
    for i, idx in enumerate(joint_mapping_index):
        mujoco_joints[i] = model_joints[idx]
    return mujoco_joints


def mujoco_to_model(mujoco_joints: np.ndarray, joint_mapping_index: np.ndarray) -> np.ndarray:
    """Convert joint values from MuJoCo order to model (ONNX) order."""
    model_joints = np.zeros_like(mujoco_joints)
    for mujoco_idx, model_idx in enumerate(joint_mapping_index):
        model_joints[model_idx] = mujoco_joints[mujoco_idx]
    return model_joints


def create_joint_mapping_index(onnx_joint_names: list[str], actuator_names: list[str]) -> np.ndarray:
    """Create index array for joint order mapping.

    Args:
        onnx_joint_names: Joint names from ONNX model metadata.
        actuator_names: Actuator names from MuJoCo XML (in MuJoCo order).

    Returns:
        Array where result[i] is the ONNX index for actuator i.
    """
    return np.array([onnx_joint_names.index(joint) for joint in actuator_names])
