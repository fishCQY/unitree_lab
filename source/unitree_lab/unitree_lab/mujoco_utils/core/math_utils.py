"""Math utilities for quaternion operations and transformations."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: Quaternion in (w, x, y, z) format. Shape is (..., 4) or (4,).
        v: Vector in (x, y, z) format. Shape is (..., 3) or (3,).

    Returns:
        Rotated vector in (x, y, z) format with same batch shape as input.
    """
    if q.ndim == 1:
        q_w = q[0]
        q_vec = q[1:]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c
    else:
        q_w = q[..., 0]
        q_vec = q[..., 1:]
        a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)
        b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
        if q_vec.ndim == 2:
            dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)
            c = q_vec * dot_product * 2.0
        else:
            dot_product = np.expand_dims(np.einsum("...i,...i->...", q_vec, v), axis=-1)
            c = q_vec * dot_product * 2.0
        return a - b + c


def quat_rotate_forward_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion (forward rotation).

    Args:
        q: Quaternion in (w, x, y, z) format. Shape is (..., 4) or (4,).
        v: Vector in (x, y, z) format. Shape is (..., 3) or (3,).

    Returns:
        Rotated vector in (x, y, z) format with same batch shape as input.
    """
    if q.ndim == 1:
        q_w = q[0]
        q_vec = q[1:]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a + b + c
    else:
        q_w = q[..., 0]
        q_vec = q[..., 1:]
        a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)
        b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
        if q_vec.ndim == 2:
            dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)
            c = q_vec * dot_product * 2.0
        else:
            dot_product = np.expand_dims(np.einsum("...i,...i->...", q_vec, v), axis=-1)
            c = q_vec * dot_product * 2.0
        return a + b + c


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions: q1 x q2.

    Args:
        q1: First quaternion in (w, x, y, z) format.
        q2: Second quaternion in (w, x, y, z) format.

    Returns:
        Result quaternion in (w, x, y, z) format.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Calculate quaternion inverse (conjugate for unit quaternions)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def yaw_from_quat(quat: np.ndarray) -> float:
    """Extract yaw angle from quaternion in (w, x, y, z) format."""
    w, x, y, z = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quaternion(q: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from full orientation quaternion."""
    w, x, y, z = q
    r = R.from_quat([x, y, z, w])
    euler = r.as_euler("xyz", degrees=False)
    yaw = euler[2]
    yaw_rot = R.from_euler("z", yaw)
    yaw_quat = yaw_rot.as_quat()
    return np.array([yaw_quat[3], yaw_quat[0], yaw_quat[1], yaw_quat[2]])


def subtract_frame_transforms_np(
    pos_01: np.ndarray,
    quat_01: np.ndarray,
    pos_02: np.ndarray | None,
    quat_02: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate relative transform T12 = T01^(-1) x T02."""
    quat_10 = quat_01.copy()
    quat_10[1:] *= -1
    if quat_02 is not None:
        quat_12 = quaternion_multiply(quat_10, quat_02)
    else:
        quat_12 = quat_10
    if pos_02 is not None:
        delta_pos = pos_02 - pos_01
    else:
        delta_pos = -pos_01
    pos_12 = quat_rotate_inverse_np(quat_01, delta_pos)
    return pos_12, quat_12


def se3_inverse(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate inverse of SE3 transform."""
    inv_quat = quaternion_inverse(quat)
    inv_pos = quat_rotate_inverse_np(quat.reshape(1, -1), (-pos).reshape(1, -1)).reshape(-1)
    return inv_pos, inv_quat


def apply_se3_transform(
    pos: np.ndarray,
    quat: np.ndarray,
    transform_pos: np.ndarray,
    transform_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SE3 transform: result = transform * input."""
    rotated_pos = quat_rotate_forward_np(transform_quat.reshape(1, -1), pos.reshape(1, -1)).reshape(-1)
    result_pos = rotated_pos + transform_pos
    result_quat = quaternion_multiply(transform_quat, quat)
    return result_pos, result_quat
