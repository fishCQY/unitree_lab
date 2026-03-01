#!/usr/bin/env python3
"""Preprocess AMP motion data: compute dof_vel, root_angle_vel, proj_grav, dof_pos_rel."""

import pickle
import numpy as np
from pathlib import Path

# G1 29-DOF default joint positions (Lab joint order, from unitree.py UNITREE_G1_CFG)
DEFAULT_JOINT_POS = np.array([
    -0.20,   # 0:  left_hip_pitch
    -0.20,   # 1:  right_hip_pitch
     0.00,   # 2:  waist_yaw
     0.00,   # 3:  left_hip_roll
     0.00,   # 4:  right_hip_roll
     0.00,   # 5:  waist_roll
     0.00,   # 6:  left_hip_yaw
     0.00,   # 7:  right_hip_yaw
     0.00,   # 8:  waist_pitch
     0.42,   # 9:  left_knee
     0.42,   # 10: right_knee
     0.35,   # 11: left_shoulder_pitch
     0.35,   # 12: right_shoulder_pitch
    -0.23,   # 13: left_ankle_pitch
    -0.23,   # 14: right_ankle_pitch
     0.18,   # 15: left_shoulder_roll
    -0.18,   # 16: right_shoulder_roll
     0.00,   # 17: left_ankle_roll
     0.00,   # 18: right_ankle_roll
     0.00,   # 19: left_shoulder_yaw
     0.00,   # 20: right_shoulder_yaw
     0.87,   # 21: left_elbow
     0.87,   # 22: right_elbow
     0.00,   # 23: left_wrist_roll
     0.00,   # 24: right_wrist_roll
     0.00,   # 25: left_wrist_pitch
     0.00,   # 26: right_wrist_pitch
     0.00,   # 27: left_wrist_yaw
     0.00,   # 28: right_wrist_yaw
], dtype=np.float64)


def quat_conjugate(q):
    """Conjugate of quaternion in (w, x, y, z) format."""
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quat_multiply(q1, q2):
    """Hamilton product of two quaternions in (w, x, y, z) format."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q.  q: (w,x,y,z)."""
    q_conj = quat_conjugate(q)
    v_quat = np.zeros((*v.shape[:-1], 4), dtype=v.dtype)
    v_quat[..., 1:] = v
    result = quat_multiply(q_conj, quat_multiply(v_quat, q))
    return result[..., 1:]


def compute_body_angular_velocity(quats, dt):
    """Body-frame angular velocity: omega = 2 * Im(conj(q) * dq/dt).  q: (w,x,y,z)."""
    dq = np.zeros_like(quats)
    dq[1:-1] = (quats[2:] - quats[:-2]) / (2 * dt)
    dq[0] = (quats[1] - quats[0]) / dt
    dq[-1] = (quats[-1] - quats[-2]) / dt
    omega_quat = 2.0 * quat_multiply(quat_conjugate(quats), dq)
    return omega_quat[..., 1:]


def preprocess_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fps = data["fps"]
    dt = 1.0 / fps
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)

    # dof_vel via central differences
    dof_vel = np.zeros_like(dof_pos)
    dof_vel[1:-1] = (dof_pos[2:] - dof_pos[:-2]) / (2 * dt)
    dof_vel[0] = (dof_pos[1] - dof_pos[0]) / dt
    dof_vel[-1] = (dof_pos[-1] - dof_pos[-2]) / dt

    # proj_grav: gravity projected into body frame
    T = root_rot.shape[0]
    gravity_world = np.tile(np.array([0.0, 0.0, -1.0]), (T, 1))
    proj_grav = quat_rotate_inverse(root_rot, gravity_world)

    # root_angle_vel: body-frame angular velocity
    root_angle_vel = compute_body_angular_velocity(root_rot, dt)

    # dof_pos relative to default standing pose
    dof_pos_rel = dof_pos - DEFAULT_JOINT_POS[np.newaxis, :]

    data["dof_vel"] = dof_vel.astype(np.float32)
    data["proj_grav"] = proj_grav.astype(np.float32)
    data["root_angle_vel"] = root_angle_vel.astype(np.float32)
    data["dof_pos_rel"] = dof_pos_rel.astype(np.float32)

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    print(f"  {pkl_path.name:55s}  {T:4d} frames  "
          f"proj_grav[0]=[{proj_grav[0,0]:+.3f},{proj_grav[0,1]:+.3f},{proj_grav[0,2]:+.3f}]")


def main():
    data_dir = Path("source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/walk_and_run")
    pkl_files = sorted(data_dir.glob("*.pkl"))
    print(f"Preprocessing {len(pkl_files)} motion files in {data_dir}\n")
    for f in pkl_files:
        preprocess_file(f)
    print(f"\nDone – {len(pkl_files)} files updated.")


if __name__ == "__main__":
    main()
