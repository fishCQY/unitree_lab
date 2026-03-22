#!/usr/bin/env python3
"""Add key_points_b to existing AMP PKL files using Pinocchio FK.

This script reads each clip in an AMP PKL, computes robot body positions
via forward kinematics (Pinocchio + URDF), transforms them to body frame,
and saves the result as key_points_b.

Usage:
    python scripts/add_key_points_to_amp_pkl.py \
        --urdf source/unitree_lab/unitree_lab/assets/robots_urdf/g1/g1_29dof.urdf \
        --pkl source/unitree_lab/unitree_lab/data/AMP/lafan_walk_clips.pkl \
        --output source/unitree_lab/unitree_lab/data/AMP/lafan_walk_clips_with_kp.pkl

    # Process all AMP PKL files:
    for f in source/unitree_lab/unitree_lab/data/AMP/lafan_*.pkl; do
        python scripts/add_key_points_to_amp_pkl.py --urdf ... --pkl "$f" --output "$f"
    done
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

BODY_NAMES = [
    "pelvis",
    "left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link",
]

AMP_BODY_NAMES = [
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
]


def quat_rotate_inverse_np(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec by the inverse of quat (w,x,y,z convention). Supports batch."""
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
    t = 2.0 * np.cross(np.stack([x, y, z], axis=-1), vec)
    return vec - w[..., None] * t + np.cross(np.stack([x, y, z], axis=-1), t)


def compute_fk_and_key_points_b(
    model, pin_data, body_frame_ids: list[int],
    root_pos: np.ndarray, root_rot_wxyz: np.ndarray, dof_pos: np.ndarray,
    amp_body_indices: list[int],
) -> np.ndarray:
    """Compute key_points_b for AMP bodies in body frame."""
    import pinocchio as pin

    num_frames = dof_pos.shape[0]
    num_amp_bodies = len(amp_body_indices)
    key_points_b = np.zeros((num_frames, num_amp_bodies, 3), dtype=np.float32)

    for i in range(num_frames):
        q = np.zeros(model.nq, dtype=np.float64)
        q[:dof_pos.shape[1]] = dof_pos[i]
        pin.forwardKinematics(model, pin_data, q)
        pin.updateFramePlacements(model, pin_data)

        w, x, y, z = root_rot_wxyz[i]
        R = pin.Quaternion(w, x, y, z).toRotationMatrix()
        t = root_pos[i]

        for j, body_idx in enumerate(amp_body_indices):
            fid = body_frame_ids[body_idx]
            pos_local = pin_data.oMf[fid].translation.copy()
            pos_world = R @ pos_local + t
            pos_relative = pos_world - t
            pos_body = quat_rotate_inverse_np(root_rot_wxyz[i:i+1], pos_relative[None])[0]
            key_points_b[i, j] = pos_body

    return key_points_b


def load_urdf_model(urdf_path: str):
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    body_frame_ids = []
    for bname in BODY_NAMES:
        found = False
        for fi in range(model.nframes):
            if model.frames[fi].name == bname:
                body_frame_ids.append(fi)
                found = True
                break
        if not found:
            raise ValueError(f"Body '{bname}' not found in URDF. Available: {[model.frames[i].name for i in range(model.nframes)]}")

    amp_body_indices = [BODY_NAMES.index(n) for n in AMP_BODY_NAMES]
    return model, data, body_frame_ids, amp_body_indices


def process_pkl(pkl_path: str, urdf_path: str, output_path: str):
    model, pin_data, body_frame_ids, amp_body_indices = load_urdf_model(urdf_path)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        first_val = next(iter(data.values()))
        is_nested = isinstance(first_val, dict)
    else:
        is_nested = False

    clips = data if is_nested else {"clip_0": data} if isinstance(data, dict) else {f"clip_{i}": c for i, c in enumerate(data)}

    total_added = 0
    for clip_name, clip in clips.items():
        if "key_points_b" in clip:
            print(f"  {clip_name}: already has key_points_b, skipping")
            continue

        dof_pos = np.asarray(clip["dof_pos"], dtype=np.float64)
        root_pos = np.asarray(clip["root_pos"], dtype=np.float64)
        root_rot = np.asarray(clip["root_rot"], dtype=np.float64)

        if root_rot.shape[-1] == 4:
            if abs(np.mean(root_rot[:, 0])) > abs(np.mean(root_rot[:, 3])):
                root_rot_wxyz = root_rot
            else:
                root_rot_wxyz = root_rot[:, [3, 0, 1, 2]]
        else:
            raise ValueError(f"Unexpected root_rot shape: {root_rot.shape}")

        kp_b = compute_fk_and_key_points_b(
            model, pin_data, body_frame_ids,
            root_pos, root_rot_wxyz, dof_pos, amp_body_indices,
        )
        clip["key_points_b"] = kp_b
        total_added += 1
        print(f"  {clip_name}: added key_points_b shape={kp_b.shape}")

    output = clips if is_nested else (list(clips.values()) if not is_nested else clips)

    with open(output_path, "wb") as f:
        pickle.dump(output if is_nested else data, f)

    print(f"Saved to {output_path} ({total_added} clips updated)")


def main():
    parser = argparse.ArgumentParser(description="Add key_points_b to AMP PKL files")
    parser.add_argument("--urdf", required=True, help="Path to robot URDF")
    parser.add_argument("--pkl", required=True, help="Input AMP PKL file")
    parser.add_argument("--output", default=None, help="Output PKL file (default: overwrite input)")
    args = parser.parse_args()

    output = args.output or args.pkl
    print(f"Processing {args.pkl} with URDF {args.urdf}")
    process_pkl(args.pkl, args.urdf, output)


if __name__ == "__main__":
    main()
