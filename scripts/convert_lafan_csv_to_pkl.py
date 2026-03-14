#!/usr/bin/env python3
"""Convert LAFAN1 retargeted CSV files to AMP training PKL files.

CSV format (from unitree_lafan README):
    Each row = 1 frame at 30 FPS, 36 columns:
    [0:7]   root_joint: X, Y, Z, QX, QY, QZ, QW
    [7:36]  29 joint positions in URDF order (body-part grouped):
            left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee,
            left_ankle_pitch, left_ankle_roll,
            right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee,
            right_ankle_pitch, right_ankle_roll,
            waist_yaw, waist_roll, waist_pitch,
            left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
            left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
            right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
            right_wrist_roll, right_wrist_pitch, right_wrist_yaw

Output PKL contains data in URDF order (same as CSV):
    - fps: int
    - dof_pos: (T, 29) joint positions
    - dof_names: list of 29 joint name strings
    - root_pos: (T, 3) root position [x, y, z]
    - root_rot: (T, 4) root rotation quaternion [w, x, y, z]

After conversion, run scripts/preprocess_amp_motion.py to compute derived
quantities (dof_vel, proj_grav, root_angle_vel, dof_pos_rel).
"""

import argparse
import pickle
import numpy as np
from pathlib import Path


# URDF order: body-part grouped (matches CSV column order)
URDF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

CSV_FPS = 30


def convert_csv_to_pkl(csv_path: Path, output_path: Path):
    """Convert a single LAFAN CSV to PKL in URDF joint order."""
    data = np.genfromtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    assert data.shape[1] == 36, f"Expected 36 columns, got {data.shape[1]} in {csv_path}"
    T = data.shape[0]

    root_pos = data[:, 0:3].astype(np.float64)

    # CSV quaternion is (qx, qy, qz, qw), convert to (w, x, y, z)
    qx, qy, qz, qw = data[:, 3], data[:, 4], data[:, 5], data[:, 6]
    root_rot = np.stack([qw, qx, qy, qz], axis=-1).astype(np.float64)

    # Joint positions: columns 7..36, already in URDF order
    dof_pos = data[:, 7:36].astype(np.float64)
    assert dof_pos.shape == (T, 29)

    pkl_data = {
        "fps": CSV_FPS,
        "dof_pos": dof_pos,
        "dof_names": URDF_JOINT_NAMES,
        "root_pos": root_pos,
        "root_rot": root_rot,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pkl_data, f)

    print(f"  {csv_path.name:45s} -> {output_path.name:55s}  {T:5d} frames")


def main():
    parser = argparse.ArgumentParser(description="Convert LAFAN CSV files to AMP training PKL files (URDF joint order)")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="source/unitree_lab/unitree_lab/data/unitree_lafan/g1",
        help="Directory containing G1 CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/lafan",
        help="Output directory for PKL files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for CSV files (default: *.csv)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    csv_files = sorted(input_dir.glob(args.pattern))
    if not csv_files:
        print(f"No CSV files found in {input_dir} matching '{args.pattern}'")
        return

    print(f"Converting {len(csv_files)} CSV files from {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Joint order: URDF (body-part grouped)\n")

    for csv_path in csv_files:
        pkl_name = csv_path.stem + ".pkl"
        output_path = output_dir / pkl_name
        convert_csv_to_pkl(csv_path, output_path)

    print(f"\nDone – {len(csv_files)} files converted.")
    print(f"\nNext step: run preprocess_amp_motion.py to compute dof_vel, proj_grav, etc.:")
    print(f"  python scripts/preprocess_amp_motion.py --data-dir {output_dir}")


if __name__ == "__main__":
    main()
