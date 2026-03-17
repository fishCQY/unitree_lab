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

This script also supports cubic-spline resampling to target FPS, merging all
clips into a single PKL, and exporting a segment manifest.
"""

import argparse
import json
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
DEFAULT_JOINT_POS = np.array(
    [
        -0.20, 0.00, 0.00, 0.42, -0.23, 0.00,
        -0.20, 0.00, 0.00, 0.42, -0.23, 0.00,
        0.00, 0.00, 0.00,
        0.35, 0.18, 0.00, 0.87, 0.00, 0.00, 0.00,
        0.35, -0.18, 0.00, 0.87, 0.00, 0.00, 0.00,
    ],
    dtype=np.float64,
)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate for (w, x, y, z)."""
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by inverse(q), q in (w, x, y, z)."""
    q_conj = quat_conjugate(q)
    v_quat = np.zeros((*v.shape[:-1], 4), dtype=v.dtype)
    v_quat[..., 1:] = v
    result = quat_multiply(q_conj, quat_multiply(v_quat, q))
    return result[..., 1:]


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """Central diff with forward/backward at boundaries."""
    out = np.zeros_like(values, dtype=np.float64)
    if values.shape[0] <= 1:
        return out
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    return out


def compute_body_angular_velocity(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity from quaternion derivative."""
    dq = finite_difference(quats_wxyz, dt)
    omega_quat = 2.0 * quat_multiply(quat_conjugate(quats_wxyz), dq)
    return omega_quat[..., 1:]


def enrich_motion_features(motion: dict, default_joint_pos: np.ndarray = DEFAULT_JOINT_POS) -> dict:
    """Add AMP-required derived features to motion dict."""
    fps = float(motion["fps"])
    dt = 1.0 / fps
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    num_frames = dof_pos.shape[0]

    dof_vel = finite_difference(dof_pos, dt)
    root_angle_vel = compute_body_angular_velocity(root_rot, dt)
    gravity_world = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (num_frames, 1))
    proj_grav = quat_rotate_inverse(root_rot, gravity_world)
    dof_pos_rel = dof_pos - default_joint_pos.reshape(1, -1)

    motion["dof_vel"] = dof_vel.astype(np.float32)
    motion["root_angle_vel"] = root_angle_vel.astype(np.float32)
    motion["proj_grav"] = proj_grav.astype(np.float32)
    motion["dof_pos_rel"] = dof_pos_rel.astype(np.float32)
    return motion


def load_csv_motion(csv_path: Path, csv_fps: int) -> dict:
    """Load a single LAFAN CSV into motion dict in URDF joint order."""
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

    return {
        "fps": int(csv_fps),
        "dof_pos": dof_pos,
        "dof_names": URDF_JOINT_NAMES,
        "root_pos": root_pos,
        "root_rot": root_rot,
    }


def infer_action_name(clip_name: str) -> str:
    """Infer action label from clip name (best-effort)."""
    if "_subject" in clip_name:
        return clip_name.split("_subject", maxsplit=1)[0]
    parts = clip_name.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return clip_name


def ensure_quaternion_continuity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Flip quaternion signs to keep adjacent frames in the same hemisphere."""
    out = np.asarray(quat_wxyz, dtype=np.float64).copy()
    for i in range(1, out.shape[0]):
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] = -out[i]
    return out


def natural_cubic_spline_sample(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Evaluate natural cubic spline for vector-valued samples.

    Args:
        t_src: (N,) strictly increasing timestamps.
        y_src: (N, D) sample values.
        t_dst: (M,) query timestamps in [t_src[0], t_src[-1]].
    """
    t_src = np.asarray(t_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)

    n = t_src.shape[0]
    if n == 1:
        return np.repeat(y_src, repeats=t_dst.shape[0], axis=0)
    if n == 2:
        # With only two points, natural cubic degenerates to linear.
        w = ((t_dst - t_src[0]) / (t_src[1] - t_src[0]))[:, None]
        return (1.0 - w) * y_src[0:1] + w * y_src[1:2]

    h = t_src[1:] - t_src[:-1]  # (N-1,)
    if np.any(h <= 0):
        raise ValueError("t_src must be strictly increasing.")

    d = y_src.shape[1]
    a = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros((n, d), dtype=np.float64)
    a[0, 0] = 1.0
    a[-1, -1] = 1.0

    for i in range(1, n - 1):
        h_prev = h[i - 1]
        h_next = h[i]
        a[i, i - 1] = h_prev
        a[i, i] = 2.0 * (h_prev + h_next)
        a[i, i + 1] = h_next
        rhs[i] = 6.0 * ((y_src[i + 1] - y_src[i]) / h_next - (y_src[i] - y_src[i - 1]) / h_prev)

    m2 = np.linalg.solve(a, rhs)  # second derivatives, shape (N, D)

    # Locate interval index i so t_src[i] <= t < t_src[i+1].
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    idx = np.clip(idx, 0, n - 2)

    x0 = t_src[idx]
    x1 = t_src[idx + 1]
    hi = (x1 - x0)[:, None]
    a_w = ((x1 - t_dst)[:, None] / hi)
    b_w = ((t_dst - x0)[:, None] / hi)

    y0 = y_src[idx]
    y1 = y_src[idx + 1]
    m0 = m2[idx]
    m1 = m2[idx + 1]

    return (
        a_w * y0
        + b_w * y1
        + (((a_w**3 - a_w) * m0 + (b_w**3 - b_w) * m1) * (hi**2) / 6.0)
    )


def resample_motion_cubic(motion: dict, target_fps: int) -> dict:
    """Resample motion with cubic spline interpolation to target FPS."""
    src_fps = int(motion["fps"])
    if src_fps == target_fps:
        return {
            "fps": int(target_fps),
            "dof_pos": np.asarray(motion["dof_pos"], dtype=np.float64),
            "dof_names": motion["dof_names"],
            "root_pos": np.asarray(motion["root_pos"], dtype=np.float64),
            "root_rot": np.asarray(motion["root_rot"], dtype=np.float64),
        }

    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot_wxyz = np.asarray(motion["root_rot"], dtype=np.float64)

    num_src = dof_pos.shape[0]
    if num_src < 2:
        return {
            "fps": int(target_fps),
            "dof_pos": dof_pos.copy(),
            "dof_names": motion["dof_names"],
            "root_pos": root_pos.copy(),
            "root_rot": root_rot_wxyz.copy(),
        }

    t_src = np.arange(num_src, dtype=np.float64) / float(src_fps)
    duration = (num_src - 1) / float(src_fps)
    num_dst = int(round(duration * float(target_fps))) + 1
    t_dst = np.linspace(0.0, duration, num=num_dst, dtype=np.float64)

    dof_pos_dst = natural_cubic_spline_sample(t_src, dof_pos, t_dst)
    root_pos_dst = natural_cubic_spline_sample(t_src, root_pos, t_dst)

    # Component-wise cubic spline on quaternion with hemisphere continuity.
    root_rot_cont = ensure_quaternion_continuity(root_rot_wxyz)
    root_rot_dst = natural_cubic_spline_sample(t_src, root_rot_cont, t_dst)

    # Keep normalized quaternions to avoid numerical drift.
    norm = np.linalg.norm(root_rot_dst, axis=-1, keepdims=True)
    root_rot_dst = root_rot_dst / np.clip(norm, 1e-12, None)

    return {
        "fps": int(target_fps),
        "dof_pos": dof_pos_dst,
        "dof_names": motion["dof_names"],
        "root_pos": root_pos_dst,
        "root_rot": root_rot_dst,
    }


def save_motion_pkl(motion: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(motion, f)


def build_manifest_and_merged(resampled_items: list[dict], source_fps: int, target_fps: int) -> tuple[dict, list[dict]]:
    """Build merged motion dict and segment manifest."""
    merged_dof = []
    merged_root_pos = []
    merged_root_rot = []
    merged_dof_vel = []
    merged_root_angle_vel = []
    merged_proj_grav = []
    merged_dof_pos_rel = []
    segments: list[dict] = []
    cursor = 0

    for idx, item in enumerate(resampled_items):
        motion = item["motion"]
        frames = int(motion["dof_pos"].shape[0])
        if frames <= 0:
            continue

        start_frame = cursor
        end_frame = cursor + frames - 1
        segments.append(
            {
                "id": idx,
                "clip_name": item["clip_name"],
                "action": item["action"],
                "source_csv": item["source_csv"],
                "fps": int(target_fps),
                "num_frames": frames,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": start_frame / float(target_fps),
                "end_time_sec": end_frame / float(target_fps),
                "duration_sec": frames / float(target_fps),
            }
        )

        merged_dof.append(np.asarray(motion["dof_pos"], dtype=np.float64))
        merged_root_pos.append(np.asarray(motion["root_pos"], dtype=np.float64))
        merged_root_rot.append(np.asarray(motion["root_rot"], dtype=np.float64))
        if "dof_vel" in motion:
            merged_dof_vel.append(np.asarray(motion["dof_vel"], dtype=np.float32))
        if "root_angle_vel" in motion:
            merged_root_angle_vel.append(np.asarray(motion["root_angle_vel"], dtype=np.float32))
        if "proj_grav" in motion:
            merged_proj_grav.append(np.asarray(motion["proj_grav"], dtype=np.float32))
        if "dof_pos_rel" in motion:
            merged_dof_pos_rel.append(np.asarray(motion["dof_pos_rel"], dtype=np.float32))
        cursor = end_frame + 1

    if not merged_dof:
        raise RuntimeError("No valid motion clips to merge.")

    merged_motion = {
        "fps": int(target_fps),
        "source_fps": int(source_fps),
        "dof_names": URDF_JOINT_NAMES,
        "dof_pos": np.concatenate(merged_dof, axis=0),
        "root_pos": np.concatenate(merged_root_pos, axis=0),
        "root_rot": np.concatenate(merged_root_rot, axis=0),
        "segments": segments,
    }
    if merged_dof_vel:
        merged_motion["dof_vel"] = np.concatenate(merged_dof_vel, axis=0)
    if merged_root_angle_vel:
        merged_motion["root_angle_vel"] = np.concatenate(merged_root_angle_vel, axis=0)
    if merged_proj_grav:
        merged_motion["proj_grav"] = np.concatenate(merged_proj_grav, axis=0)
    if merged_dof_pos_rel:
        merged_motion["dof_pos_rel"] = np.concatenate(merged_dof_pos_rel, axis=0)
    return merged_motion, segments


def write_manifest_markdown(manifest_path: Path, segments: list[dict], target_fps: int):
    """Write markdown manifest with id order and time spans."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LAFAN merged segments",
        "",
        f"单位：`time_sec` 是合并后总序列（{target_fps}Hz 时间轴）的秒数区间。",
        "",
        "| id | action | clip_name | start_frame | end_frame | start_time_sec | end_time_sec | num_frames | duration_sec |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seg in segments:
        lines.append(
            f"| {seg['id']} | {seg['action']} | {seg['clip_name']} | "
            f"{seg['start_frame']} | {seg['end_frame']} | "
            f"{seg['start_time_sec']:.4f} | {seg['end_time_sec']:.4f} | "
            f"{seg['num_frames']} | {seg['duration_sec']:.4f} |"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest_json(manifest_json_path: Path, segments: list[dict]):
    """Write machine-readable manifest JSON."""
    manifest_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert LAFAN CSV files to PKL, cubic-resample to target FPS, and merge all clips."
    )
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
        "--source-fps",
        type=int,
        default=CSV_FPS,
        help=f"Source CSV FPS (default: {CSV_FPS})",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=50,
        help="Target FPS after cubic-spline resampling (default: 50)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for CSV files (default: *.csv)",
    )
    parser.add_argument(
        "--merge-name",
        type=str,
        default="lafan_all_50fps.pkl",
        help="Filename for merged output PKL",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="lafan_all_50fps_segments.md",
        help="Filename for markdown segment manifest",
    )
    parser.add_argument(
        "--manifest-json-name",
        type=str,
        default="lafan_all_50fps_segments.json",
        help="Filename for JSON segment manifest",
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

    resampled_items = []
    for csv_path in csv_files:
        pkl_name = csv_path.stem + ".pkl"
        output_path = output_dir / pkl_name
        motion_30 = load_csv_motion(csv_path, csv_fps=args.source_fps)
        motion_resampled = resample_motion_cubic(motion_30, target_fps=args.target_fps)
        motion_resampled = enrich_motion_features(motion_resampled)
        save_motion_pkl(motion_resampled, output_path)
        print(
            f"  {csv_path.name:45s} -> {output_path.name:55s}  "
            f"{motion_30['dof_pos'].shape[0]:5d} -> {motion_resampled['dof_pos'].shape[0]:5d} frames"
        )

        resampled_items.append(
            {
                "clip_name": csv_path.stem,
                "action": infer_action_name(csv_path.stem),
                "source_csv": str(csv_path),
                "motion": motion_resampled,
            }
        )

    merged_motion, segments = build_manifest_and_merged(
        resampled_items,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
    )

    merge_output_path = output_dir / args.merge_name
    save_motion_pkl(merged_motion, merge_output_path)

    manifest_path = output_dir / args.manifest_name
    write_manifest_markdown(manifest_path, segments, target_fps=args.target_fps)

    manifest_json_path = output_dir / args.manifest_json_name
    write_manifest_json(manifest_json_path, segments)

    total_frames = int(merged_motion["dof_pos"].shape[0])
    total_seconds = total_frames / float(args.target_fps)
    print(f"\nDone – {len(csv_files)} files converted and resampled to {args.target_fps}Hz.")
    print(f"Merged PKL: {merge_output_path}  ({total_frames} frames, {total_seconds:.2f} sec)")
    print(f"Manifest (markdown): {manifest_path}")
    print(f"Manifest (json): {manifest_json_path}")
    print("\nDerived fields included: dof_vel, root_angle_vel, proj_grav, dof_pos_rel")


if __name__ == "__main__":
    main()
