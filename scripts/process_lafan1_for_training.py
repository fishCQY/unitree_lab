#!/usr/bin/env python3
"""Process LAFAN1 retargeted G1 CSV data into AMP and DeepMimic PKL formats.

AMP format (50Hz, per-category clips):
    PKL = {clip_name: {fps, dof_pos, dof_vel, root_pos, root_rot, root_angle_vel, proj_grav, ...}}
    - root_rot in (x, y, z, w) format
    - Resampled from 30fps to 50fps via cubic spline
    - Grouped by action category: walk_clips, run_clips, etc.

DeepMimic format (original fps, complete motions):
    PKL = {clip_name: {fps, dof_pos, dof_vel, root_pos, root_rot, root_vel, root_angle_vel,
                       robot_points, smpl_points, feet_contact, hands_contact, ...}}
    - root_rot in (x, y, z, w) format
    - robot_points computed via pinocchio FK
    - feet_contact estimated from foot height/velocity
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pinocchio as pin


# ── G1 joint order (CSV / URDF) ──────────────────────────────────────────────

URDF_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

BODY_NAMES_FOR_ROBOT_POINTS = [
    "pelvis",
    "left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link",
]

FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

CSV_FPS = 30
DEFAULT_JOINT_POS = np.array([
    -0.20, 0.00, 0.00, 0.42, -0.23, 0.00,
    -0.20, 0.00, 0.00, 0.42, -0.23, 0.00,
    0.00, 0.00, 0.00,
    0.35, 0.18, 0.00, 0.87, 0.00, 0.00, 0.00,
    0.35, -0.18, 0.00, 0.87, 0.00, 0.00, 0.00,
], dtype=np.float64)

# AMP action category mapping
ACTION_CATEGORIES = {
    "walk": ["walk1", "walk2", "walk3", "walk4"],
    "run": ["run1", "run2", "sprint1"],
    "dance": ["dance1", "dance2"],
    "jump": ["jumps1"],
    "fight": ["fight1", "fightAndSports1"],
    "getup": ["fallAndGetUp1", "fallAndGetUp2", "fallAndGetUp3"],
}


# ── Quaternion utils (w, x, y, z) ────────────────────────────────────────────

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_conj = quat_conjugate(q_wxyz)
    v_quat = np.zeros((*v.shape[:-1], 4), dtype=v.dtype)
    v_quat[..., 1:] = v
    result = quat_multiply(q_conj, quat_multiply(v_quat, q_wxyz))
    return result[..., 1:]


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def ensure_quaternion_continuity(quat: np.ndarray) -> np.ndarray:
    out = np.array(quat, dtype=np.float64)
    for i in range(1, out.shape[0]):
        if np.dot(out[i - 1], out[i]) < 0.0:
            out[i] = -out[i]
    return out


# ── Finite differences ────────────────────────────────────────────────────────

def central_diff(values: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if values.shape[0] <= 1:
        return out
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    return out


def compute_body_angular_velocity(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    dq = central_diff(quats_wxyz, dt)
    omega_quat = 2.0 * quat_multiply(quat_conjugate(quats_wxyz), dq)
    return omega_quat[..., 1:]


# ── Cubic spline resampling ───────────────────────────────────────────────────

def natural_cubic_spline_sample(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    t_src = np.asarray(t_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)

    n = t_src.shape[0]
    if n == 1:
        return np.repeat(y_src, repeats=t_dst.shape[0], axis=0)
    if n == 2:
        w = ((t_dst - t_src[0]) / (t_src[1] - t_src[0]))[:, None]
        return (1.0 - w) * y_src[0:1] + w * y_src[1:2]

    h = t_src[1:] - t_src[:-1]
    d = y_src.shape[1]
    a = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros((n, d), dtype=np.float64)
    a[0, 0] = 1.0
    a[-1, -1] = 1.0

    for i in range(1, n - 1):
        h_prev, h_next = h[i - 1], h[i]
        a[i, i - 1] = h_prev
        a[i, i] = 2.0 * (h_prev + h_next)
        a[i, i + 1] = h_next
        rhs[i] = 6.0 * ((y_src[i + 1] - y_src[i]) / h_next - (y_src[i] - y_src[i - 1]) / h_prev)

    m2 = np.linalg.solve(a, rhs)
    idx = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, n - 2)
    x0, x1 = t_src[idx], t_src[idx + 1]
    hi = (x1 - x0)[:, None]
    a_w = (x1 - t_dst)[:, None] / hi
    b_w = (t_dst - x0)[:, None] / hi
    y0, y1 = y_src[idx], y_src[idx + 1]
    m0, m1 = m2[idx], m2[idx + 1]
    return a_w * y0 + b_w * y1 + ((a_w**3 - a_w) * m0 + (b_w**3 - b_w) * m1) * hi**2 / 6.0


def resample_motion(motion: dict, target_fps: int) -> dict:
    src_fps = int(motion["fps"])
    if src_fps == target_fps:
        return dict(motion)

    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)

    num_src = dof_pos.shape[0]
    if num_src < 2:
        out = dict(motion)
        out["fps"] = int(target_fps)
        return out

    t_src = np.arange(num_src, dtype=np.float64) / float(src_fps)
    duration = (num_src - 1) / float(src_fps)
    num_dst = int(round(duration * float(target_fps))) + 1
    t_dst = np.linspace(0.0, duration, num=num_dst, dtype=np.float64)

    dof_pos_dst = natural_cubic_spline_sample(t_src, dof_pos, t_dst)
    root_pos_dst = natural_cubic_spline_sample(t_src, root_pos, t_dst)
    root_rot_cont = ensure_quaternion_continuity(root_rot)
    root_rot_dst = natural_cubic_spline_sample(t_src, root_rot_cont, t_dst)
    norm = np.linalg.norm(root_rot_dst, axis=-1, keepdims=True)
    root_rot_dst = root_rot_dst / np.clip(norm, 1e-12, None)

    return {
        "fps": int(target_fps),
        "dof_pos": dof_pos_dst,
        "dof_names": motion["dof_names"],
        "root_pos": root_pos_dst,
        "root_rot": root_rot_dst,
    }


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv_motion(csv_path: Path) -> dict:
    data = np.genfromtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape[1] == 36, f"Expected 36 columns, got {data.shape[1]}"
    T = data.shape[0]

    root_pos = data[:, 0:3].astype(np.float64)
    qx, qy, qz, qw = data[:, 3], data[:, 4], data[:, 5], data[:, 6]
    root_rot_wxyz = np.stack([qw, qx, qy, qz], axis=-1).astype(np.float64)
    dof_pos = data[:, 7:36].astype(np.float64)
    assert dof_pos.shape == (T, 29)

    return {
        "fps": CSV_FPS,
        "dof_pos": dof_pos,
        "dof_names": URDF_JOINT_NAMES,
        "root_pos": root_pos,
        "root_rot": root_rot_wxyz,
    }


# ── Derived features ─────────────────────────────────────────────────────────

def compute_amp_features(motion: dict) -> dict:
    """Add AMP-required features: dof_vel, root_angle_vel, proj_grav, dof_pos_rel.
    root_rot stored as (x, y, z, w) for AMP compatibility."""
    fps = float(motion["fps"])
    dt = 1.0 / fps
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    root_rot_wxyz = np.asarray(motion["root_rot"], dtype=np.float64)
    num_frames = dof_pos.shape[0]

    dof_vel = central_diff(dof_pos, dt)
    root_angle_vel = compute_body_angular_velocity(root_rot_wxyz, dt)
    gravity_world = np.tile(np.array([0.0, 0.0, -1.0]), (num_frames, 1))
    proj_grav = quat_rotate_inverse(root_rot_wxyz, gravity_world)
    dof_pos_rel = dof_pos - DEFAULT_JOINT_POS.reshape(1, -1)

    motion["dof_vel"] = dof_vel.astype(np.float32)
    motion["root_angle_vel"] = root_angle_vel.astype(np.float32)
    motion["proj_grav"] = proj_grav.astype(np.float32)
    motion["dof_pos_rel"] = dof_pos_rel.astype(np.float32)
    motion["root_rot"] = quat_wxyz_to_xyzw(root_rot_wxyz).astype(np.float64)
    motion["root_pos"] = np.asarray(motion["root_pos"], dtype=np.float64)
    motion["dof_pos"] = dof_pos.astype(np.float64)
    return motion


def compute_fk_robot_points(
    model: pin.Model,
    pin_data: pin.Data,
    body_frame_ids: list[int],
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    """Compute robot body positions in world frame using pinocchio FK."""
    num_frames = dof_pos.shape[0]
    num_bodies = len(body_frame_ids)
    robot_points = np.zeros((num_frames, num_bodies, 3), dtype=np.float64)

    for i in range(num_frames):
        q = np.zeros(model.nq, dtype=np.float64)
        q[:] = dof_pos[i]
        pin.forwardKinematics(model, pin_data, q)
        pin.updateFramePlacements(model, pin_data)

        w, x, y, z = root_rot_wxyz[i]
        root_rotation = pin.Quaternion(w, x, y, z).toRotationMatrix()
        root_translation = root_pos[i]

        for j, fid in enumerate(body_frame_ids):
            pos_local = pin_data.oMf[fid].translation.copy()
            pos_world = root_rotation @ pos_local + root_translation
            robot_points[i, j] = pos_world

    return robot_points


def estimate_feet_contact(
    robot_points: np.ndarray,
    foot_indices: list[int],
    height_threshold: float = 0.05,
    velocity_threshold: float = 0.3,
    dt: float | None = None,
) -> np.ndarray:
    """Estimate foot contact from height and velocity."""
    num_frames = robot_points.shape[0]
    num_feet = len(foot_indices)
    contact = np.zeros((num_frames, num_feet), dtype=bool)

    for fi, foot_idx in enumerate(foot_indices):
        foot_z = robot_points[:, foot_idx, 2]
        min_z = np.min(foot_z)
        height_above_ground = foot_z - min_z

        height_contact = height_above_ground < height_threshold

        if dt is not None and num_frames > 1:
            foot_vel = central_diff(robot_points[:, foot_idx], dt)
            speed = np.linalg.norm(foot_vel, axis=-1)
            vel_contact = speed < velocity_threshold
            contact[:, fi] = height_contact & vel_contact
        else:
            contact[:, fi] = height_contact

    return contact


def compute_mimic_features(
    motion: dict,
    model: pin.Model,
    pin_data: pin.Data,
    body_frame_ids: list[int],
    foot_indices_in_body: list[int],
) -> dict:
    """Add DeepMimic-required features: root_vel, root_angle_vel, dof_vel, robot_points,
    smpl_points, feet_contact, hands_contact.
    root_rot stored as (x, y, z, w) for MotionLib compatibility."""
    fps = float(motion["fps"])
    dt = 1.0 / fps
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot_wxyz = np.asarray(motion["root_rot"], dtype=np.float64)
    num_frames = dof_pos.shape[0]

    dof_vel = central_diff(dof_pos, dt)
    root_angle_vel = compute_body_angular_velocity(root_rot_wxyz, dt)

    root_vel_world = central_diff(root_pos, dt)
    root_vel = np.zeros_like(root_vel_world)
    for i in range(num_frames):
        root_vel[i] = quat_rotate_inverse(
            root_rot_wxyz[i:i+1], root_vel_world[i:i+1]
        )[0]

    robot_points = compute_fk_robot_points(
        model, pin_data, body_frame_ids,
        root_pos, root_rot_wxyz, dof_pos,
    )

    feet_contact = estimate_feet_contact(
        robot_points, foot_indices_in_body, dt=dt,
    )

    motion["dof_pos"] = dof_pos.astype(np.float64)
    motion["dof_vel"] = dof_vel.astype(np.float64)
    motion["root_pos"] = root_pos.astype(np.float64)
    motion["root_rot"] = quat_wxyz_to_xyzw(root_rot_wxyz).astype(np.float64)
    motion["root_vel"] = root_vel.astype(np.float64)
    motion["root_angle_vel"] = root_angle_vel.astype(np.float64)
    motion["robot_points"] = robot_points.astype(np.float64)
    motion["smpl_points"] = robot_points.astype(np.float32)
    motion["feet_contact"] = feet_contact
    motion["hands_contact"] = np.zeros((num_frames, 2), dtype=bool)
    return motion


# ── Action name inference ─────────────────────────────────────────────────────

def infer_action_prefix(clip_name: str) -> str:
    if "_subject" in clip_name:
        return clip_name.split("_subject", maxsplit=1)[0]
    return clip_name


def get_action_category(clip_name: str) -> str:
    prefix = infer_action_prefix(clip_name)
    for category, prefixes in ACTION_CATEGORIES.items():
        if any(prefix.startswith(p) or prefix == p for p in prefixes):
            return category
    return "other"


# ── Pinocchio model setup ────────────────────────────────────────────────────

def setup_pinocchio(urdf_path: str) -> tuple[pin.Model, pin.Data, list[int], list[int]]:
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    body_frame_ids = []
    for bname in BODY_NAMES_FOR_ROBOT_POINTS:
        found = False
        for i in range(model.nframes):
            if model.frames[i].name == bname:
                body_frame_ids.append(i)
                found = True
                break
        if not found:
            raise ValueError(f"Body '{bname}' not found in URDF")

    foot_indices_in_body = []
    for fname in FOOT_BODY_NAMES:
        idx = BODY_NAMES_FOR_ROBOT_POINTS.index(fname)
        foot_indices_in_body.append(idx)

    return model, data, body_frame_ids, foot_indices_in_body


# ── Save helper ───────────────────────────────────────────────────────────────

def save_pkl(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved: {path} ({len(data)} motions)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Process LAFAN1 G1 CSV data into AMP and DeepMimic PKL formats."
    )
    parser.add_argument(
        "--input-dir", type=str,
        default="source/unitree_lab/unitree_lab/data/LAFAN1_Retargeting_Dataset/g1",
    )
    parser.add_argument(
        "--urdf", type=str,
        default="source/unitree_lab/unitree_lab/data/LAFAN1_Retargeting_Dataset/robot_description/g1/g1_29dof_rev_1_0.urdf",
    )
    parser.add_argument(
        "--amp-output-dir", type=str,
        default="source/unitree_lab/unitree_lab/data/AMP",
    )
    parser.add_argument(
        "--mimic-output-dir", type=str,
        default="source/unitree_lab/unitree_lab/data/Mimic",
    )
    parser.add_argument("--amp-fps", type=int, default=50)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    amp_output_dir = Path(args.amp_output_dir)
    mimic_output_dir = Path(args.mimic_output_dir)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    print(f"AMP output: {amp_output_dir} (target {args.amp_fps}Hz)")
    print(f"DeepMimic output: {mimic_output_dir} (original {CSV_FPS}Hz)")
    print()

    # Setup pinocchio for FK
    print("Loading URDF for forward kinematics...")
    model, pin_data, body_frame_ids, foot_indices_in_body = setup_pinocchio(args.urdf)
    print(f"  Model: {model.nq} DoFs, tracking {len(body_frame_ids)} bodies")
    print()

    # ── Process all CSVs ──────────────────────────────────────────────────
    amp_by_category: dict[str, dict] = defaultdict(dict)
    mimic_all: dict[str, dict] = {}

    for csv_path in csv_files:
        clip_name = csv_path.stem
        category = get_action_category(clip_name)
        print(f"  [{category:6s}] {clip_name:40s}", end="", flush=True)

        # Load CSV (30fps, root_rot in wxyz)
        raw_motion = load_csv_motion(csv_path)
        src_frames = raw_motion["dof_pos"].shape[0]

        # ── AMP path: resample to 50Hz, compute AMP features ─────────
        amp_motion = resample_motion(raw_motion, target_fps=args.amp_fps)
        amp_motion = compute_amp_features(amp_motion)
        amp_frames = amp_motion["dof_pos"].shape[0]
        amp_by_category[category][clip_name] = amp_motion

        # ── DeepMimic path: keep original 30fps, compute full features ─
        mimic_motion = dict(raw_motion)
        mimic_motion = compute_mimic_features(
            mimic_motion, model, pin_data,
            body_frame_ids, foot_indices_in_body,
        )
        mimic_all[clip_name] = mimic_motion

        print(f"  {src_frames:5d} -> AMP:{amp_frames:5d}@{args.amp_fps}Hz  Mimic:{src_frames:5d}@{CSV_FPS}Hz")

    # ── Save AMP PKLs (per category) ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Saving AMP PKLs (per category)...")
    for category, motions in sorted(amp_by_category.items()):
        pkl_path = amp_output_dir / f"lafan_{category}_clips.pkl"
        save_pkl(motions, pkl_path)

    # Also save a merged all-in-one AMP PKL
    amp_all = {}
    for motions in amp_by_category.values():
        amp_all.update(motions)
    save_pkl(amp_all, amp_output_dir / "lafan_all_clips.pkl")

    # ── Save DeepMimic PKL (single file) ─────────────────────────────────
    print(f"\nSaving DeepMimic PKL...")
    save_pkl(mimic_all, mimic_output_dir / "lafan.pkl")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  AMP ({args.amp_fps}Hz):")
    total_amp = 0
    for cat, motions in sorted(amp_by_category.items()):
        n_clips = len(motions)
        n_frames = sum(m["dof_pos"].shape[0] for m in motions.values())
        total_amp += n_frames
        print(f"    {cat:10s}: {n_clips:3d} clips, {n_frames:7d} frames ({n_frames/args.amp_fps:.1f}s)")
    print(f"    {'TOTAL':10s}: {sum(len(m) for m in amp_by_category.values()):3d} clips, {total_amp:7d} frames")

    total_mimic = sum(m["dof_pos"].shape[0] for m in mimic_all.values())
    print(f"  DeepMimic ({CSV_FPS}Hz):")
    print(f"    {len(mimic_all):3d} motions, {total_mimic:7d} frames ({total_mimic/CSV_FPS:.1f}s)")
    print("\nDone.")


if __name__ == "__main__":
    main()
