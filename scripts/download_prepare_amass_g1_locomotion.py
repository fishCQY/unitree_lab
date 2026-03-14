#!/usr/bin/env python3
"""Download and prepare AMASS-retargeted G1 motions for locomotion AMP.

This script downloads selected .npz files from:
  ember-lab-berkeley/AMASS_Retargeted_for_G1

It filters locomotion-relevant clips, converts them to the AMP PKL format used
in this repo, and computes preprocessed fields required by AMPDemoObsTerm:
  - dof_vel
  - proj_grav
  - root_angle_vel
  - dof_pos_rel
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import hf_hub_download


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


INCLUDE_KEYWORDS = (
    "walk",
    "run",
    "jog",
    "sprint",
    "turn",
    "sidestep",
    "side_step",
    "backward",
    "forward",
    "standto",
    "to_stand",
    "jump",
    "hop",
    "fall",
    "getup",
    "get_up",
    "crouchto",
    "to_crouch",
)

EXCLUDE_KEYWORDS = (
    "dance",
    "fight",
    "martial",
    "punch",
    "kick",
    "box",
    "basketball",
    "tennis",
    "golf",
    "soccer",
    "baseball",
    "throw",
    "catch",
    "wave",
    "clap",
    "phone",
)

DEFAULT_SOURCE_SUBDIRS = (
    "g1/ACCAD/Female1Walking_c3d",
    "g1/ACCAD/Female1Running_c3d",
    "g1/ACCAD/Male1Walking_c3d",
    "g1/ACCAD/Male1Running_c3d",
    "g1/ACCAD/Male2Walking_c3d",
    "g1/ACCAD/Male2Running_c3d",
    "g1/ACCAD/MartialArtsWalksTurns_c3d",
)


@dataclass
class MotionMeta:
    src_file: str
    out_file: str
    category: str
    src_fps: float
    out_fps: int
    frames: int


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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
    q_conj = quat_conjugate(q)
    v_quat = np.zeros((*v.shape[:-1], 4), dtype=v.dtype)
    v_quat[..., 1:] = v
    out = quat_multiply(q_conj, quat_multiply(v_quat, q))
    return out[..., 1:]


def compute_body_angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
    dq = np.zeros_like(quats)
    dq[1:-1] = (quats[2:] - quats[:-2]) / (2 * dt)
    dq[0] = (quats[1] - quats[0]) / dt
    dq[-1] = (quats[-1] - quats[-2]) / dt
    omega_quat = 2.0 * quat_multiply(quat_conjugate(quats), dq)
    return omega_quat[..., 1:]


def central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(x)
    out[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    out[0] = (x[1] - x[0]) / dt
    out[-1] = (x[-1] - x[-2]) / dt
    return out


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.clip(n, 1e-8, None)
    return q / n


def resample_2d(data: np.ndarray, src_fps: float, dst_fps: int) -> np.ndarray:
    if int(round(src_fps)) == int(dst_fps):
        return data
    t_src = np.arange(data.shape[0], dtype=np.float64) / float(src_fps)
    t_end = t_src[-1]
    n_dst = int(round(t_end * dst_fps)) + 1
    t_dst = np.arange(n_dst, dtype=np.float64) / float(dst_fps)
    out = np.zeros((n_dst, data.shape[1]), dtype=np.float64)
    for i in range(data.shape[1]):
        out[:, i] = np.interp(t_dst, t_src, data[:, i])
    return out


def classify_clip(path: str) -> str | None:
    name = Path(path).name.lower()
    if not name.endswith("_poses_120_jpos.npz"):
        return None
    if any(k in name for k in EXCLUDE_KEYWORDS):
        return None
    if any(k in name for k in INCLUDE_KEYWORDS):
        if "fall" in name or "getup" in name or "get_up" in name:
            return "recovery"
        if "jump" in name or "hop" in name:
            return "jump"
        if "run" in name or "sprint" in name or "jog" in name:
            return "run"
        return "walk"
    return None


def list_files_under_subdir(repo_id: str, subdir: str, endpoint: str, timeout_s: int = 30) -> list[str]:
    url = f"{endpoint.rstrip('/')}/api/datasets/{repo_id}/tree/main/{subdir}?recursive=true"
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    out = []
    for item in payload:
        if item.get("type") == "file":
            p = item.get("path", "")
            if p:
                out.append(p)
    return out


def convert_one(npz_path: Path, out_pkl: Path, target_fps: int) -> tuple[int, float]:
    d = np.load(npz_path)
    src_fps = float(d["fps"][0])

    dof_pos = np.asarray(d["dof_positions"], dtype=np.float64)
    root_pos = np.asarray(d["body_positions"][:, 0, :], dtype=np.float64)

    # body_rotations in this dataset are xyzw; convert to wxyz
    root_rot_xyzw = np.asarray(d["body_rotations"][:, 0, :], dtype=np.float64)
    root_rot = np.stack(
        [root_rot_xyzw[:, 3], root_rot_xyzw[:, 0], root_rot_xyzw[:, 1], root_rot_xyzw[:, 2]],
        axis=-1,
    )

    dof_pos = resample_2d(dof_pos, src_fps, target_fps)
    root_pos = resample_2d(root_pos, src_fps, target_fps)
    root_rot = normalize_quat_wxyz(resample_2d(root_rot, src_fps, target_fps))

    dt = 1.0 / target_fps
    dof_vel = central_diff(dof_pos, dt)
    root_angle_vel = compute_body_angular_velocity(root_rot, dt)
    gravity_world = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (root_rot.shape[0], 1))
    proj_grav = quat_rotate_inverse(root_rot, gravity_world)
    dof_pos_rel = dof_pos - DEFAULT_JOINT_POS[np.newaxis, :]

    out = {
        "fps": int(target_fps),
        "dof_pos": dof_pos.astype(np.float32),
        "dof_vel": dof_vel.astype(np.float32),
        "dof_pos_rel": dof_pos_rel.astype(np.float32),
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "root_angle_vel": root_angle_vel.astype(np.float32),
        "proj_grav": proj_grav.astype(np.float32),
        "loop_mode": "wrap",
    }

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)
    return dof_pos.shape[0], src_fps


def main():
    parser = argparse.ArgumentParser(description="Download and prepare AMASS->G1 locomotion AMP dataset.")
    parser.add_argument("--repo-id", type=str, default="ember-lab-berkeley/AMASS_Retargeted_for_G1")
    parser.add_argument("--endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument(
        "--source-subdirs",
        type=str,
        default=",".join(DEFAULT_SOURCE_SUBDIRS),
        help="Comma-separated repo subdirs to scan (avoid listing the whole giant repo).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/amass_locomotion",
    )
    parser.add_argument("--cache-dir", type=str, default=".hf_cache")
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--max-files", type=int, default=300, help="Limit number of downloaded clips (0=all).")
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    subdirs = [x.strip() for x in args.source_subdirs.split(",") if x.strip()]
    print(f"[AMASS] Listing files from {len(subdirs)} subdirs ...")
    files: list[str] = []
    for sd in subdirs:
        try:
            files.extend(list_files_under_subdir(args.repo_id, sd, args.endpoint))
        except Exception as e:
            print(f"[AMASS] Warning: failed listing {sd}: {e}")
    candidates = []
    for f in files:
        if not f.startswith("g1/"):
            continue
        category = classify_clip(f)
        if category is not None:
            candidates.append((f, category))
    candidates.sort(key=lambda x: x[0])
    if args.max_files > 0:
        candidates = candidates[: args.max_files]
    print(f"[AMASS] Selected {len(candidates)} locomotion clips")

    manifest: list[MotionMeta] = []
    for i, (src_file, category) in enumerate(candidates, start=1):
        local_npz = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=src_file,
            cache_dir=str(cache_dir),
            endpoint=args.endpoint,
        )
        out_name = src_file.replace("/", "__").replace(".npz", ".pkl")
        out_pkl = out_dir / out_name
        frames, src_fps = convert_one(Path(local_npz), out_pkl, target_fps=args.target_fps)
        manifest.append(
            MotionMeta(
                src_file=src_file,
                out_file=out_name,
                category=category,
                src_fps=float(src_fps),
                out_fps=int(args.target_fps),
                frames=int(frames),
            )
        )
        if i % 20 == 0 or i == len(candidates):
            print(f"[AMASS] Converted {i}/{len(candidates)}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([m.__dict__ for m in manifest], f, indent=2, ensure_ascii=True)
    print(f"[AMASS] Done. PKL files: {len(manifest)}")
    print(f"[AMASS] Output: {out_dir}")
    print(f"[AMASS] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

