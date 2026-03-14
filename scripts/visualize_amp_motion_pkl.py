#!/usr/bin/env python3
"""Quick visualization for AMP motion PKL files."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def load_motion(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def summarize(data: dict):
    fps = float(data["fps"])
    dof_pos = np.asarray(data["dof_pos"])
    root_pos = np.asarray(data["root_pos"])
    duration = dof_pos.shape[0] / fps
    root_speed = np.linalg.norm((root_pos[1:, :2] - root_pos[:-1, :2]) * fps, axis=1) if dof_pos.shape[0] > 1 else np.array([0.0])
    print(f"frames: {dof_pos.shape[0]}")
    print(f"fps: {fps:.3f}")
    print(f"duration_s: {duration:.2f}")
    print(f"mean_root_xy_speed_mps: {root_speed.mean():.3f}")
    print(f"max_root_xy_speed_mps: {root_speed.max():.3f}")
    print(f"mean_abs_joint_pos_rad: {np.abs(dof_pos).mean():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize AMP PKL motion data.")
    parser.add_argument("--file", type=str, required=True, help="Path to a .pkl file")
    parser.add_argument("--save", type=str, default="", help="Optional output image path (.png)")
    args = parser.parse_args()

    data = load_motion(Path(args.file))
    summarize(data)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib is not available: {e}")
        print("[hint] Install matplotlib to get plots.")
        return

    fps = float(data["fps"])
    dof_pos = np.asarray(data["dof_pos"])
    root_pos = np.asarray(data["root_pos"])
    t = np.arange(dof_pos.shape[0]) / fps
    root_speed = np.linalg.norm((root_pos[1:, :2] - root_pos[:-1, :2]) * fps, axis=1) if dof_pos.shape[0] > 1 else np.array([0.0])

    # Representative joints
    idx = {
        "left_knee": 3,
        "right_knee": 9,
        "left_elbow": 18,
        "right_elbow": 25,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(root_pos[:, 0], root_pos[:, 1], linewidth=1.0)
    axes[0, 0].set_title("Root XY trajectory")
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].axis("equal")

    axes[0, 1].plot(t[1:], root_speed, linewidth=1.0)
    axes[0, 1].set_title("Root XY speed")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("m/s")

    axes[1, 0].plot(t, dof_pos[:, idx["left_knee"]], label="left_knee")
    axes[1, 0].plot(t, dof_pos[:, idx["right_knee"]], label="right_knee")
    axes[1, 0].set_title("Knee joint trajectories")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("rad")
    axes[1, 0].legend()

    axes[1, 1].plot(t, dof_pos[:, idx["left_elbow"]], label="left_elbow")
    axes[1, 1].plot(t, dof_pos[:, idx["right_elbow"]], label="right_elbow")
    axes[1, 1].set_title("Elbow joint trajectories")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("rad")
    axes[1, 1].legend()

    fig.tight_layout()
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
        print(f"saved plot: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

