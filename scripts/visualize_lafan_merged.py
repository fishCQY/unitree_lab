#!/usr/bin/env python3
"""Visualize merged LAFAN PKL and rerun selected id/time ranges."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


DEFAULT_MERGED_PKL = "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/lafan/lafan_all_50fps.pkl"


def load_motion(path: Path) -> dict:
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            # Compatibility fallback for pickle files saved under environments
            # where NumPy internal module path used "numpy._core.*".
            if e.name and e.name.startswith("numpy._core"):
                import numpy.core as np_core
                import numpy.core.multiarray as np_core_multiarray
                import numpy.core.numeric as np_core_numeric
                import numpy.core.umath as np_core_umath

                sys.modules.setdefault("numpy._core", np_core)
                sys.modules.setdefault("numpy._core.multiarray", np_core_multiarray)
                sys.modules.setdefault("numpy._core.numeric", np_core_numeric)
                sys.modules.setdefault("numpy._core.umath", np_core_umath)
                f.seek(0)
                return pickle.load(f)
            raise


def clamp_range(start: int, end: int, total: int) -> tuple[int, int]:
    start = max(0, min(start, total - 1))
    end = max(start, min(end, total - 1))
    return start, end


def print_id_table(segments: list[dict]) -> None:
    print("id | action            | clip_name                    | start_sec | end_sec | frames")
    print("---|-------------------|------------------------------|----------:|--------:|------:")
    for seg in segments:
        print(
            f"{seg['id']:2d} | "
            f"{str(seg.get('action', ''))[:17]:17s} | "
            f"{str(seg.get('clip_name', ''))[:28]:28s} | "
            f"{float(seg.get('start_time_sec', 0.0)):9.3f} | "
            f"{float(seg.get('end_time_sec', 0.0)):7.3f} | "
            f"{int(seg.get('num_frames', 0)):6d}"
        )


def choose_range(data: dict, args: argparse.Namespace) -> tuple[int, int, dict | None]:
    fps = float(data["fps"])
    total_frames = int(np.asarray(data["dof_pos"]).shape[0])
    segments = data.get("segments", [])

    seg = None
    if args.id is not None:
        if not segments:
            raise ValueError("This PKL has no 'segments' field; cannot select by --id.")
        seg_map = {int(s["id"]): s for s in segments}
        if args.id not in seg_map:
            raise ValueError(f"Invalid --id {args.id}. Use --list-ids to inspect available ids.")
        seg = seg_map[args.id]
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
    else:
        start = 0
        end = total_frames - 1

    # Absolute override
    if args.start_frame is not None:
        start = int(args.start_frame)
    if args.end_frame is not None:
        end = int(args.end_frame)

    # Absolute second override
    if args.start_sec is not None:
        start = int(round(float(args.start_sec) * fps))
    if args.end_sec is not None:
        end = int(round(float(args.end_sec) * fps))

    # Relative window within selected base range.
    if args.rel_start_sec is not None:
        start = start + int(round(float(args.rel_start_sec) * fps))
    if args.rel_end_sec is not None:
        end = start + int(round(float(args.rel_end_sec) * fps))

    return clamp_range(start, end, total_frames), seg


def summarize_slice(data: dict, start: int, end: int) -> None:
    fps = float(data["fps"])
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)[start : end + 1]
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)[start : end + 1]
    duration = dof_pos.shape[0] / fps
    if dof_pos.shape[0] > 1:
        root_speed = np.linalg.norm((root_pos[1:, :2] - root_pos[:-1, :2]) * fps, axis=1)
    else:
        root_speed = np.array([0.0], dtype=np.float64)
    print(f"slice_frames: [{start}, {end}] ({dof_pos.shape[0]} frames)")
    print(f"slice_time_s: [{start / fps:.3f}, {end / fps:.3f}] ({duration:.3f}s)")
    print(f"mean_root_xy_speed_mps: {root_speed.mean():.3f}")
    print(f"max_root_xy_speed_mps: {root_speed.max():.3f}")
    print(f"mean_abs_joint_pos_rad: {np.abs(dof_pos).mean():.3f}")


def render_plots(
    data: dict,
    start: int,
    end: int,
    title_prefix: str,
    save_plot: str,
    rerun_loops: int,
    stride: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as e:
        print(f"[warn] matplotlib is not available: {e}")
        print("[hint] Install matplotlib to visualize: pip install matplotlib")
        return

    fps = float(data["fps"])
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)[start : end + 1]
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)[start : end + 1]
    t = (np.arange(dof_pos.shape[0], dtype=np.float64) + start) / fps
    if dof_pos.shape[0] > 1:
        root_speed = np.linalg.norm((root_pos[1:, :2] - root_pos[:-1, :2]) * fps, axis=1)
    else:
        root_speed = np.array([0.0], dtype=np.float64)

    idx = {
        "left_knee": 3,
        "right_knee": 9,
        "left_elbow": 18,
        "right_elbow": 25,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title_prefix)

    axes[0, 0].plot(root_pos[:, 0], root_pos[:, 1], linewidth=1.0, alpha=0.8)
    marker_xy, = axes[0, 0].plot([root_pos[0, 0]], [root_pos[0, 1]], "ro", markersize=4)
    axes[0, 0].set_title("Root XY trajectory")
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].axis("equal")

    axes[0, 1].plot(t[1:], root_speed, linewidth=1.0, alpha=0.8)
    marker_v = axes[0, 1].axvline(t[0], color="r", linewidth=1.0)
    axes[0, 1].set_title("Root XY speed")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("m/s")

    axes[1, 0].plot(t, dof_pos[:, idx["left_knee"]], label="left_knee")
    axes[1, 0].plot(t, dof_pos[:, idx["right_knee"]], label="right_knee")
    marker_k = axes[1, 0].axvline(t[0], color="r", linewidth=1.0)
    axes[1, 0].set_title("Knee joint trajectories")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("rad")
    axes[1, 0].legend()

    axes[1, 1].plot(t, dof_pos[:, idx["left_elbow"]], label="left_elbow")
    axes[1, 1].plot(t, dof_pos[:, idx["right_elbow"]], label="right_elbow")
    marker_e = axes[1, 1].axvline(t[0], color="r", linewidth=1.0)
    axes[1, 1].set_title("Elbow joint trajectories")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("rad")
    axes[1, 1].legend()

    fig.tight_layout()

    if save_plot:
        out = Path(save_plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
        print(f"saved plot: {out}")

    if rerun_loops <= 0:
        plt.show()
        return

    stride = max(1, int(stride))
    frame_indices = list(range(0, dof_pos.shape[0], stride))
    if frame_indices[-1] != dof_pos.shape[0] - 1:
        frame_indices.append(dof_pos.shape[0] - 1)
    total_anim_frames = len(frame_indices) * rerun_loops

    def update(k: int):
        local_idx = frame_indices[k % len(frame_indices)]
        marker_xy.set_data([root_pos[local_idx, 0]], [root_pos[local_idx, 1]])
        marker_v.set_xdata([t[local_idx], t[local_idx]])
        marker_k.set_xdata([t[local_idx], t[local_idx]])
        marker_e.set_xdata([t[local_idx], t[local_idx]])
        loop_idx = k // len(frame_indices) + 1
        fig.suptitle(f"{title_prefix} | rerun {loop_idx}/{rerun_loops}")
        return marker_xy, marker_v, marker_k, marker_e

    interval_ms = (1000.0 / fps) * stride
    _ani = FuncAnimation(
        fig,
        update,
        frames=total_anim_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize merged LAFAN data and rerun one id or a custom time slice."
    )
    parser.add_argument("--file", type=str, default=DEFAULT_MERGED_PKL, help="Path to merged LAFAN .pkl")
    parser.add_argument("--list-ids", action="store_true", help="Print id/action/time table from segments and exit")
    parser.add_argument("--id", type=int, default=None, help="Select one segment id from merged file")
    parser.add_argument("--start-frame", type=int, default=None, help="Absolute start frame in merged timeline")
    parser.add_argument("--end-frame", type=int, default=None, help="Absolute end frame in merged timeline")
    parser.add_argument("--start-sec", type=float, default=None, help="Absolute start second in merged timeline")
    parser.add_argument("--end-sec", type=float, default=None, help="Absolute end second in merged timeline")
    parser.add_argument(
        "--rel-start-sec",
        type=float,
        default=None,
        help="Relative start second from the selected base range (id/full-range)",
    )
    parser.add_argument(
        "--rel-end-sec",
        type=float,
        default=None,
        help="Relative end second from the computed start position",
    )
    parser.add_argument("--save-plot", type=str, default="", help="Optional static figure output (.png)")
    parser.add_argument(
        "--rerun-loops",
        type=int,
        default=1,
        help="How many times to replay selected slice in animation; 0 means static only",
    )
    parser.add_argument("--stride", type=int, default=1, help="Animation frame stride for faster playback")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_motion(Path(args.file))
    fps = float(data["fps"])
    total_frames = int(np.asarray(data["dof_pos"]).shape[0])
    total_seconds = total_frames / fps
    segments = data.get("segments", [])

    print(f"file: {args.file}")
    print(f"fps: {fps:.3f}")
    print(f"total_frames: {total_frames}")
    print(f"total_duration_s: {total_seconds:.3f}")
    print(f"num_segments: {len(segments)}")

    if args.list_ids:
        if not segments:
            print("No segments available in this PKL.")
        else:
            print_id_table(segments)
        return

    (start, end), seg = choose_range(data, args)

    if seg is not None:
        print(
            "selected_id: "
            f"{seg['id']} | action={seg.get('action', '')} | clip={seg.get('clip_name', '')} | "
            f"segment_time=[{float(seg['start_time_sec']):.3f}, {float(seg['end_time_sec']):.3f}]"
        )

    summarize_slice(data, start, end)
    title_prefix = f"LAFAN merged [{start}:{end}] ({(end - start + 1) / fps:.2f}s)"
    if seg is not None:
        title_prefix += f" | id={seg['id']} {seg.get('action', '')}"

    render_plots(
        data=data,
        start=start,
        end=end,
        title_prefix=title_prefix,
        save_plot=args.save_plot,
        rerun_loops=args.rerun_loops,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()

