#!/usr/bin/env python3
"""Run Sim2Sim evaluation for locomotion policies.

This script evaluates IsaacLab-trained locomotion policies in MuJoCo.

Usage:
    # Single task evaluation
    python run_sim2sim_locomotion.py --robot go2 --onnx policy.onnx --task flat_forward
    
    # Batch evaluation (all tasks)
    python run_sim2sim_locomotion.py --robot go2 --onnx policy.onnx --batch
    
    # With rendering
    python run_sim2sim_locomotion.py --robot go2 --onnx policy.onnx --task flat_forward --render
    
    # Save videos
    python run_sim2sim_locomotion.py --robot go2 --onnx policy.onnx --batch --save-videos

    # Deploy-like continuous control loop (recommended for sim2sim debug)
    # - Builds observations from MuJoCo state (qpos/qvel)
    # - Runs ONNX inference continuously
    # - Steps MuJoCo while keeping a single viewer open
    python run_sim2sim_locomotion.py --robot go2 --onnx policy.onnx --render --deploy --forever

Available tasks:
    flat_forward, flat_backward, flat_lateral, flat_turn, flat_fast,
    rough_forward, stairs_up, stairs_down, slope_up, mixed_terrain
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import json
import csv

# Add mujoco_utils directly to path (avoid importing main unitree_rl_lab which requires Isaac Sim)
# IMPORTANT: resolve() to avoid relative __file__ paths (e.g. ".../rsl_rl/../mujoco_eval/...") breaking path math.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MUJOCO_UTILS_DIR = _REPO_ROOT / "source" / "unitree_rl_lab" / "unitree_rl_lab"
sys.path.insert(0, str(_MUJOCO_UTILS_DIR))

import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Sim2Sim locomotion evaluation in MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument(
        "--robot",
        type=str,
        required=True,
        help="Robot name (e.g., go2, g1, h1, b2)",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to ONNX policy file",
    )
    
    # Optional arguments
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Path to MuJoCo XML (default: auto-detect from robot name)",
    )
    parser.add_argument(
        "--deploy-yaml",
        type=str,
        default=None,
        help=(
            "Optional mjlab-style deploy.yaml to override PD gains, default pose, action scale/offset, etc. "
            "Useful for running mjlab official exported ONNX (which usually lacks metadata_json)."
        ),
    )
    parser.add_argument(
        "--history-order",
        type=str,
        default="oldest_first",
        choices=["newest_first", "oldest_first"],
        help="History stacking order for temporal observations (debug sim2sim mismatch).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="flat_forward",
        help="Evaluation task name",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch evaluation on all tasks",
    )
    parser.add_argument(
        "--batch-two-phase",
        action="store_true",
        help=(
            "In --batch mode, run a two-phase evaluation: "
            "Phase 1 computes metrics for all tasks without rendering/video; "
            "Phase 2 records videos only for a small subset (mixed_terrain + worst-K)."
        ),
    )
    parser.add_argument(
        "--batch-worst-k",
        type=int,
        default=3,
        help="(Batch two-phase) Record videos for the worst K tasks in addition to always-record tasks.",
    )
    parser.add_argument(
        "--batch-worst-metric",
        type=str,
        default="survival_then_vel",
        choices=["survival_then_vel", "survival_rate", "mean_velocity_error"],
        help=(
            "(Batch two-phase) Ranking metric for selecting worst tasks. "
            "'survival_then_vel' sorts by lowest survival_rate, then highest mean_velocity_error."
        ),
    )
    parser.add_argument(
        "--batch-always-record",
        type=str,
        nargs="*",
        default=["mixed_terrain"],
        help="(Batch two-phase) Task names that always get a video (default: mixed_terrain).",
    )
    parser.add_argument(
        "--batch-record-episodes",
        type=int,
        default=1,
        help="(Batch two-phase) Number of episodes to run when recording each selected task video.",
    )
    parser.add_argument(
        "--batch-video-steps",
        type=int,
        default=300,
        help="(Batch two-phase) Number of policy steps to record per selected task video.",
    )
    parser.add_argument(
        "--dump-metrics-json",
        type=str,
        default=None,
        help="(Batch mode) Dump per-task metrics + selected video list to a JSON file (for W&B upload).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering during evaluation",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help=(
            "Run a deployment-like loop: build obs from MuJoCo state, run ONNX inference continuously, "
            "and step MuJoCo with a single viewer window."
        ),
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="When rendering, keep a single MuJoCo viewer open and run episodes continuously",
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="When rendering, ignore --num-episodes and keep running episodes until the viewer is closed",
    )
    parser.add_argument(
        "--no-reset-on-fall",
        action="store_true",
        help="In --deploy mode, do not auto-reset when a termination condition is detected (tilt/height).",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="In --deploy mode, run as fast as possible (do not try to match policy_dt wall-clock).",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save evaluation videos",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="(Single-task mode) Save an evaluation video (headless offscreen renderer).",
    )
    parser.add_argument(
        "--video-steps",
        type=int,
        default=0,
        help="(Single-task mode) Max policy steps to record when --save-video is enabled (0 = record full eval duration).",
    )
    parser.add_argument(
        "--video-segment-steps",
        type=int,
        default=0,
        help=(
            "(Single-task mode) If >0, split headless video into multiple MP4 segments "
            "of N policy steps each, to enable near-real-time uploading."
        ),
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=640,
        help="(Single-task mode) Video width for --save-video.",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=480,
        help="(Single-task mode) Video height for --save-video.",
    )
    parser.add_argument(
        "--dump-npz",
        type=str,
        default=None,
        help=(
            "(Single-task mode) Dump episode trajectories to an .npz file. "
            "Includes joint_pos, base_pos, base_lin_vel, actions, policy_dt, joint_names, and stats."
        ),
    )
    parser.add_argument(
        "--live-csv",
        type=str,
        default=None,
        help=(
            "(Single-task mode) Stream joint_pos to a CSV file during headless recording. "
            "Useful for near-real-time W&B plots (training process can tail this file)."
        ),
    )
    parser.add_argument(
        "--live-stride",
        type=int,
        default=2,
        help="(Single-task mode) Write one joint_pos row every N policy steps to --live-csv.",
    )
    parser.add_argument(
        "--record-video",
        type=str,
        default=None,
        help=(
            "Record an MP4 during --deploy/--forever (streams frames to disk). "
            "Example: --record-video eval_results/deploy.mp4"
        ),
    )
    parser.add_argument(
        "--record-width",
        type=int,
        default=640,
        help="Video width for --record-video",
    )
    parser.add_argument(
        "--record-height",
        type=int,
        default=480,
        help="Video height for --record-video",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=None,
        help="Video FPS for --record-video (default: 1/policy_dt)",
    )
    parser.add_argument(
        "--record-steps",
        type=int,
        default=0,
        help="Max number of policy steps to record in --deploy (0 = record until viewer closes)",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="In --render mode, enable viewer camera auto-follow (tracking) the robot body.",
    )
    parser.add_argument(
        "--follow-body",
        type=str,
        default="pelvis",
        help="MuJoCo body name to follow when --follow is enabled (default: pelvis).",
    )
    parser.add_argument(
        "--follow-distance",
        type=float,
        default=3.0,
        help="Camera distance when --follow is enabled.",
    )
    parser.add_argument(
        "--follow-azimuth",
        type=float,
        default=-130.0,
        help="Camera azimuth (deg) when --follow is enabled.",
    )
    parser.add_argument(
        "--follow-elevation",
        type=float,
        default=-20.0,
        help="Camera elevation (deg) when --follow is enabled.",
    )
    parser.add_argument(
        "--teleop",
        type=str,
        default="off",
        choices=["off", "keyboard"],
        help=(
            "Velocity-command teleoperation source for --deploy mode. "
            "'keyboard' uses MuJoCo viewer key events (W/S vx, A/D vy, Q/E wz, SPACE=zero)."
        ),
    )
    parser.add_argument("--teleop-step-vx", type=float, default=0.1, help="vx step per key press (m/s)")
    parser.add_argument("--teleop-step-vy", type=float, default=0.1, help="vy step per key press (m/s)")
    parser.add_argument("--teleop-step-wz", type=float, default=0.2, help="wz step per key press (rad/s)")
    parser.add_argument("--teleop-max-vx", type=float, default=1.0, help="max |vx| (m/s)")
    parser.add_argument("--teleop-max-vy", type=float, default=0.5, help="max |vy| (m/s)")
    parser.add_argument("--teleop-max-wz", type=float, default=1.0, help="max |wz| (rad/s)")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes per task",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        nargs=3,
        default=None,
        metavar=("VX", "VY", "WZ"),
        help="Override velocity command (vx vy wz)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    
    return parser.parse_args()


def _transcode_mp4_to_h264_faststart(in_path: Path) -> Path:
    """Best-effort transcode to H.264/AAC MP4 with faststart for browser/W&B playback.

    Our OpenCV writer may produce MPEG-4 Part 2 ("mp4v"), which is not supported by all browsers
    (W&B Media panel uses browser playback). If ffmpeg is available, we transcode in-place.
    """
    try:
        import subprocess
    except Exception:
        return in_path

    if not in_path.exists():
        return in_path

    ffmpeg = "ffmpeg"
    # quick presence check
    try:
        subprocess.run([ffmpeg, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        return in_path

    tmp = in_path.with_suffix(".h264_tmp.mp4")
    try:
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(in_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(tmp),
        ]
        subprocess.run(cmd, check=True)
        tmp.replace(in_path)
        return in_path
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return in_path


def _record_video_headless(simulator, out_path: Path, steps: int, width: int, height: int) -> tuple[Path, int]:
    """Record a headless MP4 using MuJoCo offscreen renderer."""
    try:
        import mujoco  # type: ignore
        import cv2  # type: ignore
    except Exception as e:
        raise ImportError("Recording requires mujoco + opencv-python.") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(simulator.model, int(height), int(width))
    # Use mp4v for maximum OpenCV compatibility, then transcode to H.264 for browser playback.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Encode at a fixed FPS for consistent W&B playback.
    fps = 60

    def _open_writer(p: Path):
        w = cv2.VideoWriter(str(p), fourcc, fps, (int(width), int(height)))
        if not w.isOpened():
            raise RuntimeError(f"Failed to open video writer: {p}")
        return w

    writer = _open_writer(out_path)

    return_path = out_path
    return_fps = fps

    # NOTE: the actual encoding happens in the caller loop; once the file is finalized,
    # we will transcode to H.264 where possible.
    return return_path, return_fps


def _record_video_headless_segmented(
    simulator,
    out_path: Path,
    steps: int,
    width: int,
    height: int,
    segment_steps: int,
    live_csv: str | None = None,
    live_stride: int = 2,
) -> tuple[list[Path], int]:
    """Record headless MP4 segments + optionally stream joint_pos to CSV."""
    try:
        import mujoco  # type: ignore
        import cv2  # type: ignore
    except Exception as e:
        raise ImportError("Recording requires mujoco + opencv-python.") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(simulator.model, int(height), int(width))
    # Encode at a fixed FPS for consistent W&B playback.
    fps = 60
    # Use mp4v for maximum OpenCV compatibility, then transcode each closed segment to H.264.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    seg_n = max(0, int(segment_steps))
    if seg_n <= 0:
        seg_n = int(steps)

    stem = out_path.stem
    suffix = out_path.suffix or ".mp4"
    segments: list[Path] = []

    # CSV streaming (optional)
    csv_f = None
    csv_writer = None
    if live_csv:
        p = Path(live_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        csv_f = open(p, "w", newline="")
        csv_writer = csv.writer(csv_f)
        joint_names = getattr(simulator.onnx_config, "joint_names", None) or []
        header = ["t"] + [str(n) for n in joint_names] if joint_names else ["t"]
        if not joint_names:
            # fallback: qpos_0...qpos_{n-1}
            try:
                n = int(simulator.onnx_action_dim)
            except Exception:
                n = 0
            header += [f"qpos_{i}" for i in range(n)]
        csv_writer.writerow(header)
        csv_f.flush()

    def _open_writer(p: Path):
        w = cv2.VideoWriter(str(p), fourcc, fps, (int(width), int(height)))
        if not w.isOpened():
            raise RuntimeError(f"Failed to open video writer: {p}")
        return w

    writer = None
    try:
        for i in range(int(steps)):
            # Start a new segment
            if i % seg_n == 0:
                if writer is not None:
                    writer.release()
                    # Transcode the previously completed segment (best-effort)
                    try:
                        _transcode_mp4_to_h264_faststart(segments[-1])
                    except Exception:
                        pass
                part_idx = int(i // seg_n)
                part_path = out_path.with_name(f"{stem}_part{part_idx:03d}{suffix}")
                writer = _open_writer(part_path)
                segments.append(part_path)

            # Step first (so rendered frame corresponds to the updated state)
            _obs, info = simulator.step()

            # Stream joint_pos to CSV
            if csv_writer is not None and (i % max(1, int(live_stride)) == 0):
                t = float(i) * float(simulator.policy_dt)
                q = info.get("joint_pos", None)
                if q is not None:
                    row = [t] + [float(x) for x in np.asarray(q).reshape(-1).tolist()]
                    csv_writer.writerow(row)
                    csv_f.flush()  # flush for near-real-time tailing

            # Render + write one frame per policy step
            renderer.update_scene(simulator.data)
            frame_rgb = renderer.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            assert writer is not None
            writer.write(frame_bgr)
    finally:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        # Transcode the last segment (best-effort)
        try:
            if segments:
                _transcode_mp4_to_h264_faststart(segments[-1])
        except Exception:
            pass
        try:
            renderer.close()
        except Exception:
            pass
        try:
            if csv_f is not None:
                csv_f.close()
        except Exception:
            pass

    return segments, fps


def _run_headless_eval_with_single_video(
    simulator,
    out_path: Path,
    num_episodes: int,
    max_steps_per_episode: int,
    velocity_command: tuple[float, float, float],
    width: int,
    height: int,
    max_video_steps: int = 0,
    live_csv: str | None = None,
    live_stride: int = 2,
) -> tuple[list[dict], Path, int]:
    """Run sim2sim eval and record ONE long MP4 whose duration matches the eval run.

    - Concatenates multiple episodes back-to-back into a single file.
    - Stops early if max_video_steps>0 is reached.
    """
    try:
        import mujoco  # type: ignore
        import cv2  # type: ignore
    except Exception as e:
        raise ImportError("Recording requires mujoco + opencv-python.") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(simulator.model, int(height), int(width))
    # Encode at a fixed FPS for consistent W&B playback.
    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    # Optional CSV streaming
    csv_f = None
    csv_writer = None
    if live_csv:
        p = Path(live_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        csv_f = open(p, "w", newline="")
        csv_writer = csv.writer(csv_f)
        joint_names = getattr(simulator.onnx_config, "joint_names", None) or []
        header = ["t"] + [str(n) for n in joint_names] if joint_names else ["t"]
        if not joint_names:
            try:
                n = int(simulator.onnx_action_dim)
            except Exception:
                n = 0
            header += [f"qpos_{i}" for i in range(n)]
        csv_writer.writerow(header)
        csv_f.flush()

    results: list[dict] = []
    total_written = 0
    try:
        for ep in range(int(num_episodes)):
            simulator.reset()
            simulator.set_velocity_command(*velocity_command)

            episode_data = {
                "base_pos": [],
                "base_quat": [],
                "base_lin_vel": [],
                "joint_pos": [],
                "actions": [],
            }

            for step in range(int(max_steps_per_episode)):
                _obs, info = simulator.step()

                episode_data["base_pos"].append(info["base_pos"])
                episode_data["base_quat"].append(info["base_quat"])
                episode_data["base_lin_vel"].append(info["base_lin_vel"])
                episode_data["joint_pos"].append(info["joint_pos"])
                episode_data["actions"].append(simulator._last_action.copy())

                # stream joint_pos
                if csv_writer is not None and (total_written % max(1, int(live_stride)) == 0):
                    t = float(total_written) * float(simulator.policy_dt)
                    q = info.get("joint_pos", None)
                    if q is not None:
                        row = [t] + [float(x) for x in np.asarray(q).reshape(-1).tolist()]
                        csv_writer.writerow(row)
                        csv_f.flush()

                # render + write
                renderer.update_scene(simulator.data)
                frame_rgb = renderer.render()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                total_written += 1

                if max_video_steps > 0 and total_written >= int(max_video_steps):
                    break

                if simulator._check_termination():
                    break

            # stats (match BaseMujocoSimulator style)
            for k in episode_data:
                episode_data[k] = np.asarray(episode_data[k])
            base_pos = episode_data["base_pos"]
            stats = {
                "num_steps": int(len(base_pos)),
                "survived": bool(not simulator._check_termination()),
                "distance_traveled": float(
                    np.linalg.norm(base_pos[-1, :2] - base_pos[0, :2]) if len(base_pos) > 1 else 0.0
                ),
                "mean_velocity": float(
                    np.mean(np.linalg.norm(episode_data["base_lin_vel"][:, :2], axis=1))
                    if len(episode_data["base_lin_vel"]) > 0
                    else 0.0
                ),
            }
            results.append({"data": episode_data, "stats": stats})

            if max_video_steps > 0 and total_written >= int(max_video_steps):
                break
    finally:
        try:
            writer.release()
        except Exception:
            pass
        try:
            renderer.close()
        except Exception:
            pass
        try:
            if csv_f is not None:
                csv_f.close()
        except Exception:
            pass

    # Ensure browser/W&B playback compatibility
    try:
        _transcode_mp4_to_h264_faststart(out_path)
    except Exception:
        pass

    return results, out_path, int(fps)


def _apply_task_terrain_to_mujoco_model(simulator, task) -> None:
    """Best-effort: apply eval task terrain to the simulator's MuJoCo model via heightfield.

    Requires the loaded XML to include a heightfield named 'terrain_hfield'
    (e.g. `assets/robots/unitree_robots/g1/scene_29dof_terrain.xml`).
    If not present, this function is a no-op.
    """
    try:
        from mujoco_utils.terrain.generator import MujocoTerrainGenerator, TerrainConfig  # type: ignore
        from mujoco_utils.terrain.xml_generation import setup_terrain_data_in_model  # type: ignore
    except Exception:
        return

    # If model doesn't have the expected heightfield, skip.
    try:
        import mujoco as mj  # type: ignore

        hid = mj.mj_name2id(simulator.model, mj.mjtObj.mjOBJ_HFIELD, "terrain_hfield")
        if hid < 0:
            return
        nrow = int(simulator.model.hfield_nrow[hid])
        ncol = int(simulator.model.hfield_ncol[hid])
        sx = float(simulator.model.hfield_size[hid, 0]) * 2.0
        sy = float(simulator.model.hfield_size[hid, 1]) * 2.0
    except Exception:
        return

    # Map eval task terrain -> generator config
    ttype = str(getattr(task, "terrain_type", "") or "flat")
    tc = dict(getattr(task, "terrain_config", {}) or {})

    # Derive horizontal scale from model grid + chosen physical size.
    # (ncol/nrow include endpoints)
    hscale_x = float(sx) / max(1, (ncol - 1))
    hscale_y = float(sy) / max(1, (nrow - 1))
    hscale = float(min(hscale_x, hscale_y))

    cfg = TerrainConfig(
        terrain_type=ttype,
        size=(float(sx), float(sy)),
        horizontal_scale=hscale,
        difficulty=float(tc.get("difficulty", 0.5) if isinstance(tc, dict) else 0.5),
    )
    # Task-specific knobs (best-effort): copy any known TerrainConfig fields.
    if isinstance(tc, dict):
        for k, v in tc.items():
            if k == "difficulty":
                continue
            if not hasattr(cfg, k):
                continue
            try:
                setattr(cfg, k, v)
            except Exception:
                # Ignore bad user values (keep defaults)
                pass

    gen = MujocoTerrainGenerator(cfg)
    gen.generate()
    setup_terrain_data_in_model(simulator.model, gen, hfield_name="terrain_hfield")

    # ---------------------------------------------------------------------
    # Spawn-height fix: if the heightfield gets shifted upward (because MuJoCo
    # hfields cannot represent negative heights), the robot may start with feet
    # embedded in the terrain. Compute the actual terrain height at the origin
    # and lift the floating base on reset accordingly.
    # ---------------------------------------------------------------------
    try:
        base_z = float(simulator.model.hfield_size[hid, 3])
        size_z = float(simulator.model.hfield_size[hid, 2])
        heights = np.asarray(gen.heightfield, dtype=np.float32)

        # Apply the same "fit to [base_z, base_z + size_z]" logic as in setup_terrain_data_in_model
        # so this matches the actual terrain height used by MuJoCo.
        if size_z > 1e-9:
            min_h = float(np.min(heights)) if heights.size else 0.0
            max_h = float(np.max(heights)) if heights.size else 0.0
            if min_h < base_z:
                heights = heights + (base_z - min_h)
                max_h = float(np.max(heights)) if heights.size else base_z
            max_allowed = base_z + size_z
            if max_h > max_allowed and max_h > base_z + 1e-9:
                scale = (max_allowed - base_z) / (max_h - base_z)
                heights = base_z + (heights - base_z) * float(scale)

        # Spawn safety: take the max height in a small patch around the origin to account
        # for stance footprint (so we don't start with any foot intersecting a local bump).
        ix = int((0.0 + float(sx) / 2.0) / float(hscale))
        iy = int((0.0 + float(sy) / 2.0) / float(hscale))
        ix = int(np.clip(ix, 0, heights.shape[1] - 1))
        iy = int(np.clip(iy, 0, heights.shape[0] - 1))

        patch_halfwidth_m = 0.6
        r = int(max(1, round(float(patch_halfwidth_m) / float(hscale))))
        y0 = int(np.clip(iy - r, 0, heights.shape[0] - 1))
        y1 = int(np.clip(iy + r + 1, 0, heights.shape[0]))
        x0 = int(np.clip(ix - r, 0, heights.shape[1] - 1))
        x1 = int(np.clip(ix + r + 1, 0, heights.shape[1]))
        h0 = float(np.max(heights[y0:y1, x0:x1])) if (y1 > y0 and x1 > x0) else float(heights[iy, ix])

        dz = float(h0 - base_z)
        if dz > 0.0:
            dz = dz + 0.01  # 1cm extra clearance to avoid initial interpenetration
        else:
            dz = 0.0
        simulator.spawn_root_z_offset = float(dz)
        if dz > 0.0:
            print(f"[Terrain][Spawn] Lift base by dz={dz:.3f}m (h0={h0:.3f}, base={base_z:.3f})")
    except Exception:
        pass

    # Make sure derived quantities are updated before stepping/rendering.
    try:
        import mujoco  # type: ignore

        mujoco.mj_forward(simulator.model, simulator.data)
    except Exception:
        pass


def run_deploy_forever(
    simulator,
    velocity_command: tuple[float, float, float],
    max_steps_per_episode: int,
    reset_on_fall: bool,
    realtime: bool,
    record_video: str | None = None,
    record_width: int = 640,
    record_height: int = 480,
    record_fps: float | None = None,
    record_steps: int = 0,
    follow: bool = False,
    follow_body: str = "pelvis",
    follow_distance: float = 3.0,
    follow_azimuth: float = -130.0,
    follow_elevation: float = -20.0,
    teleop: str = "off",
    teleop_step_vx: float = 0.1,
    teleop_step_vy: float = 0.1,
    teleop_step_wz: float = 0.2,
    teleop_max_vx: float = 1.0,
    teleop_max_vy: float = 0.5,
    teleop_max_wz: float = 1.0,
) -> None:
    """Run a deployment-like continuous control loop in a single MuJoCo viewer.

    Loop:
      MuJoCo state (qpos/qvel) -> observation builder -> ONNX inference -> PD control -> mj_step

    Notes:
    - We sync the viewer on every MuJoCo physics step (not just every policy step) for smooth render.
    - If reset_on_fall is enabled, we reset when simulator._check_termination() triggers.
    """
    try:
        import mujoco  # type: ignore
        import mujoco.viewer  # type: ignore
    except Exception as e:
        raise ImportError("mujoco not installed. Run: pip install mujoco") from e

    video_writer = None
    renderer = None
    frames_written = 0
    if record_video:
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise ImportError(
                "OpenCV is required for --record-video. Install with: pip install opencv-python"
            ) from e

        out_path = Path(record_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Default FPS to policy rate (one frame per policy step).
        fps = float(record_fps) if record_fps is not None else float(1.0 / simulator.policy_dt)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (int(record_width), int(record_height)))
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {out_path}")
        # MuJoCo offscreen renderer is limited by model.vis.global_[offwidth/offheight].
        # If user requests a larger recording size, try to enlarge the model framebuffer;
        # otherwise fall back to the maximum supported size.
        def _get_vis_global(model_):
            return getattr(model_.vis, "global_", getattr(model_.vis, "global", None))

        def _get_offscreen_wh(model_):
            g = _get_vis_global(model_)
            if g is None:
                return 0, 0
            return int(getattr(g, "offwidth", 0) or 0), int(getattr(g, "offheight", 0) or 0)

        def _set_offscreen_wh(model_, w: int, h: int) -> None:
            g = _get_vis_global(model_)
            if g is None:
                return
            try:
                setattr(g, "offwidth", int(w))
                setattr(g, "offheight", int(h))
            except Exception:
                # Some builds may treat these as read-only.
                pass

        req_w, req_h = int(record_width), int(record_height)
        off_w, off_h = _get_offscreen_wh(simulator.model)
        if off_w and off_h and (req_w > off_w or req_h > off_h):
            # Best-effort: enlarge framebuffer to requested size
            _set_offscreen_wh(simulator.model, max(off_w, req_w), max(off_h, req_h))
            off_w2, off_h2 = _get_offscreen_wh(simulator.model)
            if off_w2 and off_h2 and (req_w > off_w2 or req_h > off_h2):
                # Still too small -> fall back
                print(
                    f"[Deploy][Record][Warn] Requested {req_w}x{req_h} exceeds offscreen framebuffer "
                    f"{off_w2}x{off_h2}. Falling back to {off_w2}x{off_h2}."
                )
                req_w, req_h = off_w2, off_h2

        try:
            renderer = mujoco.Renderer(simulator.model, int(req_h), int(req_w))
        except ValueError as e:
            # Fall back to the model's maximum supported framebuffer size.
            off_w3, off_h3 = _get_offscreen_wh(simulator.model)
            if off_w3 and off_h3 and (req_w != off_w3 or req_h != off_h3):
                print(
                    f"[Deploy][Record][Warn] Renderer init failed for {req_w}x{req_h}: {e}\n"
                    f"                     Retrying with framebuffer size {off_w3}x{off_h3}."
                )
                renderer = mujoco.Renderer(simulator.model, int(off_h3), int(off_w3))
                req_w, req_h = off_w3, off_h3
            else:
                # Ensure writer is closed before raising.
                try:
                    video_writer.release()
                except Exception:
                    pass
                raise

        print(f"[Deploy][Record] Recording to: {out_path} @ {fps:.1f} FPS ({req_w}x{req_h})")

    # We intentionally drive generalized forces (data.qfrc_applied) instead of data.ctrl for robustness:
    # many Unitree XMLs keep motor ctrlrange at [-1, 1] with gear=1, which would clamp torques to ~1Nm.
    try:
        from mujoco_utils.core.physics import pd_control
    except Exception as e:
        raise ImportError("Failed to import mujoco_utils helpers for deploy loop.") from e

    # Use simulator's robust model-derived mapping if available.
    model_nu = int(simulator.model.nu)
    if model_nu <= 0:
        raise RuntimeError("[Deploy] MuJoCo model.nu==0 (no actuators). Cannot run deploy loop.")

    n_u = model_nu
    qpos_adrs = getattr(simulator, "_qpos_adrs", None)
    dof_adrs = getattr(simulator, "_dof_adrs", None)
    tau_limits = getattr(simulator, "tau_limits", None)
    actuator_names = getattr(simulator, "actuator_names", None)

    if qpos_adrs is None or dof_adrs is None or tau_limits is None or actuator_names is None:
        # Fallback to building mapping here (older simulator versions)
        actuator_joint_ids: list[int] = []
        actuator_names = []
        for act_id in range(model_nu):
            name = mujoco.mj_id2name(simulator.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
            actuator_names.append(name or f"actuator_{act_id}")
            jid = int(simulator.model.actuator_trnid[act_id, 0])
            if jid < 0:
                raise RuntimeError(
                    f"[Deploy] Actuator '{actuator_names[-1]}' has invalid actuator_trnid joint id: {jid}"
                )
            actuator_joint_ids.append(jid)

        qpos_adrs = []
        dof_adrs = []
        tau_limits = np.zeros((n_u, 2), dtype=np.float32)
        for i in range(n_u):
            jid = actuator_joint_ids[i]
            qpos_adrs.append(int(simulator.model.jnt_qposadr[jid]))
            dof_adrs.append(int(simulator.model.jnt_dofadr[jid]))
            tau_limits[i, :] = simulator.model.jnt_actfrcrange[jid].astype(np.float32)

    ctrlrange = simulator.model.actuator_ctrlrange[:n_u].copy()
    gear = simulator.model.actuator_gear[:n_u, 0].copy()
    if np.allclose(ctrlrange, np.array([[-1.0, 1.0]], dtype=ctrlrange.dtype)) and np.allclose(
        gear, 1.0
    ):
        print(
            "[Deploy][Info] Detected motor ctrlrange≈[-1,1] with gear≈1. "
            "Applying torques via data.qfrc_applied (bypass data.ctrl clamp)."
        )

    simulator.reset()
    simulator.set_velocity_command(*velocity_command)

    # -------------------------------------------------------------------------
    # Teleop via viewer keyboard events (no extra deps).
    # -------------------------------------------------------------------------
    class _TeleopState:
        def __init__(self, vx: float, vy: float, wz: float):
            self.vx = float(vx)
            self.vy = float(vy)
            self.wz = float(wz)

        def clamp(self):
            self.vx = float(np.clip(self.vx, -teleop_max_vx, teleop_max_vx))
            self.vy = float(np.clip(self.vy, -teleop_max_vy, teleop_max_vy))
            self.wz = float(np.clip(self.wz, -teleop_max_wz, teleop_max_wz))

        def as_tuple(self):
            return (float(self.vx), float(self.vy), float(self.wz))

    teleop_state = _TeleopState(*velocity_command)

    # GLFW key codes used by MuJoCo viewer (GLFW_KEY_*):
    # https://www.glfw.org/docs/latest/group__keys.html
    GLFW_KEY_SPACE = 32
    GLFW_KEY_W = 87
    GLFW_KEY_A = 65
    GLFW_KEY_S = 83
    GLFW_KEY_D = 68
    GLFW_KEY_Q = 81
    GLFW_KEY_E = 69

    def _key_cb(key: int) -> None:
        if teleop != "keyboard":
            return
        if key == GLFW_KEY_W:
            teleop_state.vx += teleop_step_vx
        elif key == GLFW_KEY_S:
            teleop_state.vx -= teleop_step_vx
        elif key == GLFW_KEY_A:
            teleop_state.vy += teleop_step_vy
        elif key == GLFW_KEY_D:
            teleop_state.vy -= teleop_step_vy
        elif key == GLFW_KEY_Q:
            teleop_state.wz += teleop_step_wz
        elif key == GLFW_KEY_E:
            teleop_state.wz -= teleop_step_wz
        elif key == GLFW_KEY_SPACE:
            teleop_state.vx = teleop_state.vy = teleop_state.wz = 0.0
        teleop_state.clamp()
        simulator.set_velocity_command(*teleop_state.as_tuple())
        print(f"[Teleop] cmd vx={teleop_state.vx:+.2f} vy={teleop_state.vy:+.2f} wz={teleop_state.wz:+.2f}")

    viewer = mujoco.viewer.launch_passive(simulator.model, simulator.data, key_callback=_key_cb if teleop == "keyboard" else None)
    # Optional camera follow (tracking).
    follow_body_id = None
    if follow:
        try:
            follow_body_id = mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_BODY, follow_body)
            if follow_body_id < 0:
                follow_body_id = None
        except Exception:
            follow_body_id = None
        if follow_body_id is None:
            print(f"[Deploy][Follow][Warn] Body not found: '{follow_body}'. Disable follow.")
        else:
            try:
                # Use MuJoCo built-in tracking mode.
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = int(follow_body_id)
                viewer.cam.distance = float(follow_distance)
                viewer.cam.azimuth = float(follow_azimuth)
                viewer.cam.elevation = float(follow_elevation)
                # Seed lookat with current body position
                viewer.cam.lookat[:] = simulator.data.xpos[int(follow_body_id)]
                print(f"[Deploy][Follow] Tracking body='{follow_body}' (id={follow_body_id})")
            except Exception as e:
                print(f"[Deploy][Follow][Warn] Failed to enable tracking camera: {e}")
                follow_body_id = None
    try:
        next_tick = time.perf_counter()
        while viewer.is_running():
            # Optional “episode-like” reset conditions, but keep the same viewer open.
            if reset_on_fall and simulator._check_termination():
                simulator.reset()
                simulator.set_velocity_command(*velocity_command)

            # Update command (teleop overrides the fixed command).
            if teleop == "keyboard":
                simulator.set_velocity_command(*teleop_state.as_tuple())

            # Read MuJoCo state in actuator order, then convert to ONNX order for observations.
            joint_pos_act = np.array([simulator.data.qpos[a] for a in qpos_adrs], dtype=np.float32)
            joint_vel_act = np.array([simulator.data.qvel[a] for a in dof_adrs], dtype=np.float32)
            base_quat = simulator.data.qpos[3:7].astype(np.float32).copy()
            base_ang_vel = simulator.data.qvel[3:6].astype(np.float32).copy()
            base_lin_vel = simulator.data.qvel[:3].astype(np.float32).copy()

            # Policy step (one action per policy_dt): MuJoCo state -> obs -> ONNX inference
            inv_map = getattr(simulator, "inv_joint_mapping", None)
            if inv_map is None:
                # Best-effort: assume already aligned
                joint_pos = joint_pos_act
                joint_vel = joint_vel_act
            else:
                joint_pos = joint_pos_act[np.array(inv_map, dtype=np.int64)]
                joint_vel = joint_vel_act[np.array(inv_map, dtype=np.int64)]

            obs = simulator.obs_builder.build(
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                base_quat=base_quat,
                base_ang_vel=base_ang_vel,
                base_lin_vel=base_lin_vel,
                last_action=simulator._last_action,
                velocity_command=simulator.velocity_command,
                episode_length=simulator.episode_length,
                step_dt=simulator.policy_dt,
            )
            # ONNX policy outputs actions in ONNX joint_names order.
            action_onnx = simulator.policy(obs).astype(np.float32).reshape(-1)
            if action_onnx.shape[0] != int(getattr(simulator, "onnx_action_dim", n_u)):
                raise RuntimeError(
                    f"[Deploy] Policy action_dim={action_onnx.shape[0]} but expected {getattr(simulator, 'onnx_action_dim', n_u)}. "
                    f"ONNX must match actuator count. Actuators (first 10): {actuator_names[:10]}"
                )

            # Store last_action in ONNX order for observation alignment.
            simulator._last_action = action_onnx.copy()

            # Reorder for actuator application.
            if hasattr(simulator, "joint_mapping") and simulator.joint_mapping and len(simulator.joint_mapping) == n_u:
                action_act = action_onnx[np.array(simulator.joint_mapping, dtype=np.int64)]
            else:
                action_act = action_onnx

            # Build target_q consistent with BaseMujocoSimulator / IsaacLab semantics:
            # target_q = action_offset + action * action_scale
            # (If action_offset is missing, fall back to default_joint_pos as the base.)
            if getattr(simulator, "default_joint_pos", None) is not None and len(simulator.default_joint_pos) == n_u:
                default_q = simulator.default_joint_pos.astype(np.float32)
            else:
                default_q = np.zeros(n_u, dtype=np.float32)

            if getattr(simulator, "action_scale", None) is not None and len(simulator.action_scale) == n_u:
                scale = simulator.action_scale.astype(np.float32)
            else:
                scale = np.ones(n_u, dtype=np.float32) * 0.25

            if getattr(simulator, "action_offset", None) is not None and len(simulator.action_offset) == n_u:
                offset = simulator.action_offset.astype(np.float32)
            else:
                offset = np.zeros(n_u, dtype=np.float32)

            base_q = offset if np.any(offset) else default_q
            target_q = base_q + action_act * scale

            # Physics steps (decimation sub-steps per policy step)
            for _ in range(simulator.decimation):
                # Re-read state each substep for proper damping
                joint_pos = np.array([simulator.data.qpos[a] for a in qpos_adrs], dtype=np.float32)
                joint_vel = np.array([simulator.data.qvel[a] for a in dof_adrs], dtype=np.float32)

                tau = pd_control(
                    target_q=target_q,
                    current_q=joint_pos,
                    current_dq=joint_vel,
                    kp=simulator.kp.astype(np.float32),
                    kd=simulator.kd.astype(np.float32),
                    tau_limits=tau_limits,
                ).astype(np.float32)

                # Drive generalized forces directly (robust to actuator ctrlrange/gear conventions).
                simulator.data.ctrl[:] = 0.0
                simulator.data.qfrc_applied[:] = 0.0
                for i, dof_adr in enumerate(dof_adrs):
                    simulator.data.qfrc_applied[dof_adr] = float(tau[i])

                mujoco.mj_step(simulator.model, simulator.data)
                viewer.sync()

            simulator.episode_length += 1
            simulator._update_phase()
            # Some viewer builds don't fully auto-update lookat for tracking camera; keep it fresh.
            if follow_body_id is not None:
                try:
                    viewer.cam.lookat[:] = simulator.data.xpos[int(follow_body_id)]
                except Exception:
                    pass

            # Record one frame per policy step (after physics substeps).
            if video_writer is not None and renderer is not None:
                try:
                    # IMPORTANT:
                    # The offscreen renderer has its own camera. To ensure the recorded video matches
                    # what the user sees in the interactive viewer (including --follow tracking),
                    # render using the viewer camera.
                    renderer.update_scene(simulator.data, camera=viewer.cam)
                    frame_rgb = renderer.render()
                    # cv2 expects BGR
                    frame_bgr = frame_rgb[..., ::-1]
                    video_writer.write(frame_bgr)
                    frames_written += 1
                    if record_steps and frames_written >= int(record_steps):
                        print(f"[Deploy][Record] Reached record_steps={record_steps}. Stopping recording.")
                        video_writer.release()
                        video_writer = None
                        renderer = None
                except Exception as e:
                    print(f"[Deploy][Record][Warning] Failed to record frame: {e}")

            # Best-effort real-time pacing at policy rate (more like real robot control loop)
            if realtime:
                next_tick += float(simulator.policy_dt)
                now = time.perf_counter()
                sleep_s = next_tick - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    # If we're behind, don't accumulate unbounded lag.
                    next_tick = now

            # Optional cap per “episode” (useful for quick debug without --forever)
            if max_steps_per_episode > 0 and simulator.episode_length >= max_steps_per_episode:
                simulator.reset()
                simulator.set_velocity_command(*velocity_command)
    finally:
        try:
            if video_writer is not None:
                video_writer.release()
        except Exception:
            pass
        try:
            if renderer is not None:
                renderer.close()
        except Exception:
            pass
        viewer.close()


def find_robot_xml(robot_name: str) -> Path:
    """Find MuJoCo XML for robot.
    
    Args:
        robot_name: Robot name
        
    Returns:
        Path to XML file
    """
    # Common asset directories
    # Repo contains MuJoCo XMLs under:
    #   source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree_robots/<robot>/
    repo_root = Path(__file__).parent.parent.parent
    asset_dirs = [
        # Common asset directories
        repo_root / "assets" / "robots" / robot_name,
        repo_root / "source" / "unitree_rl_lab" / "assets" / robot_name,
        # This repo's mujoco-ready XMLs (preferred)
        repo_root / "source" / "unitree_rl_lab" / "unitree_rl_lab" / "assets" / "robots" / "unitree_robots" / robot_name,
        # User-local models
        Path.home() / ".mujoco" / "models" / robot_name,
    ]
    
    # Search for XML
    for asset_dir in asset_dirs:
        # Prefer scene XMLs when available (include ground, lights, etc.)
        preferred = [
            "scene_29dof.xml",
            "scene.xml",
            "scene_23dof.xml",
            f"{robot_name}.xml",
            f"{robot_name}_29dof.xml",
            f"{robot_name}_23dof.xml",
            "g1_29dof.xml",
            "g1_23dof.xml",
        ]
        for fname in preferred:
            xml_path = asset_dir / fname
            if xml_path.exists():
                return xml_path
        
        # Try variations
        for pattern in ["*.xml", f"*{robot_name}*.xml"]:
            matches = list(asset_dir.glob(pattern))
            if matches:
                return matches[0]
    
    raise FileNotFoundError(
        f"Could not find MuJoCo XML for robot '{robot_name}'. "
        f"Searched: {[str(d) for d in asset_dirs]}. "
        "Please provide --xml argument."
    )


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check ONNX path
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"[Error] ONNX file not found: {onnx_path}")
        sys.exit(1)
    
    # Find XML
    if args.xml:
        xml_path = Path(args.xml)
        if not xml_path.exists():
            print(f"[Error] XML file not found: {xml_path}")
            sys.exit(1)
    else:
        try:
            xml_path = find_robot_xml(args.robot)
            print(f"[Info] Found robot XML: {xml_path}")
        except FileNotFoundError as e:
            print(f"[Error] {e}")
            sys.exit(1)
    
    # Import MuJoCo utilities (directly from mujoco_utils, not unitree_rl_lab)
    try:
        from mujoco_utils import BaseMujocoSimulator
        from mujoco_utils.evaluation import (
            BatchEvaluator,
            get_eval_task,
            list_eval_tasks,
        )
    except ImportError as e:
        print(f"[Error] Failed to import mujoco_utils: {e}")
        print("Make sure mujoco and onnxruntime are installed:")
        print("  pip install mujoco onnxruntime")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("MuJoCo Sim2Sim Locomotion Evaluation")
    print("=" * 60)
    print(f"Robot:  {args.robot}")
    print(f"Policy: {onnx_path.name}")
    print(f"XML:    {xml_path.name}")
    print("=" * 60 + "\n")
    
    if args.batch:
        # Batch evaluation
        evaluator = BatchEvaluator(
            simulator_class=BaseMujocoSimulator,
            xml_path=xml_path,
            onnx_path=onnx_path,
            output_dir=args.output_dir,
        )

        # Phase 1: metrics for all tasks (fast, no videos)
        results = evaluator.evaluate_all(
            num_episodes_per_task=args.num_episodes,
            save_videos=(bool(args.save_videos) and not bool(args.batch_two_phase)),
        )

        # Save results (npz)
        evaluator.save_results()

        # Optional: dump metrics JSON + (optional) Phase 2 videos list
        metrics_json_path = None
        if args.dump_metrics_json:
            metrics_json_path = Path(args.dump_metrics_json)
        elif args.batch_two_phase:
            metrics_json_path = Path(args.output_dir) / "sim2sim_metrics.json"

        metrics_payload = None
        if metrics_json_path is not None:
            # Build a compact JSON payload for downstream logging (e.g. W&B).
            tasks_dict: dict[str, dict] = {}
            for task_name, res in results.items():
                m = res.metrics
                tasks_dict[str(task_name)] = {
                    "survival_rate": float(m.survival_rate),
                    "mean_episode_length": float(m.mean_episode_length),
                    "mean_velocity_error": float(m.mean_velocity_error),
                    "velocity_error_x": float(m.velocity_error_x),
                    "velocity_error_y": float(m.velocity_error_y),
                    "velocity_error_yaw": float(m.velocity_error_yaw),
                    "total_distance": float(m.total_distance),
                    "mean_forward_distance": float(m.mean_forward_distance),
                    "mean_energy": float(m.mean_energy),
                    "mean_torque_magnitude": float(m.mean_torque_magnitude),
                    "mean_base_height": float(m.mean_base_height),
                    "base_height_variance": float(m.base_height_variance),
                    "mean_orientation_error": float(m.mean_orientation_error),
                    "num_episodes": int(len(m.episode_lengths)),
                }

            # Overall aggregates (best-effort)
            try:
                all_metrics = [r.metrics for r in results.values()]
                mean_survival = float(np.mean([m.survival_rate for m in all_metrics])) if all_metrics else 0.0
                mean_vel_error = float(np.mean([m.mean_velocity_error for m in all_metrics])) if all_metrics else 0.0
                mean_distance = float(np.mean([m.mean_forward_distance for m in all_metrics])) if all_metrics else 0.0
                total_distance = float(np.sum([m.total_distance for m in all_metrics])) if all_metrics else 0.0
            except Exception:
                mean_survival = mean_vel_error = mean_distance = total_distance = 0.0

            metrics_payload = {
                "robot": str(args.robot),
                "onnx": str(onnx_path),
                "xml": str(xml_path),
                "num_episodes_per_task": int(args.num_episodes),
                "tasks": tasks_dict,
                "overall": {
                    "mean_survival_rate": mean_survival,
                    "mean_velocity_error": mean_vel_error,
                    "mean_distance": mean_distance,
                    "total_distance": total_distance,
                },
                "batch_two_phase": bool(args.batch_two_phase),
                "selection": {
                    "always_record": list(args.batch_always_record or []),
                    "worst_k": int(args.batch_worst_k),
                    "worst_metric": str(args.batch_worst_metric),
                },
            }

        # Phase 2: record videos for mixed_terrain + worst-K tasks (stable, less storage)
        if args.batch_two_phase:
            task_names = list(results.keys())

            always = [t for t in (args.batch_always_record or []) if t in task_names]

            # Determine worst tasks
            def _rank_key(t: str):
                m = results[t].metrics
                if args.batch_worst_metric == "survival_rate":
                    # Lower is worse
                    return (float(m.survival_rate),)
                if args.batch_worst_metric == "mean_velocity_error":
                    # Higher is worse (invert sort later)
                    return (-float(m.mean_velocity_error),)
                # survival_then_vel: lowest survival first, then highest vel error
                return (float(m.survival_rate), -float(m.mean_velocity_error))

            # For mean_velocity_error mode, _rank_key already flips sign so normal ascending works.
            sorted_tasks = sorted(task_names, key=_rank_key)
            worst_candidates = [t for t in sorted_tasks if t not in set(always)]
            worst = worst_candidates[: max(0, int(args.batch_worst_k))]

            record_tasks: list[str] = []
            for t in always + worst:
                if t not in record_tasks:
                    record_tasks.append(t)

            video_dir = Path(args.output_dir) / "eval_videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_evaluator = BatchEvaluator(
                simulator_class=BaseMujocoSimulator,
                xml_path=xml_path,
                onnx_path=onnx_path,
                output_dir=video_dir,
            )

            recorded = {"always": [], "worst": []}
            for t in always:
                res2 = video_evaluator.evaluate_task(
                    t,
                    num_episodes=max(1, int(args.batch_record_episodes)),
                    save_video=True,
                    video_steps=int(args.batch_video_steps),
                )
                recorded["always"].append({"task": t, "video": res2.video_path})

            for t in worst:
                res2 = video_evaluator.evaluate_task(
                    t,
                    num_episodes=max(1, int(args.batch_record_episodes)),
                    save_video=True,
                    video_steps=int(args.batch_video_steps),
                )
                recorded["worst"].append({"task": t, "video": res2.video_path})

            # Write/augment metrics JSON with the recorded video list
            if metrics_payload is not None and metrics_json_path is not None:
                metrics_payload["recorded_videos"] = recorded
                metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_json_path, "w") as f:
                    json.dump(metrics_payload, f, indent=2)
                print(f"[Info] Wrote metrics JSON: {metrics_json_path}")
        else:
            if metrics_payload is not None and metrics_json_path is not None:
                metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_json_path, "w") as f:
                    json.dump(metrics_payload, f, indent=2)
                print(f"[Info] Wrote metrics JSON: {metrics_json_path}")
        
    else:
        # Single task evaluation
        print(f"Task: {args.task}")
        print(f"Available tasks: {list_eval_tasks()}")
        
        # Config overrides (for sim2sim debugging / mjlab deploy.yaml compatibility)
        config_override = {
            "history_newest_first": (args.history_order == "newest_first"),
        }
        if args.deploy_yaml:
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise ImportError("PyYAML is required for --deploy-yaml. Install with: pip install pyyaml") from e

            deploy_yaml_path = Path(args.deploy_yaml)
            if not deploy_yaml_path.exists():
                print(f"[Error] deploy.yaml not found: {deploy_yaml_path}")
                sys.exit(1)

            with open(deploy_yaml_path, "r") as f:
                dj = yaml.safe_load(f)

            # Map mjlab deploy.yaml schema -> our OnnxConfig override keys
            # Extend existing overrides with deploy.yaml content
            if isinstance(dj, dict):
                if "default_joint_pos" in dj:
                    config_override["default_joint_pos"] = dj["default_joint_pos"]
                if "stiffness" in dj:
                    config_override["joint_stiffness"] = dj["stiffness"]
                if "damping" in dj:
                    config_override["joint_damping"] = dj["damping"]
                if "step_dt" in dj:
                    # step_dt is policy_dt in mjlab nomenclature
                    try:
                        config_override["policy_dt"] = float(dj["step_dt"])
                    except Exception:
                        pass
                actions = dj.get("actions", {})
                if isinstance(actions, dict):
                    jpa = actions.get("JointPositionAction", {})
                    if isinstance(jpa, dict):
                        if "scale" in jpa:
                            config_override["action_scale"] = jpa["scale"]
                        if "offset" in jpa:
                            config_override["action_offset"] = jpa["offset"]
                # Observations scales (optional; most mjlab deploy policies expect raw obs + internal normalizer)
                obs = dj.get("observations", {})
                if isinstance(obs, dict):
                    # Convert per-term scale vectors to our observation_scales mapping
                    obs_scales = {}
                    for k, v in obs.items():
                        if isinstance(v, dict) and "scale" in v:
                            obs_scales[k] = v["scale"]
                    if obs_scales:
                        config_override["observation_scales"] = obs_scales

            print(f"[Info] Loaded deploy.yaml overrides: {deploy_yaml_path}")

        # Create simulator
        simulator = BaseMujocoSimulator(
            xml_path=xml_path,
            onnx_path=onnx_path,
            config_override=config_override,
        )
        
        # Get task
        task = get_eval_task(args.task)
        
        # Override velocity if specified
        if args.velocity:
            velocity_cmd = tuple(args.velocity)
        else:
            velocity_cmd = task.velocity_command
        
        print(f"\nVelocity command: vx={velocity_cmd[0]:.2f}, vy={velocity_cmd[1]:.2f}, wz={velocity_cmd[2]:.2f}")

        # Deploy-like loop (recommended when rendering)
        if args.render and (args.deploy or args.forever):
            # Apply task terrain (rough/stairs/slope) if the XML contains a heightfield placeholder.
            _apply_task_terrain_to_mujoco_model(simulator, task)
            print("\n[Deploy] Running continuous ONNX inference + MuJoCo simulation (viewer stays open)")
            run_deploy_forever(
                simulator=simulator,
                velocity_command=velocity_cmd,
                max_steps_per_episode=(0 if args.forever else args.max_steps),
                reset_on_fall=(not args.no_reset_on_fall),
                realtime=(not args.no_realtime),
                record_video=args.record_video,
                record_width=args.record_width,
                record_height=args.record_height,
                record_fps=args.record_fps,
                record_steps=args.record_steps,
                follow=bool(args.follow),
                follow_body=str(args.follow_body),
                follow_distance=float(args.follow_distance),
                follow_azimuth=float(args.follow_azimuth),
                follow_elevation=float(args.follow_elevation),
                teleop=str(args.teleop),
                teleop_step_vx=float(args.teleop_step_vx),
                teleop_step_vy=float(args.teleop_step_vy),
                teleop_step_wz=float(args.teleop_step_wz),
                teleop_max_vx=float(args.teleop_max_vx),
                teleop_max_vy=float(args.teleop_max_vy),
                teleop_max_wz=float(args.teleop_max_wz),
            )
            return
        
        # Run episodes
        # If headless recording is requested, record ONE long video during the actual eval loop.
        if (not args.render) and bool(args.save_video):
            # Apply task terrain (headless) if available.
            _apply_task_terrain_to_mujoco_model(simulator, task)
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{args.task}_eval.mp4"
            all_results, _vp, _fps = _run_headless_eval_with_single_video(
                simulator=simulator,
                out_path=out_path,
                num_episodes=int(args.num_episodes),
                max_steps_per_episode=int(args.max_steps),
                velocity_command=velocity_cmd,
                width=int(args.video_width),
                height=int(args.video_height),
                max_video_steps=int(args.video_steps or 0),
                live_csv=getattr(args, "live_csv", None),
                live_stride=int(getattr(args, "live_stride", 2)),
            )
            print(
                f"[Info] Video saved: {out_path} "
                f"(fps={_fps}, frames={sum(int(r['stats']['num_steps']) for r in all_results)})"
            )
        elif args.render and args.forever:
            print("\n[Render] Forever mode enabled: running until the MuJoCo viewer is closed")
            simulator.run_forever_until_closed(
                max_steps_per_episode=args.max_steps,
                velocity_command=velocity_cmd,
            )
            return
        elif args.render and args.continuous and args.num_episodes > 1:
            print("\n[Render] Continuous viewer enabled: running episodes in a single MuJoCo window")
            all_results = simulator.run_episodes_continuous(
                num_episodes=args.num_episodes,
                max_steps_per_episode=args.max_steps,
                velocity_command=velocity_cmd,
                render=True,
            )
            for i, result in enumerate(all_results):
                stats = result["stats"]
                print(f"\nEpisode {i + 1}/{args.num_episodes}")
                print(f"  Steps: {stats['num_steps']}")
                print(f"  Distance: {stats['distance_traveled']:.2f}m")
                print(f"  Survived: {stats['survived']}")
        else:
            all_results = []
            for ep in range(args.num_episodes):
                print(f"\nEpisode {ep + 1}/{args.num_episodes}")
                
                result = simulator.run_episode(
                    max_steps=args.max_steps,
                    render=args.render,
                    velocity_command=velocity_cmd,
                )
                
                stats = result["stats"]
                print(f"  Steps: {stats['num_steps']}")
                print(f"  Distance: {stats['distance_traveled']:.2f}m")
                print(f"  Survived: {stats['survived']}")
                
                all_results.append(result)
        # (Legacy) Do not record a second pass video; single-mode records during eval above.
        
        # Summary
        survival_rate = sum(r["stats"]["survived"] for r in all_results) / len(all_results)
        mean_distance = np.mean([r["stats"]["distance_traveled"] for r in all_results])
        mean_steps = np.mean([r["stats"]["num_steps"] for r in all_results])
        
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        print(f"Survival Rate: {survival_rate:.1%}")
        print(f"Mean Distance: {mean_distance:.2f}m")
        print(f"Mean Steps:    {mean_steps:.1f}")
        print("=" * 40)

        # Optional: dump trajectories for the first episode (for plotting joint curves, etc.)
        if args.dump_npz:
            try:
                dump_path = Path(args.dump_npz)
                dump_path.parent.mkdir(parents=True, exist_ok=True)

                ep0 = all_results[0]
                joint_names = getattr(simulator.onnx_config, "joint_names", None) or []
                payload = {
                    "policy_dt": float(simulator.policy_dt),
                    "joint_names_json": json.dumps(list(joint_names)),
                    "stats_json": json.dumps(ep0.get("stats", {})),
                }
                # Trajectories
                data0 = ep0.get("data", {})
                for k in ("joint_pos", "base_pos", "base_lin_vel", "actions"):
                    if k in data0:
                        payload[k] = np.asarray(data0[k])

                np.savez(dump_path, **payload)
                print(f"[Info] Dumped trajectories: {dump_path}")
            except Exception as e:
                print(f"[Warning] Failed to dump trajectories: {e}")


if __name__ == "__main__":
    main()
