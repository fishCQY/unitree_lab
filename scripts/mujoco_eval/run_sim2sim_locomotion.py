#!/usr/bin/env python3
"""Sim2Sim locomotion evaluation in MuJoCo.

Loads an ONNX policy exported from IsaacLab and replays it in MuJoCo with
environment alignment (observation semantics, PD control, terrain, timing).

Usage examples:
    # Interactive viewer with keyboard teleop (flat ground)
    python scripts/mujoco_eval/run_sim2sim_locomotion.py \
        --robot g1 --onnx path/to/policy.onnx --task flat_forward \
        --render --teleop keyboard

    # Headless batch evaluation with video recording
    python scripts/mujoco_eval/run_sim2sim_locomotion.py \
        --robot g1 --onnx path/to/policy.onnx --task rough_forward \
        --num-episodes 10 --save-video --output-dir eval_results

    # Deploy mode with optional deploy.yaml for PD gains override
    python scripts/mujoco_eval/run_sim2sim_locomotion.py \
        --robot g1 --onnx path/to/policy.onnx --task flat_forward \
        --render --deploy --follow --deploy-yaml path/to/deploy.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Ensure unitree_lab.mujoco_utils is importable even when the full Isaac Lab
# environment is not available (unitree_lab.__init__ imports isaaclab_tasks).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_source_pkg = PROJECT_ROOT / "source" / "unitree_lab"
for _p in [str(PROJECT_ROOT), str(_source_pkg)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Prevent the top-level unitree_lab package from triggering Isaac Lab imports.
# We only need the mujoco_utils subpackage here.
import types

_ul = types.ModuleType("unitree_lab")
_ul.__path__ = [str(_source_pkg / "unitree_lab")]
_ul.__package__ = "unitree_lab"
sys.modules.setdefault("unitree_lab", _ul)

import numpy as np

ROBOT_CONFIGS: dict[str, dict[str, Any]] = {
    "g1": {
        "xml_flat": "g1/scene_29dof.xml",
        "xml_terrain": "g1/scene_29dof_terrain.xml",
        "dof": 29,
    },
}


def _find_asset_dir() -> Path:
    """Locate the robots_xml asset directory."""
    candidates = [
        Path(__file__).resolve().parents[2] / "source" / "unitree_lab" / "unitree_lab" / "assets" / "robots_xml",
        Path(__file__).resolve().parents[1] / "source" / "unitree_lab" / "unitree_lab" / "assets" / "robots_xml",
        PROJECT_ROOT / "source" / "unitree_lab" / "unitree_lab" / "assets" / "robots_xml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot find robots_xml directory. Searched: {[str(c) for c in candidates]}"
    )


def _load_deploy_yaml(path: str | Path) -> dict[str, Any]:
    """Load deploy.yaml and return as a flat dict."""
    import yaml

    path = Path(path)
    if not path.exists():
        print(f"[Warning] deploy.yaml not found: {path}")
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def _deploy_yaml_to_config_override(deploy: dict) -> dict[str, Any]:
    """Convert deploy.yaml contents to OnnxConfig-compatible override dict.

    Handles both direct metadata keys (from ONNX metadata_json) and the
    mjlab-style deploy.yaml schema produced by train.py's
    ``_build_deploy_yaml_from_metadata``.
    """
    override: dict[str, Any] = {}

    # Direct keys (ONNX metadata style)
    _DIRECT_KEYS = [
        "joint_stiffness", "joint_damping", "joint_names", "default_joint_pos",
        "action_scale", "action_offset", "observation_names", "observation_dims",
        "observation_scales", "sim_dt", "decimation", "joint_armature",
        "history_length", "single_frame_dims",
    ]
    for k in _DIRECT_KEYS:
        if k in deploy:
            override[k] = deploy[k]

    # mjlab-style aliases produced by train.py
    if "stiffness" in deploy and "joint_stiffness" not in override:
        override["joint_stiffness"] = deploy["stiffness"]
    if "damping" in deploy and "joint_damping" not in override:
        override["joint_damping"] = deploy["damping"]
    if "armature" in deploy and "joint_armature" not in override:
        override["joint_armature"] = deploy["armature"]

    # actions.JointPositionAction.{scale, offset}
    actions = deploy.get("actions", {})
    jpa = actions.get("JointPositionAction", {})
    if "scale" in jpa and "action_scale" not in override:
        override["action_scale"] = jpa["scale"]
    if "offset" in jpa and "action_offset" not in override:
        override["action_offset"] = jpa["offset"]

    # observations.<term>.scale -> observation_scales dict
    obs_dict = deploy.get("observations", {})
    if isinstance(obs_dict, dict) and obs_dict and "observation_scales" not in override:
        scales: dict[str, float] = {}
        for term, cfg in obs_dict.items():
            if isinstance(cfg, dict) and "scale" in cfg:
                scales[term] = cfg["scale"]
        if scales:
            override["observation_scales"] = scales

    return override


def _setup_terrain(
    simulator: Any,
    task: Any,
) -> float:
    """Generate and inject terrain heightfield data into the simulator model.

    Returns the recommended spawn z-offset for the floating base.
    """
    from unitree_lab.mujoco_utils.terrain.generator import MujocoTerrainGenerator
    from unitree_lab.mujoco_utils.terrain.xml_generation import setup_terrain_data_in_model

    terrain_cfg_dict = task.get_terrain_config()
    terrain_type = terrain_cfg_dict.get("terrain_type", "flat")

    if terrain_type == "flat":
        return 0.0

    # Build TerrainConfig from the task's terrain params, matching the XML heightfield dimensions.
    hfield_name = "terrain_hfield"
    import mujoco as mj

    hfield_id = mj.mj_name2id(simulator.model, mj.mjtObj.mjOBJ_HFIELD, hfield_name)
    if hfield_id < 0:
        print("[Warning] No heightfield found in model; skipping terrain injection.")
        return 0.0

    nrow = int(simulator.model.hfield_nrow[hfield_id])
    ncol = int(simulator.model.hfield_ncol[hfield_id])
    sx = float(simulator.model.hfield_size[hfield_id, 0])
    sy = float(simulator.model.hfield_size[hfield_id, 1])
    world_x = sx * 2
    world_y = sy * 2

    gen_cfg = {
        "terrain_type": terrain_type,
        "size": (world_x, world_y),
        "horizontal_scale": world_x / max(1, ncol - 1),
    }
    for k, v in terrain_cfg_dict.items():
        if k != "terrain_type":
            gen_cfg[k] = v

    generator = MujocoTerrainGenerator(gen_cfg)
    generator.nx = ncol
    generator.ny = nrow
    generator.generate()
    setup_terrain_data_in_model(simulator.model, generator, hfield_name=hfield_name)

    spawn_z = generator.get_spawn_height(0.0, 0.0)
    return max(0.0, spawn_z)


class KeyboardTeleop:
    """Keyboard velocity command controller for the MuJoCo viewer.

    Uses GLFW key codes (passed by mujoco.viewer's key_callback):
      W/S  -> vx (forward/backward)
      A/D  -> vy (left/right)
      Q/E  -> wz (yaw left/right)
      SPACE -> zero all

    IMPORTANT: ``handle_key`` is invoked from the GLFW render thread.
    All I/O (print, logging) is deferred to the main simulation thread via
    ``_pending_msg`` to avoid stdout contention / GIL deadlocks that freeze
    the terminal.
    """

    VX_STEP = 0.1
    VY_STEP = 0.1
    WZ_STEP = 0.1
    VX_RANGE = (-1.0, 2.0)
    VY_RANGE = (-0.6, 0.6)
    WZ_RANGE = (-1.0, 1.0)

    # GLFW key codes (portable, no glfw import needed at class level)
    _KEY_W = 87
    _KEY_S = 83
    _KEY_A = 65
    _KEY_D = 68
    _KEY_Q = 81
    _KEY_E = 69
    _KEY_SPACE = 32

    def __init__(self) -> None:
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self._dirty = False
        self._pending_msg: str | None = None

    def handle_key(self, key: int) -> None:
        """Handle a key press (GLFW key code).

        Called from the GLFW render thread – must NOT do any I/O here.
        """
        if key == self._KEY_W:
            self.vx = min(self.vx + self.VX_STEP, self.VX_RANGE[1])
        elif key == self._KEY_S:
            self.vx = max(self.vx - self.VX_STEP, self.VX_RANGE[0])
        elif key == self._KEY_A:
            self.vy = min(self.vy + self.VY_STEP, self.VY_RANGE[1])
        elif key == self._KEY_D:
            self.vy = max(self.vy - self.VY_STEP, self.VY_RANGE[0])
        elif key == self._KEY_Q:
            self.wz = min(self.wz + self.WZ_STEP, self.WZ_RANGE[1])
        elif key == self._KEY_E:
            self.wz = max(self.wz - self.WZ_STEP, self.WZ_RANGE[0])
        elif key == self._KEY_SPACE:
            self.vx = self.vy = self.wz = 0.0
        else:
            return

        self._dirty = True
        self._pending_msg = (
            f"[Teleop] vx={self.vx:+.1f}  vy={self.vy:+.1f}  wz={self.wz:+.1f}"
        )

    def flush_msg(self) -> None:
        """Print any pending status message (call from main thread only)."""
        msg = self._pending_msg
        if msg is not None:
            self._pending_msg = None
            print(msg)

    @property
    def command(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.wz)


def _build_overlay_text(
    teleop: KeyboardTeleop | None,
    simulator: Any,
    step: int,
    fps: float = 0.0,
) -> list[tuple]:
    """Build text overlay tuples for viewer.set_texts."""
    import mujoco

    texts = []

    # Velocity command overlay (bottom-left)
    if teleop is not None:
        vx, vy, wz = teleop.command
        cmd_left = "Velocity Cmd\nvx\nvy\nwz"
        cmd_right = f"(W/S A/D Q/E)\n{vx:+.2f} m/s\n{vy:+.2f} m/s\n{wz:+.2f} rad/s"
    else:
        vx, vy, wz = simulator.velocity_command
        cmd_left = "Velocity Cmd\nvx\nvy\nwz"
        cmd_right = f"(fixed)\n{vx:+.2f} m/s\n{vy:+.2f} m/s\n{wz:+.2f} rad/s"
    texts.append((
        mujoco.mjtFontScale.mjFONTSCALE_150,
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        cmd_left,
        cmd_right,
    ))

    # Step counter + base height + FPS (bottom-right)
    z = simulator.base_pos[2]
    texts.append((
        mujoco.mjtFontScale.mjFONTSCALE_150,
        mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        f"Step\nHeight\nFPS",
        f"{step}\n{z:.3f} m\n{int(fps)}",
    ))

    return texts


def run_interactive(
    simulator: Any,
    task: Any,
    teleop: str = "off",
    follow: bool = False,
    velocity: tuple[float, float, float] | None = None,
    max_steps_per_episode: int = 1000,
) -> None:
    """Run interactive sim2sim with MuJoCo viewer.

    Real-time pacing: each policy step targets ``simulator.policy_dt`` of
    wall-clock time so the viewer runs at the correct real-time speed
    (typically 50 Hz for policy_dt=0.02 s).  If the physics computation
    takes longer than policy_dt the viewer will still remain responsive.
    """
    import mujoco
    import mujoco.viewer

    teleop_ctrl: KeyboardTeleop | None = None
    key_callback = None
    if teleop == "keyboard":
        teleop_ctrl = KeyboardTeleop()
        if velocity:
            teleop_ctrl.vx, teleop_ctrl.vy, teleop_ctrl.wz = velocity
        key_callback = teleop_ctrl.handle_key
        print("[Teleop] Keyboard controls: W/S=vx, A/D=vy, Q/E=wz, SPACE=zero")

    if velocity and teleop != "keyboard":
        simulator.set_velocity_command(*velocity)
    elif hasattr(task, "velocity_command") and teleop != "keyboard":
        simulator.set_velocity_command(*task.velocity_command)

    viewer = mujoco.viewer.launch_passive(
        simulator.model, simulator.data, key_callback=key_callback,
    )

    policy_dt = getattr(simulator, "policy_dt", 0.02)
    target_fps = int(round(1.0 / policy_dt)) if policy_dt > 0 else 50
    print(f"[Sim2Sim] Interactive mode: target FPS = {target_fps} (policy_dt={policy_dt:.4f}s)")

    _fps_window: list[float] = []
    _FPS_WINDOW_SIZE = 50
    _fps_display = 0
    _fps_print_interval = 100

    try:
        while viewer.is_running():
            simulator.reset()
            if teleop_ctrl is not None:
                simulator.set_velocity_command(*teleop_ctrl.command)
            elif velocity:
                simulator.set_velocity_command(*velocity)
            elif hasattr(task, "velocity_command"):
                simulator.set_velocity_command(*task.velocity_command)

            _prev_time = time.monotonic()
            _fps_window.clear()

            for _step in range(max_steps_per_episode):
                if not viewer.is_running():
                    return

                if teleop_ctrl is not None and teleop_ctrl._dirty:
                    simulator.set_velocity_command(*teleop_ctrl.command)
                    teleop_ctrl._dirty = False
                    teleop_ctrl.flush_msg()

                simulator.step()

                if follow:
                    _update_camera_follow(viewer, simulator)

                viewer.set_texts(
                    _build_overlay_text(teleop_ctrl, simulator, _step + 1, fps=_fps_display)
                )
                viewer.sync()

                if simulator._check_termination():
                    print(f"[Sim2Sim] Terminated at step {_step + 1}, FPS: {_fps_display}, resetting...")
                    time.sleep(0.3)
                    break

                # Real-time pacing: sleep remainder of policy_dt budget
                now = time.monotonic()
                compute_elapsed = now - _prev_time
                remaining = policy_dt - compute_elapsed
                if remaining > 0:
                    time.sleep(remaining)

                # Measure true frame-to-frame interval (includes sleep)
                now = time.monotonic()
                frame_dt = now - _prev_time
                _prev_time = now

                _fps_window.append(frame_dt)
                if len(_fps_window) > _FPS_WINDOW_SIZE:
                    _fps_window.pop(0)
                _fps_display = int(round(len(_fps_window) / sum(_fps_window)))

                if _step > 0 and _step % _fps_print_interval == 0:
                    print(f"[Sim2Sim] Step {_step:5d}  FPS: {_fps_display} (target: {target_fps})")
    finally:
        viewer.close()


def _update_camera_follow(viewer: Any, simulator: Any) -> None:
    """Update viewer camera to follow the robot base."""
    try:
        pos = simulator.base_pos
        cam = viewer.cam
        cam.lookat[:] = pos
    except Exception:
        pass


def run_headless(
    simulator: Any,
    task: Any,
    num_episodes: int = 10,
    save_video: bool = False,
    output_dir: str = "eval_results",
    video_steps: int = 500,
) -> dict[str, Any]:
    """Run headless evaluation episodes."""
    from unitree_lab.mujoco_utils.evaluation.batch_evaluator import BatchEvaluator
    from unitree_lab.mujoco_utils.evaluation.metrics import (
        compute_locomotion_metrics,
        print_metrics,
    )

    vel_cmd = getattr(task, "velocity_command", (0.5, 0.0, 0.0))
    episode_results = []

    for ep_idx in range(num_episodes):
        print(f"  Episode {ep_idx + 1}/{num_episodes}", end="\r", flush=True)
        result = simulator.run_episode(
            max_steps=task.max_episode_steps,
            render=False,
            velocity_command=vel_cmd,
        )
        episode_results.append(result)

    print()

    metrics = compute_locomotion_metrics(
        episode_results, np.array(vel_cmd), simulator.policy_dt
    )
    print_metrics(metrics, task.name)

    summary = {
        "task": task.name,
        "num_episodes": num_episodes,
        "survival_rate": metrics.survival_rate,
        "mean_velocity_error": metrics.mean_velocity_error,
        "mean_forward_distance": metrics.mean_forward_distance,
    }

    if save_video:
        _record_video_headless(simulator, task, output_dir, video_steps)

    return summary


def _record_video_headless(
    simulator: Any,
    task: Any,
    output_dir: str,
    duration_steps: int = 500,
) -> str | None:
    """Record an evaluation video without a viewer window."""
    try:
        import cv2
        import mujoco
    except ImportError:
        print("[Warning] cv2 or mujoco not available; skipping video recording.")
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 720
    renderer = mujoco.Renderer(simulator.model, height, width)

    simulator.reset()
    vel_cmd = getattr(task, "velocity_command", (0.5, 0.0, 0.0))
    simulator.set_velocity_command(*vel_cmd)

    frames: list[np.ndarray] = []
    for _ in range(duration_steps):
        renderer.update_scene(simulator.data)
        frame = renderer.render()
        frames.append(frame)
        simulator.step()

    video_file = out_path / f"{task.name}_sim2sim.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(1.0 / simulator.policy_dt) if simulator.policy_dt > 0 else 50
    writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    # Best-effort transcode to H.264
    try:
        import subprocess

        tmp = video_file.with_suffix(".h264_tmp.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_file),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-preset", "veryfast", "-crf", "23",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(video_file)
    except Exception:
        pass

    print(f"  Video saved: {video_file}")
    return str(video_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sim2Sim locomotion evaluation in MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--robot", type=str, default="g1", choices=list(ROBOT_CONFIGS.keys()),
                        help="Robot name (default: g1)")
    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to ONNX policy file")
    parser.add_argument("--xml", type=str, default=None,
                        help="Path to MuJoCo XML (overrides auto-detection)")
    parser.add_argument("--task", type=str, default="flat_forward",
                        help="Evaluation task name (see --list-tasks)")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List available evaluation tasks and exit")

    # Rendering & interaction
    parser.add_argument("--render", action="store_true",
                        help="Open interactive MuJoCo viewer")
    parser.add_argument("--follow", action="store_true",
                        help="Camera follows the robot")
    parser.add_argument("--teleop", type=str, default="off", choices=["keyboard", "off"],
                        help="Teleop mode: keyboard or off")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy mode (interactive with auto-reset)")

    # Velocity command override
    parser.add_argument("--velocity", type=float, nargs=3, default=None,
                        metavar=("VX", "VY", "WZ"),
                        help="Fixed velocity command [vx, vy, wz]")
    parser.add_argument("--skip-terrain-inject", action="store_true",
                        help="Do NOT overwrite heightfield data at runtime; use XML terrain as-is")

    # Deploy config
    parser.add_argument("--deploy-yaml", type=str, default=None,
                        help="Path to deploy.yaml for PD gains/joint config override")
    parser.add_argument("--config-override", type=str, default=None,
                        help="JSON string of config overrides")

    # Headless evaluation
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of evaluation episodes (headless)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max episode steps from task")
    parser.add_argument("--save-video", action="store_true",
                        help="Record evaluation video")
    parser.add_argument("--video-steps", type=int, default=500,
                        help="Steps per video recording")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Output directory for results/videos")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # List tasks
    if args.list_tasks:
        from unitree_lab.mujoco_utils.evaluation.eval_task import list_eval_tasks
        print("Available evaluation tasks:")
        for name in list_eval_tasks():
            print(f"  - {name}")
        return

    # Validate ONNX path
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"[Error] ONNX file not found: {onnx_path}")
        sys.exit(1)

    # Get task
    from unitree_lab.mujoco_utils.evaluation.eval_task import get_eval_task
    task = get_eval_task(args.task)
    print(f"\n{'='*60}")
    print(f"  Sim2Sim Locomotion: {task.name}")
    print(f"  {task.description}")
    print(f"{'='*60}\n")

    # Override max episode steps if provided
    if args.max_steps is not None:
        task.max_episode_steps = args.max_steps

    # Robot config
    robot_cfg = ROBOT_CONFIGS[args.robot]
    asset_dir = _find_asset_dir()

    if args.xml:
        xml_path = Path(args.xml)
        uses_terrain = False
        if "terrain" in xml_path.name.lower():
            uses_terrain = task.terrain_type not in ("flat",)
    else:
        uses_terrain = task.terrain_type not in ("flat",)
        xml_key = "xml_terrain" if uses_terrain else "xml_flat"
        xml_path = asset_dir / robot_cfg[xml_key]
        if not xml_path.exists():
            print(f"[Warning] Terrain XML not found ({xml_path}), falling back to flat.")
            xml_path = asset_dir / robot_cfg["xml_flat"]

    print(f"[Sim2Sim] Robot: {args.robot}")
    print(f"[Sim2Sim] XML: {xml_path}")
    print(f"[Sim2Sim] ONNX: {onnx_path}")

    # Build config override
    config_override: dict[str, Any] = {}

    if args.deploy_yaml:
        deploy = _load_deploy_yaml(args.deploy_yaml)
        config_override.update(_deploy_yaml_to_config_override(deploy))
        print(f"[Sim2Sim] Loaded deploy.yaml: {args.deploy_yaml}")

    if args.config_override:
        import json
        try:
            extra = json.loads(args.config_override)
            config_override.update(extra)
        except Exception as e:
            print(f"[Warning] Failed to parse --config-override: {e}")

    # Create simulator
    from unitree_lab.mujoco_utils.simulation.base_simulator import BaseMujocoSimulator

    simulator = BaseMujocoSimulator(
        xml_path=str(xml_path),
        onnx_path=str(onnx_path),
        config_override=config_override if config_override else None,
    )

    # Setup terrain if needed (unless user wants to use XML terrain as-is)
    if uses_terrain and (not bool(args.skip_terrain_inject)):
        spawn_z = _setup_terrain(simulator, task)
        if spawn_z > 0:
            simulator.spawn_root_z_offset = spawn_z
            print(f"[Sim2Sim] Terrain spawn z-offset: {spawn_z:.2f}m")

    # Velocity override
    velocity = tuple(args.velocity) if args.velocity else None

    # Run
    if args.render or args.deploy:
        max_steps = task.max_episode_steps
        if args.deploy:
            max_steps = max(max_steps, 2000)

        run_interactive(
            simulator=simulator,
            task=task,
            teleop=args.teleop,
            follow=args.follow,
            velocity=velocity,
            max_steps_per_episode=max_steps,
        )
    else:
        summary = run_headless(
            simulator=simulator,
            task=task,
            num_episodes=args.num_episodes,
            save_video=args.save_video,
            output_dir=args.output_dir,
            video_steps=args.video_steps,
        )
        print(f"\n[Result] {summary}")


if __name__ == "__main__":
    main()
