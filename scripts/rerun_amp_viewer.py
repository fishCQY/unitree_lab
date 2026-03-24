#!/usr/bin/env python3
"""Visualize retargeted AMP motion data in Rerun.

Usage examples:
    # List all clips in a pkl file
    python scripts/rerun_amp_viewer.py --file source/unitree_lab/unitree_lab/data/AMP/lafan_walk_clips.pkl --list

    # Play a specific clip
    python scripts/rerun_amp_viewer.py --file source/unitree_lab/unitree_lab/data/AMP/lafan_walk_clips.pkl --clip walk1_subject1

    # Play all clips sequentially
    python scripts/rerun_amp_viewer.py --file source/unitree_lab/unitree_lab/data/AMP/lafan_walk_clips.pkl

    # Adjust playback speed
    python scripts/rerun_amp_viewer.py --file source/unitree_lab/unitree_lab/data/AMP/lafan_dance_clips.pkl --speed 0.5

    # Play a time range within a clip (seconds)
    python scripts/rerun_amp_viewer.py --file source/unitree_lab/unitree_lab/data/AMP/lafan_walk_clips.pkl --clip walk1_subject1 --start 2.0 --end 5.0
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import rerun as rr
import trimesh

# Paths relative to repo root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = REPO_ROOT / "source" / "unitree_lab" / "unitree_lab" / "data"
DEFAULT_AMP_DIR = DATA_ROOT / "AMP"
URDF_DIR = DATA_ROOT / "LAFAN1_Retargeting_Dataset" / "robot_description" / "g1"
URDF_PATH = URDF_DIR / "g1_29dof_rev_1_0.urdf"


def load_pickle_compat(path: Path) -> dict:
    """Load pickle with numpy 2.x compatibility."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if e.name and e.name.startswith("numpy._core"):
                import numpy.core as np_core
                sys.modules.setdefault("numpy._core", np_core)
                sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
                sys.modules.setdefault("numpy._core.numeric", np_core.numeric)
                sys.modules.setdefault("numpy._core.umath", np_core.umath)
                f.seek(0)
                return pickle.load(f)
            raise


class RerunRobot:
    """Manages a Pinocchio robot model and logs meshes to Rerun."""

    def __init__(self, urdf_path: Path, entity_prefix: str = "robot"):
        self.prefix = entity_prefix
        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(urdf_path), str(urdf_path.parent), pin.JointModelFreeFlyer()
        )
        self.link2mesh: dict[str, trimesh.Trimesh] = {}
        self._load_meshes()
        self._log_static_meshes()

    def _load_meshes(self):
        for visual in self.robot.visual_model.geometryObjects:
            name = visual.name[:-2]
            mesh = trimesh.load_mesh(visual.meshPath)
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            self.link2mesh[name] = mesh

    def _log_static_meshes(self):
        """Log mesh geometry once as static entities."""
        self.robot.framesForwardKinematics(pin.neutral(self.robot.model))
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            mesh = self.link2mesh[frame_name]

            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parent
            parent_joint_name = self.robot.model.names[parent_joint_id]

            frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]

            # Log joint transform
            rr.log(
                f"{self.prefix}/{parent_joint_name}",
                rr.Transform3D(
                    translation=joint_tf.translation,
                    mat3x3=joint_tf.rotation,
                    axis_length=0.0,
                ),
            )

            # Apply relative transform and log mesh as static
            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(
                f"{self.prefix}/{parent_joint_name}/{frame_name}",
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.faces,
                    vertex_normals=mesh.vertex_normals,
                    vertex_colors=mesh.visual.vertex_colors,
                ),
                static=True,
            )

    def update(self, configuration: np.ndarray):
        """Update joint transforms for one frame. configuration is pinocchio q vector."""
        self.robot.framesForwardKinematics(configuration)
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parent
            parent_joint_name = self.robot.model.names[parent_joint_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(
                f"{self.prefix}/{parent_joint_name}",
                rr.Transform3D(
                    translation=joint_tf.translation,
                    mat3x3=joint_tf.rotation,
                    axis_length=0.0,
                ),
            )


def build_pinocchio_q(root_pos: np.ndarray, root_rot_wxyz: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
    """Build pinocchio free-flyer configuration vector.

    Pinocchio expects: [x, y, z, qx, qy, qz, qw, joint_angles...]
    AMP data has root_rot in wxyz format.
    """
    w, x, y, z = root_rot_wxyz
    q = np.concatenate([
        root_pos,           # x, y, z
        [x, y, z, w],      # quaternion xyzw for pinocchio
        dof_pos,            # 29 joint angles
    ])
    return q.astype(np.float64)


def play_clip(
    robot: RerunRobot,
    clip_name: str,
    clip_data: dict,
    speed: float = 1.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    time_offset: float = 0.0,
    realtime: bool = True,
):
    """Play a single clip through the Rerun viewer."""
    fps = float(clip_data["fps"])
    dof_pos = np.asarray(clip_data["dof_pos"], dtype=np.float64)
    root_pos = np.asarray(clip_data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(clip_data["root_rot"], dtype=np.float64)
    num_frames = dof_pos.shape[0]

    start_frame = 0
    end_frame = num_frames
    if start_sec is not None:
        start_frame = max(0, int(round(start_sec * fps)))
    if end_sec is not None:
        end_frame = min(num_frames, int(round(end_sec * fps)))

    dt = 1.0 / fps
    print(f"  Playing '{clip_name}': frames [{start_frame}, {end_frame}) "
          f"({(end_frame - start_frame) / fps:.1f}s @ {fps}fps)")

    for i in range(start_frame, end_frame):
        t = time_offset + (i - start_frame) * dt
        rr.set_time_seconds("time", t)
        rr.set_time_sequence("frame", int(time_offset * fps) + i - start_frame)

        q = build_pinocchio_q(root_pos[i], root_rot[i], dof_pos[i])
        robot.update(q)

        # Log root trajectory
        rr.log("root_trajectory", rr.Points3D([root_pos[i]], radii=0.01, colors=[[0, 120, 255]]))

        if realtime:
            time.sleep(dt / speed)

    return time_offset + (end_frame - start_frame) * dt


def main():
    parser = argparse.ArgumentParser(description="Visualize AMP motion data in Rerun.")
    parser.add_argument("--file", type=str, required=True, help="Path to AMP .pkl file")
    parser.add_argument("--clip", type=str, default=None, help="Specific clip name to play (default: all)")
    parser.add_argument("--list", action="store_true", help="List available clips and exit")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds (within clip)")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds (within clip)")
    parser.add_argument("--urdf", type=str, default=str(URDF_PATH), help="Path to robot URDF")
    parser.add_argument("--no-realtime", action="store_true", help="Log all frames instantly (no sleep)")
    parser.add_argument("--loop", type=int, default=1, help="Number of times to loop playback")
    args = parser.parse_args()

    pkl_path = Path(args.file)
    if not pkl_path.is_absolute():
        pkl_path = REPO_ROOT / pkl_path

    print(f"Loading {pkl_path} ...")
    data = load_pickle_compat(pkl_path)

    clip_names = list(data.keys())
    print(f"Found {len(clip_names)} clips")

    if args.list:
        for i, name in enumerate(clip_names):
            clip = data[name]
            fps = float(clip["fps"])
            n = np.asarray(clip["dof_pos"]).shape[0]
            print(f"  [{i:3d}] {name:40s}  {n:6d} frames  {n / fps:.1f}s  @ {fps:.0f}fps")
        return

    # Select clips to play
    if args.clip is not None:
        if args.clip not in data:
            # Try matching by index
            try:
                idx = int(args.clip)
                args.clip = clip_names[idx]
            except (ValueError, IndexError):
                print(f"Error: clip '{args.clip}' not found. Use --list to see available clips.")
                return
        clips_to_play = [args.clip]
    else:
        clips_to_play = clip_names

    # Initialize Rerun
    rr.init("AMP Motion Viewer", spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Add ground plane
    rr.log(
        "ground",
        rr.Boxes3D(
            centers=[[0, 0, -0.005]],
            sizes=[[20, 20, 0.01]],
            colors=[[200, 200, 200, 100]],
        ),
        static=True,
    )

    print(f"Loading robot URDF: {args.urdf}")
    urdf_path = Path(args.urdf)
    robot = RerunRobot(urdf_path, entity_prefix="robot")

    for loop_i in range(args.loop):
        if args.loop > 1:
            print(f"Loop {loop_i + 1}/{args.loop}")
        t_offset = 0.0
        for clip_name in clips_to_play:
            t_offset = play_clip(
                robot=robot,
                clip_name=clip_name,
                clip_data=data[clip_name],
                speed=args.speed,
                start_sec=args.start,
                end_sec=args.end,
                time_offset=t_offset,
                realtime=not args.no_realtime,
            )

    print("Done.")


if __name__ == "__main__":
    main()
