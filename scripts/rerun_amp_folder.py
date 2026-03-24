#!/usr/bin/env python3
"""Visualize all AMP motion clips from a folder in Rerun.

Scans a directory for .pkl motion files and logs every clip into a single
Rerun recording so you can scrub through all motions on one timeline.
Shows the full 3D robot mesh loaded from URDF with joint-level animation.

Dependencies:
    pip install rerun-sdk numpy

Usage examples:

    # List every clip across all pkl files in the default AMP folder
    python scripts/rerun_amp_folder.py --list

    # Play all clips (with full robot mesh)
    python scripts/rerun_amp_folder.py --exclude-all

    # Play only walk clips
    python scripts/rerun_amp_folder.py --filter walk --exclude-all

    # Play a specific file
    python scripts/rerun_amp_folder.py --file lafan_jump_clips.pkl

    # Play a specific clip by name
    python scripts/rerun_amp_folder.py --clip jumps1_subject1

    # Adjust speed, limit clip duration, or loop
    python scripts/rerun_amp_folder.py --speed 2.0 --max-sec 10 --loop 3

    # Log to an .rrd file instead of spawning the viewer
    python scripts/rerun_amp_folder.py --save output.rrd --no-realtime

    # Skip all_clips to avoid duplicates
    python scripts/rerun_amp_folder.py --exclude-all
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = REPO_ROOT / "source" / "unitree_lab" / "unitree_lab" / "data"
DEFAULT_AMP_DIR = DATA_ROOT / "AMP"
URDF_PATH = DATA_ROOT / "LAFAN1_Retargeting_Dataset" / "robot_description" / "g1" / "g1_29dof_rev_1_0.urdf"

# ===========================================================================
# PLAYLIST — configure clips to rerun here
#
# When non-empty, ONLY these clips are played (CLI filters are ignored).
# When empty (PLAYLIST = []), falls back to CLI args / full folder scan.
#
# Each entry is a dict with:
#   "file":      pkl filename (searched in DEFAULT_AMP_DIR and its subdirs)
#   "clip":      (optional) clip name inside the pkl; omit to play all clips
#   "start":     (optional) start time in seconds within the clip
#   "end":       (optional) end time in seconds within the clip
#
# Examples:
#   {"file": "lafan_walk_clips.pkl", "clip": "walk1_subject1", "start": 3.36, "end": 39.1}
#   {"file": "lafan_run_clips.pkl"}                       # all clips, full duration
#   {"file": "individual/run1_subject2.pkl", "end": 30}   # first 30 seconds
# ===========================================================================
PLAYLIST: list[dict] = [
    # --- 取消注释下面的行即可播放对应片段，注释掉则回退到 CLI 模式 ---
    {"file": "lafan_walk_clips.pkl", "clip": "walk1_subject1", "start": 3.36, "end": 39.1},
    {"file": "lafan_walk_clips.pkl", "clip": "walk1_subject1", "start": 81.86, "end": 120.4},
    {"file": "lafan_walk_clips.pkl", "clip": "walk1_subject2", "start": 78.13, "end": 132.23},
    {"file": "lafan_walk_clips.pkl", "clip": "walk1_subject2", "start": 173.03, "end": 218.2},
    {"file": "lafan_run_clips.pkl",  "clip": "run1_subject2",  "start": 116, "end": 159},
    {"file": "lafan_run_clips.pkl",  "clip": "run1_subject5",  "start": 5.6, "end": 56.6},
    # {"file": "lafan_dance_clips.pkl"},                           # all dance clips, full duration
    # {"file": "individual/walk1_subject1.pkl", "end": 30},        # first 30s from individual pkl
]


# ---------------------------------------------------------------------------
# Pickle compat (numpy 2.x / 1.x)
# ---------------------------------------------------------------------------

def _load_pickle_compat(path: Path) -> Any:
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


def _flatten_pkl(raw: Any) -> dict[str, dict]:
    """Normalize pkl payload to {clip_name: clip_data}."""
    if isinstance(raw, dict):
        first_val = next(iter(raw.values()), None)
        if isinstance(first_val, dict):
            return raw
        return {"clip_0": raw}
    if isinstance(raw, list):
        return {f"clip_{i}": clip for i, clip in enumerate(raw)}
    raise TypeError(f"Unexpected payload type: {type(raw)}")


# ---------------------------------------------------------------------------
# Scan folder
# ---------------------------------------------------------------------------

def scan_amp_folder(folder: Path, exclude_all: bool = False) -> list[tuple[Path, str, dict]]:
    """Return list of (pkl_path, clip_name, clip_data) for every clip."""
    results: list[tuple[Path, str, dict]] = []
    for pkl_path in sorted(folder.glob("*.pkl")):
        if exclude_all and "all" in pkl_path.stem.lower():
            continue
        raw = _load_pickle_compat(pkl_path)
        clips = _flatten_pkl(raw)
        for clip_name, clip_data in clips.items():
            results.append((pkl_path, clip_name, clip_data))
    return results


def _find_pkl(folder: Path, filename: str) -> Path | None:
    """Find a pkl file by name, searching folder and its subdirectories."""
    candidate = folder / filename
    if candidate.exists():
        return candidate
    for p in folder.rglob(filename):
        return p
    return None


def _slice_clip(clip_data: dict, start_sec: float | None, end_sec: float | None) -> dict:
    """Return a time-sliced copy of clip_data."""
    fps = float(clip_data.get("fps", 50))
    total = np.asarray(clip_data["dof_pos"]).shape[0]
    s = 0 if start_sec is None else max(0, int(round(start_sec * fps)))
    e = total if end_sec is None else min(total, int(round(end_sec * fps)))
    if s == 0 and e == total:
        return clip_data
    out = {}
    for k, v in clip_data.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == total:
            out[k] = v[s:e]
        else:
            out[k] = v
    return out


def resolve_playlist(folder: Path, playlist: list[dict]) -> list[tuple[Path, str, dict]]:
    """Turn PLAYLIST config into the standard entries list."""
    entries: list[tuple[Path, str, dict]] = []
    for item in playlist:
        filename = item["file"]
        pkl_path = _find_pkl(folder, filename)
        if pkl_path is None:
            print(f"  [SKIP] pkl not found: {filename}")
            continue

        raw = _load_pickle_compat(pkl_path)
        clips = _flatten_pkl(raw)

        target_clip = item.get("clip")
        start_sec = item.get("start")
        end_sec = item.get("end")

        for clip_name, clip_data in clips.items():
            if target_clip is not None and clip_name != target_clip:
                continue
            sliced = _slice_clip(clip_data, start_sec, end_sec)
            entries.append((pkl_path, clip_name, sliced))

    return entries


def print_clip_table(entries: list[tuple[Path, str, dict]]) -> None:
    print(f"{'idx':>4s}  {'file':<30s}  {'clip_name':<35s}  {'frames':>7s}  {'duration':>8s}  {'fps':>4s}")
    print("-" * 100)
    for i, (pkl_path, clip_name, clip_data) in enumerate(entries):
        fps = float(clip_data.get("fps", 50))
        n = np.asarray(clip_data["dof_pos"]).shape[0]
        dur = n / fps
        print(f"{i:4d}  {pkl_path.name:<30s}  {clip_name:<35s}  {n:7d}  {dur:7.1f}s  {fps:4.0f}")
    total_frames = sum(np.asarray(e[2]["dof_pos"]).shape[0] for e in entries)
    total_dur = sum(np.asarray(e[2]["dof_pos"]).shape[0] / float(e[2].get("fps", 50)) for e in entries)
    print(f"\nTotal: {len(entries)} clips, {total_frames} frames, {total_dur:.0f}s")


# ---------------------------------------------------------------------------
# Rerun URDF robot (uses built-in rr.urdf, no pinocchio needed)
# ---------------------------------------------------------------------------

class _RerunUrdfRobot:
    """Robot loaded via rerun's built-in URDF support.

    Uses ``rr.log_file_from_path`` for static meshes and
    ``rr.urdf.UrdfTree`` + ``joint.compute_transform`` for per-frame FK.
    Only requires rerun-sdk >= 0.28.
    """

    def __init__(self, rr_module, urdf_path: Path, entity_prefix: str = "robot"):
        self.rr = rr_module
        self.prefix = entity_prefix

        rr_module.log_file_from_path(str(urdf_path), entity_path_prefix=entity_prefix, static=True)

        self.tree = rr_module.urdf.UrdfTree.from_file_path(
            str(urdf_path), entity_path_prefix=entity_prefix,
        )
        self.robot_name = self.tree.name
        self.root_entity = f"{entity_prefix}/{self.robot_name}"

        self.joint_map: dict[str, Any] = {}
        for joint in self.tree.joints():
            if joint.joint_type in ("revolute", "continuous", "prismatic"):
                self.joint_map[joint.name] = joint

    def update(self, root_pos: np.ndarray, root_rot_wxyz: np.ndarray,
               dof_names: list[str], dof_pos: np.ndarray) -> None:
        """Update robot pose for one frame."""
        rr = self.rr

        w, x, y, z = root_rot_wxyz
        rr.log(
            self.root_entity,
            rr.Transform3D(
                translation=root_pos.tolist(),
                quaternion=rr.Quaternion(xyzw=[float(x), float(y), float(z), float(w)]),
            ),
        )

        for j_idx, j_name in enumerate(dof_names):
            joint = self.joint_map.get(j_name)
            if joint is not None:
                tf = joint.compute_transform(float(dof_pos[j_idx]), clamp=False)
                rr.log("transforms", tf)


# ---------------------------------------------------------------------------
# Rerun version compat helpers
# ---------------------------------------------------------------------------

def _rr_set_time(rr, timeline: str, *, seconds: float | None = None, sequence: int | None = None) -> None:
    if hasattr(rr, "set_time"):
        if seconds is not None:
            rr.set_time(timeline, duration=seconds)
        elif sequence is not None:
            rr.set_time(timeline, sequence=sequence)
    else:
        if seconds is not None:
            rr.set_time_seconds(timeline, seconds)
        elif sequence is not None:
            rr.set_time_sequence(timeline, sequence)


def _rr_scalar(rr, entity: str, value: float) -> None:
    if hasattr(rr, "Scalars"):
        rr.log(entity, rr.Scalars(value))
    else:
        rr.log(entity, rr.Scalar(value))


# ---------------------------------------------------------------------------
# Main playback
# ---------------------------------------------------------------------------

def play_clips(
    entries: list[tuple[Path, str, dict]],
    urdf_path: Path,
    speed: float,
    max_sec: float | None,
    loop: int,
    realtime: bool,
    save_path: str | None,
) -> None:
    import rerun as rr

    app_name = "AMP Folder Viewer"
    if save_path:
        rr.init(app_name)
        rr.save(save_path)
    else:
        rr.init(app_name, spawn=True)

    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "ground",
        rr.Boxes3D(centers=[[0, 0, -0.005]], sizes=[[20, 20, 0.01]], colors=[[200, 200, 200, 100]]),
        static=True,
    )

    robot: _RerunUrdfRobot | None = None
    if urdf_path.exists():
        try:
            robot = _RerunUrdfRobot(rr, urdf_path)
            print(f"[robot] URDF loaded: {urdf_path.name} ({robot.robot_name})")
        except Exception as e:
            print(f"[warn] Failed to load URDF: {e}")
    else:
        print(f"[warn] URDF not found: {urdf_path}")

    global_time = 0.0
    global_frame = 0

    for loop_i in range(loop):
        if loop > 1:
            print(f"\n=== Loop {loop_i + 1}/{loop} ===")

        for entry_idx, (pkl_path, clip_name, clip_data) in enumerate(entries):
            fps = float(clip_data.get("fps", 50))
            dof_pos = np.asarray(clip_data["dof_pos"], dtype=np.float64)
            root_pos = np.asarray(clip_data["root_pos"], dtype=np.float64)
            root_rot = np.asarray(clip_data["root_rot"], dtype=np.float64)
            dof_names = clip_data.get("dof_names", [])
            num_frames = dof_pos.shape[0]
            dt = 1.0 / fps

            end_frame = num_frames
            if max_sec is not None:
                end_frame = min(num_frames, int(round(max_sec * fps)))

            clip_dur = end_frame / fps
            label = f"{pkl_path.stem}/{clip_name}"
            print(f"  [{entry_idx + 1}/{len(entries)}] {label}  "
                  f"{end_frame} frames  {clip_dur:.1f}s @ {fps:.0f}fps")

            proj_grav = np.asarray(clip_data["proj_grav"], dtype=np.float64) if "proj_grav" in clip_data else None
            root_ang_vel = np.asarray(clip_data["root_angle_vel"], dtype=np.float64) if "root_angle_vel" in clip_data else None

            for i in range(end_frame):
                _rr_set_time(rr, "time", seconds=global_time)
                _rr_set_time(rr, "frame", sequence=global_frame)

                rr.log("info/file", rr.TextLog(pkl_path.stem))
                rr.log("info/clip", rr.TextLog(clip_name))

                # -- 3D robot --
                if robot is not None:
                    robot.update(root_pos[i], root_rot[i], dof_names, dof_pos[i])

                rr.log("root_trajectory", rr.Points3D([root_pos[i]], radii=0.008, colors=[[0, 120, 255]]))

                # -- scalar plots --
                _rr_scalar(rr, "plots/root_pos/x", float(root_pos[i, 0]))
                _rr_scalar(rr, "plots/root_pos/y", float(root_pos[i, 1]))
                _rr_scalar(rr, "plots/root_pos/z", float(root_pos[i, 2]))

                if dof_names:
                    for j_idx, j_name in enumerate(dof_names):
                        if any(kw in j_name for kw in ("knee", "hip_pitch", "ankle_pitch", "elbow")):
                            _rr_scalar(rr, f"plots/joint_pos/{j_name}", float(dof_pos[i, j_idx]))

                if root_ang_vel is not None:
                    _rr_scalar(rr, "plots/root_ang_vel/wx", float(root_ang_vel[i, 0]))
                    _rr_scalar(rr, "plots/root_ang_vel/wy", float(root_ang_vel[i, 1]))
                    _rr_scalar(rr, "plots/root_ang_vel/wz", float(root_ang_vel[i, 2]))

                if proj_grav is not None:
                    _rr_scalar(rr, "plots/proj_grav/gx", float(proj_grav[i, 0]))
                    _rr_scalar(rr, "plots/proj_grav/gy", float(proj_grav[i, 1]))
                    _rr_scalar(rr, "plots/proj_grav/gz", float(proj_grav[i, 2]))

                if i > 0:
                    xy_speed = float(np.linalg.norm((root_pos[i, :2] - root_pos[i - 1, :2]) * fps))
                    _rr_scalar(rr, "plots/root_xy_speed", xy_speed)

                global_time += dt
                global_frame += 1

                if realtime:
                    time.sleep(dt / speed)

    print("\nDone. All clips logged to Rerun.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize all AMP .pkl clips from a folder in Rerun.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", type=str, default=str(DEFAULT_AMP_DIR),
                        help="Path to AMP data folder (default: %(default)s)")
    parser.add_argument("--list", action="store_true",
                        help="Print clip table and exit")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter clips: substring match on filename or clip name")
    parser.add_argument("--clip", type=str, default=None,
                        help="Play only a specific clip by exact name")
    parser.add_argument("--file", type=str, default=None,
                        help="Play only clips from a specific pkl file (by name or path)")
    parser.add_argument("--idx", type=int, nargs="+", default=None,
                        help="Play clips by table index (use --list to see indices)")
    parser.add_argument("--urdf", type=str, default=str(URDF_PATH),
                        help="Robot URDF path")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier")
    parser.add_argument("--max-sec", type=float, default=None,
                        help="Max seconds per clip (truncate longer clips)")
    parser.add_argument("--loop", type=int, default=1,
                        help="Number of playback loops")
    parser.add_argument("--no-realtime", action="store_true",
                        help="Log all frames instantly without sleep")
    parser.add_argument("--save", type=str, default=None,
                        help="Save to .rrd file instead of spawning viewer")
    parser.add_argument("--exclude-all", action="store_true",
                        help="Exclude *_all_clips.pkl to avoid duplicate clips")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    amp_dir = Path(args.dir)
    if not amp_dir.is_absolute():
        amp_dir = REPO_ROOT / amp_dir

    if not amp_dir.is_dir():
        print(f"Error: directory not found: {amp_dir}")
        sys.exit(1)

    # --- resolve entries: PLAYLIST has priority over CLI ---
    if PLAYLIST:
        print(f"Using PLAYLIST ({len(PLAYLIST)} entries) from script config\n")
        entries = resolve_playlist(amp_dir, PLAYLIST)
    else:
        print(f"Scanning {amp_dir} ...")
        entries = scan_amp_folder(amp_dir, exclude_all=args.exclude_all)
        if not entries:
            print("No .pkl files found.")
            sys.exit(1)

        print(f"Found {len(entries)} clips in {len(set(p for p, _, _ in entries))} files\n")

        if args.file is not None:
            file_key = args.file
            entries = [e for e in entries if file_key in e[0].name or file_key in str(e[0])]

        if args.filter is not None:
            keyword = args.filter.lower()
            entries = [e for e in entries
                       if keyword in e[0].name.lower() or keyword in e[1].lower()]

        if args.clip is not None:
            entries = [e for e in entries if e[1] == args.clip]

        if args.idx is not None:
            all_entries = scan_amp_folder(amp_dir, exclude_all=args.exclude_all)
            entries = [all_entries[i] for i in args.idx if 0 <= i < len(all_entries)]

    if not entries:
        print("No clips match the given filters.")
        sys.exit(1)

    if args.list:
        print_clip_table(entries)
        return

    print(f"Will play {len(entries)} clips (speed={args.speed}x)\n")
    play_clips(
        entries=entries,
        urdf_path=Path(args.urdf),
        speed=args.speed,
        max_sec=args.max_sec,
        loop=args.loop,
        realtime=not args.no_realtime,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
