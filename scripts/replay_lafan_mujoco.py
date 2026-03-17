#!/usr/bin/env python3
"""Replay merged LAFAN motions in MuJoCo by id or time range."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_PKL = "source/unitree_lab/unitree_lab/data/MotionData/g1_29dof/amp/lafan/lafan_all_50fps.pkl"
DEFAULT_XML = "source/unitree_lab/unitree_lab/assets/robots_xml/g1/scene_29dof.xml"


def load_pickle_compat(path: Path) -> dict:
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
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


def clamp_range(start: int, end: int, total: int) -> tuple[int, int]:
    start = max(0, min(start, total - 1))
    end = max(start, min(end, total - 1))
    return start, end


def choose_range(data: dict, args: argparse.Namespace) -> tuple[tuple[int, int], dict | None]:
    fps = float(data["fps"])
    total_frames = int(np.asarray(data["dof_pos"]).shape[0])
    segments = data.get("segments", [])
    seg = None

    if args.id is not None:
        if not segments:
            raise ValueError("This PKL has no 'segments' field; cannot select by --id.")
        seg_map = {int(s["id"]): s for s in segments}
        if args.id not in seg_map:
            raise ValueError(f"Invalid --id {args.id}. Use --list-ids first.")
        seg = seg_map[args.id]
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
    else:
        start = 0
        end = total_frames - 1

    if args.start_frame is not None:
        start = int(args.start_frame)
    if args.end_frame is not None:
        end = int(args.end_frame)
    if args.start_sec is not None:
        start = int(round(float(args.start_sec) * fps))
    if args.end_sec is not None:
        end = int(round(float(args.end_sec) * fps))
    if args.rel_start_sec is not None:
        start = start + int(round(float(args.rel_start_sec) * fps))
    if args.rel_end_sec is not None:
        end = start + int(round(float(args.rel_end_sec) * fps))

    return clamp_range(start, end, total_frames), seg


def replay_in_mujoco(
    pkl_data: dict,
    xml_path: Path,
    start_frame: int,
    end_frame: int,
    mode: str = "kinematic",
    loop: int = 1,
    speed: float = 1.0,
    kp: float = 80.0,
    kd: float = 2.0,
    track_root: bool = False,
    root_z_offset: float = 0.0,
    auto_ground: bool = True,
    ground_clearance: float = 0.005,
) -> None:
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    dof_pos_all = np.asarray(pkl_data["dof_pos"], dtype=np.float64)
    root_pos_all = np.asarray(pkl_data["root_pos"], dtype=np.float64)
    root_rot_all = np.asarray(pkl_data["root_rot"], dtype=np.float64)  # wxyz
    dof_names = [str(n) for n in pkl_data["dof_names"]]
    fps = float(pkl_data["fps"])
    dt_frame = 1.0 / max(1e-6, fps)
    speed = max(1e-4, float(speed))
    sleep_dt = dt_frame / speed

    base_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if base_jid < 0:
        raise RuntimeError("Cannot find floating_base_joint in MuJoCo model.")
    base_qadr = int(model.jnt_qposadr[base_jid])

    joint_qadr: list[int] = []
    joint_dadr: list[int] = []
    joint_ids: list[int] = []
    for jn in dof_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"Joint '{jn}' not found in MuJoCo model.")
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise RuntimeError(f"Joint '{jn}' is not hinge; unsupported.")
        joint_ids.append(jid)
        joint_qadr.append(int(model.jnt_qposadr[jid]))
        joint_dadr.append(int(model.jnt_dofadr[jid]))

    # Map each hinge joint to its actuator index (torque motor control).
    joint_to_act = {jid: -1 for jid in joint_ids}
    for act_id in range(model.nu):
        trn_joint_id = int(model.actuator_trnid[act_id, 0])
        if trn_joint_id in joint_to_act and joint_to_act[trn_joint_id] < 0:
            joint_to_act[trn_joint_id] = act_id
    act_ids = [joint_to_act[jid] for jid in joint_ids]
    if any(a < 0 for a in act_ids):
        missing = [dof_names[i] for i, a in enumerate(act_ids) if a < 0]
        raise RuntimeError(f"No actuator found for joints: {missing}")

    dof_vel_all = np.asarray(pkl_data.get("dof_vel", np.zeros_like(dof_pos_all)), dtype=np.float64)
    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64) if model.nu > 0 else np.zeros((0, 2))
    sim_dt = float(model.opt.timestep)
    substeps = max(1, int(round(dt_frame / max(1e-6, sim_dt))))
    print(
        f"[Replay] mode={mode}  gravity={model.opt.gravity}  sim_dt={sim_dt:.5f}s  "
        f"data_dt={dt_frame:.5f}s  substeps/frame={substeps}"
    )

    # Estimate a constant root-z offset so the first frame is close to the ground.
    # This matches kinematic replay use-cases where dataset/world height origins differ.
    auto_root_z_offset = 0.0
    if mode == "kinematic" and auto_ground:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        init_frame = start_frame
        root_pos_init = root_pos_all[init_frame].copy()
        root_rot_init = root_rot_all[init_frame]
        data.qpos[base_qadr + 0 : base_qadr + 3] = root_pos_init
        data.qpos[base_qadr + 3 : base_qadr + 7] = np.array(
            [root_rot_init[1], root_rot_init[2], root_rot_init[3], root_rot_init[0]], dtype=np.float64
        )
        q_init = dof_pos_all[init_frame]
        for k, qadr in enumerate(joint_qadr):
            data.qpos[qadr] = q_init[k]
        mujoco.mj_forward(model, data)

        # Conservative lower bound using bounding radius on *collision* geoms only.
        # Exclude plane and mesh geoms: mesh rbound is often too loose and can cause
        # large false offsets (robot appears to float).
        geom_type = model.geom_type
        is_plane = geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE)
        is_mesh = geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)
        can_collide = np.logical_and(model.geom_contype > 0, model.geom_conaffinity > 0)
        valid = np.logical_and(can_collide, np.logical_not(np.logical_or(is_plane, is_mesh)))

        z_low = data.geom_xpos[:, 2] - model.geom_rbound
        z_low = z_low[valid]
        if z_low.size > 0:
            min_z = float(np.min(z_low))
            auto_root_z_offset = float(ground_clearance) - min_z
            print(
                f"[Replay] auto_ground enabled: min_geom_z={min_z:.4f}, "
                f"computed_root_z_offset={auto_root_z_offset:+.4f}"
            )
        else:
            print("[Replay] auto_ground skipped: no valid collision geoms found.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initialize from the first target frame once, then run real dynamics.
        init_frame = start_frame
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        root_pos_init = root_pos_all[init_frame]
        root_rot_init = root_rot_all[init_frame]
        data.qpos[base_qadr + 0 : base_qadr + 3] = root_pos_init
        data.qpos[base_qadr + 3 : base_qadr + 7] = np.array(
            [root_rot_init[1], root_rot_init[2], root_rot_init[3], root_rot_init[0]], dtype=np.float64
        )
        q_init = dof_pos_all[init_frame]
        for k, qadr in enumerate(joint_qadr):
            data.qpos[qadr] = q_init[k]
        mujoco.mj_forward(model, data)

        for lp in range(loop):
            print(f"[Replay] loop {lp + 1}/{loop}")
            for frame in range(start_frame, end_frame + 1):
                if not viewer.is_running():
                    return

                tic = time.monotonic()

                root_pos = root_pos_all[frame].copy()
                root_pos[2] += float(root_z_offset) + float(auto_root_z_offset)
                root_rot = root_rot_all[frame]  # wxyz
                q_target = dof_pos_all[frame]

                if mode == "kinematic":
                    # Match rerun_visualize.py behavior: direct per-frame pose update.
                    data.qpos[base_qadr + 0 : base_qadr + 3] = root_pos
                    data.qpos[base_qadr + 3 : base_qadr + 7] = np.array(
                        [root_rot[1], root_rot[2], root_rot[3], root_rot[0]], dtype=np.float64
                    )  # xyzw
                    for k, qadr in enumerate(joint_qadr):
                        data.qpos[qadr] = q_target[k]
                    data.qvel[:] = 0.0
                    if model.nu > 0:
                        data.ctrl[:] = 0.0
                    mujoco.mj_forward(model, data)
                else:
                    if track_root:
                        data.qpos[base_qadr + 0 : base_qadr + 3] = root_pos
                        data.qpos[base_qadr + 3 : base_qadr + 7] = np.array(
                            [root_rot[1], root_rot[2], root_rot[3], root_rot[0]], dtype=np.float64
                        )  # xyzw
                        data.qvel[0:6] = 0.0

                    qd_target = dof_vel_all[frame]
                    q_cur = np.array([data.qpos[qadr] for qadr in joint_qadr], dtype=np.float64)
                    qd_cur = np.array([data.qvel[dadr] for dadr in joint_dadr], dtype=np.float64)
                    tau = kp * (q_target - q_cur) + kd * (qd_target - qd_cur)

                    for k, act_id in enumerate(act_ids):
                        lo, hi = ctrlrange[act_id]
                        data.ctrl[act_id] = np.clip(tau[k], lo, hi)

                    for _ in range(substeps):
                        mujoco.mj_step(model, data)

                viewer.sync()
                elapsed = time.monotonic() - tic
                remain = sleep_dt - elapsed
                if remain > 0:
                    time.sleep(remain)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay merged LAFAN PKL in MuJoCo.")
    parser.add_argument("--file", type=str, default=DEFAULT_PKL, help="Path to merged LAFAN .pkl")
    parser.add_argument("--xml", type=str, default=DEFAULT_XML, help="MuJoCo scene XML path")
    parser.add_argument("--list-ids", action="store_true", help="Print id/action/time table and exit")
    parser.add_argument("--id", type=int, default=None, help="Select one segment id")
    parser.add_argument("--start-frame", type=int, default=None, help="Absolute start frame")
    parser.add_argument("--end-frame", type=int, default=None, help="Absolute end frame")
    parser.add_argument("--start-sec", type=float, default=None, help="Absolute start second")
    parser.add_argument("--end-sec", type=float, default=None, help="Absolute end second")
    parser.add_argument("--rel-start-sec", type=float, default=None, help="Relative start from current base range")
    parser.add_argument("--rel-end-sec", type=float, default=None, help="Relative end from computed start")
    parser.add_argument("--loop", type=int, default=1, help="Replay loop count")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = real-time)")
    parser.add_argument(
        "--mode",
        type=str,
        default="kinematic",
        choices=["kinematic", "dynamic"],
        help="Replay mode: kinematic (like rerun_visualize) or dynamic (physics + PD)",
    )
    parser.add_argument("--kp", type=float, default=80.0, help="Joint PD Kp for motor torque control")
    parser.add_argument("--kd", type=float, default=2.0, help="Joint PD Kd for motor torque control")
    parser.add_argument(
        "--track-root",
        action="store_true",
        help="Kinematically track root pose from dataset each frame (default: physics free-base)",
    )
    parser.add_argument("--root-z-offset", type=float, default=0.0, help="Add z offset to dataset root position (m)")
    parser.add_argument(
        "--auto-ground",
        action="store_true",
        default=True,
        help="Auto compute root z offset to place first frame near ground in kinematic mode",
    )
    parser.add_argument(
        "--no-auto-ground",
        action="store_false",
        dest="auto_ground",
        help="Disable automatic ground alignment",
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=0.005,
        help="Target minimal geom-ground clearance for auto-ground (meters)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_pickle_compat(Path(args.file))
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
            f"selected_id={seg['id']} action={seg.get('action', '')} clip={seg.get('clip_name', '')} "
            f"segment_time=[{float(seg['start_time_sec']):.3f}, {float(seg['end_time_sec']):.3f}]"
        )
    print(f"replay_frames=[{start}, {end}]  duration={(end - start + 1) / fps:.3f}s")
    replay_in_mujoco(
        pkl_data=data,
        xml_path=Path(args.xml),
        start_frame=start,
        end_frame=end,
        mode=str(args.mode),
        loop=max(1, int(args.loop)),
        speed=float(args.speed),
        kp=float(args.kp),
        kd=float(args.kd),
        track_root=bool(args.track_root),
        root_z_offset=float(args.root_z_offset),
        auto_ground=bool(args.auto_ground),
        ground_clearance=float(args.ground_clearance),
    )


if __name__ == "__main__":
    main()

