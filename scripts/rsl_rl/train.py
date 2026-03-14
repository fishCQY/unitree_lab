# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# -- MuJoCo Sim2Sim evaluation (async, triggered on checkpoint save) --
parser.add_argument(
    "--sim2sim", action="store_true", default=False,
    help="Run MuJoCo sim2sim eval on each checkpoint save and upload video to W&B (requires --logger wandb).",
)
parser.add_argument("--sim2sim_duration", type=float, default=30.0, help="Sim2sim episode duration in seconds.")
parser.add_argument("--sim2sim_robot", type=str, default="g1", help="MuJoCo robot name for sim2sim.")
parser.add_argument(
    "--sim2sim_xml", type=str, default=None,
    help="Path to MuJoCo XML for sim2sim (default: auto-detect from unitree_lab assets).",
)
parser.add_argument("--sim2sim_every", type=int, default=1, help="Run sim2sim every N checkpoint saves.")
parser.add_argument(
    "--sim2sim_task", type=str, default="rough_forward",
    help="Eval task for sim2sim (e.g. flat_forward, rough_forward). Default: rough_forward.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Use the local custom rsl_rl library (workspace root)."""

import sys
from pathlib import Path

_workspace_root = str(Path(__file__).resolve().parents[2])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

try:
    import rsl_rl  # noqa: F401
except ImportError:
    print(
        "[ERROR] Could not import the local rsl_rl package.\n"
        "Make sure the rsl_rl/ directory exists at the workspace root, or install it via:\n"
        f"  pip install -e {_workspace_root}/rsl_rl\n"
    )
    exit(1)

import gymnasium as gym
import json
import logging
import os
import queue
import subprocess
import threading
import time
import numpy as np
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner, AMPRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import unitree_lab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# ONNX export helper (lightweight, no Isaac Sim deps)
# ---------------------------------------------------------------------------
def _export_policy_to_onnx(runner: OnPolicyRunner, out_dir: str, filename: str = "policy.onnx") -> str:
    """Export the actor network to ONNX format (best-effort)."""
    import onnx

    policy = runner.alg.policy
    policy.eval()

    if not hasattr(policy, "actor"):
        raise ValueError("Policy does not have 'actor' attribute")

    obs_dim = policy.actor[0].in_features
    example_input = torch.zeros(1, obs_dim, device=runner.device)
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, filename)

    class _PolicyWrapper(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.actor = p.actor
            self.actor_obs_normalizer = p.actor_obs_normalizer
            self.state_dependent_std = getattr(p, "state_dependent_std", False)

        def forward(self, obs_: torch.Tensor) -> torch.Tensor:
            obs_ = self.actor_obs_normalizer(obs_)
            if self.state_dependent_std:
                return self.actor(obs_)[..., 0, :]
            return self.actor(obs_)

    wrapped = _PolicyWrapper(policy)
    wrapped.eval()
    torch.onnx.export(
        wrapped, example_input, onnx_path,
        input_names=["obs"], output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=11,
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    return onnx_path


def _build_deploy_yaml_from_metadata(meta: dict) -> dict:
    """Convert unitree_lab ONNX metadata dict -> mjlab-style deploy.yaml schema.

    This file is consumed by scripts/mujoco_eval/run_sim2sim_locomotion.py via --deploy-yaml.
    It is especially useful when ONNX metadata_json could not be attached (e.g., missing `onnx` pkg).
    """
    deploy: dict = {}

    # Core joint/action parameters
    if isinstance(meta.get("joint_names"), list):
        deploy["joint_names"] = meta["joint_names"]
    if isinstance(meta.get("default_joint_pos"), list):
        deploy["default_joint_pos"] = meta["default_joint_pos"]
    if isinstance(meta.get("joint_stiffness"), list):
        deploy["stiffness"] = meta["joint_stiffness"]
    if isinstance(meta.get("joint_damping"), list):
        deploy["damping"] = meta["joint_damping"]
    if isinstance(meta.get("joint_armature"), list):
        deploy["armature"] = meta["joint_armature"]
    if meta.get("policy_dt") is not None:
        try:
            deploy["step_dt"] = float(meta["policy_dt"])
        except Exception:
            pass

    # Actions: match the parser in run_sim2sim_locomotion.py
    actions: dict = {}
    jpa: dict = {}
    if isinstance(meta.get("action_scale"), list):
        jpa["scale"] = meta["action_scale"]
    if isinstance(meta.get("action_offset"), list):
        jpa["offset"] = meta["action_offset"]
    if jpa:
        actions["JointPositionAction"] = jpa
    if actions:
        deploy["actions"] = actions

    # Observations scales (optional)
    obs_scales = meta.get("observation_scales")
    if isinstance(obs_scales, dict) and obs_scales:
        deploy_obs: dict = {}
        for k, v in obs_scales.items():
            deploy_obs[str(k)] = {"scale": v}
        deploy["observations"] = deploy_obs

    # Observation structure (recommended for sim2sim alignment)
    # These keys are consumed by scripts/mujoco_eval/run_sim2sim_locomotion.py
    if isinstance(meta.get("observation_names"), list):
        deploy["observation_names"] = meta["observation_names"]
    if isinstance(meta.get("observation_dims"), list):
        deploy["observation_dims"] = meta["observation_dims"]
    if meta.get("history_length") is not None:
        try:
            deploy["history_length"] = int(meta["history_length"])
        except Exception:
            pass
    if isinstance(meta.get("single_frame_dims"), dict):
        deploy["single_frame_dims"] = meta["single_frame_dims"]

    return deploy


def _find_sim2sim_resources(task: str = "flat_forward"):
    """Locate the sim2sim script and appropriate G1 XML from unitree_lab.

    For terrain tasks (rough_forward, stairs_*, slope_*, mixed_terrain) the
    terrain XML is preferred so that ``run_sim2sim_locomotion.py`` can inject
    course/heightfield geometry.  For flat tasks the plain scene XML is used.
    """
    rl_lab_root = Path.home() / "unitree_lab"
    script = rl_lab_root / "scripts" / "mujoco_eval" / "run_sim2sim_locomotion.py"

    xml_dir = rl_lab_root / "source" / "unitree_lab" / "unitree_lab" / "assets" / "robots_xml" / "g1"
    flat_xml = xml_dir / "scene_29dof.xml"
    terrain_xml = xml_dir / "scene_29dof_terrain.xml"

    _TERRAIN_TASKS = {"rough_forward", "stairs_up", "stairs_down", "slope_up", "mixed_terrain"}
    if task in _TERRAIN_TASKS and terrain_xml.exists():
        xml = terrain_xml
    else:
        xml = flat_xml

    return script if script.exists() else None, xml if xml.exists() else None


def _log_sim2sim_video_to_wandb(mp4_path: str, iteration: int) -> None:
    """Upload a sim2sim video to the active W&B run."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    if not os.path.isfile(mp4_path):
        return
    cur_step = max(int(getattr(wandb.run, "step", 0) or 0), iteration)
    wandb.log({"sim2sim_video": wandb.Video(mp4_path, format="mp4")}, step=cur_step, commit=True)
    wandb.save(mp4_path, base_path=os.path.dirname(mp4_path), policy="now")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    agent_dict = agent_cfg.to_dict()

    if agent_cfg.class_name == "AMPRunner":
        runner = AMPRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # -------------------------------------------------------------------------
    # Sim2Sim integration: on each checkpoint save, export ONNX -> run MuJoCo
    # eval in a subprocess -> upload the video to W&B.
    # -------------------------------------------------------------------------
    sim2sim_enabled = bool(getattr(args_cli, "sim2sim", False))
    sim2sim_task = str(getattr(args_cli, "sim2sim_task", "rough_forward"))
    sim2sim_script, sim2sim_default_xml = _find_sim2sim_resources(task=sim2sim_task)
    sim2sim_jobs: queue.Queue = queue.Queue()
    sim2sim_sema = threading.BoundedSemaphore(value=1)

    if sim2sim_enabled and sim2sim_script is None:
        print("[Sim2Sim][WARN] Could not find run_sim2sim_locomotion.py in ~/unitree_lab. Sim2sim disabled.")
        sim2sim_enabled = False

    def _sim2sim_poller():
        """Background thread: wait for subprocess to finish, then upload to W&B."""
        while True:
            job = sim2sim_jobs.get()
            if job is None:
                return
            proc = job.get("proc")
            if proc is None:
                continue
            proc.wait()
            it = int(job.get("it", 0))
            rc = proc.returncode
            if rc != 0:
                log_path = job.get("out_dir", "")
                print(f"[Sim2Sim][WARN] subprocess exited with code {rc} (it={it}). Check {log_path}/sim2sim.log")
            mp4_path = job.get("mp4_path")
            if mp4_path and os.path.isfile(mp4_path):
                _log_sim2sim_video_to_wandb(mp4_path, it)
                print(f"[Sim2Sim] Video uploaded to W&B (it={it}): {mp4_path}")
            elif mp4_path:
                print(f"[Sim2Sim][WARN] Video not found (it={it}): {mp4_path}")
            try:
                lf = job.get("log_f")
                if lf:
                    lf.close()
            except Exception:
                pass
            try:
                if job.get("sema_acquired"):
                    sim2sim_sema.release()
            except Exception:
                pass

    sim2sim_thread = None
    if sim2sim_enabled:
        sim2sim_thread = threading.Thread(target=_sim2sim_poller, daemon=True)
        sim2sim_thread.start()

    _orig_save = runner.save
    _save_count = {"n": 0}

    def _save_with_sim2sim(path: str, infos=None) -> None:
        _orig_save(path, infos=infos)
        if not sim2sim_enabled:
            return
        _save_count["n"] += 1
        every = max(1, int(getattr(args_cli, "sim2sim_every", 1)))
        if (_save_count["n"] % every) != 0:
            return

        sema_acquired = sim2sim_sema.acquire(blocking=False)
        if not sema_acquired:
            print("[Sim2Sim] skipped: previous sim2sim still running.")
            return

        it = int(getattr(runner, "current_learning_iteration", 0))
        try:
            export_dir = os.path.join(log_dir, "export")
            onnx_path = _export_policy_to_onnx(runner, export_dir, filename=f"policy_iter_{it}.onnx")
            # Attach IsaacLab metadata to ONNX so MuJoCo sim2sim can match:
            # - joint order (joint_names)
            # - action scale/offset
            # - observation layout + history stacking
            # - PD gains and timing (policy_dt/decimation/sim_dt)
            try:
                from unitree_lab.utils.onnx_utils import build_onnx_metadata

                # build_onnx_metadata expects the underlying IsaacLab env (ManagerBasedRLEnv),
                # not the RSL-RL wrapper.
                base_env = getattr(env, "unwrapped", env)
                meta = build_onnx_metadata(base_env)

                # Always dump a deploy.yaml next to the ONNX for robust sim2sim debugging.
                # This does NOT require attaching metadata to ONNX.
                deploy_yaml = _build_deploy_yaml_from_metadata(meta)
                deploy_yaml_path = os.path.join(export_dir, f"deploy_iter_{it}.yaml")
                dump_yaml(deploy_yaml_path, deploy_yaml)
                # Also keep a stable "latest" pointer for convenience.
                dump_yaml(os.path.join(export_dir, "deploy_latest.yaml"), deploy_yaml)

                # Best-effort: attach metadata_json to the ONNX (requires the `onnx` package).
                try:
                    from unitree_lab.utils.onnx_utils import attach_onnx_metadata
                    attach_onnx_metadata(onnx_path, meta)
                except Exception as e:
                    print(f"[Sim2Sim][WARN] Failed to attach ONNX metadata_json (deploy.yaml still written): {e}")
            except Exception as e:
                print(f"[Sim2Sim][WARN] Failed to build sim2sim deploy metadata: {e}")
        except Exception as e:
            print(f"[Sim2Sim][WARN] ONNX export failed: {e}")
            sim2sim_sema.release()
            return

        out_dir = os.path.join(log_dir, "sim2sim", f"iter_{it}")
        os.makedirs(out_dir, exist_ok=True)

        xml_path = getattr(args_cli, "sim2sim_xml", None) or (str(sim2sim_default_xml) if sim2sim_default_xml else None)
        if xml_path is None:
            print("[Sim2Sim][WARN] No MuJoCo XML found. Skipping.")
            sim2sim_sema.release()
            return

        duration = float(getattr(args_cli, "sim2sim_duration", 30.0))
        max_steps = int(duration / 0.02)  # ~50Hz policy rate -> 250 steps for 5s
        mp4_path = os.path.join(out_dir, f"{sim2sim_task}_sim2sim.mp4")
        sim2sim_log = os.path.join(out_dir, "sim2sim.log")

        deploy_yaml_path = os.path.join(export_dir, "deploy_latest.yaml")
        cmd = [
            sys.executable, str(sim2sim_script),
            "--robot", str(getattr(args_cli, "sim2sim_robot", "g1")),
            "--onnx", str(onnx_path),
            "--xml", str(xml_path),
            "--task", sim2sim_task,
            "--num-episodes", "1",
            "--max-steps", str(max_steps),
            "--output-dir", str(out_dir),
            "--save-video",
            "--video-steps", str(max_steps),
            "--velocity", "1.0", "0.0", "0.0",
        ]
        if os.path.isfile(deploy_yaml_path):
            cmd += ["--deploy-yaml", deploy_yaml_path]

        try:
            log_f = open(sim2sim_log, "w")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            sim2sim_jobs.put({
                "it": it, "proc": proc, "mp4_path": mp4_path,
                "log_f": log_f, "sema_acquired": True, "out_dir": out_dir,
            })
            print(f"[Sim2Sim] spawned (it={it}, duration={duration}s) -> {out_dir}")
        except Exception as e:
            print(f"[Sim2Sim][WARN] Failed to spawn: {e}")
            sim2sim_sema.release()

    runner.save = _save_with_sim2sim

    try:
        # run training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        if sim2sim_thread is not None:
            # Signal poller to exit, then wait for any in-flight sim2sim job to finish
            # so the video is uploaded to W&B before we tear down the run.
            sim2sim_jobs.put(None)
            sim2sim_thread.join(timeout=120)
            if sim2sim_thread.is_alive():
                print("[Sim2Sim][WARN] Poller thread still alive after 120s timeout.")
        # finalize W&B if active
        try:
            writer = getattr(getattr(runner, "logger", None), "writer", None)
            if writer is not None and hasattr(writer, "stop"):
                writer.stop()
        except Exception:
            pass

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
