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
import time
import numpy as np
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner, AMPPluginRunner

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
    # Resolve project root from this file's location instead of hardcoding ~/unitree_lab.
    rl_lab_root = Path(__file__).resolve().parents[2]
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
    step = int(iteration) if iteration is not None else None
    caption = f"iter_{iteration}" if iteration is not None else "sim2sim"
    wandb.log(
        {"sim2sim_video": wandb.Video(mp4_path, format="mp4", caption=caption)},
        step=step,
        commit=False,
    )


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

    if agent_cfg.class_name == "AMPPluginRunner":
        runner = AMPPluginRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
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
    # Sim2Sim integration (bfm_training style: runner-built-in, no subprocess)
    # -------------------------------------------------------------------------
    if bool(getattr(args_cli, "sim2sim", False)):
        sim2sim_task = str(getattr(args_cli, "sim2sim_task", "rough_forward"))
        _, sim2sim_xml = _find_sim2sim_resources(task=sim2sim_task)
        runner.sim2sim_cfg = {
            "task": sim2sim_task,
            "duration": float(getattr(args_cli, "sim2sim_duration", 20.0)),
            "robot": str(getattr(args_cli, "sim2sim_robot", "g1")),
            "every": max(1, int(getattr(args_cli, "sim2sim_every", 1))),
            "velocity": (1.0, 0.0, 0.0),
            "xml_path": getattr(args_cli, "sim2sim_xml", None) or sim2sim_xml,
        }

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
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
