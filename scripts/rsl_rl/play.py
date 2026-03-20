# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

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
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# -- Export extras for MuJoCo sim2sim --
parser.add_argument(
    "--export-deploy-yaml",
    action="store_true",
    default=True,
    help=(
        "When exporting ONNX, also attach unitree_lab metadata_json and dump a mjlab-style deploy.yaml "
        "(joint_names, stiffness/damping, default pose, action scale/offset, obs scales)."
    ),
)
parser.add_argument(
    "--export-dir",
    type=str,
    default=None,
    help="Optional export directory (default: <checkpoint_dir>/exported).",
)
parser.add_argument(
    "--export-only",
    action="store_true",
    default=False,
    help="Export artifacts (ONNX + deploy.yaml) and exit without running the play loop.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
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
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner, AMPPluginRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent_dict = agent_cfg.to_dict()

    if agent_cfg.class_name == "AMPPluginRunner":
        runner = AMPPluginRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = args_cli.export_dir or os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # Attach IsaacLab metadata to ONNX and dump deploy.yaml for MuJoCo sim2sim.
    if bool(getattr(args_cli, "export_deploy_yaml", True)):
        try:
            from unitree_lab.utils.onnx_utils import build_onnx_metadata, attach_onnx_metadata

            onnx_path = os.path.join(export_model_dir, "policy.onnx")
            # build_onnx_metadata expects the underlying IsaacLab env (ManagerBasedRLEnv),
            # not the RSL-RL wrapper.
            base_env = getattr(env, "unwrapped", env)
            meta = build_onnx_metadata(base_env)
            attach_onnx_metadata(onnx_path, meta)

            # Convert metadata -> mjlab deploy.yaml schema (consumed by run_sim2sim_locomotion.py --deploy-yaml).
            # Keep this logic in-sync with scripts/rsl_rl/train.py.
            deploy: dict = {}
            if isinstance(meta.get("joint_names"), list):
                deploy["joint_names"] = meta["joint_names"]
            if isinstance(meta.get("default_joint_pos"), list):
                deploy["default_joint_pos"] = meta["default_joint_pos"]
            if isinstance(meta.get("joint_stiffness"), list):
                deploy["stiffness"] = meta["joint_stiffness"]
            if isinstance(meta.get("joint_damping"), list):
                deploy["damping"] = meta["joint_damping"]
            if meta.get("policy_dt") is not None:
                try:
                    deploy["step_dt"] = float(meta["policy_dt"])
                except Exception:
                    pass
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
            obs_scales = meta.get("observation_scales")
            if isinstance(obs_scales, dict) and obs_scales:
                deploy_obs: dict = {}
                for k, v in obs_scales.items():
                    deploy_obs[str(k)] = {"scale": v}
                deploy["observations"] = deploy_obs

            deploy_yaml_path = os.path.join(export_model_dir, "deploy.yaml")
            dump_yaml(deploy_yaml_path, deploy)
            print(f"[Export] Wrote deploy.yaml: {deploy_yaml_path}")
        except Exception as e:
            print(f"[Export][WARN] Failed to attach metadata / write deploy.yaml: {e}")

    if bool(getattr(args_cli, "export_only", False)):
        env.close()
        return

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
