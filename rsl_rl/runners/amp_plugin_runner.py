"""AMP Plugin Runner.

Integrates the standalone AMPPlugin with the OnPolicyRunner.
The AMP discriminator is managed as a plugin — PPO stays unmodified.

Data flow:
    1. Environment outputs single-step AMP observations (obs["amp"])
    2. During rollout, AMPPlugin.reward() builds multi-frame sequences
       from the RolloutStorage and computes style reward
    3. Style reward is combined additively with task reward
    4. After PPO update, AMPPlugin.update() trains the discriminator
       using sequences from the rollout and the offline dataset

Sim2Sim (bfm_training style):
    On each checkpoint save, the runner directly calls MuJoCo sim2sim
    evaluation and uploads video + metrics to W&B. No subprocess needed.

Usage:
    In the runner config, set:
        class_name = "AMPPluginRunner"
        amp_cfg = { ... }   # AMPPlugin configuration
"""

from __future__ import annotations

import logging
import os
import statistics
import time
import torch
from collections import deque
from pathlib import Path

from rsl_rl.env import VecEnv
from rsl_rl.plugins import AMPPlugin
from rsl_rl.runners import OnPolicyRunner

logger = logging.getLogger(__name__)


class AMPPluginRunner(OnPolicyRunner):
    """On-policy runner with AMPPlugin for style-based reward shaping.

    Extends OnPolicyRunner by adding:
    - AMPPlugin construction from config + offline data
    - Style reward computation during rollout
    - Discriminator training after each PPO update
    - AMP-specific logging, save/load
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Extract AMP config before super().__init__ calls _construct_algorithm
        self._amp_cfg = train_cfg.get("amp_cfg")
        super().__init__(env, train_cfg, log_dir, device)

        # Construct AMP plugin
        self.amp: AMPPlugin | None = None
        if self._amp_cfg is not None:
            obs = self.env.get_observations().to(self.device)
            amp_obs_group = self._amp_cfg["obs_group"]

            if amp_obs_group not in obs:
                raise ValueError(
                    f"AMP obs_group '{amp_obs_group}' not found in environment observations. "
                    f"Available keys: {list(obs.keys())}"
                )

            obs_dim = obs[amp_obs_group].shape[-1]
            self.amp = AMPPlugin(
                cfg=self._amp_cfg,
                obs_dim=obs_dim,
                num_envs=self.env.num_envs,
                device=self.device,
                multi_gpu_cfg=self.multi_gpu_cfg,
            )

            # Load offline motion data
            self._load_amp_offline_data()

        # AMP logging buffers
        self._style_rewbuffer = deque(maxlen=100)
        self._total_rewbuffer = deque(maxlen=100)
        self._cur_style_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self._cur_total_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self._step_dt = self.env.unwrapped.step_dt

        # Sim2sim config (set by train.py before learn())
        self.sim2sim_cfg: dict | None = None
        self._sim2sim_save_count = 0

    def _load_amp_offline_data(self) -> None:
        """Load offline motion data for the AMP discriminator."""
        if self.amp is None:
            return

        # Try environment's load_amp_data method first
        if hasattr(self.env.cfg, "load_amp_data") and callable(self.env.cfg.load_amp_data):
            offline_data = self.env.cfg.load_amp_data()
            self.amp.set_offline_data(offline_data)
            return

        # Try config-specified data loader function
        data_loader_func = self._amp_cfg.get("data_loader_func")
        if data_loader_func is not None:
            if callable(data_loader_func):
                offline_data = data_loader_func()
            else:
                raise ValueError(f"data_loader_func must be callable, got {type(data_loader_func)}")
            self.amp.set_offline_data(offline_data)
            return

        # Try config-specified motion files
        motion_files = self._amp_cfg.get("motion_files")
        if motion_files is not None:
            from unitree_lab.utils.amp_data_loader import load_amp_motion_data
            keys = self._amp_cfg.get("motion_keys", ["dof_pos", "dof_vel", "root_angle_vel", "proj_grav"])
            offline_data = load_amp_motion_data(
                motion_files=motion_files,
                keys=keys,
                device=self.device,
                mirror=self._amp_cfg.get("mirror", False),
                joint_mirror_indices=self._amp_cfg.get("joint_mirror_indices"),
                joint_mirror_signs=self._amp_cfg.get("joint_mirror_signs"),
                point_indices=self._amp_cfg.get("point_indices"),
            )
            self.amp.set_offline_data(offline_data)
            return

        print("[AMPPluginRunner] Warning: No offline data configured for AMP. "
              "Set 'motion_files', 'data_loader_func', or implement env.cfg.load_amp_data().")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint(
                low=1,
                high=int(self.env.max_episode_length),
                size=self.env.episode_length_buf.shape,
                dtype=self.env.episode_length_buf.dtype,
                device=self.env.episode_length_buf.device,
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            if self.amp is not None:
                self.amp.broadcast_parameters()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()

            # ---- Rollout ----
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # AMP reward (reward() uses no_grad internally)
                    if self.amp is not None:
                        style_rewards, disc_score = self.amp.reward(obs, self.alg.storage, self._step_dt)
                        total_rewards = self.amp.combine_reward(task_reward=rewards, style_reward=style_rewards)
                    else:
                        style_rewards = None
                        total_rewards = rewards

                    self.alg.process_env_step(obs, total_rewards, dones, extras)

                    # Logging
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg.get("rnd_cfg") else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                    self._log_amp_step(style_rewards, total_rewards, dones, extras)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            # ---- PPO Update ----
            # AMP update BEFORE ppo update since PPO clears storage.step
            # but storage data persists — we just need num_transitions_per_env
            amp_loss_dict = {}
            if self.amp is not None:
                amp_loss_dict = self.amp.update(self.alg.storage, self.alg.learning_rate)

            loss_dict = self.alg.update()
            loss_dict.update(amp_loss_dict)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # ---- Logging ----
            self._log_iteration(it, start_it, total_it, collect_time, learn_time, loss_dict)

            # ---- Save + Sim2Sim ----
            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
                self._maybe_run_sim2sim(it)

        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _log_amp_step(self, style_rewards, total_rewards, dones, extras):
        """Track AMP-specific reward statistics."""
        if self.logger.log_dir is None or style_rewards is None:
            return
        self._cur_style_reward_sum += style_rewards
        self._cur_total_reward_sum += total_rewards
        new_ids = (dones > 0).nonzero(as_tuple=False)
        if len(new_ids) > 0:
            self._style_rewbuffer.extend(self._cur_style_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self._total_rewbuffer.extend(self._cur_total_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self._cur_style_reward_sum[new_ids] = 0
            self._cur_total_reward_sum[new_ids] = 0

    def _log_iteration(self, it, start_it, total_it, collect_time, learn_time, loss_dict):
        """Log training metrics including AMP-specific ones."""
        # Add AMP reward stats to extras for the logger
        if self.logger.log_dir is not None and len(self._style_rewbuffer) > 0:
            self.logger.writer.add_scalar("AMP/mean_style_reward", statistics.mean(self._style_rewbuffer), it)
        if self.logger.log_dir is not None and len(self._total_rewbuffer) > 0:
            self.logger.writer.add_scalar("AMP/mean_total_reward", statistics.mean(self._total_rewbuffer), it)

        self.logger.log(
            it=it,
            start_it=start_it,
            total_it=total_it,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=self.alg.learning_rate,
            action_std=self.alg.policy.action_std,
            rnd_weight=self.alg.rnd.weight if self.alg_cfg.get("rnd_cfg") else None,
        )

    # ------------------------------------------------------------------
    # Sim2Sim (bfm_training style: direct call, no subprocess)
    # ------------------------------------------------------------------

    def _maybe_run_sim2sim(self, iteration: int) -> None:
        """Run sim2sim evaluation if configured. Called after checkpoint save."""
        if self.sim2sim_cfg is None:
            return
        self._sim2sim_save_count += 1
        every = max(1, self.sim2sim_cfg.get("every", 1))
        if (self._sim2sim_save_count % every) != 0:
            return
        try:
            self._run_sim2sim(iteration)
        except Exception as e:
            logger.warning(f"[Sim2Sim] Failed at iter {iteration}: {e}")

    def _run_sim2sim(self, iteration: int) -> None:
        """Two-phase sim2sim evaluation (bfm_training style).

        Phase 1: Run all eval tasks headless (no rendering) to collect metrics.
        Phase 2: Record videos for mixed_terrain + worst-performing tasks.
        Upload structured metrics + videos to W&B.
        """
        cfg = self.sim2sim_cfg
        log_dir = self.logger.log_dir

        # 1. Export ONNX
        export_dir = os.path.join(log_dir, "export")
        os.makedirs(export_dir, exist_ok=True)
        onnx_path = self._export_onnx(export_dir, iteration)
        if onnx_path is None:
            return

        # 2. Build deploy.yaml
        deploy_yaml_path = self._build_deploy_yaml(export_dir, iteration)

        # 3. Setup imports and environment
        os.environ["MUJOCO_GL"] = "egl"

        duration = cfg.get("duration", 20.0)
        num_worst_videos = cfg.get("num_worst_videos", 2)
        xml_path_cfg = cfg.get("xml_path")
        eval_tasks = cfg.get("eval_tasks", ["rough_forward"])

        out_dir = os.path.join(log_dir, "sim2sim", f"iter_{iteration}")
        os.makedirs(out_dir, exist_ok=True)

        try:
            workspace_root = Path(__file__).resolve().parents[2]
            import sys as _sys
            import numpy as np
            for _p in [str(workspace_root), str(workspace_root / "source" / "unitree_lab")]:
                if _p not in _sys.path:
                    _sys.path.insert(0, _p)
            _script_dir = str(workspace_root / "scripts" / "mujoco_eval")
            if _script_dir not in _sys.path:
                _sys.path.insert(0, _script_dir)

            from unitree_lab.mujoco_utils.evaluation.eval_task import get_eval_task
            from unitree_lab.mujoco_utils.evaluation.metrics import compute_locomotion_metrics
            from unitree_lab.mujoco_utils.simulation.base_simulator import BaseMujocoSimulator
            from run_sim2sim_locomotion import _generate_course_xml, _setup_terrain

            config_override = {}
            if deploy_yaml_path and os.path.isfile(deploy_yaml_path):
                config_override = self._load_deploy_override(deploy_yaml_path)

            # ---- Phase 1: Headless evaluation of all tasks (no rendering) ----
            print(f"[Sim2Sim] Phase 1: evaluating {len(eval_tasks)} tasks (it={iteration})...")
            task_results: dict[str, dict] = {}
            policy_dt = 0.02

            for task_name in eval_tasks:
                try:
                    task = get_eval_task(task_name)
                    simulator, tmp_xml = self._create_simulator_with_terrain(
                        task, onnx_path, xml_path_cfg, config_override,
                        _generate_course_xml, _setup_terrain,
                    )

                    vel_cmd = task.velocity_command
                    episodes = []
                    for _ in range(task.num_episodes):
                        ep = simulator.run_episode(
                            max_steps=task.max_episode_steps,
                            render=False,
                            velocity_command=vel_cmd,
                        )
                        episodes.append(ep)

                    metrics = compute_locomotion_metrics(episodes, np.array(vel_cmd), policy_dt)
                    task_results[task_name] = {
                        "survival_rate": metrics.survival_rate,
                        "mean_velocity_error": metrics.mean_velocity_error,
                        "mean_forward_distance": metrics.mean_forward_distance,
                        "velocity_error_x": metrics.velocity_error_x,
                        "velocity_error_y": metrics.velocity_error_y,
                        "metrics": metrics,
                    }
                    print(f"  {task_name}: survival={metrics.survival_rate:.0f}%, "
                          f"vel_err={metrics.mean_velocity_error:.3f}")

                    self._cleanup_tmp_xml(tmp_xml)

                except Exception as e:
                    logger.warning(f"[Sim2Sim] Phase 1 failed for {task_name}: {e}")

            # ---- Phase 2: Record videos for selected tasks ----
            video_tasks = self._select_video_tasks(task_results, num_worst_videos)
            max_steps = int(duration / policy_dt)
            videos: dict[str, str] = {}

            print(f"[Sim2Sim] Phase 2: recording {len(video_tasks)} videos...")
            for video_label, task_name in video_tasks:
                try:
                    task = get_eval_task(task_name)
                    simulator, tmp_xml = self._create_simulator_with_terrain(
                        task, onnx_path, xml_path_cfg, config_override,
                        _generate_course_xml, _setup_terrain,
                    )

                    video_path = self._record_sim2sim_video(
                        simulator, task, out_dir, max_steps, velocity=task.velocity_command,
                    )
                    if video_path and os.path.isfile(video_path):
                        videos[video_label] = video_path
                        print(f"  {video_label} ({task_name}): {video_path}")

                    self._cleanup_tmp_xml(tmp_xml)

                except Exception as e:
                    logger.warning(f"[Sim2Sim] Phase 2 video failed for {task_name}: {e}")

            # ---- Upload to W&B ----
            self._log_sim2sim_results_to_wandb(task_results, videos, iteration)
            print(f"[Sim2Sim] Done (it={iteration}): {len(task_results)} tasks, {len(videos)} videos")

        except Exception as e:
            logger.warning(f"[Sim2Sim] Error: {e}")
            import traceback
            traceback.print_exc()

    def _create_simulator_with_terrain(self, task, onnx_path, xml_path_cfg,
                                        config_override, _generate_course_xml, _setup_terrain):
        """Create a BaseMujocoSimulator with terrain injection for the given task."""
        from unitree_lab.mujoco_utils.simulation.base_simulator import BaseMujocoSimulator

        xml_path = xml_path_cfg
        if xml_path is None:
            _, xml_path = self._find_sim2sim_xml(task.name)

        uses_terrain = task.terrain_type != "flat"
        xml_path_obj = Path(xml_path) if xml_path else None
        tmp_xml = None

        if xml_path_obj is None:
            raise ValueError(f"No XML found for task {task.name}")

        if uses_terrain and task.terrain_type == "course":
            xml_path_obj, course_spawn_z = _generate_course_xml(xml_path_obj, task)
            tmp_xml = xml_path_obj
            uses_terrain = False
        else:
            course_spawn_z = 0.0

        simulator = BaseMujocoSimulator(
            xml_path=str(xml_path_obj),
            onnx_path=str(onnx_path),
            config_override=config_override if config_override else None,
        )

        if course_spawn_z > 0:
            simulator.spawn_root_z_offset = course_spawn_z

        if uses_terrain:
            try:
                spawn_z = _setup_terrain(simulator, task)
                if spawn_z > 0:
                    simulator.spawn_root_z_offset = spawn_z
            except Exception as e:
                logger.warning(f"[Sim2Sim] Terrain setup failed for {task.name}: {e}")

        return simulator, tmp_xml

    @staticmethod
    def _select_video_tasks(task_results: dict, num_worst: int) -> list[tuple[str, str]]:
        """Select tasks for video recording: rough_forward + worst N tasks."""
        video_tasks = []

        if "rough_forward" in task_results:
            video_tasks.append(("sim2sim_video", "rough_forward"))

        sorted_tasks = sorted(
            task_results.items(),
            key=lambda kv: (kv[1]["survival_rate"], -kv[1]["mean_velocity_error"]),
        )
        worst_count = 0
        for task_name, _ in sorted_tasks:
            if worst_count >= num_worst:
                break
            if task_name == "rough_forward":
                continue
            worst_count += 1
            video_tasks.append((f"sim2sim_video_worst_{worst_count}", task_name))

        return video_tasks

    @staticmethod
    def _cleanup_tmp_xml(tmp_xml) -> None:
        if tmp_xml is not None:
            try:
                Path(tmp_xml).unlink(missing_ok=True)
                for f in Path(tmp_xml).parent.glob("rough_ground_*.bin"):
                    f.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _log_sim2sim_results_to_wandb(task_results: dict, videos: dict, iteration: int) -> None:
        """Upload structured metrics + videos to W&B."""
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        log_dict = {}
        for task_name, result in task_results.items():
            prefix = f"sim2sim_eval/{task_name}"
            log_dict[f"{prefix}/survival_rate"] = result["survival_rate"]
            log_dict[f"{prefix}/mean_velocity_error"] = result["mean_velocity_error"]
            log_dict[f"{prefix}/mean_forward_distance"] = result["mean_forward_distance"]
            log_dict[f"{prefix}/velocity_error_x"] = result["velocity_error_x"]
            log_dict[f"{prefix}/velocity_error_y"] = result["velocity_error_y"]

        for label, video_path in videos.items():
            if os.path.isfile(video_path):
                log_dict[label] = wandb.Video(video_path, format="mp4", caption=f"iter_{iteration}")

        wandb.log(log_dict, step=int(iteration), commit=False)

    def _export_onnx(self, export_dir: str, iteration: int) -> str | None:
        """Export policy to ONNX format."""
        try:
            import onnx

            policy = self.alg.policy
            policy.eval()
            if not hasattr(policy, "actor"):
                return None

            obs_dim = policy.actor[0].in_features
            example_input = torch.zeros(1, obs_dim, device=self.device)
            filename = f"policy_iter_{iteration}.onnx"
            onnx_path = os.path.join(export_dir, filename)

            class _Wrapper(torch.nn.Module):
                def __init__(self, p):
                    super().__init__()
                    self.actor = p.actor
                    self.actor_obs_normalizer = p.actor_obs_normalizer
                    self.state_dependent_std = getattr(p, "state_dependent_std", False)
                def forward(self, obs_):
                    obs_ = self.actor_obs_normalizer(obs_)
                    if self.state_dependent_std:
                        return self.actor(obs_)[..., 0, :]
                    return self.actor(obs_)

            wrapped = _Wrapper(policy)
            wrapped.eval()
            torch.onnx.export(
                wrapped, example_input, onnx_path,
                input_names=["obs"], output_names=["actions"],
                dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
                opset_version=11,
            )
            onnx.checker.check_model(onnx.load(onnx_path))

            try:
                from unitree_lab.utils.onnx_utils import build_onnx_metadata, attach_onnx_metadata
                base_env = getattr(self.env, "unwrapped", self.env)
                meta = build_onnx_metadata(base_env)
                attach_onnx_metadata(onnx_path, meta)
                print(f"[ONNX] Attached metadata to {onnx_path}")
            except Exception:
                pass

            policy.train()
            return onnx_path
        except Exception as e:
            logger.warning(f"[Sim2Sim] ONNX export failed: {e}")
            return None

    def _build_deploy_yaml(self, export_dir: str, iteration: int) -> str | None:
        """Build deploy.yaml from ONNX metadata."""
        try:
            from unitree_lab.utils.onnx_utils import build_onnx_metadata
            from isaaclab.utils.io import dump_yaml

            base_env = getattr(self.env, "unwrapped", self.env)
            meta = build_onnx_metadata(base_env)
            deploy = self._meta_to_deploy(meta)
            path = os.path.join(export_dir, f"deploy_iter_{iteration}.yaml")
            dump_yaml(path, deploy)
            dump_yaml(os.path.join(export_dir, "deploy_latest.yaml"), deploy)
            return path
        except Exception:
            return None

    @staticmethod
    def _meta_to_deploy(meta: dict) -> dict:
        deploy: dict = {}
        for src, dst in [
            ("joint_names", "joint_names"), ("default_joint_pos", "default_joint_pos"),
            ("joint_stiffness", "stiffness"), ("joint_damping", "damping"),
            ("joint_armature", "armature"),
        ]:
            if isinstance(meta.get(src), list):
                deploy[dst] = meta[src]
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

    @staticmethod
    def _load_deploy_override(yaml_path: str) -> dict:
        """Load deploy.yaml and convert to config override."""
        import yaml
        with open(yaml_path) as f:
            deploy = yaml.safe_load(f) or {}
        override: dict = {}
        key_map = {
            "joint_stiffness": "joint_stiffness", "joint_damping": "joint_damping",
            "joint_names": "joint_names", "default_joint_pos": "default_joint_pos",
            "action_scale": "action_scale", "action_offset": "action_offset",
            "observation_names": "observation_names", "observation_dims": "observation_dims",
            "sim_dt": "sim_dt", "decimation": "decimation", "joint_armature": "joint_armature",
            "history_length": "history_length", "single_frame_dims": "single_frame_dims",
            "stiffness": "joint_stiffness", "damping": "joint_damping", "armature": "joint_armature",
        }
        for src, dst in key_map.items():
            if src in deploy and dst not in override:
                override[dst] = deploy[src]
        jpa = deploy.get("actions", {}).get("JointPositionAction", {})
        if "scale" in jpa and "action_scale" not in override:
            override["action_scale"] = jpa["scale"]
        if "offset" in jpa and "action_offset" not in override:
            override["action_offset"] = jpa["offset"]
        return override

    @staticmethod
    def _record_sim2sim_video(
        simulator, task, output_dir: str, duration_steps: int,
        velocity=None,
    ) -> str | None:
        """Record sim2sim video with EGL rendering."""
        try:
            import cv2
            import mujoco
            import numpy as np
        except ImportError:
            logger.warning("[Sim2Sim] cv2 or mujoco not available; skipping video.")
            return None

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        width, height = 1280, 720
        if simulator.model.vis.global_.offwidth < width:
            simulator.model.vis.global_.offwidth = width
        if simulator.model.vis.global_.offheight < height:
            simulator.model.vis.global_.offheight = height
        renderer = mujoco.Renderer(simulator.model, height, width)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        base_body_id = getattr(simulator, "_base_body_id", None)
        cam.trackbodyid = base_body_id if (base_body_id is not None and base_body_id > 0) else 1
        cam.distance = 3.0
        cam.azimuth = -150
        cam.elevation = -20
        cam.lookat[:] = [0, 0, 0.8]

        simulator.reset()
        vel_cmd = velocity if velocity is not None else (0.5, 0.0, 0.0)
        simulator.set_velocity_command(*vel_cmd)

        frames = []
        for _ in range(duration_steps):
            renderer.update_scene(simulator.data, camera=cam)
            frames.append(renderer.render().copy())
            simulator.step()

        video_file = out_path / f"{task.name}_sim2sim.mp4"
        fps = int(1.0 / simulator.policy_dt) if simulator.policy_dt > 0 else 50
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        try:
            import subprocess
            tmp = video_file.with_suffix(".h264_tmp.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(video_file), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", "-preset", "veryfast", "-crf", "23",
                 str(tmp)], check=True,
            )
            tmp.replace(video_file)
        except Exception:
            pass

        return str(video_file)

    def _find_sim2sim_xml(self, task: str = "rough_forward"):
        rl_lab_root = Path(__file__).resolve().parents[2]
        script = rl_lab_root / "scripts" / "mujoco_eval" / "run_sim2sim_locomotion.py"
        xml_dir = rl_lab_root / "source" / "unitree_lab" / "unitree_lab" / "assets" / "robots_xml" / "g1"
        terrain_tasks = {"rough_forward", "stairs_up", "stairs_down", "slope_up", "mixed_terrain"}
        terrain_xml = xml_dir / "scene_29dof_terrain.xml"
        flat_xml = xml_dir / "scene_29dof.xml"
        xml = terrain_xml if (task in terrain_tasks and terrain_xml.exists()) else flat_xml
        return (str(script) if script.exists() else None, str(xml) if xml.exists() else None)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.alg_cfg.get("rnd_cfg"):
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        if self.amp is not None:
            saved_dict["amp_state_dict"] = self.amp.state_dict()
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

        if self.alg_cfg.get("rnd_cfg") and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        if self.amp is not None and "amp_state_dict" in loaded_dict:
            self.amp.load_state_dict(loaded_dict["amp_state_dict"])

        if load_optimizer and resumed_training:
            if "optimizer_state_dict" in loaded_dict:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if self.alg_cfg.get("rnd_cfg") and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

        if resumed_training:
            self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos", {})

    # ------------------------------------------------------------------
    # Train / Eval mode
    # ------------------------------------------------------------------

    def train_mode(self):
        super().train_mode()
        if self.amp is not None:
            self.amp.train()

    def eval_mode(self):
        super().eval_mode()
        if self.amp is not None:
            self.amp.eval()

    def _get_default_obs_sets(self) -> list[str]:
        """Include AMP obs groups in defaults so they get stored."""
        defaults = super()._get_default_obs_sets()
        if self._amp_cfg is not None:
            obs_group = self._amp_cfg.get("obs_group", "amp")
            if obs_group not in defaults:
                defaults.append(obs_group)
            cond_group = self._amp_cfg.get("condition_obs_group")
            if cond_group is not None and cond_group not in defaults:
                defaults.append(cond_group)
        return defaults
