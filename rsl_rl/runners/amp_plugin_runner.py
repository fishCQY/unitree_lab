"""AMP Plugin Runner.

Integrates the standalone AMPPlugin with the OnPolicyRunner.
The AMP discriminator is managed as a plugin — PPO stays unmodified.

Data flow:
    1. Environment outputs single-step AMP observations (obs["amp"])
    2. During rollout, AMPPlugin.reward() builds multi-frame sequences
       from the RolloutStorage and computes style reward
    3. Style reward is blended with task reward via lerp
    4. After PPO update, AMPPlugin.update() trains the discriminator
       using sequences from the rollout and the offline dataset

Usage:
    In the runner config, set:
        class_name = "AMPPluginRunner"
        amp_cfg = { ... }   # AMPPlugin configuration
"""

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

from rsl_rl.env import VecEnv
from rsl_rl.plugins import AMPPlugin
from rsl_rl.runners import OnPolicyRunner


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
                        total_rewards = self.amp.lerp_reward(task_reward=rewards, style_reward=style_rewards)
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

            # ---- Save ----
            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

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
