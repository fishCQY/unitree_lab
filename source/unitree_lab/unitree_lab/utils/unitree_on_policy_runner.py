# Copyright (c) 2024-2026, unitree_lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""UnitreeOnPolicyRunner - Extended OnPolicyRunner with ONNX/JIT export, metadata, and W&B artifacts."""

from __future__ import annotations

import glob
import math
import os
import shutil
from datetime import datetime

import torch

from rsl_rl.env import VecEnv
from rsl_rl.isaaclab_rl import export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from unitree_lab.utils.onnx_utils import attach_onnx_metadata, build_obs_spec
from unitree_lab.utils.wandb_utils import WandbFileSaver


class UnitreeOnPolicyRunner(OnPolicyRunner):
    """Extended OnPolicyRunner with automatic ONNX export, metadata collection, and W&B integration."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self.wandb_file_saver = WandbFileSaver()
        self.experiment_name = train_cfg.get("experiment_name", "")
        self.run_name = train_cfg.get("run_name", "")
        if log_dir:
            log_dir_name = os.path.basename(log_dir)
            parts = log_dir_name.split("_")
            if len(parts) >= 2:
                self.training_timestamp = f"{parts[0]}_{parts[1]}"
            else:
                self.training_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.training_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self._params_uploaded = False
        self._maybe_enrich_onnx_metadata()

    def _maybe_enrich_onnx_metadata(self) -> None:
        if "obs_groups" not in self.cfg or "policy" not in self.cfg["obs_groups"]:
            return
        policy_groups = self.cfg["obs_groups"]["policy"]
        filtered_obs_names: list = []
        filtered_obs_dims: dict = {}
        filtered_obs_scales: dict = {}
        filtered_obs_dim_total = 0
        filtered_obs_terms: list[dict] = []

        if not hasattr(self.env.unwrapped, "observation_manager"):
            return
        obs_mgr = self.env.unwrapped.observation_manager
        active_terms = obs_mgr.active_terms
        group_obs_term_dim = obs_mgr.group_obs_term_dim

        for group in policy_groups:
            if group not in active_terms:
                print(f"[WARNING] Policy group '{group}' not found in active observation terms.")
                continue
            term_names = active_terms[group]
            term_dim_tuples = group_obs_term_dim.get(group, [])
            term_cfgs = obs_mgr._group_obs_term_cfgs.get(group, [])

            for i, term_name in enumerate(term_names):
                filtered_obs_names.append(term_name)
                term_dim_tuple = term_dim_tuples[i] if i < len(term_dim_tuples) else ()
                dim = math.prod(term_dim_tuple) if term_dim_tuple else 1
                filtered_obs_dims[term_name] = int(dim)
                filtered_obs_dim_total += int(dim)
                filtered_obs_scales[term_name] = (
                    1.0 if term_cfgs[i].scale is None else float(term_cfgs[i].scale)
                )
                term_history_length = 1 if term_cfgs[i].history_length == 0 else term_cfgs[i].history_length
                filtered_obs_term = {
                    "name": term_name,
                    "shape": [int(dim / term_history_length), term_history_length],
                    "offsets": list(range(term_history_length - 1, -1, -1)),
                }
                if term_name in ["joint_pos", "joint_vel"]:
                    filtered_obs_term["joint_names"] = term_cfgs[i].params["asset_cfg"].joint_names
                filtered_obs_terms.append(filtered_obs_term)

        obs_spec = {
            "version": 1,
            "offset_unit": "policy_step",
            "time_order": "oldest_to_newest",
            "terms": filtered_obs_terms,
        }

        if hasattr(self.env.unwrapped, "onnx_metadata"):
            meta = self.env.unwrapped.onnx_metadata
            meta["observation_names"] = filtered_obs_names
            meta["observation_dims"] = filtered_obs_dims
            meta["observation_scales"] = filtered_obs_scales
            meta["obs_dim"] = filtered_obs_dim_total
            meta["obs_spec"] = obs_spec

            policy_cfg = self.cfg.get("policy", {})
            if isinstance(policy_cfg, dict):
                frame_stack = policy_cfg.get("frame_stack_num", 1)
            else:
                frame_stack = getattr(policy_cfg, "frame_stack_num", 1)
            meta["frame_stack_num"] = frame_stack if frame_stack else 1

            is_recurrent = False
            rnn_type = None
            hidden_state_shape = None
            cell_state_shape = None
            if isinstance(policy_cfg, dict):
                rnn_type = policy_cfg.get("rnn_type")
                rnn_hidden_dim = policy_cfg.get("rnn_hidden_dim", 512)
                rnn_num_layers = policy_cfg.get("rnn_num_layers", 1)
            else:
                rnn_type = getattr(policy_cfg, "rnn_type", None)
                rnn_hidden_dim = getattr(policy_cfg, "rnn_hidden_dim", 512)
                rnn_num_layers = getattr(policy_cfg, "rnn_num_layers", 1)
            if rnn_type is not None:
                is_recurrent = True
                hidden_state_shape = [rnn_num_layers, rnn_hidden_dim]
                if str(rnn_type).lower() == "lstm":
                    cell_state_shape = [rnn_num_layers, rnn_hidden_dim]
            meta["is_recurrent"] = is_recurrent
            meta["rnn_type"] = rnn_type
            meta["hidden_state_shape"] = hidden_state_shape
            meta["cell_state_shape"] = cell_state_shape

            if isinstance(policy_cfg, dict):
                exteroception_key = policy_cfg.get("exteroception_key")
            else:
                exteroception_key = getattr(policy_cfg, "exteroception_key", None)
            env_cfg = self.env.unwrapped.cfg
            enable_depth = getattr(env_cfg, "enable_depth_exteroception", False)
            enable_height = getattr(env_cfg, "enable_height_exteroception", False)
            if exteroception_key == "exteroception_depth" and enable_depth:
                meta["exteroception_type"] = "depth"
                try:
                    camera = self.env.unwrapped.scene["depth_camera"]
                    crop_left = getattr(camera.cfg, "crop_left", 0)
                    meta["exteroception_shape"] = [
                        camera.cfg.pattern_cfg.height,
                        int(camera.cfg.pattern_cfg.width) - crop_left,
                        1,
                    ]
                except (KeyError, AttributeError):
                    pass
            elif exteroception_key == "exteroception_height" and enable_height:
                meta["exteroception_type"] = "height_scan"
                try:
                    scanner = self.env.unwrapped.scene["height_scanner"]
                    h = int(scanner.cfg.pattern_cfg.size[0] / scanner.cfg.pattern_cfg.resolution) + 1
                    w = int(scanner.cfg.pattern_cfg.size[1] / scanner.cfg.pattern_cfg.resolution) + 1
                    meta["exteroception_shape"] = [h, w, 1]
                except (KeyError, AttributeError):
                    pass

            self._build_obs_spec(meta)

    def _build_obs_spec(self, meta: dict) -> None:
        try:
            spec = build_obs_spec(self.env.unwrapped)
        except Exception:
            return
        if spec:
            meta["obs_spec"] = spec

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._maybe_upload_params_to_wandb()
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def _maybe_upload_params_to_wandb(self) -> None:
        if self._params_uploaded:
            return
        logger_type = self.cfg.get("logger", "tensorboard")
        if str(logger_type).lower() != "wandb":
            return
        self.wandb_file_saver.save_python_files()
        log_dir = self.logger.log_dir
        if not log_dir:
            self._params_uploaded = True
            return
        params_dir = os.path.join(log_dir, "params")
        if not os.path.isdir(params_dir):
            self._params_uploaded = True
            return
        import wandb

        for yaml_file in glob.glob(os.path.join(params_dir, "*.yaml")):
            try:
                wandb.save(yaml_file, base_path=log_dir)
                print(f"[INFO] Uploaded params to wandb: {os.path.basename(yaml_file)}")
            except Exception as e:
                print(f"[WARNING] Failed to upload {yaml_file} to wandb: {e}")
        self._params_uploaded = True

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        self.sync_checkpoint_to_oss(path)

        policy_path = path.split("model")[0]
        base_name = policy_path.split("/")[-2]

        model_filename = os.path.basename(path)
        model_name = os.path.splitext(model_filename)[0]
        iteration = model_name.split("_")[-1]
        filename = f"policy_{iteration}.pt"
        jit_dir = os.path.join(policy_path, "jit")
        os.makedirs(jit_dir, exist_ok=True)

        if hasattr(self.alg.policy, "actor_obs_normalizer"):
            normalizer = self.alg.policy.actor_obs_normalizer
        elif hasattr(self.alg.policy, "student_obs_normalizer"):
            normalizer = self.alg.policy.student_obs_normalizer
        else:
            normalizer = None

        export_policy_as_jit(self.alg.policy, normalizer=normalizer, path=jit_dir + "/", filename=filename)
        self.sync_checkpoint_to_oss(os.path.join(jit_dir, filename))

        logger_type = self.cfg.get("logger", "tensorboard")
        if str(logger_type).lower() == "wandb":
            import wandb

            self.wandb_file_saver.save_python_files()
            onnx_name = f"{base_name}_{iteration}.onnx"
            onnx_dir = os.path.join(policy_path, "onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            export_policy_as_onnx(self.alg.policy, normalizer=normalizer, path=onnx_dir + "/", filename=onnx_name)
            wandb_run_name = (
                str(wandb.run.name)
                if (wandb.run is not None and wandb.run.name is not None)
                else "unknown_wandb_run"
            )
            attach_onnx_metadata(self.env.unwrapped, wandb_run_name, path=onnx_dir + "/", filename=onnx_name)
            onnx_path = os.path.join(onnx_dir, onnx_name)
            wandb.save(onnx_path, base_path=os.path.dirname(policy_path))
            self.sync_checkpoint_to_oss(onnx_path)

            if getattr(self.env.unwrapped, "mujoco_eval", None) is not None:
                try:
                    try:
                        iteration_int = int(iteration)
                    except Exception:
                        iteration_int = None
                    if iteration_int == 0:
                        print("[INFO] Skipping sim2sim evaluation for iteration=0.")
                    else:
                        self.env.unwrapped.mujoco_eval.batch_eval_and_log(
                            onnx_path=onnx_path, iteration=iteration, num_workers=16
                        )
                except Exception as e:
                    print(f"[WARNING] Failed to run sim2sim evaluation: {e}")

            self._maybe_log_motion_sampling_stats(iteration)

    def _maybe_log_motion_sampling_stats(self, iteration: str) -> None:
        try:
            import wandb

            cmd_mgr = getattr(self.env.unwrapped, "command_manager", None)
            if cmd_mgr is None or not hasattr(cmd_mgr, "_terms"):
                return
            motion_cmd = cmd_mgr._terms.get("motion")
            if motion_cmd is None or not hasattr(motion_cmd, "get_motion_sampling_stats"):
                return
            import numpy as np

            sampling_stats = motion_cmd.get_motion_sampling_stats()
            step = int(iteration) if iteration is not None else None
            motion_ids = list(sampling_stats["motion_ids"])
            avg_probs = np.asarray(sampling_stats["avg_probs"], dtype=np.float64).reshape(-1)
            max_probs = np.asarray(sampling_stats["max_probs"], dtype=np.float64).reshape(-1)
            if len(motion_ids) != len(avg_probs) or len(motion_ids) != len(max_probs):
                return
            num_motions = len(motion_ids)
            if num_motions == 0:
                return
            num_bins = min(512, max(10, int(np.sqrt(num_motions))))
            avg_prob_histogram = wandb.Histogram(avg_probs.tolist(), num_bins=num_bins)
            max_prob_histogram = wandb.Histogram(max_probs.tolist(), num_bins=num_bins)
            table = wandb.Table(
                data=[[int(mid), float(avg_p), float(max_p)] for mid, avg_p, max_p in zip(motion_ids, avg_probs, max_probs)],
                columns=["motion_id", "avg_sampling_prob", "max_sampling_prob"],
            )
            wandb.log(
                {
                    "Sampling/avg_prob_distribution": avg_prob_histogram,
                    "Sampling/max_prob_distribution": max_prob_histogram,
                    "Sampling/motion_details": table,
                },
                step=step,
            )
            print(f"[INFO] Logged motion sampling stats for {num_motions} motions to wandb")
        except Exception as e:
            print(f"[WARNING] Failed to log motion sampling statistics: {e}")

    def sync_checkpoint_to_oss(self, checkpoint_path: str) -> None:
        """Copy checkpoint into ``UNITREE_CHECKPOINT_SYNC_DIR`` if set (optional mirror of bfm OSS sync)."""
        sync_root = os.environ.get("UNITREE_CHECKPOINT_SYNC_DIR")
        if not sync_root or not os.path.isfile(checkpoint_path):
            return
        run_suffix = f"_{self.run_name}" if self.run_name else ""
        dest_dir = os.path.join(
            sync_root, "rsl_rl", self.experiment_name, f"{self.training_timestamp}{run_suffix}"
        )
        try:
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.basename(checkpoint_path))
            shutil.copy2(checkpoint_path, dest_path)
            print(f"[INFO] Synced checkpoint: {checkpoint_path} -> {dest_path}")
        except Exception as e:
            print(f"[WARNING] Failed to sync checkpoint: {e}")
