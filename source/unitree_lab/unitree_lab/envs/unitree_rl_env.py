"""UnitreeRLEnv - Extended ManagerBasedRLEnv with ONNX metadata and MuJoCo eval support.

Provides:
- Automatic ONNX metadata collection for deployment
- MuJoCo sim2sim evaluation integration during training
- Environment group support for mixed baseline experiments
- post_reset event support
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .unitree_rl_env_cfg import UnitreeRLEnvCfg


class UnitreeRLEnv(ManagerBasedRLEnv):
    cfg: UnitreeRLEnvCfg

    def __init__(self, cfg: UnitreeRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.env_group = None
        self._env_group_cfg = cfg.num_groups

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        if self.cfg.mujoco_eval is not None:
            mujoco_eval_class = self.cfg.mujoco_eval.get_class()
            self.mujoco_eval = mujoco_eval_class(self.cfg.mujoco_eval, self)
        else:
            self.mujoco_eval = None

        if self._env_group_cfg is not None:
            self.env_group = torch.randint(
                0, self._env_group_cfg, (self.num_envs,), dtype=torch.int32, device=self.device
            )

        self.onnx_metadata = self.cfg.init_onnx_metadata(self)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if "post_reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="post_reset", env_ids=env_ids, global_env_step_count=env_step_count)
