"""Configuration for UnitreeRLEnv.

Provides centralized ONNX metadata collection for deployment.
All joint-related lists are exported in ONNX action output order.
"""

from __future__ import annotations

import math
import re
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .unitree_rl_env import UnitreeRLEnv


@configclass
class UnitreeRLEnvCfg(ManagerBasedRLEnvCfg):
    robot_name: str = MISSING

    mujoco_eval: object | None = None

    num_groups: int | None = None

    group_ratios: list[float] | None = None

    def init_onnx_metadata(self, env: UnitreeRLEnv) -> dict:
        """Initialize ONNX metadata for deployment.

        All joint-related lists are in ONNX action output order.
        """
        (
            joint_names,
            action_scale,
            default_joint_pos,
        ) = self._onnx_collect_action_joint_info(env)
        (
            stiffness,
            damping,
            tau_limits,
            armature,
            friction,
            viscous_friction,
        ) = self._onnx_collect_actuator_params(env, joint_names)
        (
            obs_names,
            obs_dims,
            obs_scales,
            obs_dim_total,
        ) = self._onnx_collect_observation_info(env)

        num_actions = len(joint_names)

        return {
            "joint_names": joint_names,
            "joint_stiffness": stiffness,
            "joint_damping": damping,
            "tau_limits": tau_limits,
            "action_scale": action_scale,
            "default_joint_pos": default_joint_pos,
            "armature": armature,
            "friction": friction,
            "viscous_friction": viscous_friction,
            "num_actions": num_actions,
            "observation_names": obs_names,
            "observation_dims": obs_dims,
            "observation_scales": obs_scales,
            "obs_dim": obs_dim_total,
            "frame_stack_num": 1,
            "is_recurrent": False,
            "rnn_type": None,
            "hidden_state_shape": None,
            "cell_state_shape": None,
            "command_names": env.command_manager.active_terms,
            "robot_name": getattr(self, "robot_name", None),
            "decimation": self.decimation,
            "sim_dt": self.sim.dt,
            "policy_dt": self.decimation * self.sim.dt,
            "custom_data": {},
        }

    def _onnx_collect_action_joint_info(
        self, env: UnitreeRLEnv
    ) -> tuple[list[str], list[float], list[float]]:
        """Collect joint_names/action_scale/default_joint_pos in ONNX action output order."""
        isaac_joint_names = list(env.scene["robot"].data.joint_names)
        isaac_default_pos = env.scene["robot"].data.default_joint_pos.mean(dim=0).cpu().tolist()

        joint_names: list[str] = []
        action_scale: list[float] = []
        default_joint_pos: list[float] = []

        for term_name in env.action_manager.active_terms:
            term = env.action_manager.get_term(term_name)
            if not hasattr(term, "_joint_ids"):
                continue

            if hasattr(term._joint_ids, "tolist"):
                term_joint_ids = term._joint_ids.tolist()
            else:
                term_joint_ids = list(term._joint_ids)

            term_joint_names = [isaac_joint_names[i] for i in term_joint_ids]
            joint_names.extend(term_joint_names)

            if hasattr(term, "_scale"):
                if hasattr(term._scale, "cpu"):
                    scale = term._scale.cpu()
                    if scale.dim() > 1:
                        scale = scale[0]
                    scales = scale.tolist()
                elif isinstance(term._scale, (list, tuple)):
                    scales = list(term._scale)
                else:
                    scales = [float(term._scale)] * len(term_joint_names)
            else:
                scales = [-1.0] * len(term_joint_names)
            action_scale.extend(scales)

            for jid in term_joint_ids:
                default_joint_pos.append(isaac_default_pos[jid])

        return joint_names, action_scale, default_joint_pos

    def _onnx_collect_actuator_params(
        self, env: UnitreeRLEnv, joint_names: list[str]
    ) -> tuple[list[float], list[float], list[float], dict[str, float], dict[str, float], dict[str, float]]:
        """Collect actuator parameters in ONNX action output order."""
        stiffness: list[float] = []
        damping: list[float] = []
        tau_limits: list[float] = []
        armature: dict[str, float] = {}
        friction: dict[str, float] = {}
        viscous_friction: dict[str, float] = {}

        robot_cfg = env.scene["robot"].cfg
        for joint_name in joint_names:
            found = False
            for _actuator_name, actuator_cfg in robot_cfg.actuators.items():
                for pattern in actuator_cfg.joint_names_expr:
                    if re.fullmatch(pattern, joint_name) or re.search(pattern, joint_name):
                        stiffness.append(float(getattr(actuator_cfg, "stiffness", 0.0)))
                        damping.append(float(getattr(actuator_cfg, "damping", 0.0)))
                        effort = getattr(actuator_cfg, "effort_limit", None)
                        if effort is None:
                            effort = getattr(actuator_cfg, "saturation_effort", 100.0)
                        tau_limits.append(float(effort) if effort is not None else 100.0)
                        armature[joint_name] = float(getattr(actuator_cfg, "armature", 0.0))
                        friction[joint_name] = float(getattr(actuator_cfg, "friction", 0.0))
                        viscous_friction[joint_name] = float(getattr(actuator_cfg, "viscous_friction", 0.0))
                        found = True
                        break
                if found:
                    break

            if not found:
                stiffness.append(0.0)
                damping.append(0.0)
                tau_limits.append(100.0)
                armature[joint_name] = 0.0
                friction[joint_name] = 0.0
                viscous_friction[joint_name] = 0.0

        return stiffness, damping, tau_limits, armature, friction, viscous_friction

    def _onnx_collect_observation_info(
        self, env: UnitreeRLEnv
    ) -> tuple[list[str], dict[str, int], dict[str, float], int]:
        """Collect observation term names/dims/scales."""
        obs_names: list[str] = []
        obs_dims: dict[str, int] = {}
        obs_scales: dict[str, float] = {}
        obs_dim_total = 0

        obs_mgr = env.observation_manager
        for group_name, term_names in obs_mgr.active_terms.items():
            term_dim_tuples = obs_mgr.group_obs_term_dim.get(group_name, [])
            term_cfgs = obs_mgr._group_obs_term_cfgs.get(group_name, [])

            for i, term_name in enumerate(term_names):
                obs_names.append(term_name)
                if i < len(term_dim_tuples):
                    dim = math.prod(term_dim_tuples[i]) if term_dim_tuples[i] else 1
                else:
                    dim = 1
                obs_dims[term_name] = int(dim)
                obs_dim_total += int(dim)
                if i < len(term_cfgs) and hasattr(term_cfgs[i], "scale") and term_cfgs[i].scale is not None:
                    obs_scales[term_name] = float(term_cfgs[i].scale)

        return obs_names, obs_dims, obs_scales, obs_dim_total
