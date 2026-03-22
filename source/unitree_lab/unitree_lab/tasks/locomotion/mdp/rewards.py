from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers.manager_base import ManagerTermBase
import isaaclab.utils.math as math_utils
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    # reward *= (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) + torch.abs(env.command_manager.get_command(command_name)[:, 2])) > 0.1

    return reward


def joint_power_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    joint_power = (
        asset.data.applied_torque[:, asset_cfg.joint_ids]
        * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )

    return torch.sum(torch.abs(joint_power), dim=1)


class action_smoothness_l2(ManagerTermBase):
    def __init__(
        self, cfg: RewardTermCfg, env: ManagerBasedRLEnv,
    ):
        super().__init__(cfg, env)
        self.prev_prev_action = None

    def __call__(
        self, env: ManagerBasedRLEnv, cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.prev_prev_action is None:
            self.prev_prev_action = env.action_manager.prev_action.clone()
        action_smoothness_l2 = torch.sum(
            torch.square(
                env.action_manager.action
                - 2 * env.action_manager.prev_action
                + self.prev_prev_action
            ),
            dim=1,
        )
        self.prev_prev_action = env.action_manager.prev_action.clone()
        return action_smoothness_l2


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        base_height = asset.data.root_pos_w[:, 2] - sensor.data.ray_hits_w[..., 2].mean(
            dim=-1
        )
    else:
        base_height = asset.data.root_link_pos_w[:, 2]
    # Replace NaNs with the base_height
    base_height = torch.nan_to_num(
        base_height, nan=target_height, posinf=target_height, neginf=target_height
    )

    # Compute the L2 squared penalty
    return torch.square(base_height - target_height)


def feet_regulation(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None,
    sensor_cfg3: SceneEntityCfg | None = None,
    sensor_cfg4: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    foot_heights = torch.stack(
        [
            env.scene[sensor_cfg.name].data.pos_w[:, 2]
            - env.scene[sensor_cfg.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor_cfg in [sensor_cfg1, sensor_cfg2, sensor_cfg3, sensor_cfg4]
            if sensor_cfg is not None
        ],
        dim=-1,
    )
    foot_heights = torch.nan_to_num(foot_heights, nan=0, posinf=0, neginf=0)
    foot_heights = torch.clamp(foot_heights - 0.02, min=0.0)

    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    feet_target_height = target_height * 0.025

    reward = torch.sum(
        torch.exp(-foot_heights / feet_target_height)
        * torch.square(torch.norm(foot_vel_xy, dim=-1)),
        dim=1,
    )

    return reward


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def fly(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def body_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def body_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def body_lin_vel_z_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_lin_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :],
        asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids[0], :],
    )
    return torch.square(body_lin_vel_b[:, 2])


def body_ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ang_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :],
        asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids[0], :],
    )
    return torch.sum(torch.square(body_ang_vel_b[:, :2]), dim=1)


def energy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    joint_power = (
        asset.data.applied_torque[:, asset_cfg.joint_ids]
        * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )

    return torch.norm(torch.abs(joint_power), dim=1)


def tracking_joint_pos_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_names = [asset.joint_names[i] for i in asset_cfg.joint_ids]
    joint_pos_target = env.command_manager.get_term(command_name).tracking_joint_pos(joint_names)
    joint_pos_error = torch.square(joint_pos_target - joint_pos).mean(dim=-1)
    reward = torch.exp(-joint_pos_error / std**2)
    return reward


def tracking_joint_vel_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_names = [asset.joint_names[i] for i in asset_cfg.joint_ids]
    joint_vel_target = env.command_manager.get_term(command_name).tracking_joint_vel(joint_names)
    joint_vel_error = torch.square(joint_vel_target - joint_vel).mean(dim=-1)
    reward = torch.exp(-joint_vel_error / std**2)
    return reward


def tracking_body_pos_w_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    body_link_pos_w_target = env.command_manager.get_term(command_name).tracking_body_pos_w(body_names)
    body_pos_error = torch.square(body_link_pos_w_target - body_link_pos_w).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_pos_error / std**2)
    return reward


def tracking_body_pos_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # Convert the target body positions to the asset's root link frame
    body_pos_xyz = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_link_quat_w.unsqueeze(1)),
        asset.data.body_link_pos_w[:, asset_cfg.body_ids] - asset.data.root_link_pos_w.unsqueeze(1)
    )
    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    body_pos_xyz_target = env.command_manager.get_term(command_name).tracking_body_pos(body_names)
    body_pos_error = torch.square(body_pos_xyz_target - body_pos_xyz).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_pos_error / std**2)
    return reward


def tracking_body_quat_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    body_quat_target = env.command_manager.get_term(command_name).tracking_body_quat(body_names)
    quat_error_magnitude = math_utils.quat_error_magnitude(body_quat_target, body_quat).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-quat_error_magnitude / std**2)
    return reward


def tracking_body_vel_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids],
        asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    )
    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    body_vel_b_target = env.command_manager.get_term(command_name).tracking_body_vel(body_names)
    body_vel_error = torch.square(body_vel_b_target - body_vel_b).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_vel_error / std**2)
    return reward


def tracking_body_ang_vel_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel_b = math_utils.quat_apply_inverse(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids],
        asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
    )
    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    body_ang_vel_b_target = env.command_manager.get_term(command_name).tracking_body_ang_vel(body_names)
    body_ang_vel_error = torch.square(body_ang_vel_b_target - body_ang_vel_b).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_ang_vel_error / std**2)
    return reward


def tracking_key_points_w_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    keypoints_b = env.command_manager.get_term(command_name).cfg.side_length * \
        torch.eye(3, device=asset.device, dtype=torch.float)
    keypoints_b = keypoints_b.unsqueeze(0).expand(body_link_pos_w.shape[0], -1, -1, -1)
    keypoints_w = math_utils.quat_apply(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids].unsqueeze(2).expand(-1, -1, keypoints_b.shape[2], -1),
        keypoints_b.expand(-1, len(asset_cfg.body_ids), -1, -1)
    ) + body_link_pos_w.unsqueeze(2)

    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    key_points_w_target = env.command_manager.get_term(command_name).tracking_key_points_w(body_names)
    body_pos_error = torch.square(key_points_w_target - keypoints_w).sum(dim=-2).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_pos_error / std**2)
    return reward


def tracking_key_points_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    keypoints_b = env.command_manager.get_term(command_name).cfg.side_length * \
        torch.eye(3, device=asset.device, dtype=torch.float)
    keypoints_b = keypoints_b.unsqueeze(0).expand(body_link_pos_w.shape[0], -1, -1, -1)
    keypoints_w = math_utils.quat_apply(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids].unsqueeze(2).expand(-1, -1, keypoints_b.shape[2], -1),
        keypoints_b.expand(-1, len(asset_cfg.body_ids), -1, -1)
    ) + body_link_pos_w.unsqueeze(2)
    keypoints = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_link_quat_w.unsqueeze(1).unsqueeze(1)).expand(-1, keypoints_w.shape[1], keypoints_w.shape[2], -1),
        keypoints_w - asset.data.root_link_pos_w.unsqueeze(1).unsqueeze(1)
    )

    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    key_points_target = env.command_manager.get_term(command_name).tracking_key_points(body_names)
    body_pos_error = torch.square(key_points_target - keypoints).sum(dim=-2).mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-body_pos_error / std**2)
    return reward


def stay_alive_startup(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive during the startup phase."""
    mean_episode_length = env.episode_length_buf.float().mean()
    max_episode_length = env.max_episode_length
    ratio = ((max_episode_length - mean_episode_length * 2) / max_episode_length).clip(0, 1)
    return torch.ones(env.num_envs, device=env.device) * ratio


def tracking_key_points_w_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    keypoints_b = env.command_manager.get_term(command_name).cfg.side_length * \
        torch.eye(3, device=asset.device, dtype=torch.float)
    keypoints_b = keypoints_b.unsqueeze(0).expand(body_link_pos_w.shape[0], -1, -1, -1)
    keypoints_w = math_utils.quat_apply(
        asset.data.body_link_quat_w[:, asset_cfg.body_ids].unsqueeze(2),
        keypoints_b
    ) + body_link_pos_w.unsqueeze(2)

    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    key_points_w_target = env.command_manager.get_term(command_name).tracking_key_points_w(body_names)
    reward = torch.square(key_points_w_target - keypoints_w).sum(dim=-2).mean(dim=-1).mean(dim=-1)
    return reward


def tracking_body_height_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]

    tracking_body_pos_w = env.command_manager.get_term(command_name).tracking_body_pos_w(body_names)

    body_pos_error = body_link_pos_w - tracking_body_pos_w
    reward = torch.square(body_pos_error[..., 2]).sum(dim=-1)
    return reward


class select_undesired_contacts(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.body_names = [self.contact_sensor.body_names[i] for i in self.sensor_cfg.body_ids]

    def __call__(
        self, env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg,
        command_name: str,
        height_threshold
    ):
        """Penalize undesired contacts as the number of violations that are above a threshold."""
        # check if contact force is above threshold
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
        body_pos_w_target = env.command_manager.get_term(command_name).tracking_body_pos_w(self.body_names)
        mask = body_pos_w_target[:, :, 2] > height_threshold
        is_contact = is_contact & mask
        return torch.sum(is_contact, dim=1)


def orientation_mixed(
    env: ManagerBasedRLEnv, l2_sigma: float = 2.0, l1_sigma: float = 1.0, l1_weight: float = 0.1,
) -> torch.Tensor:
    """Reward for maintaining upright orientation (roll/pitch near zero)."""
    asset: Articulation = env.scene["robot"]
    projected_gravity_b = asset.data.projected_gravity_b[:, :2]
    l2_reward = torch.exp(-l2_sigma * torch.norm(projected_gravity_b, dim=1) ** 2)
    l1_reward = torch.exp(-l1_sigma * torch.norm(projected_gravity_b, dim=1))
    return l2_reward + l1_weight * l1_reward


def action_rate_l2_clipped(env: ManagerBasedRLEnv, max_val: float = 5.0) -> torch.Tensor:
    """Action rate L2 penalty with per-joint clipping to avoid exploding gradients."""
    rate_sq = torch.square(env.action_manager.action - env.action_manager.prev_action)
    return torch.sum(rate_sq.clamp(max=max_val ** 2), dim=1)


def dof_pos_near_limits(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that are within threshold of their limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]
    near_lower = (joint_pos - limits[:, :, 0]) < threshold
    near_upper = (limits[:, :, 1] - joint_pos) < threshold
    return torch.sum(near_lower | near_upper, dim=1).float()


def joint_deviation_l1_without_command(
    env: ManagerBasedRLEnv, command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(angle), dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < 0.01
    return reward
