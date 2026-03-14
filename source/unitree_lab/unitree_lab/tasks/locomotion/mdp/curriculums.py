"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, RewardManager
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    reward: RewardManager = env.reward_manager
    lin_track_reward_sum = (
        reward._episode_sums["track_lin_vel_xy_exp"][env_ids] / env.max_episode_length_s
    )
    lin_track_reward_idx = reward._term_names.index("track_lin_vel_xy_exp")
    lin_track_reward_weight = reward._term_cfgs[lin_track_reward_idx].weight
    ang_track_reward_sum = (
        reward._episode_sums["track_ang_vel_z_exp"][env_ids] / env.max_episode_length_s
    )
    ang_track_reward_idx = reward._term_names.index("track_ang_vel_z_exp")
    ang_track_reward_weight = reward._term_cfgs[ang_track_reward_idx].weight
    # compute the distance the robot walked
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    # robots that walked far enough progress to harder terrains
    move_up = (
        (distance > terrain.cfg.terrain_generator.size[0] / 2)
        & (lin_track_reward_sum > lin_track_reward_weight * 0.7)
        & (ang_track_reward_sum > ang_track_reward_weight * 0.7)
    )
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = (lin_track_reward_sum < lin_track_reward_weight * 0.6) | (
        ang_track_reward_sum < ang_track_reward_weight * 0.6
    )
    move_down *= ~move_up

    # terrain level transition counters
    if not hasattr(terrain, "terrain_level_up_count"):
        terrain.terrain_level_up_count = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        terrain.terrain_level_down_count = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        terrain.succeeded_terrain_transition_count = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)

    terrain.terrain_level_up_count[env_ids] += 1 * move_up
    terrain.terrain_level_down_count[env_ids] += 1 * move_down

    terrain.succeeded_terrain_transition_count[env_ids] = torch.where(
        terrain.terrain_levels[env_ids] + 1 * move_up - 1 * move_down >= terrain.max_terrain_level,
        terrain.succeeded_terrain_transition_count[env_ids] + 1,
        terrain.succeeded_terrain_transition_count[env_ids]
    )
    # update curriculum metrics
    if hasattr(env, "vel_ids_terrain_passed_mask"):
        env_ids_succeeded_terrain_transition = env_ids[terrain.terrain_levels[env_ids] + 1 * move_up - 1 * move_down >= terrain.max_terrain_level]
        env.vel_ids_terrain_passed_mask = torch.isin(env.vel_update_ids, env_ids_succeeded_terrain_transition)

        # env_ids_failed_terrain_transition = env_ids[terrain.terrain_levels[env_ids] + 1 * mov_up - 1 * move_down < 0]
        env_ids_failed_terrain_transition = env_ids[move_down]
        env.vel_ids_terrain_failed_mask = torch.isin(env.vel_update_ids, env_ids_failed_terrain_transition)

    env.curriculum_manager._curriculum_state.update({
        "avg_terrain_level_up_count": torch.mean(terrain.terrain_level_up_count.float()),
        "avg_terrain_level_down_count": torch.mean(terrain.terrain_level_down_count.float()),
        "avg_succeeded_terrain_transition_count": torch.mean(terrain.succeeded_terrain_transition_count.float())
    })

    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)

    # Initialize terrain to environment ID mapping
    if not hasattr(env, "terrain_name_to_env_ids"):
        env.terrain_name_to_env_ids = {}
        terrain_cfg = terrain.cfg.terrain_generator

        # Generate terrain proportion tuples (name, proportion)
        terrain_proportion_tuples = [
            (name, terrain_cfg.sub_terrains[name].proportion)
            for name in terrain_cfg.sub_terrains.keys()
        ]

        # Calculate environment index ranges for each terrain type
        cumulative_proportion = 0.0
        for terrain_name, proportion in terrain_proportion_tuples:
            num_envs_per_terrain = int(env.num_envs * proportion)
            start_idx = int(env.num_envs * cumulative_proportion)
            cumulative_proportion += proportion
            end_idx = start_idx + num_envs_per_terrain
            env.terrain_name_to_env_ids[terrain_name] = torch.arange(
                start_idx, end_idx, device=env.device
            )

        # Validate total proportion
        assert abs(cumulative_proportion - 1.0) < 1e-5, \
            f"Total terrain proportions sum to {cumulative_proportion}, should be 1.0"

        # # Define terrains requiring command updates
        terrains_requiring_command_updates = ["random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"]
        env.vel_update_ids = torch.cat([
            env.terrain_name_to_env_ids[name]
            for name in terrains_requiring_command_updates
        ])
        env.vel_ids_terrain_passed_mask = torch.zeros_like(
            env.vel_update_ids, dtype=torch.bool
        )
        env.vel_ids_terrain_failed_mask = torch.zeros_like(
            env.vel_update_ids, dtype=torch.bool
        )

    # Update curriculum metrics for each terrain type
    env.curriculum_manager._curriculum_state.update({
        f"avg_terrain_level_{terrain_name}": torch.mean(
            terrain.terrain_levels[env_ids].float()
        )
        for terrain_name, env_ids in env.terrain_name_to_env_ids.items()
    })

    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    delta: list[float],
    max_curriculum: list[tuple[float, float]]
) -> None:
    vel_cmd = env.command_manager.get_term("base_velocity")
    delta_tensor = torch.tensor(delta, device=env.device).unsqueeze(-1)
    deltas = delta_tensor * torch.tensor([-1, 1], device=env.device)

    if not hasattr(env, "vel_update_ids"):
        reward: RewardManager = env.reward_manager
        lin_track_reward_sum = (
            reward._episode_sums["track_lin_vel_xy_exp"][env_ids] / env.max_episode_length_s
        )
        lin_track_reward_idx = reward._term_names.index("track_lin_vel_xy_exp")
        lin_track_reward_weight = reward._term_cfgs[lin_track_reward_idx].weight
        ang_track_reward_sum = (
            reward._episode_sums["track_ang_vel_z_exp"][env_ids] / env.max_episode_length_s
        )
        ang_track_reward_idx = reward._term_names.index("track_ang_vel_z_exp")
        ang_track_reward_weight = reward._term_cfgs[ang_track_reward_idx].weight
        # robots that walked far enough progress to harder terrains
        mask = (
            (lin_track_reward_sum.mean() > lin_track_reward_weight * 0.8)
            & (ang_track_reward_sum.mean() > ang_track_reward_weight * 0.7)
        )
        vel_update_ids = env_ids[mask]
        if vel_update_ids.numel() > 0:
            vel_cmd.lin_vel_x_ranges[vel_update_ids] = torch.clamp(
                vel_cmd.lin_vel_x_ranges[vel_update_ids] + deltas[0],
                max_curriculum[0][0],
                max_curriculum[0][1]
            )
            vel_cmd.lin_vel_y_ranges[vel_update_ids] = torch.clamp(
                vel_cmd.lin_vel_y_ranges[vel_update_ids] + deltas[1],
                max_curriculum[1][0],
                max_curriculum[1][1]
            )
            vel_cmd.ang_vel_z_ranges[vel_update_ids] = torch.clamp(
                vel_cmd.ang_vel_z_ranges[vel_update_ids] + deltas[2],
                max_curriculum[2][0],
                max_curriculum[2][1]
            )
        env.curriculum_manager._curriculum_state.update({
            "avg_vel_lin_x": torch.mean(vel_cmd.lin_vel_x_ranges[:, 1]),
            "avg_vel_lin_y": torch.mean(vel_cmd.lin_vel_y_ranges[:, 1]),
            "avg_vel_ang_z": torch.mean(vel_cmd.ang_vel_z_ranges[:, 1])
        })
        return

    passed_mask = torch.isin(env_ids, env.vel_update_ids[env.vel_ids_terrain_passed_mask])
    passed_env_ids = env_ids[passed_mask]
    failed_mask = torch.isin(env_ids, env.vel_update_ids[env.vel_ids_terrain_failed_mask])
    failed_env_ids = env_ids[failed_mask]

    # Fallback: if no terrain-driven updates are available, still progress command curriculum
    # from velocity tracking performance so speed curriculum doesn't stall at initial ranges.
    if passed_env_ids.numel() == 0 and failed_env_ids.numel() == 0:
        reward: RewardManager = env.reward_manager
        lin_track_reward_sum = (
            reward._episode_sums["track_lin_vel_xy_exp"][env_ids] / env.max_episode_length_s
        )
        lin_track_reward_idx = reward._term_names.index("track_lin_vel_xy_exp")
        lin_track_reward_weight = reward._term_cfgs[lin_track_reward_idx].weight
        ang_track_reward_sum = (
            reward._episode_sums["track_ang_vel_z_exp"][env_ids] / env.max_episode_length_s
        )
        ang_track_reward_idx = reward._term_names.index("track_ang_vel_z_exp")
        ang_track_reward_weight = reward._term_cfgs[ang_track_reward_idx].weight

        passed_env_ids = env_ids[
            (lin_track_reward_sum > lin_track_reward_weight * 0.8)
            & (ang_track_reward_sum > ang_track_reward_weight * 0.7)
        ]

    if passed_env_ids.numel() > 0:
        # Update the linear velocity ranges for the environments that performed well
        vel_cmd.lin_vel_x_ranges[passed_env_ids] = torch.clamp(
            vel_cmd.lin_vel_x_ranges[passed_env_ids] + deltas[0],
            max_curriculum[0][0],
            max_curriculum[0][1]
        )
        vel_cmd.lin_vel_y_ranges[passed_env_ids] = torch.clamp(
            vel_cmd.lin_vel_y_ranges[passed_env_ids] + deltas[1],
            max_curriculum[1][0],
            max_curriculum[1][1]
        )
        vel_cmd.ang_vel_z_ranges[passed_env_ids] = torch.clamp(
            vel_cmd.ang_vel_z_ranges[passed_env_ids] + deltas[2],
            max_curriculum[2][0],
            max_curriculum[2][1]
        )
    if failed_env_ids.numel() > 0:
        lin_vel_x_ranges = vel_cmd.lin_vel_x_ranges[failed_env_ids] - 2 * deltas[0]
        lin_vel_y_ranges = vel_cmd.lin_vel_y_ranges[failed_env_ids] - 2 * deltas[1]
        ang_vel_z_ranges = vel_cmd.ang_vel_z_ranges[failed_env_ids] - 2 * deltas[2]
        vel_cmd.lin_vel_x_ranges[failed_env_ids, 0] = torch.clamp(
            lin_vel_x_ranges[:, 0],
            max=vel_cmd.cfg.ranges.lin_vel_x[0],
        )
        vel_cmd.lin_vel_x_ranges[failed_env_ids, 1] = torch.clamp(
            lin_vel_x_ranges[:, 1],
            min=vel_cmd.cfg.ranges.lin_vel_x[1],
        )
        vel_cmd.lin_vel_y_ranges[failed_env_ids, 0] = torch.clamp(
            lin_vel_y_ranges[:, 0],
            max=vel_cmd.cfg.ranges.lin_vel_y[0],
        )
        vel_cmd.lin_vel_y_ranges[failed_env_ids, 1] = torch.clamp(
            lin_vel_y_ranges[:, 1],
            min=vel_cmd.cfg.ranges.lin_vel_y[1],
        )
        vel_cmd.ang_vel_z_ranges[failed_env_ids, 0] = torch.clamp(
            ang_vel_z_ranges[:, 0],
            max=vel_cmd.cfg.ranges.ang_vel_z[0],
        )
        vel_cmd.ang_vel_z_ranges[failed_env_ids, 1] = torch.clamp(
            ang_vel_z_ranges[:, 1],
            min=vel_cmd.cfg.ranges.ang_vel_z[1],
        )

    env.curriculum_manager._curriculum_state.update({
        "avg_vel_lin_x": torch.mean(vel_cmd.lin_vel_x_ranges[env.vel_update_ids, 1]),
        "avg_vel_lin_y": torch.mean(vel_cmd.lin_vel_y_ranges[env.vel_update_ids, 1]),
        "avg_vel_ang_z": torch.mean(vel_cmd.ang_vel_z_ranges[env.vel_update_ids, 1])
    })


def command_sampling_weights(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    reward: RewardManager = env.reward_manager
    tracking_key_points_exp_reward_sum = (
        reward._episode_sums["tracking_key_points_exp"][env_ids] / env.max_episode_length_s
    )
    tracking_key_points_exp_reward_idx = reward._term_names.index("tracking_key_points_w_exp")
    tracking_key_points_exp_reward_weight = reward._term_cfgs[tracking_key_points_exp_reward_idx].weight

    tracking_key_points_w_exp_reward_sum = (
        reward._episode_sums["tracking_key_points_w_exp"][env_ids] / env.max_episode_length_s
    )
    tracking_key_points_w_exp_reward_idx = reward._term_names.index("tracking_key_points_w_exp")
    tracking_key_points_w_exp_reward_weight = reward._term_cfgs[tracking_key_points_w_exp_reward_idx].weight

    episode_length = env.episode_length_buf[env_ids]
    current_weights = env.command_manager.get_term("motion_tracking").weights.clone()
    index_offset = env.command_manager.get_term("motion_tracking").index_offset.clone()[env_ids]
    dataset_length = env.command_manager.get_term("motion_tracking").dataset_length

    mask_success = (
        (tracking_key_points_exp_reward_sum > tracking_key_points_exp_reward_weight * 0.8)
        & (tracking_key_points_w_exp_reward_sum > tracking_key_points_w_exp_reward_weight * 0.9)
    )
    mask_failure = (
        (tracking_key_points_exp_reward_sum < tracking_key_points_exp_reward_weight * 0.8)
        | (tracking_key_points_w_exp_reward_sum < tracking_key_points_w_exp_reward_weight * 0.9)
    )

    if mask_success.any():
        offsets = index_offset[mask_success]
        lengths = episode_length[mask_success] // 2
        if lengths.numel() > 0:
            max_len = torch.max(lengths)
            steps = torch.arange(max_len, device=env.device)
            all_indices = offsets.unsqueeze(1) + steps.unsqueeze(0)
            valid_mask = steps.unsqueeze(0) < lengths.unsqueeze(1)
            indices_to_update = all_indices[valid_mask]
            indices_to_update %= dataset_length
            current_weights[indices_to_update] -= 0.05

    if mask_success.any():
        successful_lengths = episode_length[mask_success]

        mask_full_length = successful_lengths >= env.max_episode_length
        mask_early = successful_lengths < env.max_episode_length

        all_successful_offsets = index_offset[mask_success]
        offsets_full = all_successful_offsets[mask_full_length]
        lengths_full = successful_lengths[mask_full_length]

        if lengths_full.numel() > 0:
            max_l = torch.max(lengths_full)
            steps = torch.arange(max_l, device=env.device)
            all_indices = offsets_full.unsqueeze(1) + steps.unsqueeze(0)
            valid_mask = steps.unsqueeze(0) < lengths_full.unsqueeze(1)
            indices_to_update = all_indices[valid_mask]
            indices_to_update %= dataset_length
            current_weights[indices_to_update] -= 0.05

        offsets_early = all_successful_offsets[mask_early]
        lengths_early = successful_lengths[mask_early]

        if lengths_early.numel() > 0:
            max_l = torch.max(lengths_early)
            steps = torch.arange(max_l, device=env.device)

            all_indices = offsets_early.unsqueeze(1) + steps.unsqueeze(0)
            valid_mask = steps.unsqueeze(0) < lengths_early.unsqueeze(1)

            lengths_part1 = (lengths_early // 3).unsqueeze(1)
            lengths_part2 = ((lengths_early * 2) // 3).unsqueeze(1)

            part1_mask = (steps.unsqueeze(0) < lengths_part1) & valid_mask
            indices_part1 = all_indices[part1_mask]
            indices_part1 %= dataset_length
            current_weights[indices_part1] -= 0.05

            part3_mask = (steps.unsqueeze(0) >= lengths_part2) & valid_mask
            indices_part3 = all_indices[part3_mask]
            indices_part3 %= dataset_length
            current_weights[indices_part3] += 0.1

    if mask_failure.any():
        offsets = index_offset[mask_failure]
        lengths = episode_length[mask_failure]
        if lengths.numel() > 0:
            max_len = torch.max(lengths)
            steps = torch.arange(max_len, device=env.device)
            all_indices = offsets.unsqueeze(1) + steps.unsqueeze(0)
            valid_mask = steps.unsqueeze(0) < lengths.unsqueeze(1)
            indices_to_update = all_indices[valid_mask]
            indices_to_update %= dataset_length
            current_weights[indices_to_update] += 0.1

    current_weights = torch.clamp(current_weights, min=0.05, max=1.0)

    env.command_manager.get_term("motion_tracking").update_sampling_weights(current_weights)

    return torch.mean(current_weights)
