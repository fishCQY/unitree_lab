# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from .motion_tracking_command import MotionTrackingCommand


@configclass
class MotionTrackingCommandCfg(CommandTermCfg):
    """Configuration for the motion tracking command generator."""

    class_type: type = MotionTrackingCommand

    asset_name: str = MISSING

    dataset_path: list[str] | str = MISSING

    exclude_prefix: str | None = None

    root_link_name: str = MISSING

    tracking_body_names: list[str] | None = None

    replay_dataset: bool = False

    use_world_frame: bool = False

    sequence_len: int = 1

    side_length: float = 0.1

    tracking_body_frame_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/tracking_body_frame"
    )
    key_points_visualizer: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(
        prim_path="/Visuals/Command/key_points"
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    tracking_body_frame_visualizer.markers["frame"].scale = (0.05, 0.05, 0.05)
