# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ray-cast camera sensor with noise/latency (ported from unitree_lab)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.sensors.ray_caster import RayCasterCamera

if TYPE_CHECKING:
    from .ray_caster_camera_cfg import NoiseRayCasterCameraCfg


class NoiseRayCasterCamera(RayCasterCamera):
    """Ray-caster camera with optional noise, dropout, range filtering and latency."""

    def __init__(self, cfg: "NoiseRayCasterCameraCfg"):
        super().__init__(cfg)
        self.cfg: NoiseRayCasterCameraCfg = cfg
        self.noise_cfg = cfg.noise_cfg

        self._latency_buffer: list[torch.Tensor] | None = None
        self._buffer_index: int = 0

    def _update_buffers_impl(self, env_ids: torch.Tensor):
        # Clean measurement
        super()._update_buffers_impl(env_ids)

        # Latency first (matches "real sensor then noise" intuition)
        if self.noise_cfg and self.noise_cfg.latency_steps > 0:
            self._apply_latency(env_ids)

        # Noise stack
        if self.noise_cfg is not None and self.noise_cfg.enable:
            self._apply_range_filtering(env_ids)
            self._apply_gaussian_noise(env_ids)
            self._apply_dropout(env_ids)

    # ---------------------------------------------------------------------
    # Latency
    # ---------------------------------------------------------------------

    def _initialize_latency_buffer(self):
        buffer_size = int(self.noise_cfg.latency_steps) + 1
        self._latency_buffer = [self._data.output["distance_to_image_plane"].clone() for _ in range(buffer_size)]
        self._buffer_index = 0

    def _apply_latency(self, env_ids: torch.Tensor):
        if self._latency_buffer is None:
            self._initialize_latency_buffer()

        current_data = self._data.output["distance_to_image_plane"].clone()
        delayed_data = self._latency_buffer[self._buffer_index].clone()

        self._latency_buffer[self._buffer_index] = current_data
        self._buffer_index = (self._buffer_index + 1) % (int(self.noise_cfg.latency_steps) + 1)

        self._data.output["distance_to_image_plane"][env_ids] = delayed_data[env_ids]

    # ---------------------------------------------------------------------
    # Noise
    # ---------------------------------------------------------------------

    def _apply_gaussian_noise(self, env_ids: torch.Tensor):
        if float(self.noise_cfg.gaussian_std) <= 0.0:
            return
        data = self._data.output["distance_to_image_plane"][env_ids]
        valid_mask = torch.isfinite(data)
        if valid_mask.any():
            noise = torch.randn_like(data) * float(self.noise_cfg.gaussian_std) + float(self.noise_cfg.gaussian_mean)
            data[valid_mask] += noise[valid_mask]
            data[valid_mask] = torch.clamp(data[valid_mask], min=float(self.noise_cfg.range_min), max=float(self.noise_cfg.range_max))
            self._data.output["distance_to_image_plane"][env_ids] = data

    def _apply_dropout(self, env_ids: torch.Tensor):
        if float(self.noise_cfg.dropout_prob) <= 0.0:
            return
        data = self._data.output["distance_to_image_plane"][env_ids]
        valid_mask = torch.isfinite(data)
        dropout_mask = (torch.rand_like(data) < float(self.noise_cfg.dropout_prob)) & valid_mask
        if dropout_mask.any():
            dropout_val = self._resolve_special_value(self.noise_cfg.invalid_value)
            data[dropout_mask] = dropout_val
            self._data.output["distance_to_image_plane"][env_ids] = data

    def _apply_range_filtering(self, env_ids: torch.Tensor):
        data = self._data.output["distance_to_image_plane"][env_ids]
        too_close = data < float(self.noise_cfg.range_min)
        too_far = data > float(self.noise_cfg.range_max)
        if bool(self.noise_cfg.handle_invalid_as_max):
            not_finite = ~torch.isfinite(data)
            invalid_mask = too_close | too_far | not_finite
        else:
            invalid_mask = too_close | too_far
        if invalid_mask.any():
            invalid_val = self._resolve_special_value(self.noise_cfg.invalid_value)
            data[invalid_mask] = invalid_val
            self._data.output["distance_to_image_plane"][env_ids] = data

    def _resolve_special_value(self, value: float | str) -> float:
        if isinstance(value, str):
            if value == "inf":
                return float("inf")
            if value == "max":
                return float(self.noise_cfg.range_max)
            if value == "min":
                return float(self.noise_cfg.range_min)
            raise ValueError(f"Unknown special value: {value}")
        return float(value)

