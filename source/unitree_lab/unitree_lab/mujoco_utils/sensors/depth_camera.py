# Copyright (c) 2024-2026, Light Robotics.
# All rights reserved.

"""Depth camera renderer for MuJoCo simulation.

Renders depth images from a camera mounted on the robot,
matching the IsaacLab raycaster camera output for sim2sim evaluation.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections import deque

import mujoco
import numpy as np

try:
    from unitree_lab.mujoco_utils.logging import logger
except ImportError:
    logger = logging.getLogger(__name__)

# Fallback camera parameters when ONNX metadata is not available.
# Prefer reading from ONNX metadata (depth_fovy, depth_render_width, etc.).
_DEFAULT_DEPTH_CAMERA_CFG = {
    "body_name": "depth_camera_link",
    "fovy": 60,
    "width": 48,
    "height": 30,
    "crop_left": 0,
    "min_range": 0.15,
    "max_range": 5.0,
    # D435 noise model matching IsaacLab training (depth_image_d435 in observations.py)
    "base_noise_std": 0.000,  # σ(0) = 5mm
    "distance_noise_scale": 0.00,  # σ(z) = base + scale * z
}

# MuJoCo camera orientation quaternion to convert from IsaacLab convention
# (X-forward, Y-left, Z-up) to MuJoCo camera convention (-Z forward, Y-up, X-right).
# Derived from: R_y(-90°) * R_z(-90°) → q = [0.5, 0.5, -0.5, -0.5]
_CAMERA_QUAT_ISAACLAB_TO_MUJOCO = "0.5 0.5 -0.5 -0.5"


def inject_depth_camera_into_xml(xml_path: str, output_path: str, cfg: dict | None = None) -> str:
    """Inject a MuJoCo <camera> element into the depth_camera_link body.

    Args:
        xml_path: Path to the original MuJoCo XML.
        output_path: Path to write the modified XML.
        cfg: Camera configuration dict (uses defaults if None).

    Returns:
        Path to the modified XML file.
    """
    cfg = cfg or _DEFAULT_DEPTH_CAMERA_CFG
    body_name = cfg.get("body_name", "depth_camera_link")
    fovy = cfg.get("fovy", _DEFAULT_DEPTH_CAMERA_CFG["fovy"])

    tree = ET.parse(xml_path)
    root = tree.getroot()

    found = False
    for body in root.iter("body"):
        if body.get("name") == body_name:
            existing = body.find("camera[@name='depth_cam']")
            if existing is None:
                cam_elem = ET.SubElement(body, "camera")
                cam_elem.set("name", "depth_cam")
                cam_elem.set("fovy", f"{fovy:.1f}")
                cam_elem.set("quat", _CAMERA_QUAT_ISAACLAB_TO_MUJOCO)
                logger.info(f"Injected MuJoCo camera 'depth_cam' into body '{body_name}' (fovy={fovy:.1f}°)")
            found = True
            break

    if not found:
        raise ValueError(
            f"Body '{body_name}' not found in {xml_path}. Depth camera requires a model with a depth_camera_link body."
        )

    tree.write(output_path, xml_declaration=True, encoding="unicode")
    return output_path


class DepthCameraRenderer:
    """Renders depth images from a MuJoCo camera, matching the IsaacLab D435 pipeline.

    Handles:
    - Rendering at the correct resolution (53×30)
    - Converting MuJoCo depth buffer to meters
    - Cropping left columns (53→48) to match real hardware blind spot
    - Clipping to valid range [min_range, max_range]
    - Reciprocal depth conversion and [0,1] normalization
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str = "depth_cam",
        cfg: dict | None = None,
        delay_steps: int = 0,
    ):
        """Initialize depth camera renderer.

        Args:
            model: Compiled MuJoCo model (must contain the named camera).
            camera_name: Name of the MuJoCo camera element.
            cfg: Camera configuration dict (uses defaults if None).
            delay_steps: Number of policy steps to delay the depth image output.
                Training uses delay_range=(0.03, 0.06) at 50Hz policy rate,
                corresponding to ~2 policy steps. Set 0 to disable.
        """
        self.cfg = cfg or _DEFAULT_DEPTH_CAMERA_CFG
        self.camera_name = camera_name

        self.render_width = int(self.cfg.get("width", _DEFAULT_DEPTH_CAMERA_CFG["width"]))
        self.render_height = int(self.cfg.get("height", _DEFAULT_DEPTH_CAMERA_CFG["height"]))
        self.crop_left = int(self.cfg.get("crop_left", _DEFAULT_DEPTH_CAMERA_CFG["crop_left"]))
        self.min_range = float(self.cfg.get("min_range", 0.15))
        self.max_range = float(self.cfg.get("max_range", 5.0))

        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError(f"Invalid depth render size: {self.render_width}x{self.render_height}")
        if self.crop_left < 0:
            raise ValueError(f"depth crop_left must be >= 0, got {self.crop_left}")
        if self.crop_left >= self.render_width:
            raise ValueError(
                f"depth crop_left ({self.crop_left}) must be < render_width ({self.render_width}) "
                "to keep a non-empty image."
            )

        self.output_width = self.render_width - self.crop_left
        self.output_height = self.render_height

        # D435 noise model: σ(z) = base_noise_std + distance_noise_scale * z
        self.base_noise_std = float(self.cfg.get("base_noise_std", _DEFAULT_DEPTH_CAMERA_CFG["base_noise_std"]))
        self.distance_noise_scale = float(
            self.cfg.get("distance_noise_scale", _DEFAULT_DEPTH_CAMERA_CFG["distance_noise_scale"])
        )

        # Frame delay buffer: training has 30-60ms latency at 50Hz → ~2 policy steps.
        # Buffer holds delay_steps+1 frames; returning buffer[0] gives the oldest.
        self.delay_steps = delay_steps
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=max(delay_steps + 1, 1))

        # Verify camera exists in model
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(
                f"Camera '{camera_name}' not found in MuJoCo model. "
                "Use inject_depth_camera_into_xml() before loading the model."
            )
        self._camera_id = cam_id

        # Create depth renderer
        self._renderer = mujoco.Renderer(model, height=self.render_height, width=self.render_width)
        self._renderer.enable_depth_rendering()

        logger.info(
            f"DepthCameraRenderer initialized: {self.render_width}×{self.render_height} "
            f"(output {self.output_width}×{self.output_height}), "
            f"range [{self.min_range}, {self.max_range}]m, reciprocal mode, "
            f"delay={delay_steps} policy steps"
        )

    def render(self, data: mujoco.MjData) -> np.ndarray:
        """Render depth image and process it to match the training pipeline.

        The returned image is delayed by ``delay_steps`` policy steps to
        approximate the training camera latency (30-60 ms).

        Args:
            data: Current MuJoCo simulation data.

        Returns:
            Depth image as float32 array of shape (height, width, 1) = (30, 48, 1),
            normalized to [0, 1] using reciprocal depth.
        """
        current_frame = self._render_current(data)
        self._frame_buffer.append(current_frame)
        return self._frame_buffer[0]

    def _render_current(self, data: mujoco.MjData) -> np.ndarray:
        """Render and process the current depth frame (no delay)."""
        self._renderer.update_scene(data, camera=self._camera_id)
        # MuJoCo Renderer.render() with depth enabled returns linear depth
        # in meters (scene units), NOT a normalized [0,1] buffer.
        depth_meters = self._renderer.render().copy()

        # Crop left columns
        if self.crop_left > 0:
            depth_meters = depth_meters[:, self.crop_left :]

        # Handle invalid values (inf/nan/<=0) and far-plane sky pixels.
        # Match IsaacLab reciprocal mode: invalid → max_range (no random range for simplicity),
        # then only clamp min — no max clip, so depths > max_range are preserved.
        invalid = np.isinf(depth_meters) | np.isnan(depth_meters) | (depth_meters <= 0)
        depth_meters[invalid] = self.max_range

        # Distance-dependent Gaussian noise matching IsaacLab D435 noise model:
        #   σ(z) = base_noise_std + distance_noise_scale * z
        # Applied to raw depth in meters (before reciprocal), same as training.
        if self.base_noise_std > 0 or self.distance_noise_scale > 0:
            noise_std = self.base_noise_std + self.distance_noise_scale * depth_meters
            depth_meters = depth_meters + np.random.randn(*depth_meters.shape).astype(np.float32) * noise_std

        depth_meters = np.clip(depth_meters, a_min=self.min_range, a_max=None)

        # Reciprocal depth + normalize to [0, 1]: min_range / depth
        # depth=min_range → 1.0 (closest), depth=max_range → min_range/max_range ≈ 0.03
        depth_normalized = (self.min_range / depth_meters).astype(np.float32)

        return depth_normalized[:, :, np.newaxis]

    def prefill(self, data: mujoco.MjData) -> None:
        """Pre-fill the frame buffer so delay is active from the first step."""
        frame = self._render_current(data)
        self._frame_buffer.clear()
        for _ in range(self.delay_steps + 1):
            self._frame_buffer.append(frame.copy())

    def reset(self) -> None:
        """Clear the frame delay buffer."""
        self._frame_buffer.clear()

    def cleanup(self):
        """Release renderer resources."""
        self._frame_buffer.clear()
        if self._renderer is not None:
            del self._renderer
            self._renderer = None
