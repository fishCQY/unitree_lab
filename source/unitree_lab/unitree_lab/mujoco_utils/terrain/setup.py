"""Terrain setup utilities for MuJoCo simulation."""

from __future__ import annotations

import os
import random
import time
from typing import TYPE_CHECKING

import mujoco

from ..logging import logger
from .generator import MujocoTerrainGenerator
from .xml_generation import create_robot_with_terrain_xml, create_terrain_xml

if TYPE_CHECKING:
    from typing import Any


def setup_terrain_env(
    cfg: Any | dict | None,
    video_file: str,
    mujoco_model_path: str,
) -> tuple[str, MujocoTerrainGenerator | None, str | None]:
    """Set up terrain environment for MuJoCo simulation.

    Args:
        cfg: Terrain configuration. If None, returns flat terrain.
        video_file: Path to output video file.
        mujoco_model_path: Path to robot MuJoCo XML file.

    Returns:
        Tuple of (final_model_path, terrain_generator, temp_dir).
    """
    if cfg is None:
        logger.debug("No terrain config provided. Using flat terrain.")
        return mujoco_model_path, None, None

    try:
        generator = MujocoTerrainGenerator(cfg=cfg, num_rows=10, num_cols=10)
    except (ValueError, AttributeError, TypeError) as e:
        logger.warning(f"Could not initialize terrain generator: {e}. Running without custom terrain.")
        return mujoco_model_path, None, None

    logger.debug("Terrain generation enabled.")

    if video_file:
        output_dir = os.path.dirname(os.path.abspath(video_file))
    else:
        output_dir = os.path.abspath("mujoco_eval_outputs")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    run_id = f"terrain_run_{os.getpid()}_{int(time.time() * 1000) % 1000000}_{random.randint(0, 9999)}"
    temp_dir = os.path.join(output_dir, run_id)
    os.makedirs(temp_dir, exist_ok=True)

    logger.debug(f"Generating terrain artifacts in: {temp_dir}")

    generator.generate_all_terrains(difficulty=0.5)

    patch_pixels = generator.pixels_per_patch_x
    patch_length = generator.patch_size_x

    terrain_xml_content = create_terrain_xml(grid_size=10, patch_pixels=patch_pixels, patch_length=patch_length)

    merged_xml_path = os.path.join(temp_dir, "merged_robot_terrain.xml")
    create_robot_with_terrain_xml(mujoco_model_path, terrain_xml_content, merged_xml_path)
    final_model_path = merged_xml_path
    logger.debug(f"Generated merged XML at: {merged_xml_path}")

    return final_model_path, generator, temp_dir


def setup_terrain_data_in_model(
    model: mujoco.MjModel,
    generator: MujocoTerrainGenerator | None,
) -> None:
    """Assign generated terrain data to MuJoCo model hfields."""
    if not generator:
        return

    for i in range(generator.num_rows):
        for j in range(generator.num_cols):
            hfield_name = f"terrain_{i}_{j}"
            try:
                hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)

                if (i, j) in generator.generated_terrains:
                    height_data = generator.generated_terrains[(i, j)].flatten()
                    hfield_adr = int(model.hfield_adr[hfield_id])
                    hfield_size = len(height_data)
                    end_idx = min(hfield_adr + hfield_size, len(model.hfield_data))
                    data_to_assign = height_data[: end_idx - hfield_adr] / 10.0
                    model.hfield_data[hfield_adr:end_idx] = data_to_assign
            except Exception as e:
                logger.warning(f"Could not assign data to {hfield_name}: {e}")


def get_spawn_position(
    generator: MujocoTerrainGenerator | None,
    mujoco_model_path: str,
    grid_size: int = 10,
) -> tuple[float, float, float]:
    """Calculate spawn position based on terrain."""
    spawn_offset = 0.1

    if generator:
        spawn_r = random.randint(4, 5)
        spawn_c = random.randint(4, 5)
        patch_dim = generator.patch_size_x
        pos_x = (spawn_r - (grid_size - 1) / 2) * patch_dim
        pos_y = (spawn_c - (grid_size - 1) / 2) * patch_dim
        terrain_height = generator.get_spawn_height(spawn_r, spawn_c, x_offset=0, y_offset=0)
        spawn_z = terrain_height + 0.5 + spawn_offset
        logger.debug(
            f"Spawning at patch ({spawn_r}, {spawn_c}) -> "
            f"World: ({pos_x:.2f}, {pos_y:.2f}, {spawn_z:.2f})"
        )
        return pos_x, pos_y, spawn_z
    else:
        return 0.0, 0.0, 0.5
