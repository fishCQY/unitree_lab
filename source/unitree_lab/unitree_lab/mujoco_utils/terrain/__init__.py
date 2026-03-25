"""Terrain generation module for MuJoCo sim2sim."""

from .generator import MujocoTerrainGenerator, TerrainConfig
from .xml_generation import create_terrain_xml, create_robot_with_terrain_xml
from .setup import setup_terrain_env, setup_terrain_data_in_model, get_spawn_position

__all__ = [
    "MujocoTerrainGenerator",
    "TerrainConfig",
    "create_terrain_xml",
    "create_robot_with_terrain_xml",
    "setup_terrain_env",
    "setup_terrain_data_in_model",
    "get_spawn_position",
]
