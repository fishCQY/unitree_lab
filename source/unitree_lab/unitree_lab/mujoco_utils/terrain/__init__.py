"""Terrain generation module for MuJoCo sim2sim."""

from .generator import MujocoTerrainGenerator, TerrainConfig
from .xml_generation import create_terrain_xml, create_robot_with_terrain_xml

__all__ = [
    "MujocoTerrainGenerator",
    "TerrainConfig",
    "create_terrain_xml",
    "create_robot_with_terrain_xml",
]
