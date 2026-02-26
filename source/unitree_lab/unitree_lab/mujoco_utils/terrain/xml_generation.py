"""XML generation utilities for MuJoCo terrains.

This module creates MuJoCo XML elements for:
1. Heightfield assets
2. Ground geoms with heightfield
3. Combined robot + terrain models
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .generator import MujocoTerrainGenerator, TerrainConfig


def create_terrain_xml(
    terrain_generator: "MujocoTerrainGenerator",
    terrain_name: str = "terrain",
) -> str:
    """Create MuJoCo XML snippet for heightfield terrain.
    
    Args:
        terrain_generator: Terrain generator with generated heightfield
        terrain_name: Name for terrain elements
        
    Returns:
        XML string for terrain asset and geom
    """
    config = terrain_generator.config
    
    # Asset definition
    asset_xml = f'''
    <hfield name="{terrain_name}_hfield" 
            nrow="{terrain_generator.ny}" 
            ncol="{terrain_generator.nx}" 
            size="{config.size[0]/2} {config.size[1]/2} 1 0.01"/>
    '''
    
    # Geom definition
    geom_xml = f'''
    <geom name="{terrain_name}" 
          type="hfield" 
          hfield="{terrain_name}_hfield" 
          pos="0 0 0"
          friction="1 0.005 0.0001"
          contype="1" 
          conaffinity="1"/>
    '''
    
    return asset_xml, geom_xml


def create_robot_with_terrain_xml(
    robot_xml_path: str | Path,
    terrain_generator: "MujocoTerrainGenerator" | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Create combined robot + terrain XML.
    
    Args:
        robot_xml_path: Path to robot MuJoCo XML
        terrain_generator: Optional terrain generator
        output_path: Optional output path for combined XML
        
    Returns:
        Path to combined XML file
    """
    robot_xml_path = Path(robot_xml_path)
    
    # Parse robot XML
    tree = ET.parse(robot_xml_path)
    root = tree.getroot()
    
    if terrain_generator is not None:
        config = terrain_generator.config
        
        # Ensure terrain is generated
        if terrain_generator.heightfield is None:
            terrain_generator.generate()
        
        # Find or create asset section
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        
        # Add heightfield asset
        hfield = ET.SubElement(asset, "hfield")
        hfield.set("name", "terrain_hfield")
        hfield.set("nrow", str(terrain_generator.ny))
        hfield.set("ncol", str(terrain_generator.nx))
        hfield.set("size", f"{config.size[0]/2} {config.size[1]/2} 1 0.01")
        
        # Find worldbody
        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")
        
        # Remove existing ground plane if any
        for geom in worldbody.findall("geom"):
            if geom.get("type") == "plane" or "ground" in geom.get("name", "").lower():
                worldbody.remove(geom)
        
        # Add terrain geom
        terrain_geom = ET.SubElement(worldbody, "geom")
        terrain_geom.set("name", "terrain")
        terrain_geom.set("type", "hfield")
        terrain_geom.set("hfield", "terrain_hfield")
        terrain_geom.set("pos", "0 0 0")
        terrain_geom.set("friction", "1 0.005 0.0001")
        terrain_geom.set("contype", "1")
        terrain_geom.set("conaffinity", "1")
    
    # Write to output
    if output_path is None:
        # Create temp file
        fd, output_path = tempfile.mkstemp(suffix=".xml", prefix="robot_terrain_")
    
    output_path = Path(output_path)
    tree.write(output_path, encoding="unicode")
    
    return str(output_path)


def setup_terrain_data_in_model(
    model: "mujoco.MjModel",
    terrain_generator: "MujocoTerrainGenerator",
    hfield_name: str = "terrain_hfield",
) -> None:
    """Write heightfield data into MuJoCo model.
    
    This must be called after loading the model but before simulation.
    
    Args:
        model: MuJoCo model
        terrain_generator: Terrain generator with generated heightfield
        hfield_name: Name of heightfield asset in model
    """
    import mujoco as mj
    
    # Get heightfield data
    hfield_data = terrain_generator.get_mujoco_heightfield_data()
    
    # Find heightfield by name
    hfield_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_HFIELD, hfield_name)
    
    if hfield_id < 0:
        raise ValueError(f"Heightfield '{hfield_name}' not found in model")
    
    # Get heightfield address in data array
    hfield_adr = model.hfield_adr[hfield_id]
    hfield_size = model.hfield_nrow[hfield_id] * model.hfield_ncol[hfield_id]
    
    # Convert height values in meters -> MuJoCo hfield_data in [0,1] using the model's hfield size.
    # MuJoCo uses: height(x,y) = base + size_z * hfield_data(x,y).
    try:
        size_z = float(model.hfield_size[hfield_id, 2])
        base_z = float(model.hfield_size[hfield_id, 3])
    except Exception:
        size_z = 0.0
        base_z = 0.0

    if size_z > 1e-9:
        # First, fit heights into the representable range [base_z, base_z + size_z]
        # to avoid hard clipping (which would flatten negative roughness).
        heights = np.asarray(hfield_data, dtype=np.float32)
        min_h = float(np.min(heights)) if heights.size else 0.0
        max_h = float(np.max(heights)) if heights.size else 0.0

        # Shift up if below base (common for symmetric noise ranges like [-a, a])
        if min_h < base_z:
            heights = heights + (base_z - min_h)
            max_h = float(np.max(heights)) if heights.size else base_z

        # Scale down if above max
        max_allowed = base_z + size_z
        if max_h > max_allowed and max_h > base_z + 1e-9:
            scale = (max_allowed - base_z) / (max_h - base_z)
            heights = base_z + (heights - base_z) * float(scale)

        normalized = (heights - base_z) / size_z
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        # Fallback (shouldn't happen): normalize by min/max to avoid crashing.
        if hfield_data.max() > hfield_data.min():
            normalized = (hfield_data - hfield_data.min()) / (hfield_data.max() - hfield_data.min())
        else:
            normalized = np.zeros_like(hfield_data)
    
    # Write to model
    model.hfield_data[hfield_adr:hfield_adr + hfield_size] = normalized


def create_flat_ground_xml() -> tuple[str, str]:
    """Create XML for flat ground plane.
    
    Returns:
        Tuple of (asset_xml, geom_xml)
    """
    asset_xml = '''
    <texture name="groundplane" type="2d" builtin="checker" 
             rgb1="0.25 0.26 0.25" rgb2="0.22 0.22 0.22" 
             width="100" height="100"/>
    <material name="groundplane" texture="groundplane" texuniform="true" 
              texrepeat="5 5" reflectance="0.2"/>
    '''
    
    geom_xml = '''
    <geom name="ground" type="plane" size="100 100 0.1" 
          material="groundplane" friction="1 0.005 0.0001"
          contype="1" conaffinity="1"/>
    '''
    
    return asset_xml, geom_xml
