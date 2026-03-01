"""XML parsing utilities for MuJoCo models.

This module handles:
1. Parsing actuator definitions from MuJoCo XML
2. Extracting joint information and limits
3. Building joint name mappings between ONNX and MuJoCo
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ActuatorInfo:
    """Information about a MuJoCo actuator."""
    name: str
    joint: str
    gear: float = 1.0
    ctrl_range: tuple[float, float] = (-1.0, 1.0)
    kp: float = 0.0
    kv: float = 0.0


@dataclass
class JointInfo:
    """Information about a MuJoCo joint."""
    name: str
    type: str = "hinge"  # hinge, slide, ball, free
    axis: tuple[float, float, float] = (0, 0, 1)
    range: tuple[float, float] | None = None
    damping: float = 0.0
    armature: float = 0.0
    stiffness: float = 0.0


def parse_actuators_from_xml(xml_path: str | Path) -> list[ActuatorInfo]:
    """Parse actuator definitions from MuJoCo XML.
    
    This extracts the actuator order which must match ONNX action output order.
    
    Args:
        xml_path: Path to MuJoCo XML file
        
    Returns:
        List of ActuatorInfo in XML definition order
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    actuators = []
    
    # Find actuator section
    actuator_elem = root.find("actuator")
    if actuator_elem is None:
        return actuators
    
    # Parse each actuator type
    for elem in actuator_elem:
        if elem.tag in ["motor", "position", "velocity", "general"]:
            info = ActuatorInfo(
                name=elem.get("name", ""),
                joint=elem.get("joint", ""),
            )
            
            # Parse gear
            if elem.get("gear"):
                info.gear = float(elem.get("gear", "1").split()[0])
            
            # Parse control range
            if elem.get("ctrlrange"):
                parts = elem.get("ctrlrange", "-1 1").split()
                info.ctrl_range = (float(parts[0]), float(parts[1]))
            
            # Parse gains for position/velocity actuators
            if elem.get("kp"):
                info.kp = float(elem.get("kp", "0"))
            if elem.get("kv"):
                info.kv = float(elem.get("kv", "0"))
            
            actuators.append(info)
    
    return actuators


def parse_joints_from_xml(xml_path: str | Path) -> list[JointInfo]:
    """Parse joint definitions from MuJoCo XML.
    
    Args:
        xml_path: Path to MuJoCo XML file
        
    Returns:
        List of JointInfo in XML definition order
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    joints = []
    
    def _parse_body(body_elem):
        """Recursively parse joints from body elements."""
        for elem in body_elem:
            if elem.tag == "joint":
                info = JointInfo(
                    name=elem.get("name", ""),
                    type=elem.get("type", "hinge"),
                )
                
                # Parse axis
                if elem.get("axis"):
                    parts = elem.get("axis", "0 0 1").split()
                    info.axis = (float(parts[0]), float(parts[1]), float(parts[2]))
                
                # Parse range
                if elem.get("range"):
                    parts = elem.get("range", "").split()
                    if len(parts) >= 2:
                        info.range = (float(parts[0]), float(parts[1]))
                
                # Parse dynamics
                info.damping = float(elem.get("damping", "0"))
                info.armature = float(elem.get("armature", "0"))
                info.stiffness = float(elem.get("stiffness", "0"))
                
                joints.append(info)
            
            elif elem.tag == "body":
                _parse_body(elem)
    
    # Start from worldbody
    worldbody = root.find("worldbody")
    if worldbody is not None:
        _parse_body(worldbody)
    
    return joints


def build_joint_mapping(
    onnx_joint_names: list[str],
    xml_actuator_names: list[str],
) -> list[int]:
    """Build mapping from XML actuator order to ONNX joint order.
    
    This is critical for:
    1. Mapping ONNX action outputs to MuJoCo actuators
    2. Mapping PD gains from ONNX metadata to MuJoCo joints
    
    Args:
        onnx_joint_names: Joint names from ONNX metadata (action order)
        xml_actuator_names: Actuator names from MuJoCo XML
        
    Returns:
        Mapping indices: xml_actuator_names[i] corresponds to onnx_joint_names[mapping[i]]
        
    Raises:
        ValueError: If joint names don't match
    """
    mapping = []
    
    for xml_name in xml_actuator_names:
        # Try exact match first
        if xml_name in onnx_joint_names:
            mapping.append(onnx_joint_names.index(xml_name))
            continue
        
        # Try partial match (handle naming conventions)
        found = False
        for i, onnx_name in enumerate(onnx_joint_names):
            # Common patterns: "joint_name" vs "joint_name_motor"
            xml_base = xml_name.replace("_motor", "")
            onnx_base = onnx_name.replace("_motor", "")

            # Common patterns:
            # - "left_hip_pitch" (XML actuator) vs "left_hip_pitch_joint" (ONNX joint name)
            # - "left_hip_pitch_joint" (XML actuator) vs "left_hip_pitch" (ONNX name)
            # - namespace paths: "robot/left_hip_pitch" vs "left_hip_pitch_joint"
            xml_leaf = xml_base.split("/")[-1]
            onnx_leaf = onnx_base.split("/")[-1]

            if (
                xml_base == onnx_base
                or xml_base == onnx_base.replace("_joint", "")
                or xml_base + "_joint" == onnx_base
                or xml_leaf == onnx_leaf
                or xml_leaf == onnx_leaf.replace("_joint", "")
                or xml_leaf + "_joint" == onnx_leaf
            ):
                mapping.append(i)
                found = True
                break
        
        if not found:
            raise ValueError(
                f"Could not find ONNX joint for XML actuator '{xml_name}'.\n"
                f"ONNX joints: {onnx_joint_names}\n"
                f"XML actuators: {xml_actuator_names}"
            )
    
    # Guardrail: mapping must be a permutation (no duplicates), otherwise the policy will
    # drive multiple actuators with the same ONNX action index (and leave others unused),
    # which typically manifests as violent twitching in sim2sim.
    if len(set(mapping)) != len(mapping):
        # Show collisions for quick debugging.
        inv: dict[int, list[str]] = {}
        for xml_name, idx in zip(xml_actuator_names, mapping):
            inv.setdefault(int(idx), []).append(xml_name)
        collisions = {k: v for k, v in inv.items() if len(v) > 1}
        raise ValueError(
            "Ambiguous joint mapping (duplicate ONNX indices detected). "
            f"Collisions: {collisions}\n"
            f"ONNX joints ({len(onnx_joint_names)}): {onnx_joint_names}\n"
            f"XML actuators ({len(xml_actuator_names)}): {xml_actuator_names}"
        )

    return mapping


def get_actuator_names(xml_path: str | Path) -> list[str]:
    """Get list of actuator names from XML.
    
    Args:
        xml_path: Path to MuJoCo XML
        
    Returns:
        List of actuator names in order
    """
    actuators = parse_actuators_from_xml(xml_path)
    return [a.name for a in actuators]


def get_ctrl_ranges(xml_path: str | Path) -> list[tuple[float, float]]:
    """Get control ranges (tau limits) from XML.
    
    Args:
        xml_path: Path to MuJoCo XML
        
    Returns:
        List of (min, max) control ranges
    """
    actuators = parse_actuators_from_xml(xml_path)
    return [a.ctrl_range for a in actuators]
