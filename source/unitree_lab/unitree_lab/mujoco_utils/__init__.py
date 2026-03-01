"""MuJoCo Sim2Sim utilities for IsaacLab policy evaluation.

This module provides tools to:
1. Load ONNX policies with metadata
2. Build observations matching IsaacLab semantics
3. Control robots with PD in MuJoCo
4. Generate aligned terrains
5. Evaluate policies in batch

Architecture Overview:
    - core/: ONNX utils, XML parsing, physics helpers
    - simulation/: Base simulator and observation builder
    - sensors/: Height scanner, contact detection
    - terrain/: Terrain generation matching IsaacLab
    - evaluation/: Batch evaluation framework

Usage:
    ```python
    from unitree_lab.mujoco_utils import BaseMujocoSimulator
    
    # Create simulator
    simulator = BaseMujocoSimulator(
        xml_path="robot.xml",
        onnx_path="policy.onnx",
    )
    
    # Reset and run
    obs = simulator.reset()
    for _ in range(1000):
        obs, info = simulator.step()
    ```
"""

from .core.onnx_utils import get_onnx_config, load_onnx_model, OnnxInference, OnnxConfig
from .core.physics import pd_control, pd_control_velocity, apply_onnx_physics_params
from .core.xml_parsing import (
    parse_actuators_from_xml,
    parse_joints_from_xml,
    build_joint_mapping,
    get_actuator_names,
)
from .simulation.base_simulator import BaseMujocoSimulator
from .simulation.observation_builder import ObservationBuilder
from .sensors.height_scanner import HeightScanner
from .sensors.contact_detector import ContactDetector
from .terrain.generator import MujocoTerrainGenerator, TerrainConfig
from .terrain.xml_generation import create_terrain_xml, create_robot_with_terrain_xml
from .evaluation.eval_task import EvalTask, LocomotionEvalTask, get_eval_task, list_eval_tasks
from .evaluation.batch_evaluator import BatchEvaluator
from .evaluation.metrics import compute_locomotion_metrics, LocomotionMetrics

__all__ = [
    # Core - ONNX
    "get_onnx_config",
    "load_onnx_model",
    "OnnxInference",
    "OnnxConfig",
    # Core - Physics
    "pd_control",
    "pd_control_velocity",
    "apply_onnx_physics_params",
    # Core - XML
    "parse_actuators_from_xml",
    "parse_joints_from_xml",
    "build_joint_mapping",
    "get_actuator_names",
    # Simulation
    "BaseMujocoSimulator",
    "ObservationBuilder",
    # Sensors
    "HeightScanner",
    "ContactDetector",
    # Terrain
    "MujocoTerrainGenerator",
    "TerrainConfig",
    "create_terrain_xml",
    "create_robot_with_terrain_xml",
    # Evaluation
    "EvalTask",
    "LocomotionEvalTask",
    "get_eval_task",
    "list_eval_tasks",
    "BatchEvaluator",
    "compute_locomotion_metrics",
    "LocomotionMetrics",
]
