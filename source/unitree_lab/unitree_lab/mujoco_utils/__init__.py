"""MuJoCo Sim2Sim utilities for IsaacLab policy evaluation.

This module provides tools to:
1. Load ONNX policies with metadata
2. Build observations matching IsaacLab semantics
3. Control robots with PD in MuJoCo
4. Generate aligned terrains
5. Evaluate policies in batch
6. Visualize simulation results

Architecture Overview:
    - core/: ONNX utils, XML parsing, physics helpers, joint mapping, math utils
    - simulation/: Base simulator and observation builder
    - sensors/: Height scanner, contact detection
    - terrain/: Terrain generation and setup matching IsaacLab
    - evaluation/: Batch evaluation framework
    - visualization/: Torque, exteroception and combined visualization panels
"""

from .logging import logger, set_log_level

from .core.onnx_utils import (
    get_onnx_config,
    get_onnx_config_dataclass,
    detect_policy_type,
    init_hidden_states,
    load_onnx_model,
    OnnxInference,
    OnnxConfig,
)
from .core.physics import (
    pd_control,
    pd_control_velocity,
    get_tau_limit,
    apply_onnx_physics_params,
)
from .core.xml_parsing import (
    parse_actuators_from_xml,
    parse_joints_from_xml,
    build_joint_mapping,
    get_actuator_names,
)
from .core.joint_mapping import create_joint_mapping, model_to_mujoco, mujoco_to_model, create_joint_mapping_index
from .core.math_utils import (
    quat_rotate_inverse_np,
    quat_rotate_forward_np,
    quaternion_multiply,
    quaternion_inverse,
    yaw_from_quat,
    yaw_quaternion,
    subtract_frame_transforms_np,
    se3_inverse,
    apply_se3_transform,
)
from .simulation.base_simulator import BaseMujocoSimulator, SimulatorWithRunLoop
from .simulation.observation_builder import ObservationBuilder
from .sensors.height_scanner import HeightScanner
from .sensors.contact_detector import ContactDetector
from .terrain.generator import MujocoTerrainGenerator, TerrainConfig
from .terrain.xml_generation import create_terrain_xml, create_robot_with_terrain_xml
from .terrain.setup import setup_terrain_env, setup_terrain_data_in_model, get_spawn_position
from .evaluation.eval_task import (
    EVAL_TASKS,
    EVAL_TASKS_BABY,
    EVAL_TASKS_DEFAULT,
    EVAL_TASKS_FULL,
    EvalTask,
    get_eval_task,
    list_eval_tasks,
)
from .evaluation.batch_evaluator import BatchEvalConfig, BatchEvalResult, run_batch_eval
from .evaluation.metrics import (
    EvalResult,
    MetricsCollector,
    MetricsConfig,
    is_fallen,
    compute_locomotion_metrics,
    LocomotionMetrics,
)
from .evaluation.mujoco_eval_cfg import BaseMuJoCoEvalCfg
from .visualization import (
    create_combined_visualization,
    create_exteroception_visualization,
    create_torque_visualization,
)

__all__ = [
    # Logging
    "logger",
    "set_log_level",
    # Core - ONNX
    "get_onnx_config",
    "get_onnx_config_dataclass",
    "detect_policy_type",
    "init_hidden_states",
    "load_onnx_model",
    "OnnxInference",
    "OnnxConfig",
    # Core - Physics
    "pd_control",
    "pd_control_velocity",
    "get_tau_limit",
    "apply_onnx_physics_params",
    # Core - XML
    "parse_actuators_from_xml",
    "parse_joints_from_xml",
    "build_joint_mapping",
    "get_actuator_names",
    # Core - Joint Mapping
    "create_joint_mapping",
    "model_to_mujoco",
    "mujoco_to_model",
    "create_joint_mapping_index",
    # Core - Math
    "quat_rotate_inverse_np",
    "quat_rotate_forward_np",
    "quaternion_multiply",
    "quaternion_inverse",
    "yaw_from_quat",
    "yaw_quaternion",
    "subtract_frame_transforms_np",
    "se3_inverse",
    "apply_se3_transform",
    # Simulation
    "BaseMujocoSimulator",
    "SimulatorWithRunLoop",
    "ObservationBuilder",
    # Sensors
    "HeightScanner",
    "ContactDetector",
    # Terrain
    "MujocoTerrainGenerator",
    "TerrainConfig",
    "create_terrain_xml",
    "create_robot_with_terrain_xml",
    "setup_terrain_env",
    "setup_terrain_data_in_model",
    "get_spawn_position",
    # Evaluation
    "EvalTask",
    "EVAL_TASKS",
    "EVAL_TASKS_DEFAULT",
    "EVAL_TASKS_FULL",
    "EVAL_TASKS_BABY",
    "get_eval_task",
    "list_eval_tasks",
    "BatchEvalConfig",
    "BatchEvalResult",
    "run_batch_eval",
    "EvalResult",
    "MetricsCollector",
    "MetricsConfig",
    "is_fallen",
    "compute_locomotion_metrics",
    "LocomotionMetrics",
    "BaseMuJoCoEvalCfg",
    # Visualization
    "create_combined_visualization",
    "create_torque_visualization",
    "create_exteroception_visualization",
]
