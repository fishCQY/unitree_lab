# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from unitree_lab.assets import ISAAC_ASSET_DIR
from unitree_lab.actuators import DelayedImplicitActuatorCfg


# Delay configuration
MIN_DELAY = 0
MAX_DELAY = 4

# Motor natural frequency and damping parameters
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0


def calculate_motor_impedance(armature: float, natural_freq: float = NATURAL_FREQ, damping_ratio: float = DAMPING_RATIO):
    """Calculate stiffness and damping based on armature and natural frequency."""
    stiffness = armature * natural_freq**2
    damping = 2.0 * damping_ratio * armature * natural_freq
    return stiffness, damping


# Motor parameters database with computed impedance values
MotorParameter = {
    "5020": {
        "armature": 0.003609725,
        "effort_limit": 25,
        "velocity_limit": 37,
        "friction": 0.0,
        "computed_stiffness": None,  # Will be calculated if needed
        "computed_damping": None,     # Will be calculated if needed
    },
    "7520_14": {
        "armature": 0.010177520,
        "effort_limit": 75,
        "velocity_limit": 32,
        "friction": 0.0,
        "computed_stiffness": None,
        "computed_damping": None,
    },
    "7520_16": {
        "armature": 0.013,
        "effort_limit": 88,
        "velocity_limit": 23,
        "friction": 0.0,
        "computed_stiffness": None,
        "computed_damping": None,
    },
    "7520_22": {
        "armature": 0.025101925,
        "effort_limit": 120,
        "velocity_limit": 20,
        "friction": 0.0,
        "computed_stiffness": None,
        "computed_damping": None,
    },
    "4010": {
        "armature": 0.00425,
        "effort_limit": 5,
        "velocity_limit": 22,
        "friction": 0.0,
        "computed_stiffness": None,
        "computed_damping": None,
    },
}

# Pre-compute impedance values for all motors
for motor_name, params in MotorParameter.items():
    stiffness, damping = calculate_motor_impedance(params["armature"])
    params["computed_stiffness"] = stiffness
    params["computed_damping"] = damping

# Build the G1 configuration first
UNITREE_G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Unitree/g1_29dof.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=10000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    # Actuators ordered to produce URDF joint order (body-part grouped):
    # left leg → right leg → waist → left arm → right arm
    actuators={
        "left_leg": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
            ],
            effort_limit_sim={
                "left_hip_pitch_joint": MotorParameter["7520_16"]["effort_limit"],
                "left_hip_roll_joint": MotorParameter["7520_22"]["effort_limit"],
                "left_hip_yaw_joint": MotorParameter["7520_16"]["effort_limit"],
                "left_knee_joint": MotorParameter["7520_22"]["effort_limit"],
            },
            velocity_limit_sim={
                "left_hip_pitch_joint": MotorParameter["7520_16"]["velocity_limit"],
                "left_hip_roll_joint": MotorParameter["7520_22"]["velocity_limit"],
                "left_hip_yaw_joint": MotorParameter["7520_16"]["velocity_limit"],
                "left_knee_joint": MotorParameter["7520_22"]["velocity_limit"],
            },
            stiffness={
                "left_hip_pitch_joint": MotorParameter["7520_16"]["computed_stiffness"],
                "left_hip_roll_joint": MotorParameter["7520_22"]["computed_stiffness"],
                "left_hip_yaw_joint": MotorParameter["7520_16"]["computed_stiffness"],
                "left_knee_joint": MotorParameter["7520_22"]["computed_stiffness"],
            },
            damping={
                "left_hip_pitch_joint": MotorParameter["7520_16"]["computed_damping"],
                "left_hip_roll_joint": MotorParameter["7520_22"]["computed_damping"],
                "left_hip_yaw_joint": MotorParameter["7520_16"]["computed_damping"],
                "left_knee_joint": MotorParameter["7520_22"]["computed_damping"],
            },
            armature={
                "left_hip_pitch_joint": MotorParameter["7520_16"]["armature"],
                "left_hip_roll_joint": MotorParameter["7520_22"]["armature"],
                "left_hip_yaw_joint": MotorParameter["7520_16"]["armature"],
                "left_knee_joint": MotorParameter["7520_22"]["armature"],
            },
            friction={
                "left_hip_pitch_joint": MotorParameter["7520_16"]["friction"],
                "left_hip_roll_joint": MotorParameter["7520_22"]["friction"],
                "left_hip_yaw_joint": MotorParameter["7520_16"]["friction"],
                "left_knee_joint": MotorParameter["7520_22"]["friction"],
            },
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "left_foot": DelayedImplicitActuatorCfg(
            joint_names_expr=["left_ankle_pitch_joint", "left_ankle_roll_joint"],
            effort_limit_sim=2 * MotorParameter["5020"]["effort_limit"],
            velocity_limit_sim=MotorParameter["5020"]["velocity_limit"],
            stiffness=2 * MotorParameter["5020"]["computed_stiffness"],
            damping=2 * MotorParameter["5020"]["computed_damping"],
            armature=2 * MotorParameter["5020"]["armature"],
            friction=2 * MotorParameter["5020"]["friction"],
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "right_leg": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
            ],
            effort_limit_sim={
                "right_hip_pitch_joint": MotorParameter["7520_16"]["effort_limit"],
                "right_hip_roll_joint": MotorParameter["7520_22"]["effort_limit"],
                "right_hip_yaw_joint": MotorParameter["7520_16"]["effort_limit"],
                "right_knee_joint": MotorParameter["7520_22"]["effort_limit"],
            },
            velocity_limit_sim={
                "right_hip_pitch_joint": MotorParameter["7520_16"]["velocity_limit"],
                "right_hip_roll_joint": MotorParameter["7520_22"]["velocity_limit"],
                "right_hip_yaw_joint": MotorParameter["7520_16"]["velocity_limit"],
                "right_knee_joint": MotorParameter["7520_22"]["velocity_limit"],
            },
            stiffness={
                "right_hip_pitch_joint": MotorParameter["7520_16"]["computed_stiffness"],
                "right_hip_roll_joint": MotorParameter["7520_22"]["computed_stiffness"],
                "right_hip_yaw_joint": MotorParameter["7520_16"]["computed_stiffness"],
                "right_knee_joint": MotorParameter["7520_22"]["computed_stiffness"],
            },
            damping={
                "right_hip_pitch_joint": MotorParameter["7520_16"]["computed_damping"],
                "right_hip_roll_joint": MotorParameter["7520_22"]["computed_damping"],
                "right_hip_yaw_joint": MotorParameter["7520_16"]["computed_damping"],
                "right_knee_joint": MotorParameter["7520_22"]["computed_damping"],
            },
            armature={
                "right_hip_pitch_joint": MotorParameter["7520_16"]["armature"],
                "right_hip_roll_joint": MotorParameter["7520_22"]["armature"],
                "right_hip_yaw_joint": MotorParameter["7520_16"]["armature"],
                "right_knee_joint": MotorParameter["7520_22"]["armature"],
            },
            friction={
                "right_hip_pitch_joint": MotorParameter["7520_16"]["friction"],
                "right_hip_roll_joint": MotorParameter["7520_22"]["friction"],
                "right_hip_yaw_joint": MotorParameter["7520_16"]["friction"],
                "right_knee_joint": MotorParameter["7520_22"]["friction"],
            },
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "right_foot": DelayedImplicitActuatorCfg(
            joint_names_expr=["right_ankle_pitch_joint", "right_ankle_roll_joint"],
            effort_limit_sim=2 * MotorParameter["5020"]["effort_limit"],
            velocity_limit_sim=MotorParameter["5020"]["velocity_limit"],
            stiffness=2 * MotorParameter["5020"]["computed_stiffness"],
            damping=2 * MotorParameter["5020"]["computed_damping"],
            armature=2 * MotorParameter["5020"]["armature"],
            friction=2 * MotorParameter["5020"]["friction"],
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "waist": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": MotorParameter["7520_14"]["effort_limit"],
                "waist_roll_joint": 2 * MotorParameter["5020"]["effort_limit"],
                "waist_pitch_joint": 2 * MotorParameter["5020"]["effort_limit"],
            },
            velocity_limit_sim={
                "waist_yaw_joint": MotorParameter["7520_14"]["velocity_limit"],
                "waist_roll_joint": MotorParameter["5020"]["velocity_limit"],
                "waist_pitch_joint": MotorParameter["5020"]["velocity_limit"],
            },
            stiffness={
                "waist_yaw_joint": MotorParameter["7520_14"]["computed_stiffness"],
                "waist_roll_joint": 2 * MotorParameter["5020"]["computed_stiffness"],
                "waist_pitch_joint": 2 * MotorParameter["5020"]["computed_stiffness"],
            },
            damping={
                "waist_yaw_joint": MotorParameter["7520_14"]["computed_damping"],
                "waist_roll_joint": 2 * MotorParameter["5020"]["computed_damping"],
                "waist_pitch_joint": 2 * MotorParameter["5020"]["computed_damping"],
            },
            armature={
                "waist_yaw_joint": MotorParameter["7520_14"]["armature"],
                "waist_roll_joint": 2 * MotorParameter["5020"]["armature"],
                "waist_pitch_joint": 2 * MotorParameter["5020"]["armature"],
            },
            friction={
                "waist_yaw_joint": MotorParameter["7520_14"]["friction"],
                "waist_roll_joint": 2 * MotorParameter["5020"]["friction"],
                "waist_pitch_joint": 2 * MotorParameter["5020"]["friction"],
            },
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "left_shoulder": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
            ],
            effort_limit_sim=MotorParameter["5020"]["effort_limit"],
            velocity_limit_sim=MotorParameter["5020"]["velocity_limit"],
            stiffness=MotorParameter["5020"]["computed_stiffness"],
            damping=MotorParameter["5020"]["computed_damping"],
            armature=MotorParameter["5020"]["armature"],
            friction=MotorParameter["5020"]["friction"],
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "left_arm": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
            ],
            effort_limit_sim=MotorParameter["5020"]["effort_limit"],
            velocity_limit_sim=MotorParameter["5020"]["velocity_limit"],
            stiffness=MotorParameter["5020"]["computed_stiffness"],
            damping=MotorParameter["5020"]["computed_damping"],
            armature=MotorParameter["5020"]["armature"],
            friction=MotorParameter["5020"]["friction"],
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "left_wrist": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
            effort_limit_sim={
                "left_wrist_roll_joint": MotorParameter["5020"]["effort_limit"],
                "left_wrist_pitch_joint": MotorParameter["4010"]["effort_limit"],
                "left_wrist_yaw_joint": MotorParameter["4010"]["effort_limit"],
            },
            velocity_limit_sim={
                "left_wrist_roll_joint": MotorParameter["5020"]["velocity_limit"],
                "left_wrist_pitch_joint": MotorParameter["4010"]["velocity_limit"],
                "left_wrist_yaw_joint": MotorParameter["4010"]["velocity_limit"],
            },
            stiffness={
                "left_wrist_roll_joint": MotorParameter["5020"]["computed_stiffness"],
                "left_wrist_pitch_joint": MotorParameter["4010"]["computed_stiffness"],
                "left_wrist_yaw_joint": MotorParameter["4010"]["computed_stiffness"],
            },
            damping={
                "left_wrist_roll_joint": MotorParameter["5020"]["computed_damping"],
                "left_wrist_pitch_joint": MotorParameter["4010"]["computed_damping"],
                "left_wrist_yaw_joint": MotorParameter["4010"]["computed_damping"],
            },
            armature={
                "left_wrist_roll_joint": MotorParameter["5020"]["armature"],
                "left_wrist_pitch_joint": MotorParameter["4010"]["armature"],
                "left_wrist_yaw_joint": MotorParameter["4010"]["armature"],
            },
            friction={
                "left_wrist_roll_joint": MotorParameter["5020"]["friction"],
                "left_wrist_pitch_joint": MotorParameter["4010"]["friction"],
                "left_wrist_yaw_joint": MotorParameter["4010"]["friction"],
            },
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "right_shoulder": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
            ],
            effort_limit_sim=MotorParameter["5020"]["effort_limit"],
            velocity_limit_sim=MotorParameter["5020"]["velocity_limit"],
            stiffness=MotorParameter["5020"]["computed_stiffness"],
            damping=MotorParameter["5020"]["computed_damping"],
            armature=MotorParameter["5020"]["armature"],
            friction=MotorParameter["5020"]["friction"],
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "right_arm": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
            ],
            effort_limit_sim=MotorParameter["5020"]["effort_limit"],
            velocity_limit_sim=MotorParameter["5020"]["velocity_limit"],
            stiffness=MotorParameter["5020"]["computed_stiffness"],
            damping=MotorParameter["5020"]["computed_damping"],
            armature=MotorParameter["5020"]["armature"],
            friction=MotorParameter["5020"]["friction"],
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "right_wrist": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            effort_limit_sim={
                "right_wrist_roll_joint": MotorParameter["5020"]["effort_limit"],
                "right_wrist_pitch_joint": MotorParameter["4010"]["effort_limit"],
                "right_wrist_yaw_joint": MotorParameter["4010"]["effort_limit"],
            },
            velocity_limit_sim={
                "right_wrist_roll_joint": MotorParameter["5020"]["velocity_limit"],
                "right_wrist_pitch_joint": MotorParameter["4010"]["velocity_limit"],
                "right_wrist_yaw_joint": MotorParameter["4010"]["velocity_limit"],
            },
            stiffness={
                "right_wrist_roll_joint": MotorParameter["5020"]["computed_stiffness"],
                "right_wrist_pitch_joint": MotorParameter["4010"]["computed_stiffness"],
                "right_wrist_yaw_joint": MotorParameter["4010"]["computed_stiffness"],
            },
            damping={
                "right_wrist_roll_joint": MotorParameter["5020"]["computed_damping"],
                "right_wrist_pitch_joint": MotorParameter["4010"]["computed_damping"],
                "right_wrist_yaw_joint": MotorParameter["4010"]["computed_damping"],
            },
            armature={
                "right_wrist_roll_joint": MotorParameter["5020"]["armature"],
                "right_wrist_pitch_joint": MotorParameter["4010"]["armature"],
                "right_wrist_yaw_joint": MotorParameter["4010"]["armature"],
            },
            friction={
                "right_wrist_roll_joint": MotorParameter["5020"]["friction"],
                "right_wrist_pitch_joint": MotorParameter["4010"]["friction"],
                "right_wrist_yaw_joint": MotorParameter["4010"]["friction"],
            },
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
    },
)

# Dynamically calculate action scale based on effort limits and stiffness
# Formula: action_scale = scale_factor * effort_limit / stiffness
# This ensures that the maximum action corresponds to the torque limit
G1_ACTION_SCALE = {}
SCALE_FACTOR = 0.25  # Base scaling factor

for actuator_name, actuator_cfg in UNITREE_G1_CFG.actuators.items():
    effort_limit = actuator_cfg.effort_limit_sim
    stiffness = actuator_cfg.stiffness
    joint_names = actuator_cfg.joint_names_expr
    
    # Convert single values to dict format for uniform processing
    if not isinstance(effort_limit, dict):
        effort_limit = {name: effort_limit for name in joint_names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in joint_names}
    
    # Calculate action scale for each joint pattern
    # Note: joint_names_expr contains regex patterns, not actual joint names
    for pattern in joint_names:
        # Match patterns between effort_limit and stiffness dicts
        for key in effort_limit.keys():
            if key == pattern or key in stiffness:
                if key in stiffness and stiffness[key] and stiffness[key] > 0:
                    scale = SCALE_FACTOR * effort_limit[key] / stiffness[key]
                    G1_ACTION_SCALE[key] = scale

# If no action scales were calculated (shouldn't happen), use default
if not G1_ACTION_SCALE:
    print("[WARNING] No action scales calculated, using default value")
    G1_ACTION_SCALE = 0.25  # Fallback to scalar value
