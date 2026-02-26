# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from unitree_lab.assets import ISAAC_ASSET_DIR
from unitree_lab.actuators import DelayedImplicitActuatorCfg


MIN_DELAY = 0
MAX_DELAY = 4

MotorParameter = {
    "5020": {
        "armature": 0.003609725,
        "effort_limit": 25,
        "velocity_limit": 37,
        "friction": 0.0,
    },
    "7520_14": {
        "armature": 0.010177520,
        "effort_limit": 75,
        "velocity_limit": 32,
        "friction": 0.0,
    },
    "7520_16": {
        "armature": 0.013,
        "effort_limit": 88,
        "velocity_limit": 23,
        "friction": 0.0,
    },
    "7520_22": {
        "armature": 0.025101925,
        "effort_limit": 120,
        "velocity_limit": 20,
        "friction": 0.0,
    },
    "4010": {
        "armature": 0.00425,
        "effort_limit": 5,
        "velocity_limit": 22,
        "friction": 0.0,
    },
}

G1_ACTION_SCALE = 0.25
UNITREE_G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/Unitree/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=100000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": MotorParameter["7520_16"]["effort_limit"],
                ".*_hip_roll_joint": MotorParameter["7520_22"]["effort_limit"],
                ".*_hip_pitch_joint": MotorParameter["7520_16"]["effort_limit"],
                ".*_knee_joint": MotorParameter["7520_22"]["effort_limit"],
                "waist_yaw_joint": MotorParameter["7520_14"]["effort_limit"],
                "waist_pitch_joint": 2 * MotorParameter["5020"]["effort_limit"],
                "waist_roll_joint": 2 * MotorParameter["5020"]["effort_limit"],
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": MotorParameter["7520_16"]["velocity_limit"],
                ".*_hip_roll_joint": MotorParameter["7520_22"]["velocity_limit"],
                ".*_hip_pitch_joint": MotorParameter["7520_16"]["velocity_limit"],
                ".*_knee_joint": MotorParameter["7520_22"]["velocity_limit"],
                "waist_yaw_joint": MotorParameter["7520_14"]["velocity_limit"],
                "waist_pitch_joint": MotorParameter["5020"]["velocity_limit"],
                "waist_roll_joint": MotorParameter["5020"]["velocity_limit"],
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                ".*waist.*": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "waist.*": 5.0,
            },
            armature={
                ".*_hip_yaw_joint": MotorParameter["7520_16"]["armature"],
                ".*_hip_roll_joint": MotorParameter["7520_22"]["armature"],
                ".*_hip_pitch_joint": MotorParameter["7520_16"]["armature"],
                ".*_knee_joint": MotorParameter["7520_22"]["armature"],
                "waist_yaw_joint": MotorParameter["7520_14"]["armature"],
                "waist_pitch_joint": 2 * MotorParameter["5020"]["armature"],
                "waist_roll_joint": 2 * MotorParameter["5020"]["armature"],
            },
            friction={
                ".*_hip_yaw_joint": MotorParameter["7520_16"]["friction"],
                ".*_hip_roll_joint": MotorParameter["7520_22"]["friction"],
                ".*_hip_pitch_joint": MotorParameter["7520_16"]["friction"],
                ".*_knee_joint": MotorParameter["7520_22"]["friction"],
                "waist_yaw_joint": MotorParameter["7520_14"]["friction"],
                "waist_pitch_joint": 2 * MotorParameter["5020"]["friction"],
                "waist_roll_joint": 2 * MotorParameter["5020"]["friction"],
            },
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "feet": DelayedImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=2 * MotorParameter["5020"]["effort_limit"],
            velocity_limit_sim=MotorParameter["5020"]["velocity_limit"],
            stiffness=20.0,
            damping=2.0,
            armature=2 * MotorParameter["5020"]["armature"],
            friction=2 * MotorParameter["5020"]["friction"],
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "shoulders": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim=MotorParameter["5020"]["effort_limit"],
            velocity_limit_sim=MotorParameter["5020"]["velocity_limit"],
            stiffness=100.0,
            damping=2.0,
            armature=MotorParameter["5020"]["armature"],
            friction=MotorParameter["5020"]["friction"],
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "arms": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=MotorParameter["5020"]["effort_limit"],
            velocity_limit_sim=MotorParameter["5020"]["velocity_limit"],
            stiffness=50.0,
            damping=2.0,
            armature=MotorParameter["5020"]["armature"],
            friction=MotorParameter["5020"]["friction"],
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
        "wrist": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_.*",
            ],
            effort_limit_sim={
                ".*_wrist_yaw_joint": MotorParameter["4010"]["effort_limit"],
                ".*_wrist_roll_joint": MotorParameter["5020"]["effort_limit"],
                ".*_wrist_pitch_joint": MotorParameter["4010"]["effort_limit"],
            },
            velocity_limit_sim={
                ".*_wrist_yaw_joint": MotorParameter["4010"]["velocity_limit"],
                ".*_wrist_roll_joint": MotorParameter["5020"]["velocity_limit"],
                ".*_wrist_pitch_joint": MotorParameter["4010"]["velocity_limit"],
            },
            stiffness=40.0,
            damping=2.0,
            armature={
                ".*_wrist_yaw_joint": MotorParameter["4010"]["armature"],
                ".*_wrist_roll_joint": MotorParameter["5020"]["armature"],
                ".*_wrist_pitch_joint": MotorParameter["4010"]["armature"],
            },
            friction={
                ".*_wrist_yaw_joint": MotorParameter["4010"]["friction"],
                ".*_wrist_roll_joint": MotorParameter["5020"]["friction"],
                ".*_wrist_pitch_joint": MotorParameter["4010"]["friction"],
            },
            min_delay=MIN_DELAY,
            max_delay=MAX_DELAY,
        ),
    },
)
