"""G1 robot motion tracking task configurations and gym registrations."""

import gymnasium as gym

from unitree_lab.tasks.motion_tracking.config import agents

# --- Motion Tracking (standard) ---
gym.register(
    id="unitree_lab-Isaac-Velocity-Tracking-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Velocity-Tracking-Unitree-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingPPORunnerCfg",
    },
)

# --- Motion Tracking (PR parallel robot) ---
gym.register(
    id="unitree_lab-Isaac-Velocity-Tracking-Unitree-G1-PR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_pr_env_cfg:UnitreeG1TrackingPREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingPRPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Velocity-Tracking-Unitree-G1-PR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_pr_env_cfg:UnitreeG1TrackingPREnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingPRPPORunnerCfg",
    },
)

# --- Motion Tracking (AB parallel robot) ---
gym.register(
    id="unitree_lab-Isaac-Velocity-Tracking-Unitree-G1-AB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_pr_env_cfg:UnitreeG1TrackingABEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingABPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Velocity-Tracking-Unitree-G1-AB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_pr_env_cfg:UnitreeG1TrackingABEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingABPPORunnerCfg",
    },
)
