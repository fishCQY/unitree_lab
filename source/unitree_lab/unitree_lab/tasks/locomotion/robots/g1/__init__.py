"""G1 robot locomotion task configurations and gym registrations."""

import gymnasium as gym

from unitree_lab.tasks.locomotion.config import agents

# --- Rough terrain (AMP) ---
gym.register(
    id="unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughPPORunnerCfg",
    },
)

# --- Rough terrain (AMP + GRU) ---
gym.register(
    id="unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-GRU-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughPPORunnerGRUCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-GRU-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughPPORunnerGRUCfg",
    },
)

# --- Flat terrain (AMP) ---
gym.register(
    id="unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1FlatPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1FlatPPORunnerCfg",
    },
)

# --- Depth-based visual locomotion ---
gym.register(
    id="unitree_lab-Isaac-Velocity-Rough-Unitree-G1-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughDepthPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Velocity-Rough-Unitree-G1-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughDepthPPORunnerCfg",
    },
)
