"""G1 robot locomotion task configurations and gym registrations."""

import gymnasium as gym

from unitree_lab.tasks.locomotion.config import agents

# --- Rough terrain (AMP Plugin) ---
gym.register(
    id="unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughPluginRunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughPluginRunnerCfg",
    },
)

# --- Rough terrain (AMP Plugin + GRU) ---
gym.register(
    id="unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-GRU-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughPluginGRURunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Velocity-Rough-Unitree-G1-AMP-GRU-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughPluginGRURunnerCfg",
    },
)

# --- Flat terrain (AMP Plugin) ---
gym.register(
    id="unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1FlatPluginRunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Velocity-Flat-Unitree-G1-AMP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1FlatPluginRunnerCfg",
    },
)
