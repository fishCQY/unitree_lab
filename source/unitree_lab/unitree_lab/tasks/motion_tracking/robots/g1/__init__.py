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

# -----------------------------------------------------------------------------
# Aliases / extended variants matching documentation naming (18 envs total)
# -----------------------------------------------------------------------------

# --- Motion Tracking (standard) ---
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingPPORunnerCfg",
    },
)

# --- Motion Tracking (PR parallel robot) ---
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-PR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_pr_env_cfg:UnitreeG1TrackingPREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingPRPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-PR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_pr_env_cfg:UnitreeG1TrackingPREnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingPRPPORunnerCfg",
    },
)

# --- Motion Tracking (AB parallel robot) ---
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-AB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_pr_env_cfg:UnitreeG1TrackingABEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingABPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-AB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_pr_env_cfg:UnitreeG1TrackingABEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingABPPORunnerCfg",
    },
)

# --- Motion Tracking (AutoEncoder) ---
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-AE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingAEPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-AE-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingAEPPORunnerCfg",
    },
)

# --- Motion Tracking (FSQ / RFSQ) ---
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-FSQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingFSQPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-FSQ-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingFSQPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-RFSQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingRFSQPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-RFSQ-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingRFSQPPORunnerCfg",
    },
)

# --- Motion Tracking (VQ-VAE) ---
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-VQVAE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingVQVAEPPORunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-VQVAE-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingVQVAEPPORunnerCfg",
    },
)

# --- Motion Tracking (Distillation) ---
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-Distillation-FSQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingDistillationFSQRunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-Distillation-FSQ-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingDistillationFSQRunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-Distillation-VQVAE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingDistillationVQVAERunnerCfg",
    },
)
gym.register(
    id="unitree_lab-Isaac-Tracking-Unitree-G1-Distillation-VQVAE-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:UnitreeG1TrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1TrackingDistillationVQVAERunnerCfg",
    },
)
