"""Enhanced training runner with integrated experiment tracking.

This module provides:
1. LightOnPolicyRunner - Enhanced PPO runner with full tracking
2. Automatic ONNX export with metadata
3. WandB integration for logging and artifacts
4. Checkpoint management with versioning
5. Code snapshot and config backup

Design Philosophy:
- Runner should handle all experiment management automatically
- Training code should focus on algorithm, not bookkeeping
- All outputs should be traceable and reproducible
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class LightRunnerConfig:
    """Configuration for LightOnPolicyRunner."""
    
    # Experiment settings
    experiment_name: str = "default"
    run_name: Optional[str] = None
    
    # Directories
    log_dir: str = "logs"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "unitree_rl_lab"
    wandb_entity: Optional[str] = None
    
    # Checkpointing
    save_interval: int = 100
    keep_checkpoints: int = 5
    
    # Export
    export_onnx: bool = True
    export_jit: bool = True
    
    # Code snapshot
    save_code: bool = True
    code_dirs: List[str] = field(default_factory=lambda: [
        "source/unitree_rl_lab",
        "scripts",
    ])
    
    # Resume
    resume: bool = False
    resume_path: Optional[str] = None


class LightOnPolicyRunner:
    """Enhanced on-policy training runner with experiment tracking.
    
    This runner wraps standard RL training with:
    - Automatic experiment directory setup
    - WandB logging (metrics, configs, artifacts)
    - Checkpoint management
    - ONNX export with IsaacLab metadata
    - Code snapshots for reproducibility
    
    Example:
        >>> runner = LightOnPolicyRunner(
        ...     env=env,
        ...     agent=agent,
        ...     config=LightRunnerConfig(
        ...         experiment_name="locomotion",
        ...         use_wandb=True,
        ...     )
        ... )
        >>> runner.learn(max_iterations=1000)
    """
    
    def __init__(
        self,
        env,
        agent,
        config: LightRunnerConfig,
        train_cfg: Optional[Any] = None,
    ):
        """
        Args:
            env: IsaacLab ManagerBasedRLEnv
            agent: RL agent (e.g., rsl_rl PPO)
            config: Runner configuration
            train_cfg: Original training configuration (for logging)
        """
        self.env = env
        self.agent = agent
        self.config = config
        self.train_cfg = train_cfg
        
        self._setup_experiment()
        self._current_iteration = 0
        self._best_reward = float('-inf')
    
    def _setup_experiment(self) -> None:
        """Setup experiment directories and logging."""
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = getattr(self.env.cfg, 'task_name', 'unknown')
        
        if self.config.run_name:
            run_name = self.config.run_name
        else:
            run_name = f"{timestamp}_{self.config.experiment_name}"
        
        self.exp_dir = Path(self.config.log_dir) / task_name / run_name
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.onnx_dir = self.exp_dir / "onnx"
        self.config_dir = self.exp_dir / "configs"
        self.code_dir = self.exp_dir / "code_snapshot"
        
        for d in [self.checkpoint_dir, self.onnx_dir, self.config_dir, self.code_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Setup WandB
        self._wandb_manager = None
        if self.config.use_wandb:
            self._setup_wandb()
        
        # Save configs
        self._save_configs()
        
        # Save code snapshot
        if self.config.save_code:
            self._save_code()
        
        # Setup checkpoint manager
        from .checkpoint_utils import CheckpointManager
        self._checkpoint_manager = CheckpointManager(
            self.checkpoint_dir,
            keep_last=self.config.keep_checkpoints,
        )
        
        print(f"[Runner] Experiment directory: {self.exp_dir}")
    
    def _setup_wandb(self) -> None:
        """Setup WandB logging."""
        try:
            from .wandb_utils import WandbManager, WandbConfig
            
            task_name = getattr(self.env.cfg, 'task_name', 'unknown')
            
            wandb_cfg = WandbConfig(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=f"{task_name}_{self.config.experiment_name}",
                group=task_name,
            )
            
            self._wandb_manager = WandbManager(wandb_cfg)
            
            # Collect config for WandB
            run_config = {
                "task": task_name,
                "experiment": self.config.experiment_name,
            }
            
            if self.train_cfg:
                run_config["train_cfg"] = self._config_to_dict(self.train_cfg)
            
            self._wandb_manager.init(config=run_config)
            
        except Exception as e:
            print(f"[Runner] WandB setup failed: {e}")
            self._wandb_manager = None
    
    def _save_configs(self) -> None:
        """Save training configurations."""
        # Save env config
        try:
            env_cfg_dict = self._config_to_dict(self.env.cfg)
            with open(self.config_dir / "env_cfg.yaml", 'w') as f:
                yaml.dump(env_cfg_dict, f, default_flow_style=False)
        except Exception as e:
            print(f"[Runner] Failed to save env config: {e}")
        
        # Save train config
        if self.train_cfg:
            try:
                train_cfg_dict = self._config_to_dict(self.train_cfg)
                with open(self.config_dir / "train_cfg.yaml", 'w') as f:
                    yaml.dump(train_cfg_dict, f, default_flow_style=False)
            except Exception as e:
                print(f"[Runner] Failed to save train config: {e}")
        
        # Save runner config
        runner_cfg_dict = {
            "experiment_name": self.config.experiment_name,
            "log_dir": self.config.log_dir,
            "use_wandb": self.config.use_wandb,
            "save_interval": self.config.save_interval,
        }
        with open(self.config_dir / "runner_cfg.yaml", 'w') as f:
            yaml.dump(runner_cfg_dict, f)
        
        # Upload to WandB
        if self._wandb_manager:
            self._wandb_manager.save_config_files(self.config_dir)
    
    def _save_code(self) -> None:
        """Save code snapshot."""
        for code_dir in self.config.code_dirs:
            src_dir = Path(code_dir)
            if not src_dir.exists():
                continue
            
            dst_dir = self.code_dir / src_dir.name
            
            for pattern in ["**/*.py", "**/*.yaml"]:
                for src_file in src_dir.glob(pattern):
                    if "__pycache__" in str(src_file):
                        continue
                    
                    rel_path = src_file.relative_to(src_dir)
                    dst_file = dst_dir / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        shutil.copy2(src_file, dst_file)
                    except Exception:
                        pass
    
    def _config_to_dict(self, config: Any) -> Dict:
        """Convert config object to dict."""
        if hasattr(config, 'to_dict'):
            return config.to_dict()
        
        if not hasattr(config, '__dict__'):
            return {"value": str(config)}
        
        result = {}
        for key, value in vars(config).items():
            if key.startswith('_'):
                continue
            
            if hasattr(value, '__dict__') and not callable(value):
                result[key] = self._config_to_dict(value)
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                try:
                    import json
                    json.dumps(value)
                    result[key] = value
                except (TypeError, ValueError):
                    result[key] = str(value)
        
        return result
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        iteration: Optional[int] = None,
    ) -> None:
        """Log training metrics.
        
        Args:
            metrics: Dictionary of metrics
            iteration: Step number
        """
        step = iteration or self._current_iteration
        
        if self._wandb_manager:
            self._wandb_manager.log(metrics, step=step)
        
        # Track best reward
        if "mean_reward" in metrics:
            if metrics["mean_reward"] > self._best_reward:
                self._best_reward = metrics["mean_reward"]
    
    def save_checkpoint(
        self,
        iteration: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save training checkpoint.
        
        Args:
            iteration: Current iteration
            metrics: Optional metrics
            
        Returns:
            Path to saved checkpoint
        """
        self._current_iteration = iteration
        
        # Determine if best
        is_best = False
        if metrics and "mean_reward" in metrics:
            is_best = metrics["mean_reward"] >= self._best_reward
        
        # Save checkpoint
        checkpoint_path, _ = self._checkpoint_manager.save(
            model=self.agent.actor_critic,
            iteration=iteration,
            optimizer=getattr(self.agent, 'optimizer', None),
            metrics=metrics,
        )
        
        # Upload to WandB
        if self._wandb_manager and iteration % self.config.save_interval == 0:
            aliases = ["latest"]
            if is_best:
                aliases.append("best")
            self._wandb_manager.save_checkpoint(
                checkpoint_path,
                metadata={"iteration": iteration, **(metrics or {})},
                aliases=aliases,
            )
        
        return checkpoint_path
    
    def export_onnx(
        self,
        iteration: Optional[int] = None,
    ) -> Optional[Path]:
        """Export policy to ONNX with metadata.
        
        Args:
            iteration: Iteration number for naming
            
        Returns:
            Path to ONNX file
        """
        if not self.config.export_onnx:
            return None
        
        try:
            from .onnx_utils import export_onnx_with_metadata, build_onnx_metadata
            
            iteration = iteration or self._current_iteration
            onnx_path = self.onnx_dir / f"policy_{iteration:06d}.onnx"
            
            # Export
            export_onnx_with_metadata(
                env=self.env,
                policy=self.agent.actor_critic.actor,
                output_path=onnx_path,
            )
            
            # Also save as latest
            latest_path = self.onnx_dir / "policy_latest.onnx"
            shutil.copy2(onnx_path, latest_path)
            
            # Upload to WandB
            if self._wandb_manager:
                metadata = build_onnx_metadata(self.env)
                self._wandb_manager.save_onnx(onnx_path, metadata=metadata)
            
            return onnx_path
            
        except Exception as e:
            print(f"[Runner] ONNX export failed: {e}")
            return None
    
    def finish(self) -> None:
        """Finish training and cleanup."""
        # Final ONNX export
        self.export_onnx()
        
        # Close WandB
        if self._wandb_manager:
            self._wandb_manager.log_summary({
                "final_iteration": self._current_iteration,
                "best_reward": self._best_reward,
            })
            self._wandb_manager.finish()
        
        print(f"[Runner] Training finished. Output: {self.exp_dir}")
    
    def learn(
        self,
        max_iterations: int,
        log_interval: int = 10,
        save_interval: Optional[int] = None,
        eval_callback: Optional[callable] = None,
    ) -> None:
        """Run training loop.
        
        This is a simplified training loop. For production use,
        integrate with your RL library's training loop.
        
        Args:
            max_iterations: Maximum training iterations
            log_interval: Log metrics every N iterations
            save_interval: Save checkpoint every N iterations
            eval_callback: Optional evaluation callback
        """
        if save_interval is None:
            save_interval = self.config.save_interval
        
        for iteration in range(max_iterations):
            self._current_iteration = iteration
            
            # Training step (implement based on your RL library)
            # metrics = self.agent.train_step(self.env)
            
            # This is a placeholder - actual training would be:
            # obs = self.env.get_observations()
            # actions = self.agent.act(obs)
            # rewards, dones, infos = self.env.step(actions)
            # self.agent.store_transition(...)
            # metrics = self.agent.update()
            
            # For now, just placeholder
            metrics = {"iteration": iteration}
            
            # Log
            if iteration % log_interval == 0:
                self.log_metrics(metrics, iteration)
            
            # Save
            if iteration % save_interval == 0 and iteration > 0:
                self.save_checkpoint(iteration, metrics)
            
            # Eval
            if eval_callback and iteration % save_interval == 0:
                eval_metrics = eval_callback(self.env, self.agent)
                self.log_metrics(eval_metrics, iteration)
        
        self.finish()


def create_runner(
    env,
    agent,
    experiment_name: str = "default",
    use_wandb: bool = True,
    log_dir: str = "logs",
    **kwargs,
) -> LightOnPolicyRunner:
    """Create a configured training runner.
    
    Args:
        env: IsaacLab environment
        agent: RL agent
        experiment_name: Name for the experiment
        use_wandb: Enable WandB logging
        log_dir: Base log directory
        **kwargs: Additional config options
        
    Returns:
        Configured LightOnPolicyRunner
    """
    config = LightRunnerConfig(
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        log_dir=log_dir,
        **kwargs,
    )
    
    return LightOnPolicyRunner(env=env, agent=agent, config=config)
