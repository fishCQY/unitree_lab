"""Experiment tracker for unified experiment management.

This module provides:
1. Unified experiment directory structure
2. Automatic artifact versioning and management
3. Checkpoint save/load with metadata
4. Integration with WandB, TensorBoard, and local logging
5. Experiment resume and reproducibility

Design Philosophy:
- Single source of truth for experiment outputs
- Consistent directory structure across all experiments
- Automatic backup of configs and code
- Easy model deployment pipeline
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    
    # Base settings
    name: str  # Experiment name
    task: str  # Task name (e.g., "Unitree-G1-29dof-AMP-Velocity-v0")
    
    # Directory structure
    base_dir: str = "logs"
    
    # Logging
    use_wandb: bool = True
    use_tensorboard: bool = True
    
    # Checkpoint settings
    save_interval: int = 100  # Save every N iterations
    keep_last_n: int = 5  # Keep last N checkpoints
    
    # Code snapshot
    save_code: bool = True
    code_dirs: List[str] = field(default_factory=lambda: [
        "source/unitree_rl_lab/unitree_rl_lab",
        "scripts",
    ])
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Random seed
    seed: int = 42


class ExperimentTracker:
    """Unified experiment tracker.
    
    Directory structure:
        logs/
        └── {task}/
            └── {timestamp}_{name}/
                ├── checkpoints/
                │   ├── model_0100.pt
                │   ├── model_0200.pt
                │   └── model_latest.pt -> model_0200.pt
                ├── onnx/
                │   ├── policy.onnx
                │   └── metadata.json
                ├── configs/
                │   ├── env_cfg.yaml
                │   ├── agent_cfg.yaml
                │   └── train_cfg.yaml
                ├── code_snapshot/
                │   └── ... (copied source files)
                ├── videos/
                │   └── episode_100.mp4
                ├── metrics/
                │   └── training_log.csv
                └── experiment.json  (experiment metadata)
    
    Example:
        >>> tracker = ExperimentTracker(ExperimentConfig(
        ...     name="baseline",
        ...     task="Unitree-G1-29dof-Flat-v0"
        ... ))
        >>> tracker.setup()
        >>> tracker.save_config("env_cfg", env_cfg)
        >>> tracker.save_checkpoint(policy, iteration=100)
        >>> tracker.log_metrics({"reward": 100, "loss": 0.5})
        >>> tracker.finish()
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._setup_complete = False
        self._wandb_manager = None
        self._tensorboard_writer = None
        self._metrics_file = None
        self._iteration = 0
        
        # Will be set in setup()
        self.exp_dir: Optional[Path] = None
        self.checkpoint_dir: Optional[Path] = None
        self.onnx_dir: Optional[Path] = None
        self.config_dir: Optional[Path] = None
        self.code_dir: Optional[Path] = None
        self.video_dir: Optional[Path] = None
        self.metrics_dir: Optional[Path] = None
    
    def setup(self) -> "ExperimentTracker":
        """Setup experiment directories and logging.
        
        Returns:
            Self for chaining
        """
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_safe = self.config.task.replace("/", "_").replace(" ", "_")
        exp_name = f"{timestamp}_{self.config.name}"
        
        self.exp_dir = Path(self.config.base_dir) / task_safe / exp_name
        
        # Create subdirectories
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.onnx_dir = self.exp_dir / "onnx"
        self.config_dir = self.exp_dir / "configs"
        self.code_dir = self.exp_dir / "code_snapshot"
        self.video_dir = self.exp_dir / "videos"
        self.metrics_dir = self.exp_dir / "metrics"
        
        for d in [self.checkpoint_dir, self.onnx_dir, self.config_dir, 
                  self.code_dir, self.video_dir, self.metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        self._save_experiment_metadata()
        
        # Setup WandB
        if self.config.use_wandb:
            self._setup_wandb()
        
        # Setup TensorBoard
        if self.config.use_tensorboard:
            self._setup_tensorboard()
        
        # Save code snapshot
        if self.config.save_code:
            self._save_code_snapshot()
        
        self._setup_complete = True
        
        print(f"[Experiment] Setup complete")
        print(f"            Directory: {self.exp_dir}")
        
        return self
    
    def _save_experiment_metadata(self) -> None:
        """Save experiment metadata."""
        metadata = {
            "name": self.config.name,
            "task": self.config.task,
            "created_at": datetime.now().isoformat(),
            "seed": self.config.seed,
            "tags": self.config.tags,
            "notes": self.config.notes,
            "config": {
                "use_wandb": self.config.use_wandb,
                "use_tensorboard": self.config.use_tensorboard,
                "save_interval": self.config.save_interval,
            }
        }
        
        with open(self.exp_dir / "experiment.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _setup_wandb(self) -> None:
        """Setup WandB logging."""
        try:
            from .wandb_utils import WandbManager, WandbConfig
            
            wandb_cfg = WandbConfig(
                project="unitree_rl_lab",
                name=f"{self.config.task}_{self.config.name}",
                group=self.config.task,
                tags=self.config.tags,
                notes=self.config.notes,
            )
            
            self._wandb_manager = WandbManager(wandb_cfg)
            self._wandb_manager.init(config={
                "task": self.config.task,
                "seed": self.config.seed,
                **{f"tag_{i}": t for i, t in enumerate(self.config.tags)},
            })
            
        except ImportError:
            print("[Experiment] WandB not available")
    
    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tensorboard_writer = SummaryWriter(str(self.exp_dir / "tensorboard"))
        except ImportError:
            print("[Experiment] TensorBoard not available")
    
    def _save_code_snapshot(self) -> None:
        """Save code snapshot."""
        for code_dir in self.config.code_dirs:
            src_dir = Path(code_dir)
            if not src_dir.exists():
                continue
            
            dst_dir = self.code_dir / src_dir.name
            
            # Copy Python and YAML files
            for pattern in ["**/*.py", "**/*.yaml", "**/*.yml"]:
                for src_file in src_dir.glob(pattern):
                    if "__pycache__" in str(src_file) or ".git" in str(src_file):
                        continue
                    
                    rel_path = src_file.relative_to(src_dir)
                    dst_file = dst_dir / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        shutil.copy2(src_file, dst_file)
                    except Exception:
                        pass
        
        print(f"[Experiment] Code snapshot saved to {self.code_dir}")
    
    def save_config(
        self,
        name: str,
        config: Any,
        format: str = "yaml",
    ) -> Path:
        """Save configuration file.
        
        Args:
            name: Config name (without extension)
            config: Configuration object or dict
            format: Output format ("yaml" or "json")
            
        Returns:
            Path to saved config file
        """
        if not self._setup_complete:
            raise RuntimeError("Call setup() first")
        
        # Convert config to dict if needed
        if hasattr(config, '__dict__'):
            config_dict = self._config_to_dict(config)
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = dict(config)
        
        # Save
        if format == "yaml":
            filepath = self.config_dir / f"{name}.yaml"
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            filepath = self.config_dir / f"{name}.json"
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        # Also upload to WandB
        if self._wandb_manager:
            self._wandb_manager.save_file(filepath)
        
        print(f"[Experiment] Saved config: {filepath.name}")
        return filepath
    
    def _config_to_dict(self, config: Any) -> Dict:
        """Convert config object to dict recursively."""
        result = {}
        
        for key, value in vars(config).items():
            if key.startswith('_'):
                continue
            
            if hasattr(value, '__dict__') and not callable(value):
                result[key] = self._config_to_dict(value)
            elif isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    self._config_to_dict(v) if hasattr(v, '__dict__') else v
                    for v in value
                ]
            else:
                try:
                    json.dumps(value)  # Check if serializable
                    result[key] = value
                except (TypeError, ValueError):
                    result[key] = str(value)
        
        return result
    
    def save_checkpoint(
        self,
        model: Any,
        iteration: int,
        optimizer: Any = None,
        scheduler: Any = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> Path:
        """Save training checkpoint.
        
        Args:
            model: Model to save
            iteration: Current iteration
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            metrics: Optional metrics dict
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        import torch
        
        if not self._setup_complete:
            raise RuntimeError("Call setup() first")
        
        self._iteration = iteration
        
        # Build checkpoint
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        # Save numbered checkpoint
        checkpoint_path = self.checkpoint_dir / f"model_{iteration:06d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Update latest symlink
        latest_path = self.checkpoint_dir / "model_latest.pt"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        
        # Windows doesn't support symlinks well, so copy instead
        shutil.copy2(checkpoint_path, latest_path)
        
        # Save best if needed
        if is_best:
            best_path = self.checkpoint_dir / "model_best.pt"
            shutil.copy2(checkpoint_path, best_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
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
        
        print(f"[Experiment] Saved checkpoint: iter {iteration}")
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("model_[0-9]*.pt"),
            key=lambda p: int(p.stem.split('_')[1])
        )
        
        # Keep last N + best + latest
        keep_count = self.config.keep_last_n
        if len(checkpoints) > keep_count:
            for ckpt in checkpoints[:-keep_count]:
                ckpt.unlink()
    
    def load_checkpoint(
        self,
        path: Optional[Union[str, Path]] = None,
        iteration: Optional[int] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        """Load checkpoint.
        
        Args:
            path: Specific checkpoint path
            iteration: Load specific iteration
            load_best: Load best checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        import torch
        
        if path is not None:
            checkpoint_path = Path(path)
        elif load_best:
            checkpoint_path = self.checkpoint_dir / "model_best.pt"
        elif iteration is not None:
            checkpoint_path = self.checkpoint_dir / f"model_{iteration:06d}.pt"
        else:
            checkpoint_path = self.checkpoint_dir / "model_latest.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"[Experiment] Loaded checkpoint: {checkpoint_path.name}")
        
        return checkpoint
    
    def save_onnx(
        self,
        onnx_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save ONNX model with metadata.
        
        Args:
            onnx_path: Source ONNX file path
            metadata: ONNX metadata dictionary
            
        Returns:
            Path to saved ONNX file in experiment directory
        """
        if not self._setup_complete:
            raise RuntimeError("Call setup() first")
        
        onnx_path = Path(onnx_path)
        
        # Copy to experiment directory
        dst_path = self.onnx_dir / "policy.onnx"
        shutil.copy2(onnx_path, dst_path)
        
        # Save metadata
        if metadata:
            metadata_path = self.onnx_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Upload to WandB
        if self._wandb_manager:
            self._wandb_manager.save_onnx(dst_path, metadata=metadata)
        
        print(f"[Experiment] Saved ONNX: {dst_path}")
        return dst_path
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log training metrics.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step number (uses internal counter if None)
        """
        if step is None:
            step = self._iteration
        
        # WandB
        if self._wandb_manager:
            self._wandb_manager.log(metrics, step=step)
        
        # TensorBoard
        if self._tensorboard_writer:
            for key, value in metrics.items():
                self._tensorboard_writer.add_scalar(key, value, step)
        
        # CSV file
        self._log_metrics_csv(metrics, step)
    
    def _log_metrics_csv(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to CSV file."""
        csv_path = self.metrics_dir / "training_log.csv"
        
        # Add step to metrics
        metrics_with_step = {"step": step, **metrics}
        
        # Write header if new file
        write_header = not csv_path.exists()
        
        with open(csv_path, 'a') as f:
            if write_header:
                f.write(",".join(metrics_with_step.keys()) + "\n")
            f.write(",".join(str(v) for v in metrics_with_step.values()) + "\n")
    
    def log_video(
        self,
        video_path: Union[str, Path],
        name: Optional[str] = None,
    ) -> Path:
        """Save and log video.
        
        Args:
            video_path: Source video path
            name: Video name (uses source name if None)
            
        Returns:
            Path to saved video
        """
        if not self._setup_complete:
            raise RuntimeError("Call setup() first")
        
        video_path = Path(video_path)
        
        if name is None:
            name = video_path.name
        
        dst_path = self.video_dir / name
        shutil.copy2(video_path, dst_path)
        
        # Upload to WandB
        if self._wandb_manager:
            self._wandb_manager.log_video(dst_path, key=f"video/{name}")
        
        return dst_path
    
    def finish(self) -> None:
        """Finish experiment and cleanup."""
        # Close TensorBoard
        if self._tensorboard_writer:
            self._tensorboard_writer.close()
        
        # Finish WandB
        if self._wandb_manager:
            self._wandb_manager.finish()
        
        # Update experiment metadata
        if self.exp_dir and self.exp_dir.exists():
            metadata_path = self.exp_dir / "experiment.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                metadata["finished_at"] = datetime.now().isoformat()
                metadata["final_iteration"] = self._iteration
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        print(f"[Experiment] Finished: {self.exp_dir}")
    
    @classmethod
    def from_directory(cls, exp_dir: Union[str, Path]) -> "ExperimentTracker":
        """Resume experiment from existing directory.
        
        Args:
            exp_dir: Experiment directory path
            
        Returns:
            Resumed ExperimentTracker
        """
        exp_dir = Path(exp_dir)
        
        # Load metadata
        metadata_path = exp_dir / "experiment.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"experiment.json not found in {exp_dir}")
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Create config
        config = ExperimentConfig(
            name=metadata["name"],
            task=metadata["task"],
            base_dir=str(exp_dir.parent.parent),
            seed=metadata.get("seed", 42),
            tags=metadata.get("tags", []),
            notes=metadata.get("notes", ""),
        )
        
        # Create tracker without full setup
        tracker = cls(config)
        tracker.exp_dir = exp_dir
        tracker.checkpoint_dir = exp_dir / "checkpoints"
        tracker.onnx_dir = exp_dir / "onnx"
        tracker.config_dir = exp_dir / "configs"
        tracker.code_dir = exp_dir / "code_snapshot"
        tracker.video_dir = exp_dir / "videos"
        tracker.metrics_dir = exp_dir / "metrics"
        tracker._setup_complete = True
        
        print(f"[Experiment] Resumed from: {exp_dir}")
        
        return tracker


# Convenience function
def create_experiment(
    name: str,
    task: str,
    use_wandb: bool = True,
    **kwargs,
) -> ExperimentTracker:
    """Create and setup an experiment tracker.
    
    Args:
        name: Experiment name
        task: Task name
        use_wandb: Enable WandB logging
        **kwargs: Additional ExperimentConfig arguments
        
    Returns:
        Setup ExperimentTracker
    """
    config = ExperimentConfig(
        name=name,
        task=task,
        use_wandb=use_wandb,
        **kwargs,
    )
    
    tracker = ExperimentTracker(config)
    tracker.setup()
    
    return tracker
