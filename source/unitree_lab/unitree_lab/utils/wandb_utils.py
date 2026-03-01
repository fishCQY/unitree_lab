"""WandB utilities for experiment tracking and file management.

This module provides:
1. WandB initialization with project configuration
2. Code snapshot upload (preserves directory structure)
3. Model/checkpoint upload with versioning
4. Config and metrics logging
5. Artifact management for data/models

Design Philosophy:
- Separate "what to track" from "how to track"
- Support offline mode for cluster training
- Preserve reproducibility through code snapshots
- Enable easy model sharing through artifacts
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class WandbConfig:
    """Configuration for WandB integration."""
    
    # Project settings
    project: str = "unitree_lab"
    entity: Optional[str] = None  # WandB team/user name
    group: Optional[str] = None   # Experiment group (e.g., "locomotion", "amp")
    job_type: Optional[str] = None  # Job type (e.g., "train", "eval", "sim2sim")
    
    # Run settings
    name: Optional[str] = None  # Run name (auto-generated if None)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Behavior
    mode: str = "online"  # "online", "offline", "disabled"
    save_code: bool = True
    code_include_globs: List[str] = field(default_factory=lambda: [
        "**/*.py",
        "**/*.yaml",
        "**/*.yml",
    ])
    code_exclude_globs: List[str] = field(default_factory=lambda: [
        "**/logs/**",
        "**/data/**",
        "**/__pycache__/**",
        "**/.git/**",
    ])
    
    # Artifact settings
    artifact_type_model: str = "model"
    artifact_type_data: str = "dataset"
    artifact_type_config: str = "config"


class WandbManager:
    """Manager for WandB experiment tracking.
    
    Example:
        >>> manager = WandbManager(WandbConfig(project="my_project"))
        >>> manager.init(config={"lr": 0.001, "task": "locomotion"})
        >>> manager.log({"loss": 0.5, "reward": 100})
        >>> manager.save_checkpoint("model.pt", metadata={"step": 1000})
        >>> manager.finish()
    """
    
    def __init__(self, config: WandbConfig):
        self.config = config
        self._run = None
        self._initialized = False
        
    @property
    def run(self):
        """Get current WandB run (lazy import)."""
        if self._run is None and self._initialized:
            import wandb
            self._run = wandb.run
        return self._run
    
    def init(
        self,
        config: Optional[Dict[str, Any]] = None,
        resume: bool = False,
        run_id: Optional[str] = None,
    ) -> "WandbManager":
        """Initialize WandB run.
        
        Args:
            config: Experiment configuration to log
            resume: Whether to resume from a previous run
            run_id: Run ID for resuming (required if resume=True)
            
        Returns:
            Self for chaining
        """
        try:
            import wandb
        except ImportError:
            print("[WandB] wandb package not installed. Tracking disabled.")
            return self
        
        if self.config.mode == "disabled":
            print("[WandB] Tracking disabled by config.")
            return self
        
        # Generate run name if not provided
        name = self.config.name
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"run_{timestamp}"
        
        # Initialize
        self._run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            job_type=self.config.job_type,
            name=name,
            tags=self.config.tags,
            notes=self.config.notes,
            config=config,
            mode=self.config.mode,
            resume="allow" if resume else None,
            id=run_id,
        )
        
        self._initialized = True
        
        print(f"[WandB] Initialized run: {self._run.name}")
        print(f"        URL: {self._run.url}")
        
        return self
    
    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log metrics to WandB.
        
        Args:
            data: Dictionary of metrics
            step: Global step (auto-incremented if None)
            commit: Whether to commit this log
        """
        if not self._initialized or self._run is None:
            return
        
        self._run.log(data, step=step, commit=commit)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Update run configuration.
        
        Args:
            config: Configuration dictionary to merge
        """
        if not self._initialized or self._run is None:
            return
        
        self._run.config.update(config)
    
    def log_summary(self, data: Dict[str, Any]) -> None:
        """Update run summary (final metrics).
        
        Args:
            data: Summary dictionary
        """
        if not self._initialized or self._run is None:
            return
        
        for key, value in data.items():
            self._run.summary[key] = value
    
    def save_code_snapshot(
        self,
        root_dir: Union[str, Path],
        name: str = "code",
    ) -> Optional[str]:
        """Save code snapshot as artifact.
        
        Args:
            root_dir: Root directory to scan
            name: Artifact name
            
        Returns:
            Artifact name if successful
        """
        if not self._initialized or self._run is None:
            return None
        
        import wandb
        
        root_dir = Path(root_dir)
        
        # Create artifact
        artifact = wandb.Artifact(
            name=name,
            type="code",
            description=f"Code snapshot from {root_dir}",
        )
        
        # Collect files matching include patterns
        files_added = 0
        for pattern in self.config.code_include_globs:
            for filepath in root_dir.glob(pattern):
                # Check exclude patterns
                excluded = False
                for exclude in self.config.code_exclude_globs:
                    if filepath.match(exclude):
                        excluded = True
                        break
                
                if not excluded and filepath.is_file():
                    # Preserve relative path
                    rel_path = filepath.relative_to(root_dir)
                    artifact.add_file(str(filepath), name=str(rel_path))
                    files_added += 1
        
        if files_added > 0:
            self._run.log_artifact(artifact)
            print(f"[WandB] Saved code snapshot: {files_added} files")
            return artifact.name
        
        return None
    
    def save_file(
        self,
        filepath: Union[str, Path],
        base_path: Optional[Union[str, Path]] = None,
        policy: str = "live",
    ) -> None:
        """Save a single file to WandB.
        
        Args:
            filepath: Path to file
            base_path: Base path for relative naming
            policy: Save policy ("live", "now", "end")
        """
        if not self._initialized or self._run is None:
            return
        
        import wandb
        
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"[WandB] File not found: {filepath}")
            return
        
        if base_path:
            base_path = Path(base_path)
            wandb.save(str(filepath), base_path=str(base_path), policy=policy)
        else:
            wandb.save(str(filepath), policy=policy)
    
    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Save model checkpoint as artifact.
        
        Args:
            filepath: Path to checkpoint file
            metadata: Metadata to attach
            aliases: Artifact aliases (e.g., ["latest", "best"])
            
        Returns:
            Artifact name if successful
        """
        if not self._initialized or self._run is None:
            return None
        
        import wandb
        
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"[WandB] Checkpoint not found: {filepath}")
            return None
        
        # Create artifact
        artifact_name = f"{self._run.name}-checkpoint"
        artifact = wandb.Artifact(
            name=artifact_name,
            type=self.config.artifact_type_model,
            metadata=metadata or {},
        )
        
        artifact.add_file(str(filepath))
        
        # Log artifact
        self._run.log_artifact(artifact, aliases=aliases or ["latest"])
        
        print(f"[WandB] Saved checkpoint: {filepath.name}")
        return artifact.name
    
    def save_onnx(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Save ONNX model as artifact.
        
        Args:
            filepath: Path to ONNX file
            metadata: Metadata to attach (from onnx_utils.build_onnx_metadata)
            aliases: Artifact aliases
            
        Returns:
            Artifact name if successful
        """
        if not self._initialized or self._run is None:
            return None
        
        import wandb
        
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"[WandB] ONNX not found: {filepath}")
            return None
        
        # Create artifact
        artifact_name = f"{self._run.name}-onnx"
        artifact = wandb.Artifact(
            name=artifact_name,
            type=self.config.artifact_type_model,
            metadata=metadata or {},
        )
        
        artifact.add_file(str(filepath))
        
        # Also save metadata JSON alongside
        if metadata:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(metadata, f, indent=2)
                metadata_path = f.name
            artifact.add_file(metadata_path, name="metadata.json")
            os.unlink(metadata_path)
        
        self._run.log_artifact(artifact, aliases=aliases or ["latest"])
        
        print(f"[WandB] Saved ONNX: {filepath.name}")
        return artifact.name
    
    def save_config_files(
        self,
        config_dir: Union[str, Path],
        patterns: List[str] = None,
    ) -> int:
        """Save configuration files.
        
        Args:
            config_dir: Directory containing configs
            patterns: File patterns to match (default: *.yaml, *.yml, *.json)
            
        Returns:
            Number of files saved
        """
        if not self._initialized or self._run is None:
            return 0
        
        import wandb
        
        config_dir = Path(config_dir)
        if not config_dir.exists():
            return 0
        
        if patterns is None:
            patterns = ["*.yaml", "*.yml", "*.json"]
        
        count = 0
        for pattern in patterns:
            for filepath in config_dir.glob(pattern):
                wandb.save(str(filepath), base_path=str(config_dir.parent), policy="now")
                count += 1
        
        if count > 0:
            print(f"[WandB] Saved {count} config files from {config_dir}")
        
        return count
    
    def log_video(
        self,
        video_path: Union[str, Path],
        key: str = "video",
        fps: int = 30,
        caption: Optional[str] = None,
    ) -> None:
        """Log video to WandB.
        
        Args:
            video_path: Path to video file
            key: Metric key for the video
            fps: Video framerate
            caption: Optional caption
        """
        if not self._initialized or self._run is None:
            return
        
        import wandb
        
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"[WandB] Video not found: {video_path}")
            return
        
        self._run.log({key: wandb.Video(str(video_path), fps=fps, caption=caption)})
    
    def log_table(
        self,
        key: str,
        columns: List[str],
        data: List[List[Any]],
    ) -> None:
        """Log table to WandB.
        
        Args:
            key: Metric key for the table
            columns: Column names
            data: Table data (list of rows)
        """
        if not self._initialized or self._run is None:
            return
        
        import wandb
        
        table = wandb.Table(columns=columns, data=data)
        self._run.log({key: table})
    
    def alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
    ) -> None:
        """Send alert notification.
        
        Args:
            title: Alert title
            text: Alert body
            level: Alert level ("INFO", "WARN", "ERROR")
        """
        if not self._initialized or self._run is None:
            return
        
        import wandb
        
        wandb.alert(title=title, text=text, level=getattr(wandb.AlertLevel, level))
    
    def finish(self) -> None:
        """Finish WandB run."""
        if self._initialized and self._run is not None:
            self._run.finish()
            print("[WandB] Run finished")
        
        self._initialized = False
        self._run = None


class WandbFileSaver:
    """Utility for saving files to WandB with preserved structure.
    
    This is useful for saving code snapshots during training.
    
    Example:
        >>> saver = WandbFileSaver()
        >>> saver.save_directory("source/unitree_lab/unitree_lab")
    """
    
    def __init__(
        self,
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
    ):
        self.include_patterns = include_patterns or ["*.py", "*.yaml", "*.yml"]
        self.exclude_patterns = exclude_patterns or [
            "__pycache__/*",
            "*.pyc",
            ".git/*",
            "logs/*",
            "data/*",
        ]
    
    def save_directory(
        self,
        directory: Union[str, Path],
        base_path: Optional[Union[str, Path]] = None,
    ) -> int:
        """Save all matching files in directory.
        
        Args:
            directory: Directory to scan
            base_path: Base path for relative naming
            
        Returns:
            Number of files saved
        """
        try:
            import wandb
        except ImportError:
            print("[WandB] wandb not installed")
            return 0
        
        if wandb.run is None:
            print("[WandB] No active run")
            return 0
        
        directory = Path(directory)
        base_path = Path(base_path) if base_path else directory.parent
        
        count = 0
        for pattern in self.include_patterns:
            for filepath in directory.rglob(pattern):
                # Check exclude
                excluded = False
                for exclude in self.exclude_patterns:
                    if filepath.match(exclude):
                        excluded = True
                        break
                
                if not excluded and filepath.is_file():
                    try:
                        wandb.save(str(filepath), base_path=str(base_path))
                        count += 1
                    except Exception as e:
                        print(f"[WandB] Failed to save {filepath}: {e}")
        
        print(f"[WandB] Saved {count} files from {directory}")
        return count


# Convenience functions
def init_wandb(
    project: str = "unitree_lab",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    mode: str = "online",
) -> WandbManager:
    """Quick WandB initialization.
    
    Args:
        project: WandB project name
        name: Run name
        config: Experiment config
        tags: Run tags
        group: Experiment group
        mode: WandB mode
        
    Returns:
        Initialized WandbManager
    """
    cfg = WandbConfig(
        project=project,
        name=name,
        tags=tags or [],
        group=group,
        mode=mode,
    )
    
    manager = WandbManager(cfg)
    manager.init(config=config)
    
    return manager


def log_training_metrics(
    episode: int,
    reward: float,
    loss: Optional[float] = None,
    lr: Optional[float] = None,
    **kwargs,
) -> None:
    """Log common training metrics.
    
    Args:
        episode: Training episode/iteration
        reward: Episode reward
        loss: Training loss
        lr: Learning rate
        **kwargs: Additional metrics
    """
    try:
        import wandb
    except ImportError:
        return
    
    if wandb.run is None:
        return
    
    data = {
        "episode": episode,
        "reward": reward,
    }
    
    if loss is not None:
        data["loss"] = loss
    if lr is not None:
        data["learning_rate"] = lr
    
    data.update(kwargs)
    
    wandb.log(data)
