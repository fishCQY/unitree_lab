"""Checkpoint utilities for model save/load and versioning.

This module provides:
1. Checkpoint save/load with metadata
2. Automatic versioning and cleanup
3. Model export utilities
4. Checkpoint comparison and selection

Design Philosophy:
- Checkpoints should be self-contained (model + config + metadata)
- Support both training resume and deployment export
- Enable easy comparison between checkpoints
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    path: Path
    iteration: int
    timestamp: datetime
    metrics: Dict[str, float]
    is_best: bool = False
    
    @classmethod
    def from_checkpoint(cls, path: Path) -> "CheckpointInfo":
        """Create info from checkpoint file."""
        import torch
        
        checkpoint = torch.load(path, map_location='cpu')
        
        # Parse iteration from filename or checkpoint
        if 'iteration' in checkpoint:
            iteration = checkpoint['iteration']
        else:
            # Try to extract from filename (model_000100.pt)
            try:
                iteration = int(path.stem.split('_')[-1])
            except (ValueError, IndexError):
                iteration = 0
        
        # Parse timestamp
        if 'timestamp' in checkpoint:
            timestamp = datetime.fromisoformat(checkpoint['timestamp'])
        else:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime)
        
        # Get metrics
        metrics = checkpoint.get('metrics', {})
        
        return cls(
            path=path,
            iteration=iteration,
            timestamp=timestamp,
            metrics=metrics,
        )


class CheckpointManager:
    """Manager for checkpoint save/load operations.
    
    Example:
        >>> manager = CheckpointManager("logs/checkpoints", keep_last=5)
        >>> manager.save(model, optimizer, iteration=100, metrics={"reward": 50})
        >>> checkpoint = manager.load_latest()
        >>> model.load_state_dict(checkpoint["model_state_dict"])
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        keep_last: int = 5,
        keep_best: bool = True,
        metric_for_best: str = "reward",
        higher_is_better: bool = True,
    ):
        """
        Args:
            checkpoint_dir: Directory for checkpoints
            keep_last: Number of recent checkpoints to keep
            keep_best: Whether to keep the best checkpoint
            metric_for_best: Metric name for determining best
            higher_is_better: Whether higher metric value is better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.metric_for_best = metric_for_best
        self.higher_is_better = higher_is_better
        
        self._best_metric: Optional[float] = None
    
    def save(
        self,
        model: Any,
        iteration: int,
        optimizer: Any = None,
        scheduler: Any = None,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, bool]:
        """Save checkpoint.
        
        Args:
            model: Model to save (or state dict)
            iteration: Current iteration
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            metrics: Optional metrics dict
            extra: Optional extra data to save
            
        Returns:
            (checkpoint_path, is_best)
        """
        import torch
        
        # Build checkpoint dict
        checkpoint = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Model state
        if hasattr(model, 'state_dict'):
            checkpoint["model_state_dict"] = model.state_dict()
        else:
            checkpoint["model_state_dict"] = model
        
        # Optimizer state
        if optimizer is not None:
            if hasattr(optimizer, 'state_dict'):
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            else:
                checkpoint["optimizer_state_dict"] = optimizer
        
        # Scheduler state
        if scheduler is not None:
            if hasattr(scheduler, 'state_dict'):
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            else:
                checkpoint["scheduler_state_dict"] = scheduler
        
        # Metrics
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        # Extra data
        if extra is not None:
            checkpoint["extra"] = extra
        
        # Save numbered checkpoint
        filename = f"model_{iteration:06d}.pt"
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        # Update latest
        latest_path = self.checkpoint_dir / "model_latest.pt"
        if latest_path.exists():
            latest_path.unlink()
        shutil.copy2(checkpoint_path, latest_path)
        
        # Check if best
        is_best = False
        if metrics is not None and self.metric_for_best in metrics:
            metric_value = metrics[self.metric_for_best]
            
            if self._best_metric is None:
                is_best = True
            elif self.higher_is_better and metric_value > self._best_metric:
                is_best = True
            elif not self.higher_is_better and metric_value < self._best_metric:
                is_best = True
            
            if is_best:
                self._best_metric = metric_value
                if self.keep_best:
                    best_path = self.checkpoint_dir / "model_best.pt"
                    shutil.copy2(checkpoint_path, best_path)
        
        # Cleanup old checkpoints
        self._cleanup()
        
        return checkpoint_path, is_best
    
    def _cleanup(self) -> None:
        """Remove old checkpoints."""
        # Get all numbered checkpoints
        checkpoints = sorted(
            self.checkpoint_dir.glob("model_[0-9]*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        # Remove excess
        if len(checkpoints) > self.keep_last:
            for ckpt in checkpoints[:-self.keep_last]:
                ckpt.unlink()
    
    def load(
        self,
        path: Optional[Union[str, Path]] = None,
        iteration: Optional[int] = None,
        load_best: bool = False,
        map_location: str = 'cpu',
    ) -> Dict[str, Any]:
        """Load checkpoint.
        
        Args:
            path: Specific path to load
            iteration: Specific iteration to load
            load_best: Load best checkpoint
            map_location: Device for loading
            
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
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        return checkpoint
    
    def load_latest(self, map_location: str = 'cpu') -> Dict[str, Any]:
        """Load latest checkpoint."""
        return self.load(map_location=map_location)
    
    def load_best(self, map_location: str = 'cpu') -> Dict[str, Any]:
        """Load best checkpoint."""
        return self.load(load_best=True, map_location=map_location)
    
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all checkpoints with info."""
        checkpoints = []
        
        for path in self.checkpoint_dir.glob("model_[0-9]*.pt"):
            try:
                info = CheckpointInfo.from_checkpoint(path)
                checkpoints.append(info)
            except Exception:
                continue
        
        # Sort by iteration
        checkpoints.sort(key=lambda x: x.iteration)
        
        # Mark best
        best_path = self.checkpoint_dir / "model_best.pt"
        if best_path.exists():
            best_checkpoint = CheckpointInfo.from_checkpoint(best_path)
            for ckpt in checkpoints:
                if ckpt.iteration == best_checkpoint.iteration:
                    ckpt.is_best = True
                    break
        
        return checkpoints
    
    def get_latest_iteration(self) -> int:
        """Get the latest saved iteration."""
        latest_path = self.checkpoint_dir / "model_latest.pt"
        
        if not latest_path.exists():
            return 0
        
        info = CheckpointInfo.from_checkpoint(latest_path)
        return info.iteration


def save_for_deployment(
    model: Any,
    output_dir: Union[str, Path],
    name: str = "policy",
    metadata: Optional[Dict[str, Any]] = None,
    export_jit: bool = True,
    export_onnx: bool = True,
    obs_dim: Optional[int] = None,
) -> Dict[str, Path]:
    """Save model for deployment (JIT and ONNX).
    
    Args:
        model: Trained model
        output_dir: Output directory
        name: Model name
        metadata: Model metadata
        export_jit: Export TorchScript
        export_onnx: Export ONNX
        obs_dim: Observation dimension (for ONNX)
        
    Returns:
        Dictionary of exported file paths
    """
    import torch
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported = {}
    model.eval()
    
    # Save PyTorch model
    pt_path = output_dir / f"{name}.pt"
    torch.save(model.state_dict(), pt_path)
    exported["pytorch"] = pt_path
    
    # Export JIT
    if export_jit:
        jit_path = output_dir / f"{name}_jit.pt"
        try:
            if obs_dim is not None:
                dummy = torch.zeros(1, obs_dim, device=next(model.parameters()).device)
                traced = torch.jit.trace(model, dummy)
            else:
                traced = torch.jit.script(model)
            traced.save(str(jit_path))
            exported["jit"] = jit_path
        except Exception as e:
            print(f"[Warning] JIT export failed: {e}")
    
    # Export ONNX
    if export_onnx and obs_dim is not None:
        onnx_path = output_dir / f"{name}.onnx"
        try:
            dummy = torch.zeros(1, obs_dim, device=next(model.parameters()).device)
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["obs"],
                output_names=["action"],
                opset_version=11,
                dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            )
            exported["onnx"] = onnx_path
            
            # Attach metadata
            if metadata:
                from .onnx_utils import attach_onnx_metadata
                attach_onnx_metadata(onnx_path, metadata)
                
        except Exception as e:
            print(f"[Warning] ONNX export failed: {e}")
    
    # Save metadata
    if metadata:
        meta_path = output_dir / f"{name}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        exported["metadata"] = meta_path
    
    return exported


def compare_checkpoints(
    paths: List[Union[str, Path]],
    metric_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Compare multiple checkpoints.
    
    Args:
        paths: List of checkpoint paths
        metric_keys: Metrics to compare (all if None)
        
    Returns:
        List of comparison dicts
    """
    results = []
    
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        
        info = CheckpointInfo.from_checkpoint(path)
        
        result = {
            "path": str(path),
            "iteration": info.iteration,
            "timestamp": info.timestamp.isoformat(),
        }
        
        # Add metrics
        metrics = info.metrics
        if metric_keys:
            metrics = {k: v for k, v in metrics.items() if k in metric_keys}
        result["metrics"] = metrics
        
        results.append(result)
    
    return results


def find_best_checkpoint(
    checkpoint_dir: Union[str, Path],
    metric: str = "reward",
    higher_is_better: bool = True,
) -> Optional[Path]:
    """Find the best checkpoint by metric.
    
    Args:
        checkpoint_dir: Directory with checkpoints
        metric: Metric to optimize
        higher_is_better: Whether higher is better
        
    Returns:
        Path to best checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    best_path = None
    best_value = None
    
    for path in checkpoint_dir.glob("model_[0-9]*.pt"):
        try:
            info = CheckpointInfo.from_checkpoint(path)
            
            if metric not in info.metrics:
                continue
            
            value = info.metrics[metric]
            
            if best_value is None:
                best_value = value
                best_path = path
            elif higher_is_better and value > best_value:
                best_value = value
                best_path = path
            elif not higher_is_better and value < best_value:
                best_value = value
                best_path = path
                
        except Exception:
            continue
    
    return best_path
