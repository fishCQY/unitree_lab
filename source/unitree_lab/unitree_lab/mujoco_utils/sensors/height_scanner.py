"""Height scanner for MuJoCo sim2sim.

This module implements height scanning matching IsaacLab's RayCaster:
1. Sample terrain height at grid points relative to robot base
2. Apply offset and clipping
3. Match grid size and resolution to training

Key alignment:
- Grid size and resolution from ONNX metadata
- Height offset (typically 0.5m)
- Clip range (typically [-1, 1])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco


class HeightScanner:
    """Height scanner matching IsaacLab's height_scan observation.
    
    Scans terrain height at a grid of points relative to robot base,
    then computes relative heights with offset and clipping.
    """
    
    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        grid_size: tuple[float, float] = (1.6, 1.0),
        resolution: float = 0.1,
        offset: float = 0.5,
        clip_range: tuple[float, float] = (-1.0, 1.0),
        base_body_name: str = "base",
        terrain_geom_name: str | None = None,
    ):
        """Initialize height scanner.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            grid_size: (length_x, length_y) of scan grid in meters
            resolution: Grid resolution in meters
            offset: Height offset added to measurements
            clip_range: (min, max) clip range for heights
            base_body_name: Name of robot base body
            terrain_geom_name: Name of terrain geometry (None = use height field)
        """
        import mujoco as mj
        
        self.model = model
        self.data = data
        self.offset = offset
        self.clip_range = clip_range
        
        # Find base body
        try:
            self.base_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_body_name)
        except Exception:
            self.base_body_id = 1  # Usually trunk/base is body 1
        
        # Find terrain geom (if specified)
        if terrain_geom_name:
            try:
                self.terrain_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, terrain_geom_name)
            except Exception:
                self.terrain_geom_id = None
        else:
            self.terrain_geom_id = None
        
        # Create sample grid
        self.grid_size = grid_size
        self.resolution = resolution
        self._create_grid()
        
        print(f"[HeightScanner] Grid: {self.nx}x{self.ny} = {self.num_points} points")
        print(f"[HeightScanner] Size: {grid_size}, resolution: {resolution}")
    
    def _create_grid(self) -> None:
        """Create sampling grid in body frame."""
        # Grid extents
        half_x = self.grid_size[0] / 2
        half_y = self.grid_size[1] / 2
        
        # Number of points
        self.nx = int(self.grid_size[0] / self.resolution) + 1
        self.ny = int(self.grid_size[1] / self.resolution) + 1
        self.num_points = self.nx * self.ny
        
        # Create grid points in body frame (x forward, y left)
        x = np.linspace(-half_x, half_x, self.nx)
        y = np.linspace(-half_y, half_y, self.ny)
        
        # Meshgrid and flatten
        xx, yy = np.meshgrid(x, y)
        
        # Points in body frame: (num_points, 2)
        self.grid_points_body = np.stack([xx.flatten(), yy.flatten()], axis=-1)
    
    def scan(self) -> np.ndarray:
        """Perform height scan.
        
        Returns:
            Height measurements at grid points with offset and clipping.
            Shape: (num_points,)
        """
        import mujoco as mj
        
        # Get base position and orientation
        base_pos = self.data.xpos[self.base_body_id]  # [x, y, z]
        base_quat = self.data.xquat[self.base_body_id]  # [w, x, y, z]
        
        # Base height
        base_height = base_pos[2]
        
        # Get yaw rotation (ignore roll/pitch for height scan)
        yaw = self._quat_to_yaw(base_quat)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Transform grid points to world frame
        points_world = np.zeros((self.num_points, 2))
        points_world[:, 0] = cos_yaw * self.grid_points_body[:, 0] - sin_yaw * self.grid_points_body[:, 1] + base_pos[0]
        points_world[:, 1] = sin_yaw * self.grid_points_body[:, 0] + cos_yaw * self.grid_points_body[:, 1] + base_pos[1]
        
        # Sample terrain heights
        terrain_heights = self._sample_terrain_heights(points_world)
        
        # Compute relative heights: sensor_height - terrain_height
        # Then add offset and clip
        heights = base_height - terrain_heights
        heights = heights + self.offset
        heights = np.clip(heights, self.clip_range[0], self.clip_range[1])
        
        return heights.astype(np.float32)
    
    def _sample_terrain_heights(self, points_world: np.ndarray) -> np.ndarray:
        """Sample terrain heights at world positions.
        
        Args:
            points_world: World XY positions (num_points, 2)
            
        Returns:
            Terrain heights at each point
        """
        import mujoco as mj
        
        heights = np.zeros(len(points_world))
        
        # Use ray casting from high above
        for i, (px, py) in enumerate(points_world):
            # Cast ray from above
            start = np.array([px, py, 100.0])  # High start point
            direction = np.array([0.0, 0.0, -1.0])  # Straight down
            
            # Ray-geom intersection
            geomid = np.array([-1], dtype=np.int32)
            
            dist = mj.mj_ray(
                self.model, self.data,
                start, direction,
                None,  # geomgroup (all)
                1,  # flg_static
                -1,  # bodyexclude
                geomid,
            )
            
            if dist >= 0:
                heights[i] = start[2] - dist
            else:
                heights[i] = 0.0  # Default to ground level
        
        return heights
    
    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        """Extract yaw angle from quaternion [w, x, y, z]."""
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    @classmethod
    def from_onnx_config(
        cls,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        onnx_config: "OnnxConfig",
        base_body_name: str = "base",
    ) -> "HeightScanner":
        """Create height scanner from ONNX config.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            onnx_config: ONNX configuration with height scan params
            base_body_name: Name of robot base body
            
        Returns:
            Configured HeightScanner
        """
        grid_size = onnx_config.height_scan_size or (1.6, 1.0)
        resolution = onnx_config.height_scan_resolution or 0.1
        offset = onnx_config.height_scan_offset or 0.5
        
        return cls(
            model=model,
            data=data,
            grid_size=grid_size,
            resolution=resolution,
            offset=offset,
            base_body_name=base_body_name,
        )
