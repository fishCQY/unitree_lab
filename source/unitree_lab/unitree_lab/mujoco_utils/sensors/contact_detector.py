"""Contact detector for MuJoCo sim2sim.

This module detects contacts for:
1. Foot contact (for gait analysis)
2. Base/body contact (for fall detection)
3. Undesired contact (for safety)

Matches IsaacLab's contact sensor semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco


@dataclass
class ContactState:
    """Contact state for monitored bodies."""
    contact_flags: np.ndarray  # Binary contact flags
    contact_forces: np.ndarray  # Contact force magnitudes
    air_time: np.ndarray  # Time since last contact


class ContactDetector:
    """Contact detector matching IsaacLab's ContactSensor.
    
    Monitors contacts on specified bodies and reports:
    - Binary contact flags
    - Contact force magnitudes
    - Air time tracking
    """
    
    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        monitored_bodies: list[str],
        force_threshold: float = 1.0,
        dt: float = 0.005,
    ):
        """Initialize contact detector.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            monitored_bodies: List of body names to monitor
            force_threshold: Force threshold for contact detection
            dt: Simulation time step
        """
        import mujoco as mj
        
        self.model = model
        self.data = data
        self.force_threshold = force_threshold
        self.dt = dt
        
        # Map body names to IDs
        self.body_names = monitored_bodies
        self.body_ids = []
        
        for name in monitored_bodies:
            try:
                body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    self.body_ids.append(body_id)
                else:
                    print(f"[Warning] Body '{name}' not found, skipping")
            except Exception:
                # Try to find parent if exact match fails
                parent_id = self._find_parent_body(name)
                if parent_id is not None:
                    self.body_ids.append(parent_id)
                else:
                    print(f"[Warning] Body '{name}' not found and no parent found")
        
        self.num_bodies = len(self.body_ids)
        
        # State
        self._contact_flags = np.zeros(self.num_bodies, dtype=bool)
        self._contact_forces = np.zeros(self.num_bodies)
        self._air_time = np.zeros(self.num_bodies)
        self._last_contact_time = np.zeros(self.num_bodies)
        
        print(f"[ContactDetector] Monitoring {self.num_bodies} bodies")
    
    def _find_parent_body(self, name: str) -> int | None:
        """Try to find parent body when exact match fails.
        
        This handles cases where URDF links are merged in MJCF.
        """
        import mujoco as mj
        
        # Common naming patterns
        patterns = [
            name.replace("_link", ""),
            name + "_body",
            name.split("_")[0],
        ]
        
        for pattern in patterns:
            try:
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, pattern)
                if body_id >= 0:
                    return body_id
            except Exception:
                continue
        
        return None
    
    def update(self, time: float | None = None) -> ContactState:
        """Update contact state from simulation.
        
        Args:
            time: Current simulation time (for air time tracking)
            
        Returns:
            Current contact state
        """
        import mujoco as mj
        
        # Reset accumulators
        self._contact_forces.fill(0.0)
        self._contact_flags.fill(False)
        
        # Process all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get bodies involved in contact
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            # Check if any monitored body is involved
            for idx, body_id in enumerate(self.body_ids):
                if body1 == body_id or body2 == body_id:
                    # Get contact force
                    force = np.zeros(6)
                    mj.mj_contactForce(self.model, self.data, i, force)
                    force_magnitude = np.linalg.norm(force[:3])
                    
                    self._contact_forces[idx] += force_magnitude
                    
                    if force_magnitude > self.force_threshold:
                        self._contact_flags[idx] = True
        
        # Update air time
        if time is not None:
            for idx in range(self.num_bodies):
                if self._contact_flags[idx]:
                    self._last_contact_time[idx] = time
                    self._air_time[idx] = 0.0
                else:
                    self._air_time[idx] = time - self._last_contact_time[idx]
        
        return ContactState(
            contact_flags=self._contact_flags.copy(),
            contact_forces=self._contact_forces.copy(),
            air_time=self._air_time.copy(),
        )
    
    def get_contact_flags(self) -> np.ndarray:
        """Get binary contact flags."""
        return self._contact_flags.copy()
    
    def get_contact_forces(self) -> np.ndarray:
        """Get contact force magnitudes."""
        return self._contact_forces.copy()
    
    def get_air_time(self) -> np.ndarray:
        """Get time since last contact for each body."""
        return self._air_time.copy()
    
    def get_base_contact(self, base_body_names: list[str] | None = None) -> bool:
        """Check if base/trunk is in contact (fall detection).
        
        Args:
            base_body_names: Optional list of base body names to check
            
        Returns:
            True if any base body is in contact
        """
        if base_body_names is None:
            base_body_names = ["base", "trunk", "pelvis", "torso"]
        
        for name in base_body_names:
            if name in self.body_names:
                idx = self.body_names.index(name)
                if idx < len(self._contact_flags) and self._contact_flags[idx]:
                    return True
        
        return False
    
    def reset(self) -> None:
        """Reset contact state."""
        self._contact_flags.fill(False)
        self._contact_forces.fill(0.0)
        self._air_time.fill(0.0)
        self._last_contact_time.fill(0.0)
