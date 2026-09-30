from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PulseObservation:
    """Policy-visible state; privileged pulse truth is intentionally absent."""

    timestamp_s: float
    joint_position_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    tcp_pose: np.ndarray
    tcp_velocity: np.ndarray
    proximity_m: np.ndarray
    proximity_valid: np.ndarray
    tactile_kpa: np.ndarray
    tactile_raw: np.ndarray
    tactile_raw_delta: np.ndarray
    contact: np.ndarray
    task_phase: int
    previous_action: np.ndarray
    rgb: np.ndarray | None = None
    depth_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        joint_position = np.asarray(self.joint_position_rad, dtype=np.float32)
        joint_velocity = np.asarray(self.joint_velocity_rad_s, dtype=np.float32)
        if joint_position.ndim != 1 or not joint_position.size:
            raise ValueError("joint_position_rad must be a non-empty 1-D array")
        if joint_velocity.shape != joint_position.shape:
            raise ValueError(
                "joint_velocity_rad_s must match joint_position_rad shape "
                f"{joint_position.shape}, got {joint_velocity.shape}"
            )
        object.__setattr__(self, "joint_position_rad", joint_position)
        object.__setattr__(self, "joint_velocity_rad_s", joint_velocity)
        specifications = {
            "tcp_pose": ((7,), np.float32),
            "tcp_velocity": ((6,), np.float32),
            # Preserve the Isaac policy contract. MuJoCo retains each
            # finger's 2x5 raw rays in backend diagnostics and fuses them
            # channel-wise for the policy boundary.
            "proximity_m": ((5,), np.float32),
            "proximity_valid": ((5,), bool),
            "tactile_kpa": ((2, 10), np.float32),
            "tactile_raw": ((2, 10), np.uint16),
            "tactile_raw_delta": ((2, 10), np.float32),
            "contact": ((2,), bool),
            "previous_action": ((7,), np.float32),
        }
        for name, (shape, dtype) in specifications.items():
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.shape != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {value.shape}"
                )
            object.__setattr__(self, name, value)

    def policy_dict(self) -> dict:
        return {
            "timestamp_s": float(self.timestamp_s),
            "rgb": self.rgb,
            "depth_m": self.depth_m,
            "joint_position_rad": self.joint_position_rad.copy(),
            "joint_velocity_rad_s": self.joint_velocity_rad_s.copy(),
            "tcp_pose": self.tcp_pose.copy(),
            "tcp_velocity": self.tcp_velocity.copy(),
            "proximity_m": self.proximity_m.copy(),
            "proximity_valid": self.proximity_valid.copy(),
            "tactile_kpa": self.tactile_kpa.copy(),
            "tactile_raw": self.tactile_raw.copy(),
            "tactile_raw_delta": self.tactile_raw_delta.copy(),
            "contact": self.contact.copy(),
            "task_phase": int(self.task_phase),
            "previous_action": self.previous_action.copy(),
        }
