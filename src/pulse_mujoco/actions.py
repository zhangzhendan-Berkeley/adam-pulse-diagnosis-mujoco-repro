from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class HandMode(IntEnum):
    HOLD = 0
    OPEN = 1
    DIAGNOSIS = 2
    PALPATION = 3


@dataclass(frozen=True)
class PulseAction:
    delta_tcp_xyz_m: np.ndarray
    delta_tcp_rpy_rad: np.ndarray
    target_pressure_kpa: float = 0.0
    hand_mode: int = int(HandMode.HOLD)
    phase_command: int | None = None

    def __post_init__(self) -> None:
        xyz = np.asarray(self.delta_tcp_xyz_m, dtype=np.float32)
        rpy = np.asarray(self.delta_tcp_rpy_rad, dtype=np.float32)
        if xyz.shape != (3,) or rpy.shape != (3,):
            raise ValueError("Cartesian action components must have shape [3]")
        object.__setattr__(self, "delta_tcp_xyz_m", xyz)
        object.__setattr__(self, "delta_tcp_rpy_rad", rpy)
        object.__setattr__(
            self, "target_pressure_kpa", float(self.target_pressure_kpa)
        )
        object.__setattr__(self, "hand_mode", int(HandMode(self.hand_mode)))

    @classmethod
    def zero(cls) -> "PulseAction":
        return cls(np.zeros(3), np.zeros(3))

    @classmethod
    def from_array(cls, value) -> "PulseAction":
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (7,):
            raise ValueError("continuous action must have shape [7]")
        return cls(array[:3], array[3:6], float(array[6]))

    def as_array(self) -> np.ndarray:
        return np.concatenate(
            (
                self.delta_tcp_xyz_m,
                self.delta_tcp_rpy_rad,
                np.asarray([self.target_pressure_kpa], dtype=np.float32),
            )
        )

    def as_dict(self) -> dict:
        return {
            "delta_tcp_xyz_m": self.delta_tcp_xyz_m.copy(),
            "delta_tcp_rpy_rad": self.delta_tcp_rpy_rad.copy(),
            "target_pressure_kpa": self.target_pressure_kpa,
            "hand_mode": self.hand_mode,
            "phase_command": self.phase_command,
        }
