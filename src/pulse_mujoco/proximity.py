from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


FINGERS = ("index", "middle")
RAYS = ("center", "proximal", "distal", "medial", "lateral")


@dataclass(frozen=True)
class ProximityFrame:
    distance_m: np.ndarray
    valid: np.ndarray
    timestamp_s: float

    @property
    def fused_distance_m(self) -> np.ndarray:
        """Isaac-compatible five-channel minimum across both pads."""
        masked = np.where(self.valid, self.distance_m, np.inf)
        result = np.min(masked, axis=0)
        return np.where(
            np.isfinite(result),
            result,
            np.max(self.distance_m, axis=0),
        ).astype(np.float32)

    @property
    def fused_valid(self) -> np.ndarray:
        return np.any(self.valid, axis=0)


class MujocoFingertipProximity:
    """Read the five outward rangefinders on each diagnosis fingertip."""

    def __init__(self, model: mujoco.MjModel, maximum_range_m: float):
        self.model = model
        self.maximum_range_m = float(maximum_range_m)
        self.sensor_ids = np.asarray(
            [
                [
                    model.sensor(f"proximity_{finger}_{ray}").id
                    for ray in RAYS
                ]
                for finger in FINGERS
            ],
            dtype=int,
        )

    def read(
        self, data: mujoco.MjData, timestamp_s: float
    ) -> ProximityFrame:
        distances = np.empty((2, 5), dtype=np.float32)
        valid = np.zeros((2, 5), dtype=bool)
        for finger in range(2):
            for ray in range(5):
                sensor_id = self.sensor_ids[finger, ray]
                address = int(self.model.sensor_adr[sensor_id])
                value = float(data.sensordata[address])
                hit = 0.0 <= value <= self.maximum_range_m
                valid[finger, ray] = hit
                distances[finger, ray] = (
                    value if hit else self.maximum_range_m
                )
        return ProximityFrame(
            distance_m=distances,
            valid=valid,
            timestamp_s=float(timestamp_s),
        )
