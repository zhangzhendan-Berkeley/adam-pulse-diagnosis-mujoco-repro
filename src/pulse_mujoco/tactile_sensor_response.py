from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorResponseConfig:
    cutoff_hz: float = 20.0
    adjacent_crosstalk: float = 0.03
    white_noise_std_kpa: float = 0.005
    drift_std_kpa_per_s: float = 0.0005
    saturation_kpa: float = 50.0
    resolution_kpa: float = 0.001


class TactileSensorResponse:
    def __init__(
        self,
        config: SensorResponseConfig | None = None,
        seed: int = 0,
        shape: tuple[int, int] = (2, 10),
    ):
        self.config = config or SensorResponseConfig()
        self.rng = np.random.default_rng(seed)
        self.shape = tuple(int(value) for value in shape)
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError("shape must contain two positive dimensions")
        self.state = np.zeros(self.shape, dtype=np.float32)
        self.drift = np.zeros(self.shape, dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state.fill(0.0)
        self.drift.fill(0.0)

    def update(self, pressure_kpa: np.ndarray, dt_s: float, contact: np.ndarray) -> np.ndarray:
        value = np.asarray(pressure_kpa, dtype=np.float32)
        if value.shape != self.shape:
            raise ValueError(f"pressure_kpa must have shape {self.shape}")
        contact = np.asarray(contact, dtype=bool)
        if contact.shape != (self.shape[0],):
            raise ValueError(
                f"contact must have shape {(self.shape[0],)}, got {contact.shape}"
            )
        dt = max(float(dt_s), 1e-6)
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.config.cutoff_hz * dt)
        self.state += alpha * (value - self.state)
        c = float(self.config.adjacent_crosstalk)
        mixed = (1.0 - 2.0 * c) * self.state
        mixed[:, 1:] += c * self.state[:, :-1]
        mixed[:, :-1] += c * self.state[:, 1:]
        self.drift += self.rng.normal(
            0.0, self.config.drift_std_kpa_per_s * np.sqrt(dt), self.drift.shape
        )
        output = mixed + self.drift + self.rng.normal(
            0.0, self.config.white_noise_std_kpa, mixed.shape
        )
        # No-contact gating is applied after the stateful response so neither
        # filter tails nor bias can create an air-gap tactile signal.
        output[~contact] = 0.0
        output = np.clip(output, 0.0, self.config.saturation_kpa)
        return (
            np.round(output / self.config.resolution_kpa)
            * self.config.resolution_kpa
        ).astype(np.float32)
