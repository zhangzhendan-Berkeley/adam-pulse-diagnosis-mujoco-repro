from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaticContactConfig:
    stiffness_coefficient: float = 8000.0
    damping_n_s_per_m: float = 4.0
    exponent: float = 1.5
    maximum_force_n: float = 5.0
    minimum_sigma_m: float = 0.0015
    maximum_sigma_m: float = 0.006
    indentation_expansion_gain: float = 0.08
    pressure_full_activation_kpa: float = 2.0


@dataclass(frozen=True)
class StaticContactFrame:
    pressure_kpa: np.ndarray
    taxel_force_n: np.ndarray
    contact_weights: np.ndarray
    total_normal_force_n: float
    contact_area_m2: float
    indentation_m: float
    contact_sigma_m: float
    in_contact: bool


def compute_normal_force(
    indentation_m: float,
    indentation_velocity_mps: float,
    config: StaticContactConfig,
) -> float:
    indentation = max(float(indentation_m), 0.0)
    if indentation <= 0.0:
        return 0.0
    elastic = config.stiffness_coefficient * indentation**config.exponent
    damping = config.damping_n_s_per_m * max(float(indentation_velocity_mps), 0.0)
    return float(np.clip(elastic + damping, 0.0, config.maximum_force_n))


def compute_contact_sigma(indentation_m: float, config: StaticContactConfig) -> float:
    sigma = config.minimum_sigma_m + config.indentation_expansion_gain * np.sqrt(
        max(float(indentation_m), 0.0)
    )
    return float(np.clip(sigma, config.minimum_sigma_m, config.maximum_sigma_m))


def distribute_contact_force(
    taxel_positions_local_m: np.ndarray,
    taxel_area_m2: np.ndarray,
    contact_points_local_m: np.ndarray,
    contact_strength: np.ndarray,
    total_normal_force_n: float,
    contact_sigma_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(taxel_positions_local_m, dtype=float)
    areas = np.asarray(taxel_area_m2, dtype=float)
    points = np.asarray(contact_points_local_m, dtype=float)
    strength = np.asarray(contact_strength, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("taxel_positions_local_m must have shape [C, 3]")
    channels = positions.shape[0]
    if areas.shape != (channels,):
        raise ValueError("taxel_area_m2 must have shape [C]")
    if points.size == 0 or total_normal_force_n <= 0.0:
        zeros = np.zeros(channels, dtype=np.float32)
        return zeros.copy(), zeros.copy(), zeros.copy()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("contact_points_local_m must have shape [N, 3]")
    if strength.shape != (points.shape[0],):
        raise ValueError("contact_strength must have shape [N]")
    delta = positions[:, None, 1:3] - points[None, :, 1:3]
    kernels = np.exp(
        -0.5 * np.sum(delta * delta, axis=-1) / max(contact_sigma_m**2, 1e-12)
    )
    if strength.sum() <= 1e-12:
        strength = np.ones_like(strength)
    raw = kernels @ (strength / strength.sum())
    if raw.sum() <= 1e-12:
        zeros = np.zeros(channels, dtype=np.float32)
        return zeros.copy(), zeros.copy(), zeros.copy()
    weights = raw / raw.sum()
    force = weights * float(total_normal_force_n)
    pressure = force / np.maximum(areas, 1e-12) / 1000.0
    return (
        pressure.astype(np.float32),
        force.astype(np.float32),
        weights.astype(np.float32),
    )


class StaticContactArrayModel:
    def __init__(self, config: StaticContactConfig | None = None):
        self.config = config or StaticContactConfig()
        self.previous_indentation_m = 0.0

    def reset(self) -> None:
        self.previous_indentation_m = 0.0

    def sample(
        self,
        indentation_m: float,
        dt_s: float,
        taxel_positions_local_m: np.ndarray,
        taxel_area_m2: np.ndarray,
        physical_force_n: float | None = None,
        contact_points_local_m: np.ndarray | None = None,
        contact_strength: np.ndarray | None = None,
    ) -> StaticContactFrame:
        indentation = max(float(indentation_m), 0.0)
        velocity = (indentation - self.previous_indentation_m) / max(float(dt_s), 1e-8)
        self.previous_indentation_m = indentation
        force = (
            compute_normal_force(indentation, velocity, self.config)
            if physical_force_n is None
            else float(np.clip(physical_force_n, 0.0, self.config.maximum_force_n))
        )
        # In PhysX mode the contact report is authoritative. Falling back to
        # analytic indentation here would recreate an air-gap signal whenever
        # the visual surface estimate and collision shape disagree.
        in_contact = bool(
            force > 0.01 if physical_force_n is not None else indentation > 0.0
        )
        sigma = compute_contact_sigma(indentation, self.config)
        points = (
            np.asarray([[0.0, 0.0, 0.0]], dtype=float)
            if contact_points_local_m is None
            else np.asarray(contact_points_local_m, dtype=float)
        )
        strength = (
            np.ones(len(points), dtype=float)
            if contact_strength is None
            else np.asarray(contact_strength, dtype=float)
        )
        pressure, taxel_force, weights = distribute_contact_force(
            taxel_positions_local_m,
            taxel_area_m2,
            points,
            strength,
            force if in_contact else 0.0,
            sigma,
        )
        activation = np.clip(
            pressure / max(self.config.pressure_full_activation_kpa, 1e-8), 0.0, 1.0
        )
        area = float(np.sum(np.asarray(taxel_area_m2) * activation))
        return StaticContactFrame(
            pressure, taxel_force, weights, force, area, indentation, sigma, in_contact
        )
