from __future__ import annotations

import math


def force_target_to_nominal_pressure_kpa(
    target_force_n: float,
    taxel_area_m2: float,
    active_taxels: int,
) -> float:
    """Convert the Isaac per-finger force intent to nominal pad pressure."""
    force = float(target_force_n)
    area = float(taxel_area_m2)
    count = int(active_taxels)
    if not math.isfinite(force) or force < 0.0:
        raise ValueError("target_force_n must be finite and non-negative")
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("taxel_area_m2 must be finite and positive")
    if count <= 0:
        raise ValueError("active_taxels must be positive")
    return force / (area * count) / 1000.0


def nominal_pressure_to_force_target_n(
    target_pressure_kpa: float,
    taxel_area_m2: float,
    active_taxels: int,
) -> float:
    """Inverse mapping used by the MuJoCo force-control adapter."""
    pressure = float(target_pressure_kpa)
    area = float(taxel_area_m2)
    count = int(active_taxels)
    if not math.isfinite(pressure) or pressure < 0.0:
        raise ValueError(
            "target_pressure_kpa must be finite and non-negative"
        )
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("taxel_area_m2 must be finite and positive")
    if count <= 0:
        raise ValueError("active_taxels must be positive")
    return pressure * 1000.0 * area * count
