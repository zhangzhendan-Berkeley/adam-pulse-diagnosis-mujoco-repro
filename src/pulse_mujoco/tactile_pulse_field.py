from __future__ import annotations

import numpy as np


def compute_spatial_gain(
    taxel_positions_m: np.ndarray,
    artery_center_m: np.ndarray,
    artery_axis_xy: np.ndarray,
    sigma_lateral_m: float,
    sigma_axial_m: float,
) -> np.ndarray:
    positions = np.asarray(taxel_positions_m, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("taxel_positions_m must have shape [fingers, channels, 3]")
    axis = np.asarray(artery_axis_xy, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        raise ValueError("artery_axis_xy must be non-zero")
    axis /= norm
    relative = positions[..., :2] - np.asarray(artery_center_m, dtype=float)[:2]
    axial = np.sum(relative * axis, axis=-1)
    lateral = np.linalg.norm(relative - axial[..., None] * axis, axis=-1)
    return np.exp(
        -0.5 * (lateral / float(sigma_lateral_m)) ** 2
        - 0.5 * (axial / float(sigma_axial_m)) ** 2
    ).astype(np.float32)
