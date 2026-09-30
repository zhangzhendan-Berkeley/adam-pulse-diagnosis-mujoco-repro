from __future__ import annotations

import numpy as np


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def compute_preload_gain(
    force_n,
    minimum_force_n=0.20,
    maximum_force_n=2.0,
    rise_width_n=0.05,
    fall_width_n=0.15,
):
    force = np.asarray(force_n, dtype=float)
    return (
        _sigmoid((force - minimum_force_n) / rise_width_n)
        * _sigmoid((maximum_force_n - force) / fall_width_n)
    ).astype(np.float32)


def compute_orientation_gain(
    fingerpad_normals: np.ndarray,
    skin_normal: np.ndarray,
    exponent: float = 2.0,
) -> np.ndarray:
    normals = np.asarray(fingerpad_normals, dtype=float)
    skin = np.asarray(skin_normal, dtype=float)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    skin /= np.linalg.norm(skin) + 1e-8
    # Pad outward normals and the skin outward normal oppose one another at contact.
    alignment = -np.sum(normals * skin[None], axis=-1)
    return np.clip(alignment, 0.0, 1.0).astype(np.float32) ** float(exponent)


def compute_balance_gain(force_n: np.ndarray, sigma_n: float = 0.30) -> float:
    force = np.asarray(force_n, dtype=float)
    return float(np.exp(-0.5 * ((force[0] - force[1]) / float(sigma_n)) ** 2))

