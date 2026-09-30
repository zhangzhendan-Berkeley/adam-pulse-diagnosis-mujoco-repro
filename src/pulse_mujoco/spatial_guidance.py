from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tactile_layouts import TEN_CHANNEL_COORDS_MM


@dataclass(frozen=True)
class PulseSpatialEstimate:
    center_mm: np.ndarray
    offset_from_pad_center_mm: np.ndarray
    direction_unit: np.ndarray
    direction_confidence: float
    strongest_channel: int


def estimate_pulse_center_mm(
    channel_amplitude: np.ndarray,
    coordinates_mm: np.ndarray = TEN_CHANNEL_COORDS_MM,
    weight_exponent: float = 1.5,
) -> PulseSpatialEstimate:
    """Direct migration of the Isaac per-pad tactile spatial estimator."""
    amplitude = np.asarray(channel_amplitude, dtype=float).reshape(-1)
    coordinates = np.asarray(coordinates_mm, dtype=float)
    if coordinates.shape != (len(amplitude), 2):
        raise ValueError("coordinates_mm must have shape [channels, 2]")
    if not np.all(np.isfinite(amplitude)) or np.any(amplitude < 0.0):
        raise ValueError(
            "channel_amplitude must contain finite non-negative values"
        )
    pad_center = coordinates.mean(axis=0)
    floor = float(np.percentile(amplitude, 20.0))
    weights = np.maximum(amplitude - floor, 0.0) ** float(weight_exponent)
    center = (
        pad_center.copy()
        if float(weights.sum()) <= 1e-12
        else np.average(coordinates, axis=0, weights=weights)
    )
    offset = center - pad_center
    distance = float(np.linalg.norm(offset))
    direction = (
        offset / distance
        if distance > 1e-9
        else np.zeros(2, dtype=float)
    )
    amplitude_max = float(amplitude.max(initial=0.0))
    contrast = (
        float(
            (amplitude_max - float(np.median(amplitude)))
            / amplitude_max
        )
        if amplitude_max > 1e-12
        else 0.0
    )
    layout_radius = float(
        np.max(
            np.linalg.norm(coordinates - pad_center, axis=1),
            initial=1.0,
        )
    )
    separation = np.clip(
        distance / max(0.25 * layout_radius, 1e-9), 0.0, 1.0
    )
    return PulseSpatialEstimate(
        center_mm=center.astype(np.float32),
        offset_from_pad_center_mm=offset.astype(np.float32),
        direction_unit=direction.astype(np.float32),
        direction_confidence=float(np.clip(contrast * separation, 0.0, 1.0)),
        strongest_channel=int(np.argmax(amplitude)) + 1,
    )
