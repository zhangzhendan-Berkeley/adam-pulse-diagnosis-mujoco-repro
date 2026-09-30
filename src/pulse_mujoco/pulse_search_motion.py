from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PulseSearchState(str, Enum):
    INITIAL_CENTER_CONTACT = "INITIAL_CENTER_CONTACT"
    LIFT_FROM_CURRENT = "LIFT_FROM_CURRENT"
    VERIFY_CONTACT_RELEASE = "VERIFY_CONTACT_RELEASE"
    TRANSLATE_ABOVE_SURFACE = "TRANSLATE_ABOVE_SURFACE"
    VERIFY_TARGET_REACHED = "VERIFY_TARGET_REACHED"
    DESCEND_TO_NEW_POINT = "DESCEND_TO_NEW_POINT"
    ESTABLISH_DUAL_CONTACT = "ESTABLISH_DUAL_CONTACT"
    REGULATE_PRESSURE = "REGULATE_PRESSURE"
    SETTLE_AT_POINT = "SETTLE_AT_POINT"
    SAMPLE_POINT = "SAMPLE_POINT"
    SELECT_BEST_POINT = "SELECT_BEST_POINT"
    RECENTER_SEARCH = "RECENTER_SEARCH"
    FINAL_HOLD = "FINAL_HOLD"


@dataclass(frozen=True)
class SurfaceFrame:
    tangent_x: np.ndarray
    tangent_y: np.ndarray
    normal: np.ndarray


def build_surface_frame(
    patient_axis_xy, surface_height, xy, epsilon_m=1e-4
) -> SurfaceFrame:
    """Direct migration of the Isaac expert's local surface frame."""
    xy = np.asarray(xy, dtype=float)
    axis = np.asarray(
        [patient_axis_xy[0], patient_axis_xy[1], 0.0], dtype=float
    )
    dzdx = (
        float(surface_height(xy + np.asarray([epsilon_m, 0.0])))
        - float(surface_height(xy - np.asarray([epsilon_m, 0.0])))
    ) / (2.0 * epsilon_m)
    dzdy = (
        float(surface_height(xy + np.asarray([0.0, epsilon_m])))
        - float(surface_height(xy - np.asarray([0.0, epsilon_m])))
    ) / (2.0 * epsilon_m)
    normal = np.asarray([-dzdx, -dzdy, 1.0], dtype=float)
    normal /= np.linalg.norm(normal)
    tangent_x = axis - np.dot(axis, normal) * normal
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.linalg.norm(tangent_y)
    tangent_x = np.cross(tangent_y, normal)
    tangent_x /= np.linalg.norm(tangent_x)
    return SurfaceFrame(tangent_x, tangent_y, normal)


def unique_position_count(positions_m, threshold_m=0.0005) -> int:
    representatives: list[np.ndarray] = []
    for position in np.asarray(positions_m, dtype=float):
        if not any(
            np.linalg.norm(position - representative) <= threshold_m
            for representative in representatives
        ):
            representatives.append(position.copy())
    return len(representatives)


def best_finite_score_index(scores) -> int | None:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("pulse search scores must be one-dimensional")
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return None
    return int(finite[int(np.argmax(values[finite]))])


def clip_target_to_search_radius(
    origin_xy, target_xy, maximum_radius_m
) -> tuple[np.ndarray, bool]:
    """Project a public model target onto the deterministic search disk.

    This is a safety limit, not an expert correction: direction is preserved
    and no pulse truth or expert target is consulted.
    """

    origin = np.asarray(origin_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    radius = float(maximum_radius_m)
    if origin.shape != (2,) or target.shape != (2,):
        raise ValueError("origin_xy and target_xy must have shape (2,)")
    if not np.isfinite(origin).all() or not np.isfinite(target).all():
        raise ValueError("search radius projection inputs must be finite")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("maximum_radius_m must be finite and positive")
    offset = target - origin
    norm = float(np.linalg.norm(offset))
    if norm <= radius + 1e-12:
        return target.copy(), False
    return origin + offset * (radius / norm), True


def validate_three_stage_transitions(states) -> bool:
    """Require release and free-space translation before every descent."""
    values = [
        state.value if isinstance(state, PulseSearchState) else str(state)
        for state in states
    ]
    required = [
        PulseSearchState.LIFT_FROM_CURRENT.value,
        PulseSearchState.VERIFY_CONTACT_RELEASE.value,
        PulseSearchState.TRANSLATE_ABOVE_SURFACE.value,
        PulseSearchState.VERIFY_TARGET_REACHED.value,
        PulseSearchState.DESCEND_TO_NEW_POINT.value,
        PulseSearchState.ESTABLISH_DUAL_CONTACT.value,
        PulseSearchState.REGULATE_PRESSURE.value,
    ]
    cursor = 0
    for value in values:
        if value == required[cursor]:
            cursor += 1
            if cursor == len(required):
                return True
    return False
