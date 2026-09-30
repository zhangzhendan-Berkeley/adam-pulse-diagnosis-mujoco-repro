from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import PulseAction
from .observation import PulseObservation


def _meaningfully_clipped(
    original: np.ndarray, clipped: np.ndarray, limit: float
) -> bool:
    """Ignore only float32 quantization at an otherwise exact safety bound."""

    tolerance = float(np.spacing(np.float32(abs(limit))))
    return bool(np.any(np.abs(original - clipped) > tolerance))


@dataclass(frozen=True)
class SafetyLimits:
    maximum_cartesian_step_m: float = 0.00075
    maximum_rotation_step_rad: float = 0.02
    maximum_target_pressure_kpa: float = 150.0
    maximum_channel_pressure_kpa: float = 1500.0
    # Leave margin below the task's hard 8 N/finger acceptance ceiling so
    # one policy interval of contact dynamics cannot turn a warning into an
    # unsafe sample. This is measured force, not privileged pulse truth.
    maximum_normal_force_n: float = 7.5
    minimum_valid_proximity_channels: int = 3
    retreat_step_m: float = 0.002


class SafetySupervisor:
    def __init__(self, limits: SafetyLimits | None = None):
        self.limits = limits or SafetyLimits()

    def filter_action(
        self,
        observation: PulseObservation,
        proposed_action: PulseAction,
        backend_info: dict | None = None,
    ) -> tuple[PulseAction, dict]:
        backend_info = backend_info or {}
        reasons = []
        raw = proposed_action.as_array()
        hard_stop = not np.isfinite(raw).all()
        xyz = np.nan_to_num(
            proposed_action.delta_tcp_xyz_m, copy=True
        )
        rpy = np.nan_to_num(
            proposed_action.delta_tcp_rpy_rad, copy=True
        )
        if hard_stop:
            xyz.fill(0.0)
            rpy.fill(0.0)
            reasons.append("non_finite_action")
        clipped_xyz = np.clip(
            xyz,
            -self.limits.maximum_cartesian_step_m,
            self.limits.maximum_cartesian_step_m,
        )
        clipped_rpy = np.clip(
            rpy,
            -self.limits.maximum_rotation_step_rad,
            self.limits.maximum_rotation_step_rad,
        )
        if _meaningfully_clipped(
            xyz, clipped_xyz, self.limits.maximum_cartesian_step_m
        ):
            reasons.append("cartesian_step_clipped")
        if _meaningfully_clipped(
            rpy, clipped_rpy, self.limits.maximum_rotation_step_rad
        ):
            reasons.append("rotation_step_clipped")
        pressure_risk = bool(
            observation.tactile_kpa.max()
            > self.limits.maximum_channel_pressure_kpa
        )
        normal_force = np.asarray(
            backend_info.get("normal_force_n", (0.0, 0.0)), dtype=float
        )
        force_risk = bool(
            normal_force.size
            and np.isfinite(normal_force).all()
            and float(np.max(normal_force))
            >= self.limits.maximum_normal_force_n
        )
        collision_risk = not bool(
            backend_info.get("collision_safe", True)
        )
        near = bool(
            not observation.contact.any()
            and np.any(
                observation.proximity_valid
                & (observation.proximity_m < 0.03)
            )
        )
        proximity_lost = bool(
            near
            and np.count_nonzero(observation.proximity_valid)
            < self.limits.minimum_valid_proximity_channels
        )
        retreat = pressure_risk or force_risk or collision_risk or proximity_lost
        if retreat:
            clipped_xyz[:] = (0.0, 0.0, self.limits.retreat_step_m)
            clipped_rpy.fill(0.0)
            reasons.append(
                "pressure_retreat"
                if pressure_risk
                else "force_retreat"
                if force_risk
                else "collision_retreat"
                if collision_risk
                else "proximity_invalid_retreat"
            )
        target_pressure_kpa = (
            0.0
            if hard_stop or retreat
            else float(
                np.clip(
                    proposed_action.target_pressure_kpa,
                    0.0,
                    self.limits.maximum_target_pressure_kpa,
                )
            )
        )
        safe = PulseAction(
            clipped_xyz,
            clipped_rpy,
            target_pressure_kpa,
            proposed_action.hand_mode,
            proposed_action.phase_command,
        )
        return safe, {
            "modified": bool(reasons),
            "hard_stop": hard_stop,
            "retreat": retreat,
            "reasons": reasons,
        }
