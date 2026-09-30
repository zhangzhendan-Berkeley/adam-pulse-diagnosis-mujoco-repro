from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from .actions import PulseAction
from .observation import PulseObservation


class PerturbationKind(str, Enum):
    PROXIMITY_BIAS = "proximity_bias"
    PROXIMITY_DROPOUT = "proximity_dropout"
    TACTILE_GAIN = "tactile_gain"
    TACTILE_SPIKE = "tactile_spike"
    ACTION_SCALE = "action_scale"
    COMMAND_DROP = "command_drop"


@dataclass(frozen=True)
class PerturbationEvent:
    event_id: str
    kind: PerturbationKind
    start_step: int
    end_step: int
    parameters: dict
    recoverable: bool = True

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("perturbation event_id must not be empty")
        object.__setattr__(self, "kind", PerturbationKind(self.kind))
        if self.start_step < 0 or self.end_step < self.start_step:
            raise ValueError("invalid perturbation step interval")

    def active(self, step: int) -> bool:
        return self.start_step <= int(step) <= self.end_step

    def as_dict(self, active_step: int | None = None) -> dict:
        return {
            "event_id": self.event_id,
            "kind": self.kind.value,
            "start_step": int(self.start_step),
            "end_step": int(self.end_step),
            "parameters": dict(self.parameters),
            "recoverable": bool(self.recoverable),
            "active": (
                None if active_step is None else self.active(active_step)
            ),
        }


class PerturbationScheduler:
    """Direct migration of bounded Isaac observation/action perturbations."""

    def __init__(self, events=()):
        self.events = tuple(events)
        identifiers = [event.event_id for event in self.events]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("perturbation event IDs must be unique")

    def active_events(self, step: int) -> tuple[PerturbationEvent, ...]:
        return tuple(event for event in self.events if event.active(step))

    def state(self, step: int) -> dict:
        active = self.active_events(step)
        recovery_events = tuple(
            event
            for event in active
            if event.kind == PerturbationKind.COMMAND_DROP
            or (
                event.kind == PerturbationKind.ACTION_SCALE
                and bool(event.parameters.get("recovery_label", False))
            )
        )
        return {
            "step": int(step),
            "active": bool(active),
            "events": [
                event.as_dict(active_step=step) for event in active
            ],
            "recoverable": all(event.recoverable for event in active),
            # Only transient execution failures are recovery labels.  The
            # always-on, mild action-scale domain randomization is not a
            # recovery by itself; callers can opt in with recovery_label for
            # a deliberately transient scale disturbance.
            "recovery_active": bool(recovery_events),
            "recovery_modes": [event.kind.value for event in recovery_events],
        }

    def apply_observation(
        self, observation: PulseObservation, step: int
    ) -> PulseObservation:
        proximity = observation.proximity_m.copy()
        proximity_valid = observation.proximity_valid.copy()
        tactile = observation.tactile_kpa.copy()
        for event in self.active_events(step):
            parameters = event.parameters
            if event.kind == PerturbationKind.PROXIMITY_BIAS:
                bias = np.asarray(
                    parameters["bias_m"], dtype=np.float32
                )
                if bias.shape == ():
                    bias = np.full(5, float(bias), dtype=np.float32)
                if bias.shape != (5,):
                    raise ValueError(
                        "proximity bias_m must be scalar or shape (5,)"
                    )
                proximity[proximity_valid] += bias[proximity_valid]
                proximity = np.maximum(proximity, 0.0)
            elif event.kind == PerturbationKind.PROXIMITY_DROPOUT:
                channels = np.atleast_1d(
                    parameters["channels"]
                ).astype(int)
                if np.any((channels < 0) | (channels >= 5)):
                    raise ValueError(
                        "proximity dropout channel is out of range"
                    )
                proximity_valid[channels] = False
            elif event.kind == PerturbationKind.TACTILE_GAIN:
                tactile *= np.asarray(
                    parameters["gain"], dtype=np.float32
                )
            elif event.kind == PerturbationKind.TACTILE_SPIKE:
                tactile += float(parameters["amplitude_kpa"])
        return replace(
            observation,
            proximity_m=proximity,
            proximity_valid=proximity_valid,
            tactile_kpa=np.maximum(tactile, 0.0),
        )

    def apply_action(
        self, action: PulseAction, step: int
    ) -> PulseAction:
        xyz = action.delta_tcp_xyz_m.copy()
        rpy = action.delta_tcp_rpy_rad.copy()
        pressure = action.target_pressure_kpa
        for event in self.active_events(step):
            if event.kind == PerturbationKind.ACTION_SCALE:
                scale = float(event.parameters["scale"])
                xyz *= scale
                rpy *= scale
            elif event.kind == PerturbationKind.COMMAND_DROP:
                xyz.fill(0.0)
                rpy.fill(0.0)
                pressure = 0.0
        return PulseAction(
            xyz,
            rpy,
            target_pressure_kpa=pressure,
            hand_mode=action.hand_mode,
            phase_command=action.phase_command,
        )
