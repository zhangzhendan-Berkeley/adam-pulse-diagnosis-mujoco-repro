from __future__ import annotations

import numpy as np

from .task import TaskPhase


EVENT_TAG_NAMES = (
    "fine_approach",
    "pressure_regulation",
    "pressure_ramp",
    "pressure_stable",
    "active_translation",
    "active_rotation",
    "pressure_transition",
    "contact_transition",
    "search_move",
    "recenter",
    "recovery",
    "steady_hold",
    "stable_capture",
)


def _action(record: dict) -> np.ndarray:
    value = record.get("executed_action", [0.0] * 7)
    action = np.asarray(value, dtype=np.float64).reshape(-1)
    if action.size < 7:
        action = np.pad(action, (0, 7 - action.size))
    return action[:7]


def _phase(record: dict) -> TaskPhase | None:
    value = record.get("task_phase")
    if value is not None:
        try:
            return TaskPhase(int(value))
        except (TypeError, ValueError):
            pass
    name = str(record.get("task_phase_name", "")).upper()
    return TaskPhase.__members__.get(name)


def _recovery(record: dict) -> bool:
    perturbation = record.get("perturbation_state") or {}
    reasons = record.get("action_modification_reasons") or []
    label = str(record.get("label", "")).lower()
    return bool(
        perturbation.get("recovery_active", False)
        or perturbation.get("expert_takeover", False)
        or label == "single_finger_balance"
        or label.startswith("search_reacquire_")
        or bool(record.get("recovery_mode"))
        or any("recovery" in str(reason).lower() for reason in reasons)
    )


def transition_event_tags(
    previous: dict,
    current: dict,
    *,
    translation_threshold_m: float = 5e-4,
    rotation_threshold_rad: float = 2e-3,
    pressure_threshold_kpa: float = 1e-3,
) -> dict[str, bool]:
    previous_action = _action(previous)
    action = _action(current)
    translation = float(np.linalg.norm(action[:3]))
    rotation = float(np.linalg.norm(action[3:6]))
    pressure_delta = float(action[6] - previous_action[6])
    previous_contacts = int(np.asarray(previous.get("contact", []), dtype=bool).sum())
    contacts = int(np.asarray(current.get("contact", []), dtype=bool).sum())
    phase = _phase(current)
    fine_approach = phase == TaskPhase.FINE_APPROACH
    pressure_regulation = phase == TaskPhase.PRESSURE_REGULATION
    active_translation = translation >= translation_threshold_m
    active_rotation = rotation >= rotation_threshold_rad
    pressure_transition = abs(pressure_delta) >= pressure_threshold_kpa
    pressure_ramp = bool(pressure_regulation and pressure_transition)
    contact_transition = contacts != previous_contacts
    recovery = _recovery(current)
    search_move = bool(
        phase == TaskPhase.PULSE_SEARCH
        and (active_translation or active_rotation)
    )
    recenter = phase == TaskPhase.RECENTER
    steady_hold = bool(
        contacts >= 2
        and phase
        in {
            TaskPhase.PRESSURE_REGULATION,
            TaskPhase.PULSE_SEARCH,
            TaskPhase.STABLE_HOLD,
            TaskPhase.SIGNAL_CAPTURE,
        }
        and not active_translation
        and not active_rotation
        and not pressure_transition
        and not contact_transition
        and not recovery
    )
    pressure_stable = bool(
        pressure_regulation
        and contacts >= 2
        and not active_translation
        and not active_rotation
        and not pressure_transition
        and not contact_transition
        and not recovery
    )
    stable_capture = bool(
        phase == TaskPhase.SIGNAL_CAPTURE
        and contacts >= 2
        and not active_translation
        and not active_rotation
        and not pressure_transition
        and not contact_transition
        and not recovery
    )
    return {
        "fine_approach": fine_approach,
        "pressure_regulation": pressure_regulation,
        "pressure_ramp": pressure_ramp,
        "pressure_stable": pressure_stable,
        "active_translation": active_translation,
        "active_rotation": active_rotation,
        "pressure_transition": pressure_transition,
        "contact_transition": contact_transition,
        "search_move": search_move,
        "recenter": recenter,
        "recovery": recovery,
        "steady_hold": steady_hold,
        "stable_capture": stable_capture,
    }


def window_event_tags(
    records: list[dict],
    start: int,
    observation_horizon: int,
    prediction_horizon: int,
    **thresholds,
) -> dict[str, bool]:
    first_future = start + observation_horizon
    end = first_future + prediction_horizon
    if start < 0 or first_future <= 0 or end > len(records):
        raise IndexError("window exceeds episode boundaries")
    transitions = [
        transition_event_tags(records[index - 1], records[index], **thresholds)
        for index in range(first_future, end)
    ]
    result = {
        name: any(tags[name] for tags in transitions)
        for name in EVENT_TAG_NAMES
    }
    result["steady_hold"] = bool(
        transitions and all(tags["steady_hold"] for tags in transitions)
    )
    return result
