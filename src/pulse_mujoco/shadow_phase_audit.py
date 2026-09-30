from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .task import TaskPhase


def _finite_action(value: Any) -> np.ndarray | None:
    try:
        action = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if action.shape != (7,) or not np.isfinite(action).all():
        return None
    return action


def audit_shadow_phase(
    collection: Mapping[str, Any],
    shadows: Mapping[str, Mapping[str, Any]],
    *,
    target_phase: TaskPhase | str,
    expected_episodes: int,
    minimum_predictions_per_episode: int,
    maximum_safety_modified_fraction: float,
    maximum_mean_translation_error_mm: float,
    maximum_mean_rotation_error_rad: float,
    maximum_uncertainty: float,
    maximum_mean_pressure_error_kpa: float | None = None,
    target_label: str | None = None,
) -> dict[str, Any]:
    """Audit only the phase a policy is proposed to control.

    Shadow files contain predictions from other expert-owned phases so the
    unscoped summary is not a valid takeover metric. Missing or malformed
    episode records fail closed instead of silently disappearing.
    """
    phase = (
        target_phase
        if isinstance(target_phase, TaskPhase)
        else TaskPhase[str(target_phase).upper()]
    )
    if expected_episodes < 1 or minimum_predictions_per_episode < 1:
        raise ValueError("episode and prediction minimums must be positive")
    for value, name in (
        (maximum_safety_modified_fraction, "safety fraction"),
        (maximum_mean_translation_error_mm, "translation error"),
        (maximum_mean_rotation_error_rad, "rotation error"),
        (maximum_uncertainty, "uncertainty"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"maximum {name} must be finite and non-negative")
    if maximum_mean_pressure_error_kpa is not None and (
        not np.isfinite(maximum_mean_pressure_error_kpa)
        or maximum_mean_pressure_error_kpa < 0.0
    ):
        raise ValueError(
            "maximum pressure error must be finite and non-negative"
        )

    raw_episodes = collection.get("episodes")
    episodes = list(raw_episodes) if isinstance(raw_episodes, Sequence) else []
    failures: list[str] = []
    episode_reports: list[dict[str, Any]] = []
    all_translation_errors: list[float] = []
    all_rotation_errors: list[float] = []
    all_uncertainties: list[float] = []
    all_pressure_errors: list[float] = []
    safety_modified = 0
    target_records = 0

    if len(episodes) != expected_episodes:
        failures.append("episode_count_mismatch")

    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            failures.append(f"episode[{index}]:not_a_mapping")
            continue
        name = str(episode.get("episode", f"episode[{index}]"))
        if episode.get("accepted") is not True:
            failures.append(f"{name}:not_accepted")
        if episode.get("success") is not True:
            failures.append(f"{name}:not_successful")
        if episode.get("signal_quality_accepted") is not True:
            failures.append(f"{name}:signal_quality_rejected")
        if int(episode.get("forbidden_contact_events", -1)) != 0:
            failures.append(f"{name}:forbidden_contact")
        if episode.get("policy_action_executed") is not False:
            failures.append(f"{name}:policy_action_executed_in_shadow")

        shadow = shadows.get(name)
        if not isinstance(shadow, Mapping):
            failures.append(f"{name}:shadow_missing")
            continue
        raw_records = shadow.get("records")
        records = list(raw_records) if isinstance(raw_records, Sequence) else []
        phase_candidates = [
            record
            for record in records
            if isinstance(record, Mapping)
            and int(record.get("task_phase", -1)) == int(phase)
        ]
        if target_label is not None:
            if any("expert_label" not in record for record in phase_candidates):
                failures.append(f"{name}:target_label_missing")
            phase_records = [
                record
                for record in phase_candidates
                if record.get("expert_label") == target_label
            ]
        else:
            phase_records = phase_candidates
        if len(phase_records) < minimum_predictions_per_episode:
            failures.append(f"{name}:insufficient_target_predictions")

        episode_modified = 0
        episode_translation: list[float] = []
        episode_rotation: list[float] = []
        episode_uncertainty: list[float] = []
        episode_pressure: list[float] = []
        for record_index, record in enumerate(phase_records):
            predicted = _finite_action(record.get("predicted_first_action"))
            expert = _finite_action(record.get("expert_executed_action"))
            uncertainty = record.get("policy_uncertainty")
            if predicted is None or expert is None:
                failures.append(f"{name}:record[{record_index}]:invalid_action")
                continue
            try:
                uncertainty_value = float(uncertainty)
            except (TypeError, ValueError):
                failures.append(
                    f"{name}:record[{record_index}]:invalid_uncertainty"
                )
                continue
            if not np.isfinite(uncertainty_value):
                failures.append(
                    f"{name}:record[{record_index}]:invalid_uncertainty"
                )
                continue
            difference = predicted - expert
            translation = float(np.linalg.norm(difference[:3]) * 1000.0)
            rotation = float(np.linalg.norm(difference[3:6]))
            modified = bool(record.get("safety_modified", False))
            episode_modified += int(modified)
            episode_translation.append(translation)
            episode_rotation.append(rotation)
            episode_uncertainty.append(uncertainty_value)
            episode_pressure.append(float(abs(difference[6])))

        valid_count = len(episode_translation)
        target_records += valid_count
        safety_modified += episode_modified
        all_translation_errors.extend(episode_translation)
        all_rotation_errors.extend(episode_rotation)
        all_uncertainties.extend(episode_uncertainty)
        all_pressure_errors.extend(episode_pressure)
        episode_reports.append(
            {
                "episode": name,
                "target_predictions": valid_count,
                "safety_modified": episode_modified,
                "mean_translation_error_mm": (
                    float(np.mean(episode_translation))
                    if episode_translation
                    else None
                ),
                "mean_rotation_error_rad": (
                    float(np.mean(episode_rotation))
                    if episode_rotation
                    else None
                ),
                "maximum_uncertainty": (
                    float(np.max(episode_uncertainty))
                    if episode_uncertainty
                    else None
                ),
                "mean_pressure_error_kpa": (
                    float(np.mean(episode_pressure))
                    if episode_pressure
                    else None
                ),
            }
        )

    safety_fraction = (
        float(safety_modified / target_records) if target_records else 1.0
    )
    mean_translation = (
        float(np.mean(all_translation_errors))
        if all_translation_errors
        else float("inf")
    )
    mean_rotation = (
        float(np.mean(all_rotation_errors))
        if all_rotation_errors
        else float("inf")
    )
    observed_uncertainty = (
        float(np.max(all_uncertainties)) if all_uncertainties else float("inf")
    )
    mean_pressure = (
        float(np.mean(all_pressure_errors))
        if all_pressure_errors
        else float("inf")
    )
    if safety_fraction > maximum_safety_modified_fraction + 1e-12:
        failures.append("target_phase_safety_modification_exceeded")
    if mean_translation > maximum_mean_translation_error_mm + 1e-12:
        failures.append("target_phase_translation_error_exceeded")
    if mean_rotation > maximum_mean_rotation_error_rad + 1e-12:
        failures.append("target_phase_rotation_error_exceeded")
    if observed_uncertainty > maximum_uncertainty + 1e-12:
        failures.append("target_phase_uncertainty_exceeded")
    if (
        maximum_mean_pressure_error_kpa is not None
        and mean_pressure > maximum_mean_pressure_error_kpa + 1e-12
    ):
        failures.append("target_phase_pressure_error_exceeded")

    unique_failures = sorted(set(failures))
    return {
        "schema_version": "adam_pulse_shadow_phase_audit_v1",
        "passed": not unique_failures,
        "failures": unique_failures,
        "target_phase": phase.name.lower(),
        "target_label": target_label,
        "expected_episodes": int(expected_episodes),
        "observed_episodes": len(episodes),
        "minimum_predictions_per_episode": int(
            minimum_predictions_per_episode
        ),
        "target_predictions": int(target_records),
        "safety_modified": int(safety_modified),
        "safety_modified_fraction": safety_fraction,
        "mean_translation_error_mm": mean_translation,
        "maximum_translation_error_mm": (
            float(np.max(all_translation_errors))
            if all_translation_errors
            else None
        ),
        "mean_rotation_error_rad": mean_rotation,
        "maximum_rotation_error_rad": (
            float(np.max(all_rotation_errors))
            if all_rotation_errors
            else None
        ),
        "mean_uncertainty": (
            float(np.mean(all_uncertainties)) if all_uncertainties else None
        ),
        "maximum_uncertainty": observed_uncertainty,
        "mean_pressure_error_kpa": mean_pressure,
        "maximum_pressure_error_kpa": (
            float(np.max(all_pressure_errors))
            if all_pressure_errors
            else None
        ),
        "thresholds": {
            "maximum_safety_modified_fraction": float(
                maximum_safety_modified_fraction
            ),
            "maximum_mean_translation_error_mm": float(
                maximum_mean_translation_error_mm
            ),
            "maximum_mean_rotation_error_rad": float(
                maximum_mean_rotation_error_rad
            ),
            "maximum_uncertainty": float(maximum_uncertainty),
            "maximum_mean_pressure_error_kpa": (
                None
                if maximum_mean_pressure_error_kpa is None
                else float(maximum_mean_pressure_error_kpa)
            ),
        },
        "episodes": episode_reports,
    }
