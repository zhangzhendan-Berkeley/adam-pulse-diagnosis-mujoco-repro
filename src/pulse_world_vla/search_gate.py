from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def search_path_metrics(
    *,
    search_samples: Sequence[Mapping[str, Any]],
    policy_records: Sequence[Mapping[str, Any]],
    pulse_truth_xy_m: Sequence[float],
    patient_lateral_xy: Sequence[float],
    patient_axis_xy: Sequence[float],
    selected_index: int | None,
    expected_recovery_uv_m: Sequence[float] | None = None,
) -> dict:
    positions = np.asarray(
        [sample["position_m"][:2] for sample in search_samples], dtype=float
    )
    path_length_m = float(
        np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    ) if len(positions) > 1 else 0.0
    executed = [
        np.asarray(record["executed_delta_uv_m"], dtype=float)
        for record in policy_records
        if bool(record.get("policy_target_executed", False))
        and record.get("executed_delta_uv_m") is not None
    ]
    reversals = 0
    for previous, current in zip(executed, executed[1:]):
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(current))
        if denominator > 1e-12 and float(np.dot(previous, current)) < 0.0:
            reversals += 1

    first_step = executed[0] if executed else None
    absolute_truth_uv = None
    scenario_offset_uv = (
        None
        if expected_recovery_uv_m is None
        else np.asarray(expected_recovery_uv_m, dtype=float)
    )
    direction_cosine = None
    direction_sign_accuracy = None
    if len(positions):
        truth_world = np.asarray(pulse_truth_xy_m, dtype=float) - positions[0]
        absolute_truth_uv = np.asarray(
            [
                np.dot(truth_world, np.asarray(patient_lateral_xy, dtype=float)),
                np.dot(truth_world, np.asarray(patient_axis_xy, dtype=float)),
            ],
            dtype=float,
        )
    # Direction quality must always use the actual current probe-to-pulse
    # vector.  The optional scenario offset describes how pulse truth was
    # perturbed from its baseline; it is not a recovery command when the
    # stable-contact probe is already offset from that baseline.
    direction_reference_uv = absolute_truth_uv
    if first_step is not None and direction_reference_uv is not None:
        denominator = float(
            np.linalg.norm(first_step) * np.linalg.norm(direction_reference_uv)
        )
        if denominator > 1e-12:
            direction_cosine = float(
                np.dot(first_step, direction_reference_uv) / denominator
            )
        active = np.abs(direction_reference_uv) >= 0.001
        if np.any(active):
            direction_sign_accuracy = float(
                np.mean(
                    np.sign(first_step[active])
                    == np.sign(direction_reference_uv[active])
                )
            )

    initial_score = (
        _finite(search_samples[0].get("pulse_score_kpa"))
        if search_samples
        else None
    )
    selected_sample = (
        None
        if selected_index is None
        or not 0 <= int(selected_index) < len(search_samples)
        else search_samples[int(selected_index)]
    )
    final_score = (
        None
        if selected_sample is None
        else _finite(selected_sample.get("pulse_score_kpa"))
    )
    quality = (
        {}
        if selected_sample is None
        else selected_sample.get("signal_quality", {}) or {}
    )
    coverage = quality.get("tactile_coverage", {}) or {}
    initial_quality = (
        {} if not search_samples else search_samples[0].get("signal_quality", {}) or {}
    )

    def quality_margin(item: Mapping[str, Any]) -> float | None:
        any_peak = _finite(item.get("any_finger_peak_to_peak_kpa"))
        aggregate = _finite(item.get("aggregate_peak_to_peak_kpa"))
        if any_peak is None or aggregate is None:
            return None
        return float(min(any_peak / 0.30, aggregate / 0.18))

    initial_margin = quality_margin(initial_quality)
    final_margin = quality_margin(quality)
    return {
        "search_points": int(len(search_samples)),
        "path_length_mm": 1000.0 * path_length_m,
        "executed_segment_count": int(len(executed)),
        "first_step_uv_mm": (
            None if first_step is None else (1000.0 * first_step).tolist()
        ),
        "posthoc_truth_direction_uv_mm": (
            None
            if absolute_truth_uv is None
            else (1000.0 * absolute_truth_uv).tolist()
        ),
        "posthoc_expected_recovery_uv_mm": (
            None
            if scenario_offset_uv is None
            else (1000.0 * scenario_offset_uv).tolist()
        ),
        "direction_reference": "current_probe_to_pulse_truth",
        "first_step_direction_cosine": direction_cosine,
        "first_step_direction_sign_accuracy": direction_sign_accuracy,
        "direction_reversal_count": int(reversals),
        "initial_pulse_score_kpa": initial_score,
        "final_pulse_score_kpa": final_score,
        "pulse_score_improvement_kpa": (
            None
            if initial_score is None or final_score is None
            else final_score - initial_score
        ),
        "initial_signal_quality_margin": initial_margin,
        "final_signal_quality_margin": final_margin,
        "signal_quality_margin_improvement": (
            None
            if initial_margin is None or final_margin is None
            else final_margin - initial_margin
        ),
        "final_per_finger_centroid_offset_mm": coverage.get(
            "per_finger_weighted_centroid_offset_mm"
        ),
        "final_maximum_per_finger_centroid_offset_mm": coverage.get(
            "maximum_per_finger_weighted_centroid_offset_mm"
        ),
        "final_tactile_coverage_accepted": bool(coverage.get("accepted", False)),
    }


def search_gate_summary(cases: Sequence[Mapping[str, Any]]) -> dict:
    direction_cases = [
        case
        for case in cases
        if case.get("metrics", {}).get("first_step_direction_cosine") is not None
    ]
    return {
        "case_count": int(len(cases)),
        "success_count": int(sum(bool(case.get("success")) for case in cases)),
        "all_safe": bool(
            cases
            and all(
                int(case.get("forbidden_contact_events", -1)) == 0
                and int(case.get("safety_modifications", -1)) == 0
                for case in cases
            )
        ),
        "direction_case_count": int(len(direction_cases)),
        "mean_first_step_direction_cosine": (
            None
            if not direction_cases
            else float(
                np.mean(
                    [
                        case["metrics"]["first_step_direction_cosine"]
                        for case in direction_cases
                    ]
                )
            )
        ),
        "total_direction_reversals": int(
            sum(
                int(case.get("metrics", {}).get("direction_reversal_count", 0))
                for case in cases
            )
        ),
        "total_search_points": int(
            sum(int(case.get("metrics", {}).get("search_points", 0)) for case in cases)
        ),
        "total_contact_recovery_failures": int(
            sum(int(case.get("contact_recovery_failures", 0)) for case in cases)
        ),
    }
