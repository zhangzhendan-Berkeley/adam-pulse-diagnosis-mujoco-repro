from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PURE_POLICY_MODE = "expert_free_with_deterministic_safety"
REQUIRED_EPISODE_FIELDS = (
    "policy_mode",
    "success",
    "safe_terminal",
    "expert_action_queries",
    "expert_direction_queries",
    "privileged_state_reads",
    "safety_modifications",
    "emergency_retreats",
    "forbidden_contact_events",
    "maximum_force_n_per_finger",
    "signal_quality_accepted",
    "truth_in_observation",
)


def _integer(record: Mapping[str, Any], key: str) -> int | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _forces(record: Mapping[str, Any]) -> tuple[float, ...] | None:
    value = record.get("maximum_force_n_per_finger")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    # Accept the legacy two-finger vector and the custom hand's three-to-five
    # active fingers, but reject nonphysical cardinalities fail-closed.
    if not 2 <= len(value) <= 5:
        return None
    try:
        forces = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if any(force < 0.0 or force != force for force in forces):
        return None
    return forces


def audit_pure_policy_summary(
    summary: Mapping[str, Any],
    *,
    expected_episodes: int,
    minimum_successes: int,
    maximum_force_n: float = 8.0,
) -> dict[str, Any]:
    """Fail-closed audit for a full-task expert-free policy run.

    Deterministic IK, collision filtering, action clipping and emergency
    retreat are permitted.  Missing provenance fields never default to zero.
    Posthoc simulator truth used by a separate evaluator is not a policy read
    and must not be included in ``privileged_state_reads``.
    """

    if expected_episodes < 1:
        raise ValueError("expected_episodes must be positive")
    if not 0 <= minimum_successes <= expected_episodes:
        raise ValueError("minimum_successes must be within episode count")
    if maximum_force_n <= 0.0:
        raise ValueError("maximum_force_n must be positive")

    raw_episodes = summary.get("episodes")
    episodes = list(raw_episodes) if isinstance(raw_episodes, Sequence) else []
    failures: list[str] = []
    successes = 0
    maximum_force = 0.0
    totals = {
        "expert_action_queries": 0,
        "expert_direction_queries": 0,
        "privileged_state_reads": 0,
        "safety_modifications": 0,
        "emergency_retreats": 0,
        "forbidden_contact_events": 0,
    }

    if len(episodes) != expected_episodes:
        failures.append("episode_count_mismatch")

    for index, raw in enumerate(episodes):
        name = f"episode[{index}]"
        if not isinstance(raw, Mapping):
            failures.append(f"{name}:not_a_mapping")
            continue
        missing = [key for key in REQUIRED_EPISODE_FIELDS if key not in raw]
        if missing:
            failures.extend(f"{name}:missing:{key}" for key in missing)
            continue
        if raw.get("policy_mode") != PURE_POLICY_MODE:
            failures.append(f"{name}:wrong_policy_mode")
        if raw.get("truth_in_observation") is not False:
            failures.append(f"{name}:truth_in_observation")

        for key in totals:
            value = _integer(raw, key)
            if value is None or value < 0:
                failures.append(f"{name}:invalid:{key}")
            else:
                totals[key] += value

        forces = _forces(raw)
        if forces is None:
            failures.append(f"{name}:invalid:maximum_force_n_per_finger")
        else:
            maximum_force = max(maximum_force, *forces)
            if max(forces) > maximum_force_n + 1e-6:
                failures.append(f"{name}:force_limit_exceeded")

        success = raw.get("success") is True
        if success:
            successes += 1
            if raw.get("signal_quality_accepted") is not True:
                failures.append(f"{name}:success_without_signal_quality")
        elif raw.get("safe_terminal") is not True:
            failures.append(f"{name}:failure_not_safely_terminated")

    if successes < minimum_successes:
        failures.append("insufficient_successes")
    if totals["expert_action_queries"] != 0:
        failures.append("expert_action_query_observed")
    if totals["expert_direction_queries"] != 0:
        failures.append("expert_direction_query_observed")
    if totals["privileged_state_reads"] != 0:
        failures.append("privileged_state_read_observed")
    if totals["forbidden_contact_events"] != 0:
        failures.append("forbidden_contact_observed")

    unique_failures = sorted(set(failures))
    return {
        "schema_version": "adam_pulse_pure_policy_audit_v1",
        "passed": not unique_failures,
        "failures": unique_failures,
        "expected_episodes": int(expected_episodes),
        "observed_episodes": len(episodes),
        "minimum_successes": int(minimum_successes),
        "successes": int(successes),
        "success_rate": successes / expected_episodes,
        "maximum_force_limit_n": float(maximum_force_n),
        "maximum_force_n": float(maximum_force),
        **totals,
    }
