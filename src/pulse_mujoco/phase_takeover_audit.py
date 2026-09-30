from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .task import TaskPhase


REQUIRED_PHASE_EPISODE_FIELDS = (
    "success",
    "safe_terminal",
    "controller",
    "privileged_state_reads",
    "forbidden_contact_events",
    "maximum_force_n_per_finger",
    "signal_quality_accepted",
    "truth_in_observation",
)

# Policy scopes can be narrower than the task-state enum. Stable capture is
# executed while the task is in SIGNAL_CAPTURE, so its controller counters use
# the task-phase name rather than the specialist policy-scope name.
POLICY_SCOPE_PHASE_ALIASES = {
    "stable_capture": TaskPhase.SIGNAL_CAPTURE.name.lower(),
}


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _phase_count(controller: Mapping[str, Any], key: str, phase: str) -> int | None:
    values = controller.get(key)
    if not isinstance(values, Mapping) or phase not in values:
        return None
    return _nonnegative_integer(values[phase])


def _forces(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    # The official-hand task reported index and middle only.  The custom
    # tactile hand may actively use three to five fingers.  Keep the vector
    # bounded to one physical hand so malformed concatenated arrays fail.
    if not 2 <= len(value) <= 5:
        return None
    try:
        forces = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if any(force < 0.0 or force != force for force in forces):
        return None
    return forces


def audit_phase_takeover_summary(
    summary: Mapping[str, Any],
    *,
    target_phase: TaskPhase | str,
    expected_episodes: int,
    minimum_successes: int,
    minimum_policy_fraction: float,
    require_zero_target_expert_steps: bool = False,
    maximum_force_n: float = 8.0,
) -> dict[str, Any]:
    """Fail-closed audit for one M5 phase-takeover gate.

    The target phase must be visited in every episode. Other task phases may
    still use the expert during M5, but their use cannot inflate the target
    phase policy fraction. Simulator truth reads on the control path remain
    forbidden. Missing per-phase counters never default to zero.
    """

    if isinstance(target_phase, TaskPhase):
        phase = target_phase.name.lower()
    else:
        requested_phase = str(target_phase).lower()
        phase = POLICY_SCOPE_PHASE_ALIASES.get(requested_phase, requested_phase)
        valid = {item.name.lower() for item in TaskPhase}
        if phase not in valid:
            raise ValueError(f"unknown task phase: {target_phase}")
    if expected_episodes < 1:
        raise ValueError("expected_episodes must be positive")
    if not 0 <= minimum_successes <= expected_episodes:
        raise ValueError("minimum_successes must be within episode count")
    if not 0.0 <= minimum_policy_fraction <= 1.0:
        raise ValueError("minimum_policy_fraction must be in [0, 1]")
    if maximum_force_n <= 0.0:
        raise ValueError("maximum_force_n must be positive")

    raw_episodes = summary.get("episodes")
    episodes = list(raw_episodes) if isinstance(raw_episodes, Sequence) else []
    failures: list[str] = []
    successes = 0
    policy_steps = 0
    expert_steps = 0
    privileged_reads = 0
    forbidden_contacts = 0
    maximum_force = 0.0

    if len(episodes) != expected_episodes:
        failures.append("episode_count_mismatch")

    for index, raw in enumerate(episodes):
        name = f"episode[{index}]"
        if not isinstance(raw, Mapping):
            failures.append(f"{name}:not_a_mapping")
            continue
        missing = [
            key for key in REQUIRED_PHASE_EPISODE_FIELDS if key not in raw
        ]
        if missing:
            failures.extend(f"{name}:missing:{key}" for key in missing)
            continue
        if raw.get("truth_in_observation") is not False:
            failures.append(f"{name}:truth_in_observation")

        reads = _nonnegative_integer(raw.get("privileged_state_reads"))
        contacts = _nonnegative_integer(raw.get("forbidden_contact_events"))
        if reads is None:
            failures.append(f"{name}:invalid:privileged_state_reads")
        else:
            privileged_reads += reads
        if contacts is None:
            failures.append(f"{name}:invalid:forbidden_contact_events")
        else:
            forbidden_contacts += contacts

        controller = raw.get("controller")
        if not isinstance(controller, Mapping):
            failures.append(f"{name}:invalid:controller")
        else:
            policy = _phase_count(
                controller, "policy_steps_by_phase", phase
            )
            expert = _phase_count(
                controller, "expert_steps_by_phase", phase
            )
            if policy is None:
                failures.append(
                    f"{name}:missing_target_phase_policy_counter"
                )
            else:
                policy_steps += policy
            if expert is None:
                failures.append(
                    f"{name}:missing_target_phase_expert_counter"
                )
            else:
                expert_steps += expert
            if policy is not None and expert is not None and policy + expert == 0:
                failures.append(f"{name}:target_phase_not_visited")

        forces = _forces(raw.get("maximum_force_n_per_finger"))
        if forces is None:
            failures.append(f"{name}:invalid:maximum_force_n_per_finger")
        else:
            maximum_force = max(maximum_force, *forces)
            if max(forces) > maximum_force_n + 1e-6:
                failures.append(f"{name}:force_limit_exceeded")

        if raw.get("success") is True:
            successes += 1
            if raw.get("signal_quality_accepted") is not True:
                failures.append(f"{name}:success_without_signal_quality")
        elif raw.get("safe_terminal") is not True:
            failures.append(f"{name}:failure_not_safely_terminated")

    target_steps = policy_steps + expert_steps
    policy_fraction = (
        float(policy_steps / target_steps) if target_steps > 0 else 0.0
    )
    if successes < minimum_successes:
        failures.append("insufficient_successes")
    if policy_fraction + 1e-12 < minimum_policy_fraction:
        failures.append("target_phase_policy_fraction_below_minimum")
    if require_zero_target_expert_steps and expert_steps != 0:
        failures.append("target_phase_expert_step_observed")
    if privileged_reads != 0:
        failures.append("privileged_state_read_observed")
    if forbidden_contacts != 0:
        failures.append("forbidden_contact_observed")

    unique_failures = sorted(set(failures))
    return {
        "schema_version": "adam_pulse_phase_takeover_audit_v1",
        "passed": not unique_failures,
        "failures": unique_failures,
        "target_phase": phase,
        "expected_episodes": int(expected_episodes),
        "observed_episodes": len(episodes),
        "minimum_successes": int(minimum_successes),
        "successes": int(successes),
        "minimum_policy_fraction": float(minimum_policy_fraction),
        "target_phase_policy_steps": int(policy_steps),
        "target_phase_expert_steps": int(expert_steps),
        "target_phase_policy_fraction": policy_fraction,
        "require_zero_target_expert_steps": bool(
            require_zero_target_expert_steps
        ),
        "privileged_state_reads": int(privileged_reads),
        "forbidden_contact_events": int(forbidden_contacts),
        "maximum_force_limit_n": float(maximum_force_n),
        "maximum_force_n": float(maximum_force),
    }
