from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PairedRecoveryCase:
    name: str
    pair_id: str
    requested_recovery_uv_mm: tuple[float, float]


def paired_recovery_cases(radius_mm: float) -> tuple[PairedRecoveryCase, ...]:
    """Return opposite recovery targets with shared pair identifiers."""
    radius = float(radius_mm)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("recovery radius must be finite and positive")
    diagonal = radius / np.sqrt(2.0)
    return (
        PairedRecoveryCase("lateral_positive", "lateral", (radius, 0.0)),
        PairedRecoveryCase("lateral_negative", "lateral", (-radius, 0.0)),
        PairedRecoveryCase("axial_positive", "axial", (0.0, radius)),
        PairedRecoveryCase("axial_negative", "axial", (0.0, -radius)),
        PairedRecoveryCase("diagonal_pp", "diagonal_same_sign", (diagonal, diagonal)),
        PairedRecoveryCase("diagonal_nn", "diagonal_same_sign", (-diagonal, -diagonal)),
        PairedRecoveryCase("diagonal_pn", "diagonal_opposite_sign", (diagonal, -diagonal)),
        PairedRecoveryCase("diagonal_np", "diagonal_opposite_sign", (-diagonal, diagonal)),
    )


def patient_uv_to_world_xy(
    delta_uv: Sequence[float],
    *,
    patient_lateral_xy: Sequence[float],
    patient_axis_xy: Sequence[float],
) -> np.ndarray:
    delta = np.asarray(delta_uv, dtype=float)
    lateral = np.asarray(patient_lateral_xy, dtype=float)
    axis = np.asarray(patient_axis_xy, dtype=float)
    if delta.shape != (2,) or lateral.shape != (2,) or axis.shape != (2,):
        raise ValueError("patient-frame vectors must have shape (2,)")
    if not all(np.isfinite(value).all() for value in (delta, lateral, axis)):
        raise ValueError("patient-frame vectors must be finite")
    return delta[0] * lateral + delta[1] * axis


def world_xy_to_patient_uv(
    delta_xy: Sequence[float],
    *,
    patient_lateral_xy: Sequence[float],
    patient_axis_xy: Sequence[float],
) -> np.ndarray:
    delta = np.asarray(delta_xy, dtype=float)
    lateral = np.asarray(patient_lateral_xy, dtype=float)
    axis = np.asarray(patient_axis_xy, dtype=float)
    if delta.shape != (2,) or lateral.shape != (2,) or axis.shape != (2,):
        raise ValueError("patient-frame vectors must have shape (2,)")
    if not all(np.isfinite(value).all() for value in (delta, lateral, axis)):
        raise ValueError("patient-frame vectors must be finite")
    return np.asarray([np.dot(delta, lateral), np.dot(delta, axis)], dtype=float)


def direction_bucket(delta_uv_m: Sequence[float], *, threshold_m: float = 0.00025) -> str:
    delta = np.asarray(delta_uv_m, dtype=float)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("direction delta must be finite shape (2,)")
    threshold = float(threshold_m)
    if threshold < 0.0:
        raise ValueError("direction threshold must be non-negative")
    u, v = delta
    u_name = "+u" if u >= threshold else "-u" if u <= -threshold else "u0"
    v_name = "+v" if v >= threshold else "-v" if v <= -threshold else "v0"
    return f"{u_name}/{v_name}"


def annotate_student_teacher_records(
    records: Sequence[Mapping[str, Any]],
    *,
    hard_error_mm: float = 1.0,
    sign_threshold_m: float = 0.00025,
) -> list[dict[str, Any]]:
    """Attach auditable hard-state labels without changing supervision."""
    if hard_error_mm <= 0.0:
        raise ValueError("hard error threshold must be positive")
    annotated: list[dict[str, Any]] = []
    for source in records:
        item = dict(source)
        expert_stop = item.get("expert_stop")
        if expert_stop is not False or item.get("expert_delta_uv_m") is None:
            item.update(
                {
                    "dagger_hard_state": False,
                    "dagger_hard_reasons": [],
                    "teacher_direction_bucket": "stop" if expert_stop else None,
                    "teacher_model_cosine": None,
                }
            )
            annotated.append(item)
            continue
        teacher = np.asarray(item["expert_delta_uv_m"], dtype=float)
        student = np.asarray(item["delta_uv_m"], dtype=float)
        if teacher.shape != (2,) or student.shape != (2,):
            raise ValueError("student and teacher deltas must have shape (2,)")
        error_mm = float(1000.0 * np.linalg.norm(student - teacher))
        reasons: list[str] = []
        if error_mm >= hard_error_mm:
            reasons.append("vector_error")
        for index, axis_name in enumerate(("u", "v")):
            if abs(float(teacher[index])) < sign_threshold_m:
                continue
            if float(student[index]) * float(teacher[index]) <= 0.0:
                reasons.append(f"{axis_name}_sign")
        denominator = float(np.linalg.norm(student) * np.linalg.norm(teacher))
        cosine = None if denominator <= 1e-12 else float(np.dot(student, teacher) / denominator)
        item.update(
            {
                "vector_error_mm": error_mm,
                "dagger_hard_state": bool(reasons),
                "dagger_hard_reasons": reasons,
                "teacher_direction_bucket": direction_bucket(
                    teacher, threshold_m=sign_threshold_m
                ),
                "teacher_model_cosine": cosine,
            }
        )
        annotated.append(item)
    return annotated


def paired_dagger_hard_sample_audit(
    policy_record: Mapping[str, Any] | None,
    plan: Mapping[str, Any] | None,
    target_delta_uv_m: Sequence[float],
) -> dict[str, Any]:
    """Validate one replay sample as a truth-free privileged hard state."""
    reasons: list[str] = []
    target = np.asarray(target_delta_uv_m, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("paired DAgger target must be finite shape (2,)")
    if policy_record is None:
        reasons.append("missing_student_shadow_record")
    if plan is None:
        reasons.append("missing_teacher_plan")
    if policy_record is not None:
        if policy_record.get("truth_in_observation") is not False:
            reasons.append("truth_observation_contract")
        if not bool(policy_record.get("dagger_hard_state", False)):
            reasons.append("not_student_hard_state")
        if policy_record.get("teacher_label_source") != "privileged_expert":
            reasons.append("student_record_label_not_privileged")
        expert_delta = np.asarray(
            policy_record.get("expert_delta_uv_m", ()), dtype=float
        )
        if expert_delta.shape != (2,) or not np.allclose(
            expert_delta, target, atol=1e-9, rtol=0.0
        ):
            reasons.append("student_teacher_target_mismatch")
    if plan is not None:
        if plan.get("label_source") != "privileged_expert":
            reasons.append("plan_label_not_privileged")
        plan_delta = np.asarray(
            plan.get("planned_delta_patient_uv_m", ()), dtype=float
        )
        if plan_delta.shape != (2,) or not np.allclose(
            plan_delta, target, atol=1e-9, rtol=0.0
        ):
            reasons.append("plan_target_mismatch")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "student_delta_uv_m": (
            None
            if policy_record is None
            else list(policy_record.get("delta_uv_m", ()))
        ),
        "vector_error_mm": (
            None
            if policy_record is None
            else policy_record.get("vector_error_mm")
        ),
        "teacher_model_cosine": (
            None
            if policy_record is None
            else policy_record.get("teacher_model_cosine")
        ),
        "hard_reasons": (
            []
            if policy_record is None
            else list(policy_record.get("dagger_hard_reasons", ()))
        ),
    }


def paired_dagger_first_state_sample_audit(
    policy_record: Mapping[str, Any] | None,
    plan: Mapping[str, Any] | None,
    target_delta_uv_m: Sequence[float],
    *,
    point_index: int,
) -> dict[str, Any]:
    """Validate a truth-free first-state teacher label for retention replay."""

    target = np.asarray(target_delta_uv_m, dtype=float)
    reasons: list[str] = []
    if target.shape != (2,) or not np.all(np.isfinite(target)):
        raise ValueError("paired DAgger target must be finite shape (2,)")
    if int(point_index) != 0:
        reasons.append("not_first_search_state")
    if policy_record is None:
        reasons.append("missing_student_shadow_record")
        teacher = np.empty(0, dtype=float)
    else:
        if policy_record.get("truth_in_observation") is not False:
            reasons.append("truth_observation_contract")
        if policy_record.get("expert_stop") is not False:
            reasons.append("teacher_not_moving")
        teacher_source = policy_record.get("teacher_label_source")
        if teacher_source not in {"observable_expert", "privileged_expert"}:
            reasons.append("student_record_label_not_trusted_teacher")
        teacher = np.asarray(
            policy_record.get("expert_delta_uv_m", ()), dtype=float
        )
        if teacher.shape != (2,) or not np.allclose(
            teacher, target, atol=1e-9, rtol=0.0
        ):
            reasons.append("student_teacher_target_mismatch")
    if plan is None or plan.get("label_source") not in {
        "observable_expert",
        "privileged_expert",
    }:
        reasons.append("plan_label_not_trusted_teacher")
    else:
        planned = np.asarray(
            plan.get("planned_delta_patient_uv_m", ()), dtype=float
        )
        if planned.shape != (2,) or not np.allclose(
            planned, target, atol=1e-9, rtol=0.0
        ):
            reasons.append("plan_target_mismatch")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "point_index": int(point_index),
        "teacher_delta_uv_m": teacher.tolist() if teacher.shape == (2,) else None,
        "student_delta_uv_m": (
            None
            if policy_record is None
            else list(policy_record.get("delta_uv_m", ()))
        ),
        "dagger_hard_state": (
            False
            if policy_record is None
            else bool(policy_record.get("dagger_hard_state", False))
        ),
    }


def paired_dagger_first_state_episode_audit(
    metadata: Mapping[str, Any],
    *,
    validation_errors: Sequence[str] = (),
) -> dict[str, Any]:
    """Audit a collection-only episode that stops before teacher execution."""

    reasons: list[str] = []
    contract = metadata.get("paired_dagger") or {}
    if contract.get("schema_version") != "adam_pulse_search_gate_paired_dagger_v1":
        reasons.append("missing_paired_dagger_contract")
    if contract.get("collection_mode") != "first_state_teacher_label":
        reasons.append("wrong_collection_mode")
    if contract.get("first_state_label_ready") is not True:
        reasons.append("teacher_label_not_ready")
    if contract.get("truth_in_observation") is not False:
        reasons.append("truth_observation_contract")
    if int(metadata.get("forbidden_contact_events", 0)) != 0:
        reasons.append("forbidden_contact")
    if int(contract.get("safety_modifications", 0)) != 0:
        reasons.append("safety_modification")
    force = np.asarray(metadata.get("maximum_force_n_per_finger", ()), dtype=float)
    if force.size == 0 or not np.isfinite(force).all():
        reasons.append("missing_or_invalid_force_audit")
    elif float(force.max()) >= 10.0:
        reasons.append("force_limit")
    if validation_errors:
        reasons.append("invalid_episode_artifact")

    records = list(metadata.get("search_segment_policy_records") or ())
    plans = list(metadata.get("search_segment_plans") or ())
    if len(records) != 1:
        reasons.append("policy_record_count")
    if len(plans) != 1:
        reasons.append("teacher_plan_count")
    sample_audit = None
    if len(records) == 1 and len(plans) == 1:
        target = plans[0].get("planned_delta_patient_uv_m", ())
        sample_audit = paired_dagger_first_state_sample_audit(
            records[0], plans[0], target, point_index=int(records[0].get("point_index", -1))
        )
        if not sample_audit["accepted"]:
            reasons.extend(
                f"sample:{reason}" for reason in sample_audit["reasons"]
            )
        if records[0].get("teacher_label_source") != "privileged_expert":
            reasons.append("teacher_label_not_privileged")
        if plans[0].get("label_source") != "privileged_expert":
            reasons.append("plan_label_not_privileged")
    if int(contract.get("privileged_state_reads", 0)) != 1:
        reasons.append("privileged_state_read_count")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "validation_errors": list(validation_errors),
        "sample_audit": sample_audit,
        "maximum_force_n": None if force.size == 0 else float(force.max()),
    }


def initialized_policy_retention_episode_audit(
    metadata: Mapping[str, Any],
    *,
    expected_checkpoint: str,
    expected_horizon: int,
    expected_stride: int,
    transition_count: int,
    search_block_count: int,
) -> dict[str, Any]:
    """Validate a successful pure-policy episode used for retention replay.

    Retention replay is deliberately different from DAgger: its targets are
    the initialized policy's raw model outputs, not privileged teacher actions.
    Runtime step limiting is audited separately so the raw output still maps
    exactly to the deployed and executed action.  The episode therefore has to
    prove that the policy ran without expert or safety intervention and that
    the recorded transition cadence matches the checkpoint's live observation
    cadence.
    """

    reasons: list[str] = []
    runtime = dict(metadata.get("search_segment_policy") or {})
    composite = dict(metadata.get("composite_policy") or {})
    records = list(metadata.get("search_segment_policy_records") or ())
    actual_checkpoint = str(metadata.get("segment_checkpoint") or "")

    if metadata.get("success") is not True:
        reasons.append("episode_not_successful")
    if metadata.get("signal_quality_accepted") is not True:
        reasons.append("signal_quality_not_accepted")
    if metadata.get("policy_mode") != "expert_free_with_deterministic_safety":
        reasons.append("not_expert_free_policy_mode")
    if metadata.get("truth_in_observation") is not False:
        reasons.append("truth_observation_contract")
    if int(metadata.get("forbidden_contact_events", -1)) != 0:
        reasons.append("forbidden_contact_events")
    if metadata.get("failure") is not None:
        reasons.append("episode_has_failure")
    if actual_checkpoint != str(expected_checkpoint):
        reasons.append("initialized_checkpoint_mismatch")

    if composite.get("schema_version") != "adam_pulse_composite_policy_runtime_v1":
        reasons.append("invalid_composite_policy_schema")
    if composite.get("control_authority") != "deployable_policy":
        reasons.append("nondeployable_control_authority")
    if composite.get("expert_action_consumed") is not False:
        reasons.append("expert_action_consumed")
    if composite.get("truth_in_observation") is not False:
        reasons.append("composite_truth_observation_contract")
    if int(composite.get("safety_modifications", -1)) != 0:
        reasons.append("safety_modifications")

    if runtime.get("schema_version") != "pulse_search_segment_shadow_runtime_v1":
        reasons.append("invalid_search_runtime_schema")
    if runtime.get("truth_in_observation") is not False:
        reasons.append("runtime_truth_observation_contract")
    if runtime.get("ready") is not True:
        reasons.append("runtime_not_ready")
    if int(runtime.get("observation_stride_policy_steps", -1)) != int(
        expected_stride
    ):
        reasons.append("observation_stride_mismatch")
    if int(runtime.get("frames", -1)) != int(transition_count):
        reasons.append("runtime_transition_frame_mismatch")
    if int(runtime.get("callback_steps", -1)) != int(
        transition_count * expected_stride
    ):
        reasons.append("runtime_callback_stride_mismatch")
    if int(runtime.get("proposals", -1)) != len(records):
        reasons.append("runtime_proposal_count_mismatch")
    if int(runtime.get("proposal_count", -1)) != len(records):
        reasons.append("audited_proposal_count_mismatch")
    if int(runtime.get("persisted_record_count", -1)) != len(records):
        reasons.append("persisted_record_count_mismatch")
    if runtime.get("proposal_records_complete") is not True:
        reasons.append("proposal_records_incomplete")

    if int(expected_horizon) < 1:
        reasons.append("invalid_observation_horizon")
    if int(transition_count) < int(expected_horizon):
        reasons.append("insufficient_episode_history")
    if len(records) != int(search_block_count):
        reasons.append("search_point_record_count_mismatch")
    point_indices = [record.get("point_index") for record in records]
    if point_indices != list(range(len(records))):
        reasons.append("noncontiguous_search_point_indices")
    if not records:
        reasons.append("missing_search_policy_records")
    else:
        final = records[-1]
        if final.get("stop") is not True:
            reasons.append("missing_terminal_stop_record")
        if final.get("quality_gate_accepted") is not True:
            reasons.append("terminal_stop_not_quality_accepted")
        if np.asarray(final.get("delta_uv_m", ()), dtype=float).shape != (2,) or not np.allclose(
            np.asarray(final.get("delta_uv_m", ()), dtype=float),
            np.zeros(2),
            atol=1e-12,
            rtol=0.0,
        ):
            reasons.append("terminal_stop_delta_not_zero")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "label_source": "initialized_policy_retention",
        "checkpoint": actual_checkpoint,
        "observation_horizon": int(expected_horizon),
        "observation_stride_policy_steps": int(expected_stride),
        "transition_count": int(transition_count),
        "search_block_count": int(search_block_count),
        "policy_record_count": len(records),
        "moving_record_count": max(0, len(records) - 1),
    }


def initialized_policy_retention_sample_audit(
    policy_record: Mapping[str, Any] | None,
    *,
    point_index: int,
    maximum_displacement_m: float,
) -> dict[str, Any]:
    """Validate one raw initialized-policy output as a retention target.

    Training on the task-step-clipped action erases the initialized model's
    saturated-output margin.  Preserve the raw model vector as the label, but
    only after proving that radial runtime clipping reproduces the deployed
    action and that the deployed action was executed without modification.
    """

    reasons: list[str] = []
    raw_model = np.empty(0, dtype=float)
    deployed = np.empty(0, dtype=float)
    executed = np.empty(0, dtype=float)
    if policy_record is None:
        reasons.append("missing_policy_record")
    else:
        if policy_record.get("schema_version") != (
            "pulse_search_segment_policy_shadow_v1"
        ):
            reasons.append("invalid_policy_record_schema")
        if int(policy_record.get("point_index", -1)) != int(point_index):
            reasons.append("point_index_mismatch")
        if policy_record.get("mode") != "execute_model_only":
            reasons.append("not_model_only_execution")
        if policy_record.get("stop") is not False:
            reasons.append("retention_target_is_stop")
        if policy_record.get("quality_gate_accepted") is not False:
            reasons.append("moving_target_quality_gate_accepted")
        if policy_record.get("truth_in_observation") is not False:
            reasons.append("truth_observation_contract")
        if policy_record.get("student_visited_state") is not False:
            reasons.append("branch_state_not_pure_rollout")
        if policy_record.get("expert_stop") is not None:
            reasons.append("expert_stop_present")
        if policy_record.get("expert_delta_uv_m") is not None:
            reasons.append("expert_delta_present")
        if policy_record.get("teacher_label_source") is not None:
            reasons.append("teacher_label_present")
        if policy_record.get("policy_target_executed") is not True:
            reasons.append("policy_target_not_executed")
        if policy_record.get("execution_rejection_reason") is not None:
            reasons.append("policy_execution_rejected")
        if policy_record.get("safety_modification_reason") is not None:
            reasons.append("policy_target_safety_modified")

        raw_model = np.asarray(
            policy_record.get("model_delta_uv_m", ()), dtype=float
        )
        deployed = np.asarray(
            policy_record.get("delta_uv_m", ()), dtype=float
        )
        executed = np.asarray(
            policy_record.get("executed_delta_uv_m", ()), dtype=float
        )
        for name, value in (
            ("raw_model_delta", raw_model),
            ("deployed_delta", deployed),
            ("executed_delta", executed),
        ):
            if value.shape != (2,) or not np.isfinite(value).all():
                reasons.append(f"invalid_{name}")
        maximum_step = float(policy_record.get("maximum_step_m", float("nan")))
        maximum_step_valid = bool(
            np.isfinite(maximum_step)
            and maximum_step > 0.0
            and maximum_step <= float(maximum_displacement_m) + 1e-12
        )
        if not maximum_step_valid:
            reasons.append("invalid_runtime_maximum_step")
        if raw_model.shape == (2,):
            raw_norm = float(np.linalg.norm(raw_model))
            if raw_norm > float(maximum_displacement_m) + 1e-9:
                reasons.append("raw_model_delta_exceeds_checkpoint_bound")
            if maximum_step_valid and deployed.shape == (2,):
                expected_deployed = raw_model.copy()
                task_step_limited = raw_norm > maximum_step + 1e-9
                if task_step_limited:
                    expected_deployed *= maximum_step / raw_norm
                if bool(policy_record.get("task_step_limited", False)) != bool(
                    task_step_limited
                ):
                    reasons.append("task_step_limited_flag_mismatch")
                if not np.allclose(
                    expected_deployed, deployed, atol=1e-9, rtol=0.0
                ):
                    reasons.append("raw_model_runtime_clip_mismatch")
        if deployed.shape == (2,) and maximum_step_valid:
            if float(np.linalg.norm(deployed)) > maximum_step + 1e-9:
                reasons.append("deployed_delta_exceeds_runtime_step")
        if deployed.shape == (2,) and executed.shape == (2,) and not np.allclose(
            deployed, executed, atol=1e-9, rtol=0.0
        ):
            reasons.append("deployed_execution_mismatch")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "point_index": int(point_index),
        "label_source": "initialized_policy_retention",
        "target_semantics": "raw_initialized_policy_model_delta_uv_m",
        "target_delta_uv_m": (
            raw_model.tolist() if raw_model.shape == (2,) else None
        ),
        "raw_model_delta_uv_m": (
            raw_model.tolist() if raw_model.shape == (2,) else None
        ),
        "deployed_delta_uv_m": (
            deployed.tolist() if deployed.shape == (2,) else None
        ),
        "executed_delta_uv_m": (
            executed.tolist() if executed.shape == (2,) else None
        ),
        "task_step_limited": bool(
            policy_record is not None
            and policy_record.get("task_step_limited", False)
        ),
    }


def onpolicy_branch_sample_audit(
    policy_record: Mapping[str, Any] | None,
    plan: Mapping[str, Any] | None,
    target_delta_uv_m: Sequence[float],
    *,
    point_index: int,
    branch_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate one train-only student-visited branch label."""

    target = np.asarray(target_delta_uv_m, dtype=float)
    contract = dict(branch_contract or {})
    reasons: list[str] = []
    if contract.get("schema_version") != "adam_pulse_search_onpolicy_branch_v1":
        reasons.append("invalid_branch_schema")
    if contract.get("accepted") is not True:
        reasons.append("branch_not_accepted")
    expected_point = contract.get("student_visited_point_index")
    if expected_point is None or int(expected_point) != int(point_index):
        reasons.append("not_declared_student_visited_point")
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("on-policy branch target must be finite shape (2,)")
    if policy_record is None:
        reasons.append("missing_student_policy_record")
        teacher = np.empty(0, dtype=float)
    else:
        execution_mode = str(
            contract.get("student_execution_mode", "execute_reviewed")
        )
        expected_record_mode = (
            "execute" if execution_mode == "execute_bounded" else "execute_reviewed"
        )
        expected_takeover = (
            "bounded_execution_budget_exhausted"
            if execution_mode == "execute_bounded"
            else "reviewed_execution_budget_exhausted"
        )
        if execution_mode not in {"execute_reviewed", "execute_bounded"}:
            reasons.append("unsupported_student_execution_mode")
        if policy_record.get("mode") != expected_record_mode:
            reasons.append("student_execution_mode_mismatch")
        if policy_record.get("truth_in_observation") is not False:
            reasons.append("truth_observation_contract")
        if policy_record.get("student_visited_state") is not True:
            reasons.append("not_student_visited_state")
        if policy_record.get("policy_target_executed") is not False:
            reasons.append("student_reexecuted_at_label_state")
        if policy_record.get("execution_rejection_reason") != expected_takeover:
            reasons.append("expert_takeover_not_budget_guarded")
        if policy_record.get("teacher_label_source") != "privileged_expert":
            reasons.append("student_record_label_not_privileged")
        teacher = np.asarray(
            policy_record.get("expert_delta_uv_m", ()), dtype=float
        )
        if teacher.shape != (2,) or not np.allclose(
            teacher, target, atol=1e-9, rtol=0.0
        ):
            reasons.append("student_teacher_target_mismatch")
    if plan is None:
        reasons.append("missing_teacher_plan")
    else:
        if plan.get("label_source") != "privileged_expert":
            reasons.append("plan_label_not_privileged")
        planned = np.asarray(
            plan.get("planned_delta_patient_uv_m", ()), dtype=float
        )
        if planned.shape != (2,) or not np.allclose(
            planned, target, atol=1e-9, rtol=0.0
        ):
            reasons.append("plan_target_mismatch")
        if plan.get("policy_target_executed") is not False:
            reasons.append("teacher_plan_execution_contract")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "point_index": int(point_index),
        "teacher_delta_uv_m": teacher.tolist() if teacher.shape == (2,) else None,
        "student_delta_uv_m": (
            None
            if policy_record is None
            else list(policy_record.get("delta_uv_m", ()))
        ),
        "student_visited_state": bool(
            policy_record is not None
            and policy_record.get("student_visited_state") is True
        ),
    }


def onpolicy_branch_reviewed_sample_audit(
    policy_record: Mapping[str, Any] | None,
    plan: Mapping[str, Any] | None,
    target_delta_uv_m: Sequence[float],
    *,
    point_index: int,
    branch_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate every reviewed proposal up to the branch takeover boundary.

    This includes the initial reviewed proposal that the model-only runtime
    would execute before reaching the final DAgger hard state.  Labels remain
    train-only and come from the exact observable/privileged review plan; no
    truth is added to policy observations.
    """

    target = np.asarray(target_delta_uv_m, dtype=float)
    contract = dict(branch_contract or {})
    reasons: list[str] = []
    boundary = contract.get("student_visited_point_index")
    if contract.get("schema_version") != "adam_pulse_search_onpolicy_branch_v1":
        reasons.append("invalid_branch_schema")
    if contract.get("accepted") is not True:
        reasons.append("branch_not_accepted")
    if boundary is None or int(point_index) > int(boundary):
        reasons.append("outside_reviewed_branch_prefix")
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("reviewed on-policy target must be finite shape (2,)")
    teacher = np.empty(0, dtype=float)
    if policy_record is None:
        reasons.append("missing_student_policy_record")
    else:
        if policy_record.get("mode") != "execute_reviewed":
            reasons.append("not_reviewed_execution")
        if policy_record.get("truth_in_observation") is not False:
            reasons.append("truth_observation_contract")
        if policy_record.get("safety_modification_reason") is not None:
            reasons.append("reviewed_state_safety_modified")
        label_source = policy_record.get("teacher_label_source")
        if label_source not in {"observable_expert", "privileged_expert"}:
            reasons.append("unsupported_teacher_label_source")
        teacher = np.asarray(
            policy_record.get("expert_delta_uv_m", ()), dtype=float
        )
        if teacher.shape != (2,) or not np.allclose(
            teacher, target, atol=1e-9, rtol=0.0
        ):
            reasons.append("student_teacher_target_mismatch")
        executed = bool(policy_record.get("policy_target_executed", False))
        rejection = policy_record.get("execution_rejection_reason")
        is_boundary = boundary is not None and int(point_index) == int(boundary)
        if is_boundary:
            if policy_record.get("student_visited_state") is not True:
                reasons.append("boundary_not_student_visited")
            if executed:
                reasons.append("student_reexecuted_at_label_state")
            if rejection != "reviewed_execution_budget_exhausted":
                reasons.append("expert_takeover_not_budget_guarded")
        else:
            if policy_record.get("student_visited_state") is not False:
                reasons.append("prefix_state_marked_as_takeover_boundary")
            valid_rejection = rejection == "segment_direction_disagrees_with_expert"
            if not executed and not valid_rejection:
                reasons.append("unexplained_review_rejection")
            if executed and rejection is not None:
                reasons.append("executed_review_has_rejection")
    if plan is None:
        reasons.append("missing_teacher_plan")
    else:
        planned = np.asarray(
            plan.get("planned_delta_patient_uv_m", ()), dtype=float
        )
        if planned.shape != (2,) or not np.allclose(
            planned, target, atol=1e-9, rtol=0.0
        ):
            reasons.append("plan_target_mismatch")
        if policy_record is not None and plan.get("label_source") != (
            policy_record.get("teacher_label_source")
        ):
            reasons.append("plan_record_label_source_mismatch")
        if policy_record is not None and bool(
            plan.get("policy_target_executed", False)
        ) != bool(policy_record.get("policy_target_executed", False)):
            reasons.append("plan_execution_contract_mismatch")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "point_index": int(point_index),
        "branch_boundary_point_index": None if boundary is None else int(boundary),
        "teacher_delta_uv_m": teacher.tolist() if teacher.shape == (2,) else None,
        "student_delta_uv_m": (
            None
            if policy_record is None
            else list(policy_record.get("delta_uv_m", ()))
        ),
        "policy_target_executed": bool(
            policy_record is not None
            and policy_record.get("policy_target_executed", False)
        ),
    }


def paired_collection_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pair_members: dict[str, list[Mapping[str, Any]]] = {}
    direction_counts: Counter[str] = Counter()
    hard_direction_counts: Counter[str] = Counter()
    privileged_hard_direction_counts: Counter[str] = Counter()
    for item in cases:
        pair_members.setdefault(str(item["pair_id"]), []).append(item)
        for direction in item.get("teacher_direction_buckets", ()):
            direction_counts[str(direction)] += 1
        for direction in item.get("hard_teacher_direction_buckets", ()):
            hard_direction_counts[str(direction)] += 1
        for direction in item.get("privileged_hard_teacher_direction_buckets", ()):
            privileged_hard_direction_counts[str(direction)] += 1

    pair_audit = {}
    for pair_id, members in sorted(pair_members.items()):
        requested = [
            np.asarray(item["requested_recovery_uv_mm"], dtype=float)
            for item in members
        ]
        opposite = bool(
            len(requested) == 2
            and np.allclose(requested[0], -requested[1], atol=1e-9, rtol=0.0)
        )
        pair_audit[pair_id] = {
            "case_count": len(members),
            "accepted_count": sum(bool(item.get("accepted", False)) for item in members),
            "opposite_requested_recovery": opposite,
            "complete": bool(
                len(members) == 2
                and opposite
                and all(bool(item.get("accepted", False)) for item in members)
            ),
        }

    return {
        "case_count": len(cases),
        "accepted_count": sum(bool(item.get("accepted", False)) for item in cases),
        "hard_state_count": sum(int(item.get("hard_state_count", 0)) for item in cases),
        "privileged_hard_state_count": sum(
            int(item.get("privileged_hard_state_count", 0)) for item in cases
        ),
        "teacher_direction_counts": dict(sorted(direction_counts.items())),
        "hard_teacher_direction_counts": dict(sorted(hard_direction_counts.items())),
        "privileged_hard_teacher_direction_counts": dict(
            sorted(privileged_hard_direction_counts.items())
        ),
        "pair_audit": pair_audit,
        "all_requested_pairs_complete": bool(pair_audit)
        and all(item["complete"] for item in pair_audit.values()),
        "forbidden_contact_events": sum(
            int(item.get("forbidden_contact_events", 0)) for item in cases
        ),
        "truth_in_observation": any(
            bool(item.get("truth_in_observation", False)) for item in cases
        ),
    }
