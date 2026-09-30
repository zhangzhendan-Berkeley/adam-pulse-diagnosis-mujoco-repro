from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .safety import SafetyLimits
from .task import TaskPhase


M5_TRANSITION_OVERLAY_SCHEMA_VERSION_V1 = (
    "adam_pulse_m5_fine_approach_transition_overlay_v1"
)
M5_TRANSITION_OVERLAY_SCHEMA_VERSION = (
    "adam_pulse_m5_fine_approach_transition_overlay_v2"
)
M5_PRESSURE_TRANSITION_OVERLAY_SCHEMA_VERSION = (
    "adam_pulse_m5_pressure_regulation_transition_overlay_v1"
)
M5_STABLE_CAPTURE_TRANSITION_OVERLAY_SCHEMA_VERSION = (
    "adam_pulse_m5_stable_capture_transition_overlay_v1"
)
CONTACT_HANDOFF_COUNTERFACTUAL_OVERLAY_SCHEMA_VERSION = (
    "adam_pulse_contact_handoff_counterfactual_overlay_v1"
)
SUPPORTED_M5_TRANSITION_OVERLAY_SCHEMA_VERSIONS = {
    M5_TRANSITION_OVERLAY_SCHEMA_VERSION_V1,
    M5_TRANSITION_OVERLAY_SCHEMA_VERSION,
    M5_PRESSURE_TRANSITION_OVERLAY_SCHEMA_VERSION,
    M5_STABLE_CAPTURE_TRANSITION_OVERLAY_SCHEMA_VERSION,
    CONTACT_HANDOFF_COUNTERFACTUAL_OVERLAY_SCHEMA_VERSION,
}
M5_SAFETY_PROJECTION_MARGIN = 0.85

_UPDATE_FIELDS = {
    "task_phase",
    "task_phase_name",
    "robot_state",
    "previous_action",
    "expert_proposed_action",
    "safe_action",
    "execution_action",
    "executed_action",
    "expert_label_action",
    "m5_raw_realized_action",
    "m5_safety_projected",
    "m5_label_source",
    "m5_control_step_span",
    "m5_source_pose_residual_norm",
    "recovery_mode",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quaternion_xyzw(record: dict[str, Any]) -> np.ndarray:
    pose = np.asarray(record["tcp_pose"], dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("tcp_pose must contain seven finite values")
    quaternion_wxyz = pose[3:]
    norm = float(np.linalg.norm(quaternion_wxyz))
    if norm <= 1e-12:
        raise ValueError("tcp_pose quaternion must be non-zero")
    quaternion_wxyz = quaternion_wxyz / norm
    return quaternion_wxyz[[1, 2, 3, 0]]


def realized_action_per_policy_step(
    previous: dict[str, Any], current: dict[str, Any]
) -> tuple[np.ndarray, int]:
    """Recover the average 7-D Cartesian tick that produced ``current``.

    Episode capture may skip multiple policy ticks. Translation and rotation
    are accumulated deltas, so they are divided by the recorded sim-step
    span. Pressure is an absolute target and remains zero during approach.
    """

    previous_pose = np.asarray(previous["tcp_pose"], dtype=np.float64)
    current_pose = np.asarray(current["tcp_pose"], dtype=np.float64)
    if previous_pose.shape != (7,) or current_pose.shape != (7,):
        raise ValueError("tcp_pose must have shape (7,)")
    span = int(current["sim_step"]) - int(previous["sim_step"])
    if span < 1:
        raise ValueError("sim_step must increase between transition records")
    rotation_delta = (
        Rotation.from_quat(_quaternion_xyzw(previous)).inv()
        * Rotation.from_quat(_quaternion_xyzw(current))
    )
    action = np.zeros(7, dtype=np.float64)
    action[:3] = (current_pose[:3] - previous_pose[:3]) / span
    action[3:6] = rotation_delta.as_euler("xyz") / span
    if not np.isfinite(action).all():
        raise ValueError("derived fine-approach action is not finite")
    return action.astype(np.float32), span


def project_m5_action_to_safety(
    action: np.ndarray, *, margin: float = M5_SAFETY_PROJECTION_MARGIN
) -> np.ndarray:
    """Project a realized M5 TCP tick into the deployable action contract.

    Joint-waypoint recording measures the physical TCP displacement after PD
    tracking.  That displacement can exceed the action interface's per-tick
    limits even though the waypoint itself is collision-free.  Keep the raw
    measurement for audit, but train and shadow against the command that the
    unchanged safety supervisor can actually execute.  A 15% margin keeps
    regression error from landing exactly on the hard clip boundary.
    """
    if not 0.0 < float(margin) <= 1.0:
        raise ValueError("M5 safety projection margin must be in (0, 1]")
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("M5 action must contain seven finite values")
    limits = SafetyLimits()
    projected = value.copy()
    projected[:3] = np.clip(
        projected[:3],
        -limits.maximum_cartesian_step_m * float(margin),
        limits.maximum_cartesian_step_m * float(margin),
    )
    projected[3:6] = np.clip(
        projected[3:6],
        -limits.maximum_rotation_step_rad * float(margin),
        limits.maximum_rotation_step_rad * float(margin),
    )
    projected[6] = np.clip(
        projected[6], 0.0, limits.maximum_target_pressure_kpa * float(margin)
    )
    return projected.astype(np.float32)


def build_fine_approach_episode_overlay(
    episode_name: str,
    records: list[dict[str, Any]],
    *,
    source_transitions_sha256: str,
) -> dict[str, Any]:
    indices = [
        index
        for index, record in enumerate(records)
        if record.get("label") == "precontact_pose_rad"
    ]
    if not indices:
        raise ValueError(f"{episode_name}: no precontact_pose_rad records")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"{episode_name}: precontact records are not contiguous")
    if indices[0] == 0:
        raise ValueError(f"{episode_name}: fine approach has no preceding record")

    updates: dict[str, dict[str, Any]] = {}
    source_phase_counts: dict[str, int] = {}
    source_labels = {
        str(records[index].get("label", "")) for index in indices
    }
    if source_labels != {"precontact_pose_rad"}:
        raise ValueError(
            f"{episode_name}: fine-approach source labels are not explicit: "
            f"{sorted(source_labels)}"
        )
    source_zero = 0
    translation_norms = []
    rotation_norms = []
    spans = []
    for index in indices:
        current = records[index]
        source_phase = int(current.get("task_phase", -1))
        # Two source contracts are deliberately supported:
        #   * the legacy 2 s named waypoint, recorded as PRE_CONTACT;
        #   * the slow joint-waypoint pilot, recorded as FINE_APPROACH.
        # Both are relabelled to the deployable FINE_APPROACH target, but the
        # source phase is retained below so the timing semantics cannot be
        # mistaken later.
        if source_phase not in {
            int(TaskPhase.PRE_CONTACT),
            int(TaskPhase.FINE_APPROACH),
        }:
            raise ValueError(
                f"{episode_name}: source precontact frame {index} has "
                f"unsupported phase {current.get('task_phase')}"
            )
        source_phase_name = TaskPhase(source_phase).name.lower()
        source_phase_counts[source_phase_name] = (
            source_phase_counts.get(source_phase_name, 0) + 1
        )
        action, span = realized_action_per_policy_step(
            records[index - 1], current
        )
        source_action = np.asarray(
            current.get("executed_action", [0.0] * 7), dtype=np.float64
        )
        if source_action.shape != (7,):
            raise ValueError(f"{episode_name}: malformed source action at {index}")
        source_zero += int(float(np.linalg.norm(source_action[:6])) <= 1e-10)
        robot_state = np.asarray(current["robot_state"], dtype=np.float64)
        if robot_state.ndim != 1 or robot_state.size < 1:
            raise ValueError(f"{episode_name}: malformed robot_state at {index}")
        robot_state = robot_state.copy()
        robot_state[-1] = float(TaskPhase.FINE_APPROACH) / float(
            max(int(phase) for phase in TaskPhase)
        )
        raw_action = action.astype(np.float32)
        safety_projected = source_phase == int(TaskPhase.FINE_APPROACH)
        label_action = (
            project_m5_action_to_safety(raw_action)
            if safety_projected
            else raw_action
        )
        action_list = label_action.tolist()
        updates[str(index)] = {
            "task_phase": int(TaskPhase.FINE_APPROACH),
            "task_phase_name": TaskPhase.FINE_APPROACH.name.lower(),
            "robot_state": robot_state.tolist(),
            "previous_action": action_list,
            "expert_proposed_action": action_list,
            "safe_action": action_list,
            "execution_action": action_list,
            "executed_action": action_list,
            "expert_label_action": action_list,
            "m5_raw_realized_action": raw_action.tolist(),
            "m5_safety_projected": safety_projected,
            "m5_label_source": (
                "realized_tcp_pose_delta_from_slow_joint_waypoint"
                if source_phase == int(TaskPhase.FINE_APPROACH)
                else "realized_tcp_pose_delta_per_policy_step"
            ),
            "m5_control_step_span": span,
        }
        translation_norms.append(float(np.linalg.norm(action[:3])))
        rotation_norms.append(float(np.linalg.norm(action[3:6])))
        spans.append(span)

    return {
        "episode": str(episode_name),
        "source_transitions_sha256": str(source_transitions_sha256),
        "source_record_count": len(records),
        "fine_approach_record_count": len(indices),
        "source_labels": sorted(source_labels),
        "source_phase_counts": source_phase_counts,
        "source_mode": (
            "slow_joint_waypoint"
            if set(source_phase_counts) == {"fine_approach"}
            else "legacy_precontact_waypoint"
        ),
        "first_record_index": indices[0],
        "last_record_index": indices[-1],
        "source_zero_action_fraction": source_zero / len(indices),
        "maximum_translation_per_policy_step_m": max(translation_norms),
        "maximum_rotation_per_policy_step_rad": max(rotation_norms),
        "safety_projected": source_phase_counts == {"fine_approach": len(indices)},
        "safety_projection_margin": (
            M5_SAFETY_PROJECTION_MARGIN
            if source_phase_counts == {"fine_approach": len(indices)}
            else None
        ),
        "control_step_span_minimum": min(spans),
        "control_step_span_maximum": max(spans),
        "updates": updates,
    }


def build_pressure_regulation_episode_overlay(
    episode_name: str,
    records: list[dict[str, Any]],
    *,
    source_transitions_sha256: str,
) -> dict[str, Any]:
    """Attach ideal observable pressure labels without copying episode media.

    The recorded TCP action contains small physical tracking motion, while the
    expert pressure contract intentionally commands zero Cartesian motion and
    one absolute pressure target. The specialist is supervised from that safe
    command; its runtime observation remains the original measured trajectory.
    """
    indices = [
        index
        for index, record in enumerate(records)
        if record.get("label") == "pressure_regulation"
        and int(record.get("task_phase", -1))
        == int(TaskPhase.PRESSURE_REGULATION)
    ]
    if not indices:
        raise ValueError(f"{episode_name}: no pressure_regulation records")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"{episode_name}: pressure records are not contiguous")

    updates: dict[str, dict[str, Any]] = {}
    targets = []
    source_pose_residuals = []
    projected_pose_residual_count = 0
    for index in indices:
        record = records[index]
        source = np.asarray(
            record.get("safe_action", record.get("expert_proposed_action")),
            dtype=np.float64,
        )
        if source.shape != (7,) or not np.isfinite(source).all():
            raise ValueError(f"{episode_name}: malformed pressure label at {index}")
        translation_residual = float(np.linalg.norm(source[:3]))
        rotation_residual = float(np.linalg.norm(source[3:6]))
        # A handful of CH5 episodes enter pressure regulation on the final
        # sub-milliradian settling tick.  This is measured transition residue,
        # not a pressure-control pose intent.  Accept only that tightly bounded
        # rotation residue and make the supervision contract explicitly hold
        # pose.  Translation or larger rotation commands remain fatal.
        if (
            translation_residual > 1e-8
            or rotation_residual > 2e-4
            or source[6] <= 0.0
        ):
            raise ValueError(
                f"{episode_name}: pressure label violates hold contract at {index}"
            )
        hold_source = np.zeros(7, dtype=np.float64)
        hold_source[6] = source[6]
        label = project_m5_action_to_safety(hold_source)
        source_pose_residuals.append(max(translation_residual, rotation_residual))
        projected_pose_residual_count += int(rotation_residual > 1e-8)
        updates[str(index)] = {
            "expert_label_action": label.tolist(),
            "m5_safety_projected": not np.array_equal(label, source),
            "m5_label_source": "expert_safe_pressure_target_hold_pose",
            "m5_source_pose_residual_norm": max(
                translation_residual, rotation_residual
            ),
        }
        targets.append(float(label[6]))
    return {
        "episode": str(episode_name),
        "source_transitions_sha256": str(source_transitions_sha256),
        "source_record_count": len(records),
        "pressure_regulation_record_count": len(indices),
        "first_record_index": indices[0],
        "last_record_index": indices[-1],
        "minimum_target_pressure_kpa": min(targets),
        "maximum_target_pressure_kpa": max(targets),
        "maximum_source_pose_residual_norm": max(source_pose_residuals),
        "projected_source_pose_residual_count": projected_pose_residual_count,
        "updates": updates,
    }


def build_stable_capture_episode_overlay(
    episode_name: str,
    records: list[dict[str, Any]],
    *,
    source_transitions_sha256: str,
) -> dict[str, Any]:
    """Attach constant-pressure, zero-pose labels to signal capture only."""
    indices = [
        index
        for index, record in enumerate(records)
        if record.get("label") == "signal_capture"
        and int(record.get("task_phase", -1))
        == int(TaskPhase.SIGNAL_CAPTURE)
    ]
    if not indices:
        raise ValueError(f"{episode_name}: no signal_capture records")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"{episode_name}: signal capture is not contiguous")

    updates: dict[str, dict[str, Any]] = {}
    targets: list[float] = []
    for index in indices:
        record = records[index]
        contacts = np.asarray(record.get("contact"), dtype=bool)
        if contacts.shape != (2,) or not bool(np.all(contacts)):
            raise ValueError(
                f"{episode_name}: capture frame {index} lacks dual contact"
            )
        source = np.asarray(
            record.get("safe_action", record.get("expert_proposed_action")),
            dtype=np.float64,
        )
        if source.shape != (7,) or not np.isfinite(source).all():
            raise ValueError(
                f"{episode_name}: malformed capture label at {index}"
            )
        if float(np.linalg.norm(source[:6])) > 1e-8 or source[6] <= 0.0:
            raise ValueError(
                f"{episode_name}: capture label violates stable hold at {index}"
            )
        label = project_m5_action_to_safety(source)
        updates[str(index)] = {
            "expert_label_action": label.tolist(),
            "m5_safety_projected": not np.array_equal(label, source),
            "m5_label_source": "expert_safe_stable_capture_hold",
        }
        targets.append(float(label[6]))
    return {
        "episode": str(episode_name),
        "source_transitions_sha256": str(source_transitions_sha256),
        "source_record_count": len(records),
        "stable_capture_record_count": len(indices),
        "first_record_index": indices[0],
        "last_record_index": indices[-1],
        "minimum_target_pressure_kpa": min(targets),
        "maximum_target_pressure_kpa": max(targets),
        "updates": updates,
    }


def build_contact_handoff_counterfactual_episode_overlay(
    episode_name: str,
    records: list[dict[str, Any]],
    *,
    source_transitions_sha256: str,
    maximum_stationary_span_m: float = 5e-4,
) -> dict[str, Any]:
    """Relabel stationary student handoff states with the ensuing recovery.

    The label is copied from the first expert recovery action in the same
    episode. Observations and physically executed actions remain untouched, so
    the overlay cannot leak the teacher action into deployment-visible history.
    """
    if maximum_stationary_span_m <= 0.0:
        raise ValueError("maximum stationary span must be positive")

    student_indices: list[int] = []
    recovery_indices: list[int] = []
    for index, record in enumerate(records):
        if int(record.get("task_phase", -1)) != int(
            TaskPhase.CONTACT_ESTABLISHMENT
        ):
            continue
        control = record.get("policy_control")
        if not isinstance(control, dict):
            continue
        if bool(control.get("policy_action_executed")) and not bool(
            control.get("expert_intervened")
        ):
            student_indices.append(index)
        if bool(control.get("expert_intervened")) and (
            "dagger_policy_execution_budget_exhausted"
            in control.get("intervention_reasons", [])
        ):
            recovery_indices.append(index)

    if not student_indices or not recovery_indices:
        raise ValueError(
            f"{episode_name}: missing student handoff or budget recovery records"
        )
    if student_indices != list(
        range(student_indices[0], student_indices[-1] + 1)
    ):
        raise ValueError(f"{episode_name}: student handoff records are not contiguous")
    if recovery_indices[0] != student_indices[-1] + 1:
        raise ValueError(
            f"{episode_name}: expert recovery does not immediately follow handoff"
        )

    teacher_index = recovery_indices[0]
    teacher = np.asarray(
        records[teacher_index].get("expert_label_action"), dtype=np.float64
    )
    if teacher.shape != (7,) or not np.isfinite(teacher).all():
        raise ValueError(f"{episode_name}: malformed recovery teacher action")
    if teacher[2] > -5e-5 or float(np.linalg.norm(teacher[[0, 1, 3, 4, 5]])) > 1e-8:
        raise ValueError(
            f"{episode_name}: recovery teacher is not a pure effective descent"
        )
    if not np.allclose(
        project_m5_action_to_safety(teacher), teacher, atol=1e-9, rtol=0.0
    ):
        raise ValueError(f"{episode_name}: recovery teacher violates safety limits")

    positions = []
    for index in student_indices:
        record = records[index]
        active_contact = np.asarray(record.get("active_contact"), dtype=bool)
        force = np.asarray(record.get("active_normal_force_n"), dtype=np.float64)
        if (
            active_contact.ndim != 1
            or force.shape != active_contact.shape
            or bool(active_contact.any())
            or not np.isfinite(force).all()
            or float(np.max(np.abs(force), initial=0.0)) > 1e-6
        ):
            raise ValueError(
                f"{episode_name}: handoff frame {index} is not contact-free"
            )
        pose = np.asarray(record.get("tcp_pose"), dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"{episode_name}: malformed handoff TCP pose")
        positions.append(pose[:3])
    positions_array = np.stack(positions)
    stationary_span = float(
        np.linalg.norm(positions_array.max(axis=0) - positions_array.min(axis=0))
    )
    if stationary_span > maximum_stationary_span_m:
        raise ValueError(
            f"{episode_name}: handoff moved {stationary_span:.6g} m before recovery"
        )

    updates = {
        str(index): {
            "expert_label_action": teacher.astype(np.float32).tolist(),
            "m5_label_source": (
                "same_episode_first_budget_recovery_counterfactual_descent"
            ),
            "recovery_mode": "contact_handoff_counterfactual_descent",
        }
        for index in student_indices
    }
    return {
        "episode": str(episode_name),
        "source_transitions_sha256": str(source_transitions_sha256),
        "source_record_count": len(records),
        "student_handoff_record_count": len(student_indices),
        "first_student_record_index": student_indices[0],
        "last_student_record_index": student_indices[-1],
        "teacher_record_index": teacher_index,
        "teacher_action": teacher.tolist(),
        "maximum_stationary_span_m": float(maximum_stationary_span_m),
        "observed_stationary_span_m": stationary_span,
        "observation_fields_modified": False,
        "updates": updates,
    }


def load_transition_overlay(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in SUPPORTED_M5_TRANSITION_OVERLAY_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported transition overlay schema: {path}")
    if not isinstance(payload.get("episodes"), dict) or not payload["episodes"]:
        raise ValueError("transition overlay contains no episodes")
    return payload


def apply_transition_overlay(
    records: list[dict[str, Any]], episode_overlay: dict[str, Any]
) -> list[dict[str, Any]]:
    if len(records) != int(episode_overlay.get("source_record_count", -1)):
        raise ValueError("transition overlay source record count mismatch")
    result = [copy.deepcopy(record) for record in records]
    updates = episode_overlay.get("updates")
    if not isinstance(updates, dict) or not updates:
        raise ValueError("episode transition overlay contains no updates")
    for raw_index, update in updates.items():
        index = int(raw_index)
        if not 0 <= index < len(result):
            raise IndexError(f"transition overlay index out of range: {index}")
        unknown = set(update) - _UPDATE_FIELDS
        if unknown:
            raise ValueError(
                f"transition overlay contains unsupported fields: {sorted(unknown)}"
            )
        result[index].update(copy.deepcopy(update))
    return result
