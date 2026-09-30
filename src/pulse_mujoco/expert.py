from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from .actions import PulseAction
from .environment import PulseDiagnosisEnv
from .frames import dual_probe_frame
from .pressure_target import (
    force_target_to_nominal_pressure_kpa,
    nominal_pressure_to_force_target_n,
)
from .phase_gated_policy import review_search_segment_direction
from .pure_policy_audit import PURE_POLICY_MODE
from .pulse_search_motion import (
    PulseSearchState,
    best_finite_score_index,
    clip_target_to_search_radius,
    unique_position_count,
    validate_three_stage_transitions,
)
from .pulse_signal_quality import (
    PULSE_SIGNAL_QUALITY_SCHEMA_VERSION,
    PulseSignalQualityThresholds,
    despike_tactile_window,
    evaluate_pulse_signal_quality,
    pulse_spatial_direction_is_reliable,
    should_extend_pulse_quality_confirmation,
)
from .task import TaskPhase
from .tactile_layouts import handle_l1_ten_channel_layout


StepCallback = Callable[[object, dict, dict, str], None]
ActionSelector = Callable[
    [object, dict, PulseAction, str], tuple[PulseAction, dict]
]
SearchSegmentSelector = Callable[[dict], Mapping[str, Any]]
PolicyHistoryPrimer = Callable[[object], None]
CONTROL_AUTHORITY_EXPERT = "expert"
CONTROL_AUTHORITY_DEPLOYABLE_POLICY = "deployable_policy"


class SearchSegmentBranchReady(RuntimeError):
    """Signal that an on-policy branch has reached a labelled student state."""

    def __init__(self, point_index: int):
        self.point_index = int(point_index)
        super().__init__(
            "on-policy search branch label ready at point "
            f"{self.point_index}"
        )


class SearchSegmentTeacherLabelReady(RuntimeError):
    """Signal that a shadow-policy state has an unevaluated teacher label.

    The exception is raised after the segment plan and student/teacher audit
    fields have been recorded, but before the planned segment is executed.
    This boundary is collection-only and leaves ordinary expert and policy
    execution unchanged.
    """

    def __init__(self, point_index: int):
        self.point_index = int(point_index)
        super().__init__(
            "search teacher label ready at point " f"{self.point_index}"
        )


def _observable_recovery_descent_step(
    default_step_m: float,
    contacts: np.ndarray,
    adaptive_steps_m: Mapping[object, Any] | None,
) -> float:
    """Choose a bounded seating step from the observable active-contact count.

    The optional contract is intentionally discrete: it yields auditable
    sub-millimetre labels without claiming an unobserved continuous force or
    skin-depth target.  Every configured step must be positive and no larger
    than the already-validated default recovery stroke.
    """

    default = float(default_step_m)
    if not np.isfinite(default) or default <= 0.0:
        raise ValueError("default recovery descent step must be positive")
    contact = np.asarray(contacts, dtype=bool)
    if contact.ndim != 1:
        raise ValueError("recovery contact mask must be one-dimensional")
    if adaptive_steps_m is None:
        return default
    if not isinstance(adaptive_steps_m, Mapping):
        raise ValueError("adaptive recovery descent steps must be a mapping")
    count = int(contact.sum())
    raw = adaptive_steps_m.get(str(count), adaptive_steps_m.get(count, default))
    step = float(raw)
    if not np.isfinite(step) or not 0.0 < step <= default:
        raise ValueError(
            "adaptive recovery descent step must be in (0, default_step_m]"
        )
    return step


def _observable_contact_mask_seated(
    contacts: np.ndarray,
    forces_n: np.ndarray,
    *,
    minimum_force_n: float | None,
) -> bool:
    """Return whether the observable contact mask is physically seated.

    ``minimum_force_n=None`` preserves the original contact-only contract.
    Recovery after a pressure-hold dropout additionally requires a small
    measured force on every active pad, preventing a one-tick grazing contact
    from being promoted back into pressure regulation.
    """

    contact = np.asarray(contacts, dtype=bool)
    force = np.asarray(forces_n, dtype=float)
    if contact.ndim != 1 or force.shape != contact.shape:
        raise ValueError("contact mask and force vector must have matching shape")
    if not np.isfinite(force).all():
        raise ValueError("contact recovery forces must be finite")
    if not contact.all():
        return False
    if minimum_force_n is None:
        return True
    threshold = float(minimum_force_n)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum seated force must be finite and non-negative")
    return bool(np.all(force >= threshold))


def _search_landing_pressure_handoff_hold_enabled(
    task: Mapping[str, Any],
) -> bool:
    """Separate initial landing handoff from later contact-loss recovery."""

    return bool(
        task.get(
            "search_landing_pressure_handoff_hold_enabled",
            task.get("pressure_contact_loss_rotation_hold_enabled", False),
        )
    )


def _direction_limited_search_maximum_step(
    task: Mapping[str, Any],
    *,
    patient_axis_xy: np.ndarray,
    direction_unit_xy: np.ndarray,
    default_maximum_step_m: float,
) -> float:
    """Apply optional caps to direction-specific patient-frame moves."""

    maximum = float(default_maximum_step_m)
    if not bool(task.get("pulse_search_two_dimensional_enabled", False)):
        return maximum
    direction = np.asarray(direction_unit_xy, dtype=float)
    axis = np.asarray(patient_axis_xy, dtype=float)
    lateral = np.asarray([-axis[1], axis[0]], dtype=float)
    for key, component in (
        (
            "pulse_search_positive_axial_maximum_step_m",
            float(np.dot(direction, axis)),
        ),
        (
            "pulse_search_positive_lateral_maximum_step_m",
            float(np.dot(direction, lateral)),
        ),
    ):
        configured = task.get(key)
        if configured is None:
            continue
        limited = float(configured)
        if not 0.0 < limited <= float(default_maximum_step_m):
            raise ValueError(
                f"{key} must be in (0, {default_maximum_step_m}], "
                f"got {limited}"
            )
        if component > 0.5:
            maximum = min(maximum, limited)
    return maximum


def _patient_delta_uv_to_world_xy(
    delta_uv_m: np.ndarray,
    *,
    patient_lateral_xy: np.ndarray,
    patient_axis_xy: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(delta_uv_m, dtype=float)
    lateral = np.asarray(patient_lateral_xy, dtype=float)
    axis = np.asarray(patient_axis_xy, dtype=float)
    if delta.shape != (2,) or lateral.shape != (2,) or axis.shape != (2,):
        raise ValueError("patient-frame delta and axes must have shape (2,)")
    if not np.isfinite(delta).all():
        raise ValueError("patient-frame delta must be finite")
    return delta[0] * lateral + delta[1] * axis


def _review_search_segment_delta_uv(
    delta_uv_m: np.ndarray,
    expert_delta_xy_m: np.ndarray,
    *,
    patient_lateral_xy: np.ndarray,
    patient_axis_xy: np.ndarray,
    minimum_direction_cosine: float = float(np.cos(np.deg2rad(30.0))),
) -> tuple[bool, str | None, float | None]:
    """Review a 2-D patient-frame segment without changing its magnitude."""

    policy = _patient_delta_uv_to_world_xy(
        delta_uv_m,
        patient_lateral_xy=patient_lateral_xy,
        patient_axis_xy=patient_axis_xy,
    )
    expert = np.asarray(expert_delta_xy_m, dtype=float)
    policy_norm = float(np.linalg.norm(policy))
    expert_norm = float(np.linalg.norm(expert))
    if policy_norm <= 1e-12:
        return False, "policy_segment_direction_unavailable", None
    if expert_norm <= 1e-12:
        return False, "expert_segment_direction_unavailable", None
    cosine = float(np.dot(policy, expert) / (policy_norm * expert_norm))
    if cosine < minimum_direction_cosine:
        return False, "segment_direction_disagrees_with_expert", cosine
    return True, None, cosine


def _privileged_search_direction_xy(
    pulse_truth_xy: np.ndarray,
    probe_center_xy: np.ndarray,
    *,
    patient_lateral_xy: np.ndarray,
    two_dimensional: bool,
) -> np.ndarray:
    """Return an oracle label direction without changing model observations."""

    direction = np.asarray(pulse_truth_xy, dtype=float) - np.asarray(
        probe_center_xy, dtype=float
    )
    if direction.shape != (2,):
        raise ValueError("privileged pulse and probe coordinates must be 2-D")
    if two_dimensional:
        return direction
    lateral = np.asarray(patient_lateral_xy, dtype=float)
    return lateral * float(np.dot(direction, lateral))


def _use_privileged_search_fallback(
    task: Mapping[str, Any], rejection_reasons: list[str] | tuple[str, ...]
) -> bool:
    """Limit oracle guidance to explicitly configured sensor failure modes.

    A missing allowlist preserves the historical all-or-nothing privileged
    assist used by earlier audit runs.  New CH5 collection configures only the
    inter-finger-energy failure: ordinary centering errors remain supervised
    by deployable tactile gradients, while a single-finger-dominated window
    may use simulator truth to avoid turning a corrupt direction into an
    unsafe recovery probe.
    """

    configured = task.get(
        "pulse_search_privileged_fallback_rejection_reasons"
    )
    if configured is None:
        return True
    if isinstance(configured, (str, bytes)):
        raise ValueError(
            "pulse_search_privileged_fallback_rejection_reasons must be a list"
        )
    allowed = {str(value) for value in configured}
    return bool(allowed.intersection(str(value) for value in rejection_reasons))


@dataclass
class ExpertRun:
    success: bool
    result: dict
    tactile_time_s: np.ndarray
    tactile_kpa: np.ndarray
    tactile_dynamic_kpa: np.ndarray
    tactile_raw: np.ndarray
    tactile_raw_delta: np.ndarray
    force_n: np.ndarray
    contact: np.ndarray
    probe_position_m: np.ndarray
    phase: np.ndarray
    tactile_full_56_kpa: np.ndarray
    tactile_full_56_dynamic_kpa: np.ndarray
    tactile_full_56_raw: np.ndarray
    full_hand_force_n: np.ndarray
    full_hand_contact: np.ndarray


class MujocoPulseExpert:
    """MuJoCo adapter for the Isaac deterministic directional expert.

    Long-range waypoints are scene priors. Contact, release, pressure,
    per-taxel pulse direction, point selection and recovery are all decided
    from runtime state and sensing. Pulse truth is read only after the task
    has ended to produce an evaluation metric.
    """

    def __init__(
        self,
        env: PulseDiagnosisEnv,
        *,
        on_step: StepCallback | None = None,
        search_strategy: str | None = None,
        action_selector: ActionSelector | None = None,
        privileged_search_assist: bool = False,
        privileged_search_assist_after_student_execution: bool = False,
        search_segment_selector: SearchSegmentSelector | None = None,
        search_segment_policy_mode: str = "disabled",
        search_segment_policy_execution_budget: int | None = None,
        search_segment_branch_stop_after_student_label: bool = False,
        search_segment_stop_after_teacher_label: bool = False,
        policy_history_primer: PolicyHistoryPrimer | None = None,
        policy_history_warmup_steps: int = 0,
        control_authority: str = CONTROL_AUTHORITY_EXPERT,
    ):
        self.env = env
        self.backend = env.backend
        self.config = self.backend.config
        self.task = self.config["task"]
        self.diagnostic_finger_names = tuple(
            self.backend.diagnostic_finger_names
        )
        self.search_strategy = str(
            search_strategy or self.task["pulse_search_strategy"]
        )
        if self.search_strategy not in {
            "directional_gradient",
            "signal_quality_early_stop",
        }:
            raise ValueError(
                f"unsupported pulse search strategy: {self.search_strategy}"
            )
        self.quality_thresholds = PulseSignalQualityThresholds.from_mapping(
            self.task.get("pulse_signal_quality")
        )
        self.on_step = on_step
        self.action_selector = action_selector
        self.control_authority = str(control_authority)
        if self.control_authority not in {
            CONTROL_AUTHORITY_EXPERT,
            CONTROL_AUTHORITY_DEPLOYABLE_POLICY,
        }:
            raise ValueError(
                "control_authority must be expert or deployable_policy"
            )
        self.privileged_search_assist = bool(privileged_search_assist)
        self.privileged_search_assist_after_student_execution = bool(
            privileged_search_assist_after_student_execution
        )
        if (
            self.privileged_search_assist_after_student_execution
            and not self.privileged_search_assist
        ):
            raise ValueError(
                "post-student privileged assist requires privileged search assist"
            )
        self.search_segment_selector = search_segment_selector
        self.policy_history_primer = policy_history_primer
        self.policy_history_warmup_steps = int(policy_history_warmup_steps)
        if self.policy_history_warmup_steps < 0:
            raise ValueError("policy history warmup steps must be non-negative")
        if (
            self.policy_history_warmup_steps > 0
            and self.policy_history_primer is None
        ):
            raise ValueError("policy history warmup requires a primer")
        self.search_segment_policy_mode = str(search_segment_policy_mode)
        if self.search_segment_policy_mode not in {
            "disabled",
            "shadow",
            "execute_reviewed",
            "execute",
            "execute_model_only",
        }:
            raise ValueError(
                "search_segment_policy_mode must be disabled, shadow, "
                "execute_reviewed, execute or execute_model_only"
            )
        if (
            self.search_segment_policy_mode != "disabled"
            and self.search_segment_selector is None
        ):
            raise ValueError(
                "enabled search segment policy requires a selector"
            )
        if (
            self.search_segment_policy_mode == "disabled"
            and self.search_segment_selector is not None
        ):
            raise ValueError(
                "search segment selector requires shadow or execute mode"
            )
        self.search_segment_policy_execution_budget = (
            None
            if search_segment_policy_execution_budget is None
            else int(search_segment_policy_execution_budget)
        )
        if (
            self.search_segment_policy_execution_budget is not None
            and self.search_segment_policy_execution_budget < 0
        ):
            raise ValueError(
                "search segment policy execution budget must be non-negative"
            )
        if (
            self.search_segment_policy_execution_budget is not None
            and self.search_segment_policy_mode not in {
                "execute_reviewed",
                "execute",
            }
        ):
            raise ValueError(
                "search segment policy execution budget requires "
                "execute_reviewed or execute mode"
            )
        self.search_segment_branch_stop_after_student_label = bool(
            search_segment_branch_stop_after_student_label
        )
        if self.search_segment_branch_stop_after_student_label and (
            self.search_segment_policy_mode not in {
                "execute_reviewed",
                "execute",
            }
            or self.search_segment_policy_execution_budget is None
            or self.search_segment_policy_execution_budget < 1
        ):
            raise ValueError(
                "student-label branch stop requires execute_reviewed or execute mode "
                "with a positive execution budget"
            )
        self.search_segment_stop_after_teacher_label = bool(
            search_segment_stop_after_teacher_label
        )
        if self.search_segment_stop_after_teacher_label and (
            self.search_segment_policy_mode != "shadow"
            or not self.privileged_search_assist
        ):
            raise ValueError(
                "teacher-label stop requires shadow mode with privileged "
                "search assist"
            )
        if self.control_authority == CONTROL_AUTHORITY_DEPLOYABLE_POLICY:
            if self.action_selector is None:
                raise ValueError(
                    "deployable policy control requires an action selector"
                )
            if (
                getattr(self.action_selector, "consumes_expert_action", None)
                is not False
            ):
                raise ValueError(
                    "deployable action selector must prove "
                    "consumes_expert_action=False"
                )
            if self.privileged_search_assist:
                raise ValueError(
                    "deployable policy control forbids privileged search assist"
                )
            if self.search_segment_policy_mode != "execute_model_only":
                raise ValueError(
                    "deployable policy control requires model-only search segments"
                )
        self.events: list[dict] = []
        self.search_states: list[str] = []
        self.search_samples: list[dict] = []
        # Segment-level supervision is recorded from the expert's logical
        # target before any lift/escape/IK execution.  Reconstructing it by
        # summing per-tick actions is incorrect because thumb-clearance lift
        # can contain horizontal motion that is later compensated during the
        # translation phase.
        self.search_segment_plans: list[dict] = []
        self.search_segment_policy_records: list[dict] = []
        self._search_segment_policy_by_point: dict[int, dict] = {}
        self.search_segment_policy_executions = 0
        self._search_segment_student_visited_points: set[int] = set()
        self.search_segment_branch_ready_point_index: int | None = None
        # Runtime provenance counters.  A model-only search segment still uses
        # this expert for approach/contact/pressure and motion execution, so it
        # must not be described as a full-task pure policy.  The final policy
        # runner has to report both counters as zero.
        self.expert_action_queries = 0
        self.privileged_state_reads = 0
        self.posthoc_privileged_state_reads = 0
        self.search_expert_target_queries = 0
        self._last_sample_window: dict[str, np.ndarray] | None = None
        self._accepted_quality_window: dict[str, np.ndarray] | None = None
        self._last_search_stop_reason: str | None = None
        self.recovery_events: list[dict] = []
        self._quality_recovery_index = 0
        self._quality_recovery_targets_xy: list[np.ndarray] = []
        self._pending_search_recovery_mode: str | None = None
        self._active_search_recovery_mode: str | None = None
        self._search_origin_xy: np.ndarray | None = None
        self._search_free_space_rotation_reference: np.ndarray | None = None
        self.telemetry: dict[str, list] = {
            "time_s": [],
            "tactile_kpa": [],
            "tactile_dynamic_kpa": [],
            "tactile_raw": [],
            "tactile_raw_delta": [],
            "force_n": [],
            "contact": [],
            "probe_position_m": [],
            "phase": [],
            "tactile_full_56_kpa": [],
            "tactile_full_56_dynamic_kpa": [],
            "tactile_full_56_raw": [],
            "full_hand_force_n": [],
            "full_hand_contact": [],
        }
        forearm = self.config["scene"]["patient_forearm"]
        endpoint_delta = (
            np.asarray(forearm["to_m"], dtype=float)
            - np.asarray(forearm["from_m"], dtype=float)
        )
        self.patient_axis_xy = endpoint_delta[:2]
        self.patient_axis_xy /= np.linalg.norm(self.patient_axis_xy)
        self.patient_lateral_xy = np.asarray(
            [-self.patient_axis_xy[1], self.patient_axis_xy[0]]
        )
        self.target_pressure_kpa = force_target_to_nominal_pressure_kpa(
            self.task["pressure_target_force_n_per_finger"],
            self.config["sensors"]["tactile_taxel_area_m2"],
            self.config["hand"]["channel_counts"]["index"],
        )
        self.seating_pressure_kpa = force_target_to_nominal_pressure_kpa(
            self.task["dual_contact_seating_force_n_per_finger"],
            self.config["sensors"]["tactile_taxel_area_m2"],
            self.config["hand"]["channel_counts"]["index"],
        )

    def _validate_deployable_policy_control(
        self, policy_control: dict | None
    ) -> None:
        if self.control_authority != CONTROL_AUTHORITY_DEPLOYABLE_POLICY:
            return
        if not isinstance(policy_control, dict):
            raise RuntimeError(
                "deployable policy selector returned no control provenance"
            )
        if policy_control.get("expert_action_consumed") is not False:
            raise RuntimeError(
                "deployable policy selector did not prove expert-action isolation"
            )
        if policy_control.get("source") == "expert":
            raise RuntimeError("deployable policy selector returned expert control")

    def _diagnostic_arrays(self) -> dict[str, np.ndarray]:
        """Return the configured active-finger view from public tactile data."""

        if self.diagnostic_finger_names == ("index", "middle"):
            tactile = self.backend.last_tactile
            if tactile is None:
                return {
                    "pressure": np.zeros((2, 10), dtype=np.float32),
                    "dynamic": np.zeros((2, 10), dtype=np.float32),
                    "raw": np.zeros((2, 10), dtype=np.uint16),
                    "raw_delta": np.zeros((2, 10), dtype=np.float32),
                    "contact": np.zeros(2, dtype=bool),
                    "force": np.zeros(2, dtype=np.float32),
                }
            return {
                "pressure": tactile.arrays_kpa.copy(),
                "dynamic": tactile.dynamic_arrays_kpa.copy(),
                "raw": tactile.raw_values.copy(),
                "raw_delta": tactile.raw_delta.copy(),
                "contact": tactile.contact.copy(),
                "force": tactile.total_normal_force_n.copy(),
            }
        full = self.backend.last_tactile_full
        if full is None:
            raise RuntimeError("multi-finger diagnosis requires full-hand tactile")
        pressure = []
        dynamic = []
        raw = []
        raw_delta = []
        contact = []
        force = []
        for name in self.diagnostic_finger_names:
            finger_slice = full.finger_slice(name)
            pressure.append(full.pressure_kpa[finger_slice])
            dynamic.append(full.dynamic_pressure_kpa[finger_slice])
            raw.append(full.raw_values[finger_slice])
            raw_delta.append(full.raw_delta[finger_slice])
            index = full.finger_names.index(name)
            contact.append(full.contact[index])
            force.append(full.total_normal_force_n[index])
        return {
            "pressure": np.asarray(pressure, dtype=np.float32),
            "dynamic": np.asarray(dynamic, dtype=np.float32),
            "raw": np.asarray(raw, dtype=np.uint16),
            "raw_delta": np.asarray(raw_delta, dtype=np.float32),
            "contact": np.asarray(contact, dtype=bool),
            "force": np.asarray(force, dtype=np.float32),
        }

    def _diagnostic_contact(self, observation: dict | None = None) -> np.ndarray:
        del observation
        return self.backend.diagnostic_contact()

    def _diagnostic_force(self, info: dict | None = None) -> np.ndarray:
        if info is not None and "normal_force_n" in info:
            value = np.asarray(info["normal_force_n"], dtype=float)
            if value.shape == (len(self.diagnostic_finger_names),):
                return value
        return self.backend.diagnostic_force_n().astype(float)

    def _record(self, observation: dict, info: dict, label: str) -> None:
        diagnostic = self._diagnostic_arrays()
        self.telemetry["time_s"].append(float(observation["timestamp_s"]))
        self.telemetry["tactile_kpa"].append(
            diagnostic["pressure"].copy()
        )
        self.telemetry["tactile_dynamic_kpa"].append(
            diagnostic["dynamic"].copy()
        )
        self.telemetry["tactile_raw"].append(
            diagnostic["raw"].copy()
        )
        self.telemetry["tactile_raw_delta"].append(
            diagnostic["raw_delta"].copy()
        )
        self.telemetry["force_n"].append(
            diagnostic["force"].copy()
        )
        self.telemetry["contact"].append(
            diagnostic["contact"].copy()
        )
        self.telemetry["probe_position_m"].append(
            np.asarray(observation["tcp_pose"][:3]).copy()
        )
        self.telemetry["phase"].append(int(observation["task_phase"]))
        full = self.backend.last_tactile_full
        if full is None:
            self.telemetry["tactile_full_56_kpa"].append(
                np.zeros(56, dtype=np.float32)
            )
            self.telemetry["tactile_full_56_dynamic_kpa"].append(
                np.zeros(56, dtype=np.float32)
            )
            self.telemetry["tactile_full_56_raw"].append(
                np.zeros(56, dtype=np.uint16)
            )
            self.telemetry["full_hand_force_n"].append(
                np.zeros(5, dtype=np.float32)
            )
            self.telemetry["full_hand_contact"].append(
                np.zeros(5, dtype=bool)
            )
        else:
            self.telemetry["tactile_full_56_kpa"].append(
                full.pressure_kpa.copy()
            )
            self.telemetry["tactile_full_56_dynamic_kpa"].append(
                full.dynamic_pressure_kpa.copy()
            )
            self.telemetry["tactile_full_56_raw"].append(
                full.raw_values.copy()
            )
            self.telemetry["full_hand_force_n"].append(
                full.total_normal_force_n.copy()
            )
            self.telemetry["full_hand_contact"].append(full.contact.copy())
        if self.on_step is not None:
            self.on_step(self.backend, observation, info, label)

    def _record_waypoint(
        self, label: str, *, m5_joint_waypoint_shadow: bool = False
    ) -> None:
        observation_object = self.backend._observation()
        info = self.backend._info(ik_converged=True)
        if m5_joint_waypoint_shadow:
            info["executed_action"] = self.backend.previous_action.copy()
            info["m5_joint_waypoint_shadow"] = True
        self.env.observation = observation_object
        self.env.backend_info = info
        self._record(observation_object.policy_dict(), info, label)

    def _record_path_controlled_waypoint(
        self,
        label: str,
        executed_action: np.ndarray,
        policy_control: dict | None,
    ) -> None:
        observation_object = self.backend._observation()
        info = self.backend._info(ik_converged=True)
        info["executed_action"] = np.asarray(
            executed_action, dtype=np.float32
        ).copy()
        info["m5_joint_waypoint_protected"] = True
        if policy_control is not None:
            info["policy_control"] = dict(policy_control)
            if policy_control.get("expert_label_action") is not None:
                info["expert_label_action"] = np.asarray(
                    policy_control["expert_label_action"], dtype=np.float32
                )
        self.env.observation = observation_object
        self.env.backend_info = info
        self._record(observation_object.policy_dict(), info, label)

    def _sync(self) -> tuple[dict, dict]:
        observation_object = self.backend._observation()
        info = self.backend._info(ik_converged=True)
        self.env.observation = observation_object
        self.env.backend_info = info
        return observation_object.policy_dict(), info

    def _step(self, action: PulseAction, label: str) -> tuple[dict, dict]:
        if self.control_authority == CONTROL_AUTHORITY_EXPERT:
            self.expert_action_queries += 1
        selected_action = action
        policy_control = None
        if self.action_selector is not None:
            selected_action, policy_control = self.action_selector(
                self.env.observation,
                self.env.backend_info,
                action,
                label,
            )
            self._validate_deployable_policy_control(policy_control)
        observation, _, terminated, truncated, info = self.env.step(
            selected_action
        )
        recovery_mode = self._active_search_recovery_mode
        if label == "single_finger_balance":
            recovery_mode = "dual_contact_balance"
        elif label.startswith("search_reacquire_"):
            recovery_mode = "contact_reacquisition"
        if recovery_mode is not None:
            info["expert_recovery"] = {
                "active": True,
                "mode": recovery_mode,
                "reason": "expert_recovery_motion",
            }
        if policy_control is not None:
            info["policy_control"] = policy_control
            if policy_control.get("expert_label_action") is not None:
                info["expert_label_action"] = np.asarray(
                    policy_control["expert_label_action"], dtype=np.float32
                )
        if truncated:
            raise RuntimeError("expert episode exceeded maximum steps")
        if terminated and self.backend.phase == TaskPhase.FAILURE:
            raise RuntimeError("backend entered failure phase")
        if info["runtime_forbidden_contacts"]:
            raise RuntimeError(
                "forbidden runtime contact: "
                f"{info['runtime_forbidden_contacts']}"
            )
        if info.get("joint_path_preview_rejected", False):
            raise RuntimeError(
                "joint_path_preview_forbidden during "
                f"{label}: {info.get('swept_candidate_forbidden_contacts', [])}"
            )
        self._record(observation, info, label)
        return observation, info

    def _phase_action(
        self,
        phase: TaskPhase,
        *,
        pressure_kpa: float = 0.0,
    ) -> PulseAction:
        return PulseAction(
            np.zeros(3),
            np.zeros(3),
            target_pressure_kpa=pressure_kpa,
            phase_command=int(phase),
        )

    def _move_named_pose(
        self,
        pose_name: str,
        phase: TaskPhase,
        duration_s: float,
        *,
        m5_joint_waypoint_shadow: bool = False,
    ) -> None:
        self.backend.phase = phase
        self.events.append(
            {"event": pose_name, "time_s": float(self.backend.data.time)}
        )
        self.backend.move_to_joint_pose(
            self.config["robot"][pose_name],
            duration_s=duration_s,
            settle_s=0.4,
            on_control_step=lambda: self._record_waypoint(
                pose_name,
                m5_joint_waypoint_shadow=m5_joint_waypoint_shadow,
            ),
            track_tcp_action=m5_joint_waypoint_shadow,
        )
        self._sync()

    def _move_named_pose_path_controlled(
        self,
        pose_name: str,
        phase: TaskPhase,
        duration_s: float,
    ) -> None:
        if self.action_selector is None:
            raise RuntimeError(
                "path-controlled joint waypoint requires an action selector"
            )
        self.backend.phase = phase
        self.events.append(
            {
                "event": f"{pose_name}_path_controlled",
                "time_s": float(self.backend.data.time),
            }
        )
        self._sync()

        bootstrap_requested = 0
        bootstrap_recorded = 0

        def select_progress(
            nominal_action: np.ndarray,
        ) -> tuple[float, dict | None]:
            nonlocal bootstrap_requested
            if bootstrap_requested < self.policy_history_warmup_steps:
                bootstrap_requested += 1
                return 1.0, None
            if self.control_authority == CONTROL_AUTHORITY_EXPERT:
                self.expert_action_queries += 1
            expert_action = PulseAction(
                nominal_action[:3],
                nominal_action[3:6],
                target_pressure_kpa=0.0,
                phase_command=int(phase),
            )
            _, control = self.action_selector(
                self.env.observation,
                self.env.backend_info,
                expert_action,
                "m5_joint_waypoint_protected",
            )
            self._validate_deployable_policy_control(control)
            progress = (
                float(control.get("joint_waypoint_progress_scale", 1.0))
                if control.get("policy_action_executed", False)
                else 1.0
            )
            return progress, control

        def record_progress(
            action: np.ndarray, control: dict | None
        ) -> None:
            nonlocal bootstrap_recorded
            bootstrap = (
                control is None
                and bootstrap_recorded < self.policy_history_warmup_steps
            )
            self._record_path_controlled_waypoint(
                (
                    "m5_fine_approach_history_bootstrap"
                    if bootstrap
                    else pose_name
                ),
                action,
                control,
            )
            if bootstrap:
                assert self.policy_history_primer is not None
                self.policy_history_primer(self.env.observation)
                bootstrap_recorded += 1

        self.backend.move_to_joint_pose_path_controlled(
            self.config["robot"][pose_name],
            duration_s=duration_s,
            settle_s=0.4,
            progress_selector=select_progress,
            on_control_step=record_progress,
        )
        self._sync()

    def _move_safe_waypoints(
        self, phase: TaskPhase, *, reverse: bool = False
    ) -> None:
        poses = self.backend.safe_waypoint_poses()
        if reverse:
            poses = list(reversed(poses))
        configured_durations = self.config["robot"].get(
            "safe_waypoint_durations_s"
        )
        if configured_durations is None:
            durations = [1.5 / len(poses)] * len(poses)
        else:
            durations = [float(value) for value in configured_durations]
            if len(durations) != len(poses):
                raise ValueError(
                    "safe_waypoint_durations_s must match waypoint count"
                )
            if reverse:
                durations.reverse()
        for index, (pose, duration_s) in enumerate(zip(poses, durations)):
            label = f"safe_waypoint_{index + 1}_of_{len(poses)}"
            self.backend.phase = phase
            self.events.append(
                {"event": label, "time_s": float(self.backend.data.time)}
            )
            self.backend.move_to_joint_pose(
                pose,
                duration_s=duration_s,
                settle_s=(
                    float(
                        self.config["robot"].get(
                            "safe_waypoint_intermediate_settle_s", 0.1
                        )
                    )
                    if index + 1 < len(poses)
                    else 0.4
                ),
                on_control_step=lambda label=label: self._record_waypoint(
                    label
                ),
            )
        self._sync()

    def _form_diagnosis_hand_pose(self) -> None:
        self.backend.phase = TaskPhase.OBSERVE
        label = "form_diagnosis_hand_pose"
        self.events.append(
            {"event": label, "time_s": float(self.backend.data.time)}
        )
        self.backend.move_hand_to_diagnosis_pose(
            on_control_step=lambda: self._record_waypoint(label)
        )
        self._sync()

    def _apply_diagnostic_precontact_offset(self) -> None:
        axial = float(
            self.task.get("diagnostic_precontact_axial_offset_m", 0.0)
        )
        if abs(axial) <= 1e-12:
            return
        maximum_step = min(
            float(self.task.get("fingerpad_approach_step_m", 0.00020)) * 3.0,
            0.00060,
        )
        remaining = axial
        while abs(remaining) > 1e-9:
            signed_step = float(
                np.clip(remaining, -maximum_step, maximum_step)
            )
            delta_xy = signed_step * self.patient_axis_xy
            observation, info = self._step(
                PulseAction(
                    np.asarray([delta_xy[0], delta_xy[1], 0.0]),
                    np.zeros(3),
                    target_pressure_kpa=0.0,
                    phase_command=int(TaskPhase.PRE_CONTACT),
                ),
                "multifinger_precontact_centering",
            )
            if self._diagnostic_contact(observation).any():
                raise RuntimeError("precontact_centering_contact")
            if not bool(info.get("action_accepted", False)):
                raise RuntimeError("precontact_centering_ik_rejected")
            remaining -= signed_step
        self.events.append(
            {
                "event": "multifinger_precontact_centered",
                "axial_offset_m": axial,
                "finger_names": list(self.diagnostic_finger_names),
                "time_s": float(self.backend.data.time),
            }
        )

    @staticmethod
    def _bounded_vector(value: np.ndarray, maximum_norm: float) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm <= maximum_norm or norm <= 1e-12:
            return vector.copy()
        return vector * (float(maximum_norm) / norm)

    def _fine_approach_to_precontact_pose(self) -> tuple[dict, dict]:
        """Follow the proven collision-free approach curve with Cartesian ticks."""

        target_joint_pose = self.config["robot"]["precontact_pose_rad"]
        entry_hold_pose = {
            name: float(
                self.backend.data.qpos[
                    self.backend.model.jnt_qposadr[
                        self.backend.model.joint(name).id
                    ]
                ]
            )
            for name in self.backend.arm_joint_names
        }
        entry_settle_s = float(
            self.task.get("m5_fine_approach_entry_settle_s", 1.2)
        )
        entry_check_s = float(
            self.task.get(
                "m5_fine_approach_entry_stability_check_s", 0.25
            )
        )
        entry_max_position_drift = float(
            self.task.get(
                "m5_fine_approach_entry_max_position_drift_m", 0.00010
            )
        )
        entry_max_rotation_drift = float(
            self.task.get(
                "m5_fine_approach_entry_max_rotation_drift_rad", 0.003
            )
        )
        entry_settle_attempts = int(
            self.task.get("m5_fine_approach_entry_settle_attempts", 3)
        )
        if min(
            entry_settle_s,
            entry_check_s,
            entry_max_position_drift,
            entry_max_rotation_drift,
            entry_settle_attempts,
        ) <= 0:
            raise ValueError("M5 fine-approach entry limits must be positive")

        # The final coarse waypoint can still carry substantial wrist
        # velocity. Handing that state directly to Cartesian IK makes the
        # first policy tick inherit a joint-space transient. Brake at the
        # measured pose while this phase is still deterministic, then expose
        # FINE_APPROACH only after the arm is quasi-static.
        entry_position_drift = float("inf")
        entry_rotation_drift = float("inf")
        for attempt in range(entry_settle_attempts):
            self.backend.move_to_joint_pose(
                entry_hold_pose,
                duration_s=0.05,
                settle_s=entry_settle_s,
                on_control_step=lambda: self._record_waypoint(
                    "m5_fine_approach_entry_settle"
                ),
            )
            before_check = dual_probe_frame(
                self.backend.model, self.backend.data
            ).center
            self.backend.move_to_joint_pose(
                entry_hold_pose,
                duration_s=0.05,
                settle_s=entry_check_s,
                on_control_step=lambda: self._record_waypoint(
                    "m5_fine_approach_entry_stability_check"
                ),
            )
            after_check = dual_probe_frame(
                self.backend.model, self.backend.data
            ).center
            entry_position_drift = float(
                np.linalg.norm(
                    after_check.position_m - before_check.position_m
                )
            )
            entry_rotation_drift = float(
                (
                    Rotation.from_matrix(
                        before_check.rotation_world_from_frame
                    ).inv()
                    * Rotation.from_matrix(
                        after_check.rotation_world_from_frame
                    )
                ).magnitude()
            )
            if (
                entry_position_drift <= entry_max_position_drift
                and entry_rotation_drift <= entry_max_rotation_drift
            ):
                break
            self.events.append(
                {
                    "event": "m5_fine_approach_entry_settle_retry",
                    "time_s": float(self.backend.data.time),
                    "attempt": attempt + 1,
                    "position_drift_m": entry_position_drift,
                    "rotation_drift_rad": entry_rotation_drift,
                }
            )
        if (
            entry_position_drift > entry_max_position_drift
            or entry_rotation_drift > entry_max_rotation_drift
        ):
            raise RuntimeError(
                "m5_fine_approach_entry_not_settled: "
                f"position_drift_m={entry_position_drift:.6f}, "
                f"rotation_drift_rad={entry_rotation_drift:.6f}"
            )

        start_joint_pose = {
            name: float(
                self.backend.data.qpos[
                    self.backend.model.jnt_qposadr[
                        self.backend.model.joint(name).id
                    ]
                ]
            )
            for name in self.backend.arm_joint_names
        }
        translation_step = float(
            self.task.get("m5_fine_approach_translation_step_m", 0.00065)
        )
        rotation_step = float(
            self.task.get("m5_fine_approach_rotation_step_rad", 0.010)
        )
        position_tolerance = float(
            self.task.get("m5_fine_approach_position_tolerance_m", 0.0005)
        )
        rotation_tolerance = float(
            self.task.get("m5_fine_approach_rotation_tolerance_rad", 0.008)
        )
        maximum_steps = int(
            self.task.get("m5_fine_approach_maximum_steps", 900)
        )
        stable_required = int(
            self.task.get("m5_fine_approach_stable_steps", 8)
        )
        reference_count = int(
            self.task.get("m5_fine_approach_reference_count", 160)
        )
        reference_position_tolerance = float(
            self.task.get(
                "m5_fine_approach_reference_position_tolerance_m", 1e-7
            )
        )
        reference_rotation_tolerance = float(
            self.task.get(
                "m5_fine_approach_reference_rotation_tolerance_rad", 1e-6
            )
        )
        reference_actual_position_tolerance = float(
            self.task.get(
                "m5_fine_approach_reference_actual_position_tolerance_m",
                0.0010,
            )
        )
        reference_actual_rotation_tolerance = float(
            self.task.get(
                "m5_fine_approach_reference_actual_rotation_tolerance_rad",
                0.015,
            )
        )
        maximum_tracking_position_error = float(
            self.task.get(
                "m5_fine_approach_maximum_tracking_position_error_m",
                0.0015,
            )
        )
        maximum_tracking_rotation_error = float(
            self.task.get(
                "m5_fine_approach_maximum_tracking_rotation_error_rad",
                0.020,
            )
        )
        if min(
            translation_step,
            rotation_step,
            position_tolerance,
            rotation_tolerance,
            maximum_steps,
            stable_required,
            reference_count,
            reference_position_tolerance,
            reference_rotation_tolerance,
            reference_actual_position_tolerance,
            reference_actual_rotation_tolerance,
            maximum_tracking_position_error,
            maximum_tracking_rotation_error,
        ) <= 0:
            raise ValueError("M5 fine-approach limits must be positive")

        self.backend.phase = TaskPhase.FINE_APPROACH
        self.backend.task_rotation_hold_enabled = True
        observation, info = self._sync()
        self.events.append(
            {
                "event": "m5_fine_approach_started",
                "time_s": float(self.backend.data.time),
                "reference_source": "collision_checked_joint_curve",
                "reference_count": reference_count,
                "entry_position_drift_m": entry_position_drift,
                "entry_rotation_drift_rad": entry_rotation_drift,
            }
        )
        step_count = 0
        maximum_translation_command_m = 0.0
        maximum_rotation_command_rad = 0.0
        stable = 0
        observation = observation
        info = info
        target_position = None
        target_rotation = None
        for reference_index in range(1, reference_count + 1):
            u = reference_index / reference_count
            smooth = u * u * (3.0 - 2.0 * u)
            reference_joint_pose = {
                name: (1.0 - smooth) * start_joint_pose[name]
                + smooth * float(target_joint_pose[name])
                for name in self.backend.arm_joint_names
            }
            target_position, target_rotation = (
                self.backend.probe_pose_for_joint_pose(reference_joint_pose)
            )
            reference_forbidden = self.backend.forbidden_contacts(
                self.backend.ik_data
            )
            if reference_forbidden:
                raise RuntimeError(
                    "m5_fine_approach_reference_forbidden_contact: "
                    f"{reference_forbidden}"
                )
            while True:
                command_translation_error = (
                    target_position - self.backend.task_target_position
                )
                command_rotation_error = (
                    Rotation.from_matrix(
                        self.backend.task_target_rotation
                    ).inv()
                    * Rotation.from_matrix(target_rotation)
                ).as_rotvec()
                actual_position = np.asarray(
                    observation["tcp_pose"][:3], dtype=float
                )
                quaternion_wxyz = np.asarray(
                    observation["tcp_pose"][3:], dtype=float
                )
                actual_rotation = Rotation.from_quat(
                    quaternion_wxyz[[1, 2, 3, 0]]
                )
                actual_position_error = float(
                    np.linalg.norm(target_position - actual_position)
                )
                actual_rotation_error = float(
                    (
                        actual_rotation.inv()
                        * Rotation.from_matrix(target_rotation)
                    ).magnitude()
                )
                if (
                    np.linalg.norm(command_translation_error)
                    <= reference_position_tolerance
                    and np.linalg.norm(command_rotation_error)
                    <= reference_rotation_tolerance
                    and actual_position_error
                    <= reference_actual_position_tolerance
                    and actual_rotation_error
                    <= reference_actual_rotation_tolerance
                ):
                    break
                if step_count >= maximum_steps:
                    raise RuntimeError("m5_fine_approach_timeout")
                tracking_position_error = float(
                    np.linalg.norm(
                        self.backend.task_target_position - actual_position
                    )
                )
                tracking_rotation_error = float(
                    (
                        actual_rotation.inv()
                        * Rotation.from_matrix(
                            self.backend.task_target_rotation
                        )
                    ).magnitude()
                )
                tracking_lagged = bool(
                    tracking_position_error
                    > maximum_tracking_position_error
                    or tracking_rotation_error
                    > maximum_tracking_rotation_error
                )
                translation = (
                    np.zeros(3, dtype=float)
                    if tracking_lagged
                    else self._bounded_vector(
                        command_translation_error, translation_step
                    )
                )
                rotation = (
                    np.zeros(3, dtype=float)
                    if tracking_lagged
                    else self._bounded_vector(
                        command_rotation_error, rotation_step
                    )
                )
                maximum_translation_command_m = max(
                    maximum_translation_command_m,
                    float(np.linalg.norm(translation)),
                )
                maximum_rotation_command_rad = max(
                    maximum_rotation_command_rad,
                    float(np.linalg.norm(rotation)),
                )
                observation, info = self._step(
                    PulseAction(
                        translation,
                        rotation,
                        target_pressure_kpa=0.0,
                        phase_command=int(TaskPhase.FINE_APPROACH),
                    ),
                    "fine_approach",
                )
                step_count += 1
                if not bool(info.get("action_accepted", False)):
                    raise RuntimeError(
                        "m5_fine_approach_ik_rejected: "
                        f"reference={reference_index}/{reference_count}, "
                        f"start_position_error_m="
                        f"{float(info.get('ik_start_position_error_m', float('nan'))):.6f}, "
                        f"position_error_m="
                        f"{float(info.get('ik_position_error_m', float('nan'))):.6f}, "
                        f"maximum_joint_delta_rad="
                        f"{float(info.get('ik_maximum_joint_delta_rad', float('nan'))):.6f}, "
                        f"rotation_error_rad="
                        f"{float(info.get('ik_rotation_error_rad', float('nan'))):.6f}, "
                        f"candidate_contacts="
                        f"{info.get('candidate_forbidden_contacts', [])}"
                    )
                if bool(np.any(self._diagnostic_contact(observation))):
                    raise RuntimeError("fine_approach_contact_before_handoff")

        assert target_position is not None and target_rotation is not None
        while step_count < maximum_steps:
            command_translation_error = (
                target_position - self.backend.task_target_position
            )
            command_rotation_error = (
                Rotation.from_matrix(self.backend.task_target_rotation).inv()
                * Rotation.from_matrix(target_rotation)
            ).as_rotvec()
            translation = self._bounded_vector(
                command_translation_error, translation_step
            )
            rotation = self._bounded_vector(
                command_rotation_error, rotation_step
            )
            observation, info = self._step(
                PulseAction(
                    translation,
                    rotation,
                    target_pressure_kpa=0.0,
                    phase_command=int(TaskPhase.FINE_APPROACH),
                ),
                "fine_approach_settle",
            )
            step_count += 1
            if not bool(info.get("action_accepted", False)):
                raise RuntimeError("m5_fine_approach_settle_ik_rejected")
            if bool(np.any(self._diagnostic_contact(observation))):
                raise RuntimeError("fine_approach_contact_before_handoff")
            actual_position = np.asarray(observation["tcp_pose"][:3], dtype=float)
            quaternion_wxyz = np.asarray(
                observation["tcp_pose"][3:], dtype=float
            )
            actual_rotation = Rotation.from_quat(
                quaternion_wxyz[[1, 2, 3, 0]]
            )
            position_error = float(
                np.linalg.norm(target_position - actual_position)
            )
            rotation_error = float(
                (
                    actual_rotation.inv()
                    * Rotation.from_matrix(target_rotation)
                ).magnitude()
            )
            command_reached = bool(
                np.linalg.norm(command_translation_error) <= position_tolerance
                and np.linalg.norm(command_rotation_error) <= rotation_tolerance
            )
            actual_reached = bool(
                position_error <= position_tolerance
                and rotation_error <= rotation_tolerance
            )
            stable = stable + 1 if command_reached and actual_reached else 0
            if stable >= stable_required:
                self.events.append(
                    {
                        "event": "m5_fine_approach_reached",
                        "time_s": float(observation["timestamp_s"]),
                        "position_error_m": position_error,
                        "rotation_error_rad": rotation_error,
                        "step_count": step_count,
                        "maximum_translation_command_m": (
                            maximum_translation_command_m
                        ),
                        "maximum_rotation_command_rad": (
                            maximum_rotation_command_rad
                        ),
                    }
                )
                return observation, info
        raise RuntimeError("m5_fine_approach_timeout")

    def _recover_dual_contact(
        self,
        observation: dict,
        info: dict,
        *,
        phase: TaskPhase,
        maximum_iterations: int | None = None,
        maximum_force_n: float | None = None,
        minimum_single_contact_imbalance_n: float | None = None,
        descent_step_m: float | None = None,
        minimum_seated_force_n: float | None = None,
        seated_stable_iterations: int = 1,
    ) -> tuple[dict, dict]:
        """Seat all active pads using slow descent and optional two-pad tilt."""
        cumulative_tilt = 0.0
        use_balance_tilt = bool(
            self.task.get("dual_contact_balance_tilt_enabled", True)
        )
        if len(self.diagnostic_finger_names) > 2:
            use_balance_tilt = bool(
                self.task.get(
                    (
                        "search_multifinger_balance_tilt_enabled"
                        if phase == TaskPhase.PULSE_SEARCH
                        else "contact_multifinger_balance_tilt_enabled"
                    ),
                    False,
                )
            )
        use_seating_preload = bool(
            self.task.get("dual_contact_seating_preload_enabled", True)
        )
        iterations = int(
            self.task["dual_contact_recovery_iterations"]
            if maximum_iterations is None
            else maximum_iterations
        )
        recovery_descent_step_m = float(
            self.task.get("dual_contact_recovery_descent_step_m", 0.00005)
            if descent_step_m is None
            else descent_step_m
        )
        if recovery_descent_step_m <= 0.0:
            raise ValueError("dual-contact recovery descent must be positive")
        required_seated = int(seated_stable_iterations)
        if required_seated < 1:
            raise ValueError("seated stable iterations must be positive")
        adaptive_descent_steps_m = (
            self.task.get("pulse_search_adaptive_descent_steps_m")
            if phase == TaskPhase.PULSE_SEARCH
            else None
        )
        seated = 0
        for _ in range(iterations):
            contacts = self._diagnostic_contact(observation)
            forces = self._diagnostic_force(info)
            if _observable_contact_mask_seated(
                contacts,
                forces,
                minimum_force_n=minimum_seated_force_n,
            ):
                seated += 1
            else:
                seated = 0
            if seated >= required_seated:
                self.recovery_events.append(
                    {
                        "event": "dual_contact_recovery_seated",
                        "mode": "observable_contact_state_audit",
                        "phase": int(phase),
                        "contacts": contacts.astype(bool).tolist(),
                        "forces_n": forces.astype(float).tolist(),
                        "stable_iterations": int(seated),
                        "minimum_seated_force_n": minimum_seated_force_n,
                        "truth_in_observation": False,
                        "time_s": float(self.backend.data.time),
                    }
                )
                return observation, info
            confirmation_hold = 0 < seated < required_seated
            commanded_descent_step_m = (
                0.0
                if confirmation_hold
                else _observable_recovery_descent_step(
                    recovery_descent_step_m,
                    contacts,
                    adaptive_descent_steps_m,
                )
            )
            if (
                maximum_force_n is not None
                and float(np.max(np.abs(forces)))
                >= float(maximum_force_n)
            ):
                raise RuntimeError("dual_contact_recovery_force_limit")
            tilt_step = 0.0
            local_axis = np.zeros(3, dtype=float)
            if (
                not confirmation_hold
                and use_balance_tilt
                and len(self.diagnostic_finger_names) >= 2
            ):
                centers = np.asarray(
                    [
                        self.backend.data.site_xpos[
                            self.backend.model.site(
                                f"handle_l1_{finger}_tactile_site"
                            ).id
                        ]
                        for finger in self.diagnostic_finger_names
                    ]
                )
                separation = centers[-1] - centers[0]
                separation[2] = 0.0
                balance_axis_world = np.cross(
                    separation, np.asarray([0.0, 0.0, 1.0])
                )
                balance_axis_world /= max(
                    float(np.linalg.norm(balance_axis_world)), 1e-12
                )
                probe_rotation = dual_probe_frame(
                    self.backend.model, self.backend.data
                ).center.rotation_world_from_frame
                local_axis = probe_rotation.T @ balance_axis_world
                imbalance = float(forces[-1] - forces[0])
                if abs(imbalance) < 1e-6:
                    imbalance = 1.0 if contacts[-1] else -1.0
                if (
                    minimum_single_contact_imbalance_n is not None
                    and bool(contacts[0]) != bool(contacts[-1])
                ):
                    sign = 1.0 if contacts[-1] else -1.0
                    imbalance = sign * max(
                        abs(imbalance),
                        float(minimum_single_contact_imbalance_n),
                    )
                tilt_step = float(
                    np.clip(
                        float(
                            self.task["pressure_balance_tilt_gain_rad_per_n"]
                        )
                        * imbalance,
                        -float(
                            self.task["pressure_maximum_balance_tilt_step_rad"]
                        ),
                        float(
                            self.task["pressure_maximum_balance_tilt_step_rad"]
                        ),
                    )
                )
            remaining = float(
                self.task["pressure_maximum_balance_tilt_rad"]
            ) - abs(cumulative_tilt)
            tilt_step = float(np.clip(tilt_step, -remaining, remaining))
            observation, info = self._step(
                PulseAction(
                    np.asarray([0.0, 0.0, -commanded_descent_step_m]),
                    local_axis * tilt_step,
                    target_pressure_kpa=(
                        self.seating_pressure_kpa
                        if (
                            use_seating_preload
                            and (
                                confirmation_hold
                                or use_balance_tilt
                            )
                        )
                        else 0.0
                    ),
                    phase_command=int(phase),
                ),
                (
                    "contact_seating_confirmation"
                    if confirmation_hold
                    else "single_finger_balance"
                ),
            )
            cumulative_tilt += tilt_step
        # The final commanded descent produces a new observation.  Assess it
        # once before declaring timeout; otherwise a valid seating transition
        # on the last permitted control tick is discarded without issuing any
        # further command.
        final_contacts = self._diagnostic_contact(observation)
        final_forces = self._diagnostic_force(info)
        if _observable_contact_mask_seated(
            final_contacts,
            final_forces,
            minimum_force_n=minimum_seated_force_n,
        ):
            seated += 1
        else:
            seated = 0
        if seated >= required_seated:
            self.recovery_events.append(
                {
                    "event": "dual_contact_recovery_seated",
                    "mode": "observable_contact_state_audit_final_tick",
                    "phase": int(phase),
                    "contacts": final_contacts.astype(bool).tolist(),
                    "forces_n": final_forces.astype(float).tolist(),
                    "stable_iterations": int(seated),
                    "minimum_seated_force_n": minimum_seated_force_n,
                    "truth_in_observation": False,
                    "time_s": float(self.backend.data.time),
                }
            )
            return observation, info
        self.recovery_events.append(
            {
                "event": "dual_contact_recovery_timeout",
                "mode": "observable_contact_state_audit",
                "phase": int(phase),
                "iterations": int(iterations),
                "final_contacts": final_contacts.astype(bool).tolist(),
                "final_forces_n": final_forces.astype(float).tolist(),
                "final_stable_iterations": int(seated),
                "required_stable_iterations": int(required_seated),
                "minimum_seated_force_n": minimum_seated_force_n,
                "cumulative_tilt_rad": float(cumulative_tilt),
                "truth_in_observation": False,
                "time_s": float(self.backend.data.time),
            }
        )
        raise RuntimeError("dual_contact_establishment_timeout")

    def _approach_to_dual_contact(self) -> tuple[dict, dict]:
        approach_step = float(self.task["fingerpad_approach_step_m"])
        maximum_steps = int(
            self.task["fingerpad_maximum_approach_steps"]
        )
        first_contact_step = None
        observation = info = None
        for step_index in range(maximum_steps):
            current = self.env.observation
            valid_distances = current.proximity_m[
                current.proximity_valid
            ]
            minimum_distance = (
                float(valid_distances.min())
                if valid_distances.size
                else float("inf")
            )
            near_contact_step = float(
                self.task.get("fingerpad_near_contact_step_m", 0.00020)
            )
            if near_contact_step <= 0.0 or near_contact_step > approach_step:
                raise ValueError(
                    "fingerpad_near_contact_step_m must be in (0, fingerpad_approach_step_m]"
                )
            commanded_step = (
                min(approach_step, near_contact_step)
                if minimum_distance < 0.0015
                else approach_step
            )
            observation, info = self._step(
                PulseAction(
                    np.asarray([0.0, 0.0, -commanded_step]),
                    np.zeros(3),
                    phase_command=int(TaskPhase.CONTACT_ESTABLISHMENT),
                ),
                "contact_approach",
            )
            contacts = self._diagnostic_contact(observation)
            if contacts.any() and first_contact_step is None:
                first_contact_step = step_index
                self.events.append(
                    {
                        "event": "first_contact",
                        "time_s": float(observation["timestamp_s"]),
                        "contact": contacts.tolist(),
                        "finger_names": list(self.diagnostic_finger_names),
                    }
                )
            if contacts.all():
                self.events.append(
                    {
                        "event": "dual_contact",
                        "time_s": float(observation["timestamp_s"]),
                        "finger_names": list(self.diagnostic_finger_names),
                    }
                )
                return observation, info
            if first_contact_step is not None:
                observation, info = self._recover_dual_contact(
                    observation,
                    info,
                    phase=TaskPhase.CONTACT_ESTABLISHMENT,
                )
                self.events.append(
                    {
                        "event": "dual_contact",
                        "time_s": float(observation["timestamp_s"]),
                        "finger_names": list(self.diagnostic_finger_names),
                    }
                )
                return observation, info
        raise RuntimeError("contact_approach_timeout")

    def _regulate_pressure(
        self, phase: TaskPhase = TaskPhase.PRESSURE_REGULATION
    ) -> tuple[dict, dict]:
        minimum = float(
            self.task["pressure_minimum_force_n_per_finger"]
        )
        maximum = float(
            self.task["pressure_maximum_force_n_per_finger"]
        )
        required = int(self.task["pressure_stable_iterations"])
        stable = 0
        observation = info = None
        minimum_seated_force_config = self.task.get(
            "pulse_search_contact_loss_minimum_seated_force_n",
            self.task.get("pressure_contact_loss_minimum_seated_force_n"),
        ) if phase == TaskPhase.PULSE_SEARCH else self.task.get(
            "pressure_contact_loss_minimum_seated_force_n"
        )
        minimum_seated_force_n = (
            None
            if minimum_seated_force_config is None
            else float(minimum_seated_force_config)
        )
        cumulative_balance_tilt = 0.0
        multifinger_balance_locked = False
        target_total = len(self.diagnostic_finger_names) * float(
            self.task["pressure_target_force_n_per_finger"]
        )
        for _ in range(int(self.task["pressure_maximum_iterations"])):
            balance_rotation = np.zeros(3, dtype=float)
            current_contact = self.backend.diagnostic_contact()
            current_force = self.backend.diagnostic_force_n().astype(float)
            contact_loss_pending = not bool(current_contact.all())
            if (
                len(current_force) > 2
                and current_contact.all()
                and np.all(current_force >= minimum)
                and float(np.ptp(current_force))
                <= float(
                    self.task.get(
                        "pressure_multifinger_balance_lock_spread_n", 1.0
                    )
                )
            ):
                multifinger_balance_locked = True
            if (
                len(self.diagnostic_finger_names) > 2
                and bool(
                    self.task.get(
                        "pressure_multifinger_balance_tilt_enabled", False
                    )
                )
                and current_contact.all()
                and not multifinger_balance_locked
            ):
                imbalance = float(current_force[-1] - current_force[0])
                deadband = float(
                    self.task.get(
                        "pressure_multifinger_balance_deadband_n", 0.35
                    )
                )
                if abs(imbalance) > deadband:
                    centers = np.asarray(
                        [
                            self.backend.data.site_xpos[
                                self.backend.model.site(
                                    f"handle_l1_{finger}_tactile_site"
                                ).id
                            ]
                            for finger in self.diagnostic_finger_names
                        ]
                    )
                    separation = centers[-1] - centers[0]
                    separation[2] = 0.0
                    axis_world = np.cross(
                        separation, np.asarray([0.0, 0.0, 1.0])
                    )
                    axis_world /= max(
                        float(np.linalg.norm(axis_world)), 1e-12
                    )
                    tilt_step = float(
                        np.clip(
                            float(
                                self.task[
                                    "pressure_balance_tilt_gain_rad_per_n"
                                ]
                            )
                            * imbalance,
                            -float(
                                self.task[
                                    "pressure_maximum_balance_tilt_step_rad"
                                ]
                            ),
                            float(
                                self.task[
                                    "pressure_maximum_balance_tilt_step_rad"
                                ]
                            ),
                        )
                    )
                    maximum_tilt = float(
                        self.task["pressure_maximum_balance_tilt_rad"]
                    )
                    lower = -maximum_tilt - cumulative_balance_tilt
                    upper = maximum_tilt - cumulative_balance_tilt
                    tilt_step = float(np.clip(tilt_step, lower, upper))
                    probe_rotation = dual_probe_frame(
                        self.backend.model, self.backend.data
                    ).center.rotation_world_from_frame
                    balance_rotation = (
                        probe_rotation.T @ axis_world
                    ) * tilt_step
                    cumulative_balance_tilt += tilt_step
            observation, info = self._step(
                PulseAction(
                    np.zeros(3),
                    balance_rotation,
                    target_pressure_kpa=self.target_pressure_kpa,
                    phase_command=int(phase),
                ),
                "pressure_regulation",
            )
            if contact_loss_pending or bool(info.get("pressure_contact_lost", False)):
                recovery_descent_step_m = float(
                    self.task.get(
                        "pressure_contact_loss_recovery_descent_step_m",
                        self.task.get(
                            "dual_contact_recovery_descent_step_m", 0.00005
                        ),
                    )
                )
                # Preserve a seated pad with a tiny correction when only one
                # or two pads remain in contact.  If the controller has
                # already lost every pad, however, that same sub-0.1 mm
                # stroke cannot reacquire the surface before the timeout.
                # Use a separately bounded reacquisition stroke only for the
                # all-contact-lost case; this keeps normal recovery gentle
                # while making a full dropout recoverable.
                recovery_contacts = self._diagnostic_contact(observation)
                if not bool(recovery_contacts.any()):
                    recovery_descent_step_m = float(
                        self.task.get(
                            "pressure_contact_loss_no_contact_descent_step_m",
                            self.task.get(
                                "dual_contact_recovery_descent_step_m",
                                recovery_descent_step_m,
                            ),
                        )
                    )
                adaptive_schedule = (
                    self.task.get("pulse_search_adaptive_descent_steps_m")
                    if phase == TaskPhase.PULSE_SEARCH
                    else None
                )
                if adaptive_schedule:
                    recovery_descent_step_m = max(
                        recovery_descent_step_m,
                        max(
                            float(value)
                            for value in adaptive_schedule.values()
                        ),
                    )
                self.recovery_events.append(
                    {
                        "event": "pressure_contact_loss_recovery",
                        "mode": "observable_contact_mask_descent",
                        "phase": int(phase),
                        "contacts": self._diagnostic_contact(
                            observation
                        ).astype(bool).tolist(),
                        "forces_n": self._diagnostic_force(info).astype(
                            float
                        ).tolist(),
                        "truth_in_observation": False,
                        "time_s": float(observation["timestamp_s"]),
                    }
                )
                observation, info = self._recover_dual_contact(
                    observation,
                    info,
                    phase=phase,
                    maximum_iterations=int(
                        self.task.get(
                            "pressure_contact_loss_recovery_iterations",
                            self.task["dual_contact_recovery_iterations"],
                        )
                    ),
                    maximum_force_n=float(
                        self.task["pressure_maximum_force_n_per_finger"]
                    ),
                    descent_step_m=recovery_descent_step_m,
                    minimum_seated_force_n=minimum_seated_force_n,
                    seated_stable_iterations=int(
                        self.task.get(
                            "pressure_contact_loss_seated_stable_iterations",
                            1,
                        )
                    ),
                )
                if (
                    phase == TaskPhase.PULSE_SEARCH
                    and bool(
                        self.task.get(
                            "pressure_contact_loss_rotation_hold_enabled", False
                        )
                    )
                ):
                    self.backend.arm_pressure_recovery_rotation_hold()
                stable = 0
                continue
            forces = self._diagnostic_force(info)
            policy_control = info.get("policy_control") or {}
            selected_action = policy_control.get("selected_action")
            selected_pressure_kpa = (
                float(selected_action[6])
                if policy_control.get("source") == "policy"
                and isinstance(selected_action, (list, tuple))
                and len(selected_action) == 7
                else None
            )
            command_target_total = target_total
            if selected_pressure_kpa is not None:
                command_target_total = len(
                    self.diagnostic_finger_names
                ) * nominal_pressure_to_force_target_n(
                    selected_pressure_kpa,
                    self.config["sensors"]["tactile_taxel_area_m2"],
                    self.config["hand"]["channel_counts"]["index"],
                )
            full_command = bool(
                info["pressure_command_force_n"]
                >= command_target_total - 1e-6
            )
            command_gate = full_command or bool(
                self.task.get(
                    "pressure_stable_without_full_command", False
                )
            )
            in_window = bool(
                self._diagnostic_contact(observation).all()
                and np.all((forces >= minimum) & (forces <= maximum))
                and command_gate
            )
            stable = stable + 1 if in_window else 0
            if stable >= required:
                self.backend.clear_pressure_recovery_hold()
                self.events.append(
                    {
                        "event": "pressure_stable",
                        "time_s": float(observation["timestamp_s"]),
                        "force_n": forces.tolist(),
                    }
                )
                return observation, info
        raise RuntimeError("pressure_regulation_timeout")

    def _taxel_positions_world(self) -> np.ndarray:
        positions = []
        for finger in self.diagnostic_finger_names:
            site_id = self.backend.model.site(
                f"handle_l1_{finger}_tactile_site"
            ).id
            rotation = self.backend.data.site_xmat[site_id].reshape(3, 3)
            center = self.backend.data.site_xpos[site_id]
            local = handle_l1_ten_channel_layout(
                finger
            ).positions_local_m
            positions.append(center + local @ rotation.T)
        return np.asarray(positions)

    def _observable_search_direction_xy(
        self, raw_direction_xy: np.ndarray
    ) -> np.ndarray:
        """Apply the configured one- or two-dimensional search contract."""
        direction_xy = np.asarray(raw_direction_xy, dtype=float).copy()
        if direction_xy.shape != (2,):
            raise ValueError("search direction must have shape (2,)")
        if bool(self.task.get("pulse_search_two_dimensional_enabled", False)):
            return direction_xy
        lateral = np.asarray(self.patient_lateral_xy, dtype=float)
        return lateral * float(np.dot(direction_xy, lateral))

    @staticmethod
    def _centered_taxel_direction_xy(
        taxel_positions_world: np.ndarray,
        weights: np.ndarray,
        *,
        equal_finger_weight: bool = False,
    ) -> np.ndarray:
        """Fuse within-pad gradients without treating finger spacing as error."""
        positions = np.asarray(taxel_positions_world, dtype=float)
        scores = np.maximum(np.asarray(weights, dtype=float), 0.0)
        if (
            positions.ndim != 3
            or positions.shape[1:] != (10, 3)
            or scores.shape != positions.shape[:2]
        ):
            raise ValueError(
                "taxel positions/weights must be [fingers,10,3]/[fingers,10]"
            )
        finger_energy = scores.sum(axis=1)
        active = finger_energy > 1e-12
        if not np.any(active):
            return np.zeros(2, dtype=float)
        if equal_finger_weight:
            scores = scores.copy()
            scores[active] /= finger_energy[active, None]
            scores[~active] = 0.0
        total = float(scores.sum())
        pad_centers = positions.mean(axis=1, keepdims=True)
        within_pad_offsets = positions - pad_centers
        direction_xyz = np.sum(
            within_pad_offsets * scores[..., None], axis=(0, 1)
        ) / total
        return direction_xyz[:2]

    @staticmethod
    def _interfinger_energy_direction_xy(
        taxel_positions_world: np.ndarray,
        weights: np.ndarray,
        reference_energy_fraction: np.ndarray,
        patient_axis_xy: np.ndarray,
    ) -> np.ndarray:
        """Recover the second tangent-axis error from relative finger energy.

        Within-pad centering intentionally removes finger spacing. That is
        correct for two-finger lateral search, but a three-finger palpation can
        also observe axial displacement as an index/ring energy imbalance.
        Subtracting a fixed nominal fraction removes the natural unequal-gain
        bias without using pulse truth.
        """

        positions = np.asarray(taxel_positions_world, dtype=float)
        scores = np.maximum(np.asarray(weights, dtype=float), 0.0)
        reference = np.asarray(reference_energy_fraction, dtype=float)
        axis = np.asarray(patient_axis_xy, dtype=float)
        if (
            positions.ndim != 3
            or positions.shape[1:] != (10, 3)
            or scores.shape != positions.shape[:2]
            or reference.shape != (positions.shape[0],)
            or axis.shape != (2,)
        ):
            raise ValueError("invalid interfinger energy guidance shapes")
        axis_norm = float(np.linalg.norm(axis))
        total = float(scores.sum())
        reference_total = float(reference.sum())
        if total <= 1e-12 or reference_total <= 1e-12 or axis_norm <= 1e-12:
            return np.zeros(2, dtype=float)
        axis = axis / axis_norm
        fractions = scores.sum(axis=1) / total
        reference = reference / reference_total
        pad_centers_xy = positions.mean(axis=1)[:, :2]
        observed_center = fractions @ pad_centers_xy
        reference_center = reference @ pad_centers_xy
        axial_error = float(np.dot(observed_center - reference_center, axis))
        return axial_error * axis

    @staticmethod
    def _channel_amplitude(window: np.ndarray) -> np.ndarray:
        if len(window) < 4:
            raise RuntimeError("pulse sample window is too short")
        return (
            np.percentile(window, 95.0, axis=0)
            - np.percentile(window, 5.0, axis=0)
        )

    @staticmethod
    def _bidirectional_longitudinal_decay(
        channel_scores: np.ndarray,
        longitudinal_positions_mm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Measure centre-to-edge decay independently on both pad halves.

        Handle L1 has no taxel exactly at longitudinal zero.  Its two nearest
        rows therefore form the centre core.  On each side, a spatially useful
        pulse must decrease from that centre row to the shoulder row and again
        to the outer row.  Returning the minimum of all four ratios makes one
        wrong-side edge peak sufficient to reject that finger.
        """

        scores = np.maximum(np.asarray(channel_scores, dtype=float), 0.0)
        longitudinal = np.asarray(longitudinal_positions_mm, dtype=float)
        if scores.ndim != 2 or longitudinal.shape != scores.shape:
            raise ValueError(
                "channel scores and longitudinal positions must have matching "
                "[fingers,channels] shapes"
            )
        reference = np.round(longitudinal[0], decimals=6)
        if not np.allclose(longitudinal, longitudinal[0], atol=1e-6):
            raise ValueError("diagnostic fingers must share one taxel row layout")
        rows = np.unique(reference)
        negative = rows[rows < 0.0]
        positive = rows[rows > 0.0]
        if len(negative) < 3 or len(positive) < 3:
            raise ValueError("bidirectional decay requires three taxel rows per side")

        row_profile = np.stack(
            [
                scores[:, np.isclose(reference, row, atol=1e-6)].mean(axis=1)
                for row in rows
            ],
            axis=1,
        )
        row_index = {float(row): index for index, row in enumerate(rows)}
        negative_edge, negative_shoulder, negative_center = negative[-3:]
        positive_center, positive_shoulder, positive_edge = positive[:3]
        numerator = np.stack(
            (
                row_profile[:, row_index[float(negative_center)]],
                row_profile[:, row_index[float(negative_shoulder)]],
                row_profile[:, row_index[float(positive_center)]],
                row_profile[:, row_index[float(positive_shoulder)]],
            ),
            axis=1,
        )
        denominator = np.stack(
            (
                row_profile[:, row_index[float(negative_shoulder)]],
                row_profile[:, row_index[float(negative_edge)]],
                row_profile[:, row_index[float(positive_shoulder)]],
                row_profile[:, row_index[float(positive_edge)]],
            ),
            axis=1,
        )
        epsilon_kpa = 1e-6
        ratios = (numerator + epsilon_kpa) / (denominator + epsilon_kpa)
        return row_profile, ratios.min(axis=1)

    def _tactile_coverage_quality(self, channel_scores: np.ndarray) -> dict:
        """Evaluate observable pulse coverage within active diagnostic pads."""
        config = self.task.get("pulse_tactile_coverage", {})
        scores = np.maximum(np.asarray(channel_scores, dtype=float), 0.0)
        expected = (len(self.diagnostic_finger_names), 10)
        if scores.shape != expected:
            raise ValueError(
                f"diagnostic channel scores must have shape {expected}"
            )
        positions_mm = np.asarray(
            [
                handle_l1_ten_channel_layout(name).positions_local_m[:, 1:3]
                for name in self.diagnostic_finger_names
            ],
            dtype=float,
        ) * 1000.0
        lateral_half = float(
            config.get("central_lateral_half_extent_mm", 2.5)
        )
        longitudinal_half = float(
            config.get("central_longitudinal_half_extent_mm", 9.0)
        )
        central = (
            np.abs(positions_mm[..., 0]) <= lateral_half
        ) & (np.abs(positions_mm[..., 1]) <= longitudinal_half)
        energy = scores.sum(axis=1)
        total_energy = float(energy.sum())
        if total_energy <= 1e-12:
            centroid = np.zeros(2, dtype=float)
            central_ratio = 0.0
            per_finger_centroids = np.zeros(
                (len(self.diagnostic_finger_names), 2), dtype=float
            )
        else:
            centroid = np.sum(
                positions_mm * scores[..., None], axis=(0, 1)
            ) / total_energy
            central_ratio = float(scores[central].sum() / total_energy)
            per_finger_centroids = np.zeros(
                (len(self.diagnostic_finger_names), 2), dtype=float
            )
            for index in range(len(self.diagnostic_finger_names)):
                if energy[index] > 1e-12:
                    per_finger_centroids[index] = (
                        positions_mm[index] * scores[index, :, None]
                    ).sum(axis=0) / energy[index]
        centroid_offset = float(np.linalg.norm(centroid))
        minimum_ratio = float(config.get("minimum_central_energy_ratio", 0.75))
        maximum_offset = float(
            config.get("maximum_weighted_centroid_offset_mm", 8.0)
        )
        energy_fraction = (
            energy / total_energy
            if total_energy > 1e-12
            else np.zeros_like(energy)
        )
        minimum_finger_fraction = float(
            config.get("minimum_per_finger_energy_fraction", 0.0)
        )
        per_finger_central_ratio = np.divide(
            (scores * central).sum(axis=1),
            energy,
            out=np.zeros_like(energy),
            where=energy > 1e-12,
        )
        per_finger_centroid_offset = np.linalg.norm(
            per_finger_centroids, axis=1
        )
        longitudinal = positions_mm[..., 1]
        center_core_half = float(
            config.get("center_core_longitudinal_half_extent_mm", 2.5)
        )
        edge_minimum = float(
            config.get("edge_longitudinal_minimum_offset_mm", 7.5)
        )
        center_core = np.abs(longitudinal) <= center_core_half
        distal_edge = longitudinal <= -edge_minimum
        proximal_edge = longitudinal >= edge_minimum
        center_core_count = center_core.sum(axis=1)
        distal_edge_count = distal_edge.sum(axis=1)
        proximal_edge_count = proximal_edge.sum(axis=1)
        center_core_mean = np.divide(
            (scores * center_core).sum(axis=1),
            center_core_count,
            out=np.zeros_like(energy),
            where=center_core_count > 0,
        )
        distal_edge_mean = np.divide(
            (scores * distal_edge).sum(axis=1),
            distal_edge_count,
            out=np.zeros_like(energy),
            where=distal_edge_count > 0,
        )
        proximal_edge_mean = np.divide(
            (scores * proximal_edge).sum(axis=1),
            proximal_edge_count,
            out=np.zeros_like(energy),
            where=proximal_edge_count > 0,
        )
        strongest_edge_mean = np.maximum(distal_edge_mean, proximal_edge_mean)
        center_to_edge_contrast = np.divide(
            center_core_mean,
            strongest_edge_mean,
            out=np.full_like(energy, float("inf")),
            where=strongest_edge_mean > 1e-12,
        )
        no_spatial_signal = (center_core_mean <= 1e-12) & (
            strongest_edge_mean <= 1e-12
        )
        center_to_edge_contrast[no_spatial_signal] = 0.0
        peak_channel = np.argmax(scores, axis=1)
        peak_longitudinal_offset = np.abs(
            longitudinal[
                np.arange(len(self.diagnostic_finger_names)), peak_channel
            ]
        )
        longitudinal_row_profile, minimum_bidirectional_decay = (
            self._bidirectional_longitudinal_decay(scores, longitudinal)
        )
        minimum_per_finger_central_ratio = float(
            config.get("minimum_per_finger_central_energy_ratio", 0.0)
        )
        maximum_per_finger_centroid_offset = float(
            config.get(
                "maximum_per_finger_weighted_centroid_offset_mm",
                float("inf"),
            )
        )
        minimum_center_to_edge_contrast = float(
            config.get(
                "minimum_per_finger_center_to_edge_contrast_ratio", 0.0
            )
        )
        maximum_peak_longitudinal_offset = float(
            config.get(
                "maximum_per_finger_peak_longitudinal_offset_mm",
                float("inf"),
            )
        )
        minimum_center_core_mean = float(
            config.get("minimum_per_finger_center_core_mean_kpa", 0.0)
        )
        minimum_bidirectional_decay_ratio = float(
            config.get("minimum_per_finger_bidirectional_decay_ratio", 0.0)
        )
        reasons = []
        if central_ratio < minimum_ratio:
            reasons.append("tactile_central_energy")
        if centroid_offset > maximum_offset:
            reasons.append("tactile_centroid_offset")
        if np.any(energy_fraction < minimum_finger_fraction):
            reasons.append("tactile_per_finger_energy")
        if np.any(
            per_finger_central_ratio < minimum_per_finger_central_ratio
        ):
            reasons.append("tactile_per_finger_central_energy")
        if np.any(
            per_finger_centroid_offset > maximum_per_finger_centroid_offset
        ):
            reasons.append("tactile_per_finger_centroid_offset")
        if np.any(center_to_edge_contrast < minimum_center_to_edge_contrast):
            reasons.append("tactile_per_finger_center_edge_contrast")
        if np.any(peak_longitudinal_offset > maximum_peak_longitudinal_offset):
            reasons.append("tactile_per_finger_peak_location")
        if np.any(center_core_mean < minimum_center_core_mean):
            reasons.append("tactile_per_finger_center_amplitude")
        if np.any(
            minimum_bidirectional_decay < minimum_bidirectional_decay_ratio
        ):
            reasons.append("tactile_per_finger_bidirectional_decay")
        return {
            "enabled": bool(config.get("enabled", False)),
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "central_energy_ratio": central_ratio,
            "weighted_centroid_offset_mm": centroid_offset,
            "weighted_centroid_local_mm": centroid.tolist(),
            "per_finger_weighted_centroid_local_mm": (
                per_finger_centroids.tolist()
            ),
            "per_finger_energy": energy.tolist(),
            "per_finger_energy_fraction": energy_fraction.tolist(),
            "per_finger_central_energy_ratio": (
                per_finger_central_ratio.tolist()
            ),
            "per_finger_weighted_centroid_offset_mm": (
                per_finger_centroid_offset.tolist()
            ),
            "per_finger_center_core_mean_kpa": center_core_mean.tolist(),
            "per_finger_distal_edge_mean_kpa": distal_edge_mean.tolist(),
            "per_finger_proximal_edge_mean_kpa": proximal_edge_mean.tolist(),
            "per_finger_center_to_edge_contrast_ratio": (
                center_to_edge_contrast.tolist()
            ),
            "per_finger_peak_longitudinal_offset_mm": (
                peak_longitudinal_offset.tolist()
            ),
            "per_finger_longitudinal_row_profile_kpa": (
                longitudinal_row_profile.tolist()
            ),
            "per_finger_minimum_bidirectional_decay_ratio": (
                minimum_bidirectional_decay.tolist()
            ),
            "minimum_central_energy_ratio": minimum_ratio,
            "maximum_weighted_centroid_offset_mm": maximum_offset,
            "minimum_per_finger_energy_fraction": minimum_finger_fraction,
            "minimum_per_finger_central_energy_ratio": (
                minimum_per_finger_central_ratio
            ),
            "maximum_per_finger_weighted_centroid_offset_mm": (
                None
                if not np.isfinite(maximum_per_finger_centroid_offset)
                else maximum_per_finger_centroid_offset
            ),
            "minimum_per_finger_center_to_edge_contrast_ratio": (
                minimum_center_to_edge_contrast
            ),
            "maximum_per_finger_peak_longitudinal_offset_mm": (
                None
                if not np.isfinite(maximum_peak_longitudinal_offset)
                else maximum_peak_longitudinal_offset
            ),
            "minimum_per_finger_center_core_mean_kpa": minimum_center_core_mean,
            "minimum_per_finger_bidirectional_decay_ratio": (
                minimum_bidirectional_decay_ratio
            ),
        }

    @staticmethod
    def _pulse_band_component(
        tactile_kpa: np.ndarray,
        sample_rate_hz: float,
        minimum_frequency_hz: float,
        maximum_frequency_hz: float,
    ) -> np.ndarray:
        """Extract the deployable pulse band from public tactile pressure."""
        values = np.asarray(tactile_kpa, dtype=float)
        if values.ndim != 3 or len(values) < 4:
            raise ValueError(
                "tactile_kpa must have shape [samples, fingers, channels]"
            )
        sos = butter(
            3,
            [float(minimum_frequency_hz), float(maximum_frequency_hz)],
            btype="bandpass",
            fs=float(sample_rate_hz),
            output="sos",
        )
        default_pad = 3 * (2 * len(sos) + 1)
        return sosfiltfilt(
            sos,
            values,
            axis=0,
            padlen=min(default_pad, len(values) - 1),
        )

    def _despike_policy_tactile_window(
        self, tactile_kpa: np.ndarray
    ) -> tuple[np.ndarray, int]:
        values = np.asarray(tactile_kpa, dtype=float)
        if not bool(
            self.task.get("policy_tactile_despike_enabled", False)
        ):
            return values.copy(), 0
        filtered, mask = despike_tactile_window(
            values,
            window_samples=int(
                self.task.get("policy_tactile_despike_window_samples", 5)
            ),
            threshold_mad=float(
                self.task.get("policy_tactile_despike_threshold_mad", 6.0)
            ),
            minimum_delta_kpa=float(
                self.task.get(
                    "policy_tactile_despike_minimum_delta_kpa", 0.08
                )
            ),
        )
        return filtered, int(mask.sum())

    def _sample_current_point(self, point_index: int) -> dict:
        self.search_states.append(PulseSearchState.SETTLE_AT_POINT.value)
        stable = 0
        maximum_warmup_steps = int(
            self.task["pulse_search_sampling_warmup_steps"]
        )
        minimum_warmup_steps = int(
            self.task.get("pulse_search_sampling_minimum_warmup_steps", 8)
        )
        if not 1 <= minimum_warmup_steps <= maximum_warmup_steps:
            raise ValueError(
                "pulse search minimum warmup must be within maximum warmup"
            )
        for warmup_index in range(maximum_warmup_steps):
            observation, info = self._step(
                self._phase_action(
                    TaskPhase.PULSE_SEARCH,
                    pressure_kpa=self.target_pressure_kpa,
                ),
                "search_settle",
            )
            forces = self._diagnostic_force(info)
            safe_force = bool(
                self._diagnostic_contact(observation).any()
                and np.max(forces)
                >= float(self.task["pulse_search_minimum_sampling_force_n"])
                and np.max(forces)
                <= float(self.task["pulse_search_maximum_sampling_force_n"])
            )
            stable = stable + 1 if safe_force else 0
            if stable >= 8 and warmup_index + 1 >= minimum_warmup_steps:
                break
        self.search_states.append(PulseSearchState.SAMPLE_POINT.value)
        dynamic_window = []
        use_policy_tactile = bool(
            self.task.get("policy_tactile_bandpass_enabled", False)
        )
        contact_window = []
        force_window = []
        raw_window = []
        for _ in range(int(self.task["pulse_search_hold_steps"])):
            observation, info = self._step(
                self._phase_action(
                    TaskPhase.PULSE_SEARCH,
                    pressure_kpa=self.target_pressure_kpa,
                ),
                "search_sample",
            )
            diagnostic = self._diagnostic_arrays()
            dynamic_window.append(
                (
                    diagnostic["pressure"].astype(float, copy=True)
                    if use_policy_tactile
                    else diagnostic["dynamic"].astype(float, copy=True)
                )
            )
            contact_window.append(diagnostic["contact"].copy())
            force_window.append(diagnostic["force"].astype(float, copy=True))
            raw_window.append(diagnostic["raw"].copy())
        tactile_observed = np.asarray(dynamic_window, dtype=float)
        contacts = np.asarray(contact_window, dtype=bool)
        forces = np.asarray(force_window, dtype=float)
        raw_values = np.asarray(raw_window)
        tactile_for_filter, despike_replacements = (
            self._despike_policy_tactile_window(tactile_observed)
            if use_policy_tactile
            else (tactile_observed.copy(), 0)
        )
        sample_rate_hz = 1.0 / (
            self.backend.physics_steps_per_policy
            * float(self.backend.model.opt.timestep)
        )
        dynamic = (
            self._pulse_band_component(
                tactile_for_filter,
                sample_rate_hz,
                float(
                    self.task.get(
                        "policy_tactile_filter_minimum_frequency_hz",
                        self.quality_thresholds.minimum_frequency_hz,
                    )
                ),
                float(
                    self.task.get(
                        "policy_tactile_filter_maximum_frequency_hz",
                        self.quality_thresholds.maximum_frequency_hz,
                    )
                ),
            )
            if use_policy_tactile
            else tactile_observed.copy()
        )
        if use_policy_tactile:
            dynamic *= float(
                self.task.get(
                    "policy_tactile_bandpass_calibration_gain", 1.0
                )
            )
        preliminary_quality = evaluate_pulse_signal_quality(
            dynamic,
            contacts,
            forces,
            sample_rate_hz=sample_rate_hz,
            thresholds=self.quality_thresholds,
        )
        confirmation_extended = False
        confirmation_extra_steps = 0
        if bool(
            self.task.get("pulse_search_quality_confirmation_enabled", False)
        ) and should_extend_pulse_quality_confirmation(
            preliminary_quality,
            thresholds=self.quality_thresholds,
            band_energy_margin=float(
                self.task.get(
                    "pulse_search_quality_confirmation_band_margin", 0.0
                )
            ),
        ):
            confirmation_extra_steps = int(
                self.task.get(
                    "pulse_search_quality_confirmation_extra_steps", 0
                )
            )
            if confirmation_extra_steps <= 0:
                raise ValueError(
                    "pulse_search_quality_confirmation_extra_steps must be positive"
                )
            confirmation_extended = True
            for _ in range(confirmation_extra_steps):
                observation, info = self._step(
                    self._phase_action(
                        TaskPhase.PULSE_SEARCH,
                        pressure_kpa=self.target_pressure_kpa,
                    ),
                    "search_quality_confirmation",
                )
                diagnostic = self._diagnostic_arrays()
                dynamic_window.append(
                    (
                        diagnostic["pressure"].astype(float, copy=True)
                        if use_policy_tactile
                        else diagnostic["dynamic"].astype(float, copy=True)
                    )
                )
                contact_window.append(diagnostic["contact"].copy())
                force_window.append(
                    diagnostic["force"].astype(float, copy=True)
                )
                raw_window.append(diagnostic["raw"].copy())
            tactile_observed = np.asarray(dynamic_window, dtype=float)
            contacts = np.asarray(contact_window, dtype=bool)
            forces = np.asarray(force_window, dtype=float)
            raw_values = np.asarray(raw_window)
            tactile_for_filter, despike_replacements = (
                self._despike_policy_tactile_window(tactile_observed)
                if use_policy_tactile
                else (tactile_observed.copy(), 0)
            )
            dynamic = (
                self._pulse_band_component(
                    tactile_for_filter,
                    sample_rate_hz,
                    float(
                        self.task.get(
                            "policy_tactile_filter_minimum_frequency_hz",
                            self.quality_thresholds.minimum_frequency_hz,
                        )
                    ),
                    float(
                        self.task.get(
                            "policy_tactile_filter_maximum_frequency_hz",
                            self.quality_thresholds.maximum_frequency_hz,
                        )
                    ),
                )
                if use_policy_tactile
                else tactile_observed.copy()
            )
            if use_policy_tactile:
                dynamic *= float(
                    self.task.get(
                        "policy_tactile_bandpass_calibration_gain", 1.0
                    )
                )
            self.events.append(
                {
                    "event": "pulse_search_quality_confirmation_extended",
                    "point_index": int(point_index),
                    "extra_steps": int(confirmation_extra_steps),
                    "preliminary_quality": preliminary_quality.as_dict(),
                    "time_s": float(observation["timestamp_s"]),
                }
            )
        quality = evaluate_pulse_signal_quality(
            dynamic,
            contacts,
            forces,
            sample_rate_hz=sample_rate_hz,
            thresholds=self.quality_thresholds,
        )
        self._last_sample_window = {
            "tactile": tactile_for_filter.copy(),
            "dynamic": dynamic.copy(),
            "contact": contacts.copy(),
            "force": forces.copy(),
            "raw": raw_values.copy(),
        }
        channel_scores = self._channel_amplitude(dynamic)
        tactile_coverage = self._tactile_coverage_quality(channel_scores)
        quality_dict = quality.as_dict()
        quality_dict["tactile_coverage"] = tactile_coverage
        if tactile_coverage["enabled"] and not tactile_coverage["accepted"]:
            quality_dict["accepted"] = False
            quality_dict["rejection_reasons"] = list(
                quality_dict["rejection_reasons"]
                + tactile_coverage["rejection_reasons"]
            )
        coverage = contacts.mean(axis=0)
        finger_signal = dynamic.mean(axis=2)
        finger_scores = self._channel_amplitude(
            finger_signal[:, :, None]
        )[:, 0]
        score = float(np.max(finger_scores * coverage))
        taxel_world = self._taxel_positions_world()
        weights = channel_scores * coverage[:, None]
        neutral = taxel_world.reshape(-1, 3).mean(axis=0)
        within_pad_direction_xy = np.zeros(2, dtype=float)
        interfinger_direction_xy = np.zeros(2, dtype=float)
        if float(weights.sum()) > 1e-12:
            weighted = np.sum(
                taxel_world * weights[:, :, None], axis=(0, 1)
            ) / float(weights.sum())
            within_pad_direction_xy = self._centered_taxel_direction_xy(
                taxel_world,
                weights,
                equal_finger_weight=bool(
                    self.task.get("pulse_tactile_coverage", {}).get(
                        "equal_finger_weight_for_spatial_guidance", False
                    )
                ),
            )
            interfinger_config = self.task.get(
                "pulse_search_interfinger_energy_guidance", {}
            )
            if bool(interfinger_config.get("enabled", False)):
                interfinger_direction_xy = self._interfinger_energy_direction_xy(
                    taxel_world,
                    weights,
                    np.asarray(
                        interfinger_config["reference_energy_fraction"],
                        dtype=float,
                    ),
                    self.patient_axis_xy,
                ) * float(interfinger_config.get("gain", 1.0))
            direction_xy = within_pad_direction_xy + interfinger_direction_xy
        else:
            weighted = neutral.copy()
            direction_xy = np.zeros(2, dtype=float)
        direction_xy = self._observable_search_direction_xy(direction_xy)
        probe = dual_probe_frame(
            self.backend.model, self.backend.data
        ).center.position_m.copy()
        record = {
            "point_index": int(point_index),
            "position_m": probe.tolist(),
            "pulse_score_kpa": score,
            "per_finger_pulse_score_kpa": finger_scores.tolist(),
            "per_channel_pulse_score_kpa": channel_scores.tolist(),
            "pulse_weighted_taxel_center_m": weighted.tolist(),
            "pulse_centered_taxel_offset_xy_m": direction_xy.tolist(),
            "pulse_within_pad_offset_xy_m": within_pad_direction_xy.tolist(),
            "pulse_interfinger_energy_offset_xy_m": (
                interfinger_direction_xy.tolist()
            ),
            "pulse_direction_xy": direction_xy.tolist(),
            "pulse_direction_magnitude_mm": float(
                np.linalg.norm(direction_xy) * 1000.0
            ),
            "contact_ratio": float(contacts.any(axis=1).mean()),
            "force_n": np.asarray(info["normal_force_n"]).tolist(),
            "settled": bool(stable >= 8),
            "tactile_despike_replacements": int(despike_replacements),
            "quality_confirmation_extended": bool(confirmation_extended),
            "quality_confirmation_extra_steps": int(
                confirmation_extra_steps
            ),
            "preliminary_signal_quality": (
                preliminary_quality.as_dict()
                if confirmation_extended
                else None
            ),
            "signal_quality": quality_dict,
        }
        self.search_samples.append(record)
        return record

    def _query_search_segment_policy(self, sample: dict) -> dict | None:
        """Validate and record one segment-level policy proposal.

        The callback receives only the current sample summary.  In particular,
        this interface has no pulse-center or future-state argument.  The
        deterministic quality gate retains sole authority over termination.
        """

        if self.search_segment_policy_mode == "disabled":
            return None
        if self.search_segment_selector is None:
            raise RuntimeError("enabled segment policy has no selector")
        selector_input = {
            "point_index": int(sample["point_index"]),
            # This summary is computed from the completed current tactile,
            # contact and force window.  Do not pass the sample position,
            # pulse-center evaluation, expert guidance or future target.
            "signal_quality": dict(sample.get("signal_quality", {})),
            "pulse_direction_xy": list(sample.get("pulse_direction_xy", (0.0, 0.0))),
        }
        raw = dict(self.search_segment_selector(selector_input))
        if "stop" not in raw:
            raise ValueError("segment policy proposal requires stop")
        stop = bool(raw["stop"])
        maximum_step = float(
            self.task.get("pulse_search_quality_maximum_step_m", 0.004)
        )
        if "delta_uv_m" in raw:
            delta_uv = np.asarray(raw["delta_uv_m"], dtype=float)
            if delta_uv.shape != (2,) or not np.isfinite(delta_uv).all():
                raise ValueError("2-D segment displacement must be finite shape (2,)")
            contract = "patient_lateral_axial_2d"
        elif "signed_displacement_m" in raw:
            signed = float(raw["signed_displacement_m"])
            if not np.isfinite(signed):
                raise ValueError("segment policy displacement must be finite")
            delta_uv = np.asarray([signed, 0.0], dtype=float)
            contract = "patient_lateral_1d"
        else:
            raise ValueError(
                "segment policy proposal requires delta_uv_m or "
                "signed_displacement_m"
            )
        model_delta_uv = np.asarray(
            raw.get("model_delta_uv_m", delta_uv), dtype=float
        )
        if model_delta_uv.shape != (2,) or not np.isfinite(model_delta_uv).all():
            raise ValueError("model_delta_uv_m must be finite shape (2,)")
        route = str(raw.get("route", "single_policy"))
        hierarchical_routes = {
            "coarse",
            "tail_discrete",
            "tail_spatial_rejection",
            "compact_spatial_rejection",
            "compact_spatial_firststate",
        }
        if route not in {"single_policy", "contact_direction_adapter", *hierarchical_routes}:
            raise ValueError("unsupported search segment policy route")

        def optional_model_delta(key: str) -> list[float] | None:
            value = raw.get(key)
            if value is None:
                return None
            array = np.asarray(value, dtype=float)
            if array.shape != (2,) or not np.isfinite(array).all():
                raise ValueError(f"{key} must be finite shape (2,)")
            return array.tolist()

        coarse_model_delta_uv = optional_model_delta(
            "coarse_model_delta_uv_m"
        )
        tail_model_delta_uv = optional_model_delta("tail_model_delta_uv_m")
        if route in hierarchical_routes and (
            coarse_model_delta_uv is None or tail_model_delta_uv is None
        ):
            raise ValueError(
                "hierarchical search segment route requires coarse and tail deltas"
            )
        proposed_norm = float(np.linalg.norm(delta_uv))
        task_step_limited = False
        if proposed_norm > maximum_step + 1e-9:
            # A checkpoint may have been trained with a wider bounded output
            # than a later embodiment-specific safety profile.  Only accept
            # and radially clip such a proposal when the runtime supplies its
            # own maximum and preserves the exact raw model vector.  Hand-made
            # or malformed oversized selector outputs still fail closed.
            runtime_maximum = raw.get("maximum_displacement_m")
            try:
                runtime_maximum = float(runtime_maximum)
            except (TypeError, ValueError):
                runtime_maximum = float("nan")
            trusted_runtime_bound = (
                np.isfinite(runtime_maximum)
                and runtime_maximum >= proposed_norm - 1e-9
                and np.allclose(model_delta_uv, delta_uv, atol=1e-12, rtol=0.0)
            )
            if not trusted_runtime_bound:
                raise ValueError(
                    "segment policy displacement exceeds maximum search step"
                )
            delta_uv = delta_uv * (maximum_step / proposed_norm)
            task_step_limited = True
        point = int(sample["point_index"])
        record = {
            "schema_version": "pulse_search_segment_policy_shadow_v1",
            "point_index": point,
            "mode": self.search_segment_policy_mode,
            "stop": stop,
            "segment_contract": contract,
            "delta_uv_m": delta_uv.tolist(),
            # Preserve the legacy field for old audit/readout code. For a 2-D
            # proposal it is exactly the patient-lateral component.
            "signed_displacement_m": float(delta_uv[0]),
            "maximum_step_m": maximum_step,
            "task_step_limited": task_step_limited,
            "quality_gate_accepted": bool(
                sample.get("signal_quality", {}).get("accepted", False)
            ),
            "model_stop_probability": (
                None
                if raw.get("model_stop_probability") is None
                else float(raw["model_stop_probability"])
            ),
            "model_delta_uv_m": model_delta_uv.tolist(),
            "route": route,
            "coarse_model_delta_uv_m": coarse_model_delta_uv,
            "tail_model_delta_uv_m": tail_model_delta_uv,
            "model_signed_displacement_m": float(
                raw.get("model_signed_displacement_m", delta_uv[0])
            ),
            "quality_window_available": bool(
                raw.get("quality_window_available", True)
            ),
            "stop_rejection_reasons": list(
                raw.get("stop_rejection_reasons", ())
            ),
            "truth_in_observation": False,
            "student_visited_state": bool(
                point
                in getattr(
                    self, "_search_segment_student_visited_points", set()
                )
            ),
            "execution_budget_segments": (
                None
                if getattr(
                    self, "search_segment_policy_execution_budget", None
                )
                is None
                else int(self.search_segment_policy_execution_budget)
            ),
        }
        self.search_segment_policy_records.append(record)
        self._search_segment_policy_by_point[point] = record
        return record

    def _search_segment_branch_stop_ready(
        self, policy_record: dict | None
    ) -> bool:
        """Return whether the bounded student rollout reached its label state."""
        return bool(
            self.search_segment_branch_stop_after_student_label
            and policy_record is not None
            and policy_record.get("student_visited_state", False)
            and policy_record.get("execution_rejection_reason")
            in {
                "reviewed_execution_budget_exhausted",
                "bounded_execution_budget_exhausted",
            }
        )

    def _three_stage_move(
        self,
        target_xy: np.ndarray,
        point_index: int,
        *,
        execution_internal_correction: bool = False,
    ) -> np.ndarray:
        self._active_search_recovery_mode = (
            self._pending_search_recovery_mode
        )
        self._pending_search_recovery_mode = None
        expert_target_xy = np.asarray(target_xy, dtype=float).reshape(2)
        target_xy = expert_target_xy.copy()
        start = dual_probe_frame(
            self.backend.model, self.backend.data
        ).center.position_m.copy()
        source_sample = (
            None
            if execution_internal_correction or not self.search_samples
            else self.search_samples[-1]
        )
        policy_record = (
            None
            if source_sample is None
            else self._search_segment_policy_by_point.get(
                int(source_sample["point_index"])
            )
        )
        if (
            source_sample is not None
            and policy_record is not None
            and self.search_segment_policy_mode == "execute_model_only"
        ):
            policy_record.update(
                {
                    "expert_stop": None,
                    "expert_signed_displacement_m": None,
                    "signed_error_mm": None,
                    "direction_agrees_with_expert": None,
                    "policy_target_executed": True,
                    "execution_rejection_reason": None,
                    "expert_target_xy_m": None,
                    "execution_target_xy_m": target_xy.tolist(),
                }
            )
        elif source_sample is not None and policy_record is not None:
            sample_xy = np.asarray(source_sample["position_m"], dtype=float)[:2]
            expert_delta_xy = expert_target_xy - sample_xy
            expert_uv = np.asarray(
                [
                    np.dot(expert_delta_xy, self.patient_lateral_xy),
                    np.dot(expert_delta_xy, self.patient_axis_xy),
                ],
                dtype=float,
            )
            policy_uv = np.asarray(policy_record["delta_uv_m"], dtype=float)
            policy_delta_xy = _patient_delta_uv_to_world_xy(
                policy_uv,
                patient_lateral_xy=self.patient_lateral_xy,
                patient_axis_xy=self.patient_axis_xy,
            )
            policy_record.update(
                {
                    "expert_stop": False,
                    "expert_delta_uv_m": expert_uv.tolist(),
                    "expert_signed_displacement_m": float(expert_uv[0]),
                    "delta_uv_error_mm": (
                        1000.0 * (policy_uv - expert_uv)
                    ).tolist(),
                    "vector_error_mm": float(
                        1000.0 * np.linalg.norm(policy_uv - expert_uv)
                    ),
                    "signed_error_mm": float(
                        1000.0 * (policy_uv[0] - expert_uv[0])
                    ),
                    "direction_agrees_with_expert": bool(
                        np.dot(policy_delta_xy, expert_delta_xy) > 0.0
                    ),
                    "policy_target_executed": False,
                    "execution_rejection_reason": None,
                }
            )
            if self.search_segment_policy_mode in {
                "execute_reviewed",
                "execute",
            }:
                if policy_record["stop"]:
                    policy_record["execution_rejection_reason"] = (
                        "stop_not_admitted_without_quality_gate"
                    )
                else:
                    admitted = True
                    if self.search_segment_policy_mode == "execute_reviewed":
                        if policy_record["segment_contract"] == "patient_lateral_1d":
                            admitted, reason, cosine = review_search_segment_direction(
                                policy_record["signed_displacement_m"],
                                expert_delta_xy,
                                self.patient_lateral_xy,
                            )
                        else:
                            admitted, reason, cosine = _review_search_segment_delta_uv(
                                policy_uv,
                                expert_delta_xy,
                                patient_lateral_xy=self.patient_lateral_xy,
                                patient_axis_xy=self.patient_axis_xy,
                            )
                        policy_record["expert_direction_review_cosine"] = cosine
                        if not admitted:
                            policy_record["execution_rejection_reason"] = reason
                    candidate = sample_xy + policy_delta_xy
                    radius = float(
                        self.task.get("pulse_search_quality_radius_m", np.inf)
                    )
                    outside_radius = bool(
                        self._search_origin_xy is not None
                        and np.linalg.norm(candidate - self._search_origin_xy)
                        > radius + 1e-9
                    )
                    budget_exhausted = bool(
                        self.search_segment_policy_execution_budget is not None
                        and self.search_segment_policy_executions
                        >= self.search_segment_policy_execution_budget
                    )
                    # The bounded DAgger rollout boundary takes precedence
                    # over direction/radius rejection.  The review metrics
                    # above are still recorded, but no teacher or student
                    # target may execute after the requested number of
                    # student segments has reached this visited state.
                    if budget_exhausted:
                        policy_record["execution_rejection_reason"] = (
                            "reviewed_execution_budget_exhausted"
                            if self.search_segment_policy_mode == "execute_reviewed"
                            else "bounded_execution_budget_exhausted"
                        )
                        policy_record["execution_budget_remaining_before"] = 0
                    elif not admitted:
                        pass
                    elif outside_radius:
                        policy_record["execution_rejection_reason"] = (
                            "policy_target_outside_search_radius"
                        )
                    else:
                        remaining_before = (
                            None
                            if self.search_segment_policy_execution_budget is None
                            else self.search_segment_policy_execution_budget
                            - self.search_segment_policy_executions
                        )
                        target_xy = candidate
                        policy_record["policy_target_executed"] = True
                        policy_record["execution_budget_remaining_before"] = (
                            remaining_before
                        )
                        self.search_segment_policy_executions += 1
                        policy_record["executed_segment_ordinal"] = int(
                            self.search_segment_policy_executions
                        )
                        self._search_segment_student_visited_points.add(
                            int(point_index)
                        )
            policy_record["expert_target_xy_m"] = expert_target_xy.tolist()
            policy_record["execution_target_xy_m"] = target_xy.tolist()
        positive_axial_reapproach = bool(
            self.task.get(
                "pulse_search_positive_axial_coarse_reapproach", False
            )
            and source_sample is not None
            and float(
                np.dot(
                    target_xy
                    - np.asarray(source_sample["position_m"], dtype=float)[:2],
                    self.patient_axis_xy,
                )
            )
            > 0.0005
        )
        coarse_reapproach = bool(
            self.task.get("pulse_search_coarse_reapproach_enabled", False)
            and self._search_origin_xy is not None
            and (
                positive_axial_reapproach
                or abs(
                    float(
                        np.dot(
                            target_xy - self._search_origin_xy,
                            self.patient_lateral_xy,
                        )
                    )
                )
                > float(
                    self.task.get(
                        "pulse_search_local_envelope_m", 0.006
                    )
                )
            )
        )
        if coarse_reapproach:
            self._active_search_recovery_mode = (
                "whole_arm_coarse_reapproach"
            )
            self.recovery_events.append(
                {
                    "event": "whole_arm_coarse_reapproach",
                    "mode": "high_clearance_rebase",
                    "target_xy_m": target_xy.tolist(),
                    "time_s": float(self.backend.data.time),
                }
            )
        if (
            self.search_samples
            and not execution_internal_correction
            and self.search_segment_policy_mode != "execute_model_only"
        ):
            source_sample = self.search_samples[-1]
            sample_xy = np.asarray(
                source_sample["position_m"], dtype=float
            )[:2]
            # Supervision remains the expert's counterfactual next segment
            # even when a later M4 level executes the policy target.
            planned_delta = expert_target_xy - sample_xy
            maximum_step = float(
                self.task.get("pulse_search_quality_maximum_step_m", 0.004)
            )
            signed_lateral = float(
                np.dot(planned_delta, self.patient_lateral_xy)
            )
            signed_axial = float(
                np.dot(planned_delta, self.patient_axis_xy)
            )
            two_dimensional = bool(
                self.task.get("pulse_search_two_dimensional_enabled", False)
            )
            planned_norm = float(np.linalg.norm(planned_delta))
            self.search_segment_plans.append(
                {
                    "schema_version": (
                        "pulse_search_segment_plan_v3_2d_centering"
                        if two_dimensional
                        else "pulse_search_segment_plan_v2"
                    ),
                    "from_point_index": int(len(self.search_samples) - 1),
                    "to_point_index": int(point_index),
                    "sample_position_xy_m": sample_xy.tolist(),
                    "planned_target_xy_m": expert_target_xy.tolist(),
                    "planned_delta_xy_m": planned_delta.tolist(),
                    "planned_signed_lateral_m": signed_lateral,
                    "planned_signed_axial_m": signed_axial,
                    "planned_delta_patient_uv_m": [
                        signed_lateral,
                        signed_axial,
                    ],
                    "planned_delta_norm_m": planned_norm,
                    "patient_lateral_xy": self.patient_lateral_xy.tolist(),
                    "patient_axis_xy": self.patient_axis_xy.tolist(),
                    "label_source": (
                        "privileged_expert"
                        if source_sample.get("expert_guidance")
                        == "privileged_pulse_center"
                        else "observable_expert"
                    ),
                    "recovery_mode": self._active_search_recovery_mode,
                    "maximum_step_m": maximum_step,
                    "within_maximum_step": bool(
                        (
                            planned_norm
                            if two_dimensional
                            else abs(signed_lateral)
                        )
                        <= maximum_step + 1e-9
                    ),
                    "execution_target_xy_m": target_xy.tolist(),
                    "policy_target_executed": bool(
                        policy_record is not None
                        and policy_record.get("policy_target_executed", False)
                    ),
                }
            )
            if self.search_segment_stop_after_teacher_label:
                raise SearchSegmentTeacherLabelReady(
                    int(source_sample["point_index"])
                )
            if self._search_segment_branch_stop_ready(policy_record):
                self.search_segment_branch_ready_point_index = int(
                    source_sample["point_index"]
                )
                raise SearchSegmentBranchReady(
                    self.search_segment_branch_ready_point_index
                )
        self.search_states.append(PulseSearchState.LIFT_FROM_CURRENT.value)
        if (
            policy_record is not None
            and bool(policy_record.get("policy_target_executed", False))
        ):
            # This is provenance only.  It lets train-only DAgger collectors
            # select the expert's contact-recovery labels caused after a
            # physically executed student search segment without changing the
            # search trajectory or exposing pulse truth to the policy.
            policy_record["execution_started_time_s"] = float(
                self.backend.data.time
            )
        escape_direction = self._thumb_escape_direction()
        escape_steps = int(
            self.task.get("pulse_search_lift_escape_steps", 0)
        )
        escape_step = float(
            self.task.get("pulse_search_lift_escape_step_m", 0.0)
        )
        vertical_unload_steps = max(
            0,
            int(
                self.task.get(
                    "pulse_search_lift_vertical_unload_steps", 0
                )
            ),
        )
        release_count = 0
        lift_index = 0
        lift_clearance_m = float(
            self.task.get(
                "pulse_search_coarse_reapproach_clearance_m", 0.025
            )
            if coarse_reapproach
            else self.task["pulse_search_lift_clearance_m"]
        )
        maximum_lift_m = float(
            self.task.get(
                "pulse_search_coarse_reapproach_maximum_lift_m", 0.040
            )
            if coarse_reapproach
            else self.task["pulse_search_maximum_lift_m"]
        )
        lift_step_m = float(
            self.task.get("pulse_search_lift_step_m", 0.00075)
        )
        if not np.isfinite(lift_step_m) or lift_step_m <= 0.0:
            raise ValueError("pulse_search_lift_step_m must be finite and positive")
        while True:
            lift_delta = np.asarray([0.0, 0.0, lift_step_m])
            if (
                vertical_unload_steps
                <= lift_index
                < vertical_unload_steps + escape_steps
                and escape_step > 0.0
                and np.any(escape_direction)
            ):
                lift_delta[:2] = escape_step * escape_direction
            lift_index += 1
            observation, info = self._step(
                PulseAction(
                    lift_delta,
                    np.zeros(3),
                    target_pressure_kpa=0.0,
                    phase_command=int(TaskPhase.PULSE_SEARCH),
                ),
                "search_lift",
            )
            current = np.asarray(observation["tcp_pose"][:3])
            release_count = (
                release_count + 1
                if not self._diagnostic_contact(observation).any()
                else 0
            )
            lifted = float(current[2] - start[2])
            if (
                lifted
                >= lift_clearance_m
                and release_count >= 3
            ):
                break
            if lifted > maximum_lift_m:
                raise RuntimeError(
                    "contact_not_released_before_translation"
                )
        free_space_recenter = bool(
            self.task.get(
                "pulse_search_free_space_rotation_recenter_enabled", False
            )
            and self._search_free_space_rotation_reference is not None
        )
        if free_space_recenter:
            self.backend.task_target_rotation = (
                self._search_free_space_rotation_reference.copy()
            )
            self.backend.task_rotation_hold_enabled = True
            maximum_recenter_steps = int(
                self.task.get(
                    "pulse_search_free_space_rotation_recenter_steps", 8
                )
            )
            recenter_tolerance = float(
                self.task.get(
                    "pulse_search_free_space_rotation_tolerance_rad", 0.012
                )
            )
            if maximum_recenter_steps < 1 or recenter_tolerance <= 0.0:
                raise ValueError(
                    "pulse-search free-space recenter limits must be positive"
                )
            for _ in range(maximum_recenter_steps):
                probe = dual_probe_frame(
                    self.backend.model, self.backend.data
                ).center
                rotation_error = float(
                    (
                        Rotation.from_matrix(
                            probe.rotation_world_from_frame
                        ).inv()
                        * Rotation.from_matrix(
                            self._search_free_space_rotation_reference
                        )
                    ).magnitude()
                )
                if rotation_error <= recenter_tolerance:
                    break
                observation, _ = self._step(
                    PulseAction(
                        np.zeros(3),
                        np.zeros(3),
                        target_pressure_kpa=0.0,
                        phase_command=int(TaskPhase.PULSE_SEARCH),
                    ),
                    "search_free_space_rotation_recenter",
                )
                if self._diagnostic_contact(observation).any():
                    raise RuntimeError(
                        "pulse_search_contact_during_rotation_recenter"
                    )
        original_wrist_stabilization = bool(
            self.backend.contact_wrist_stabilization_enabled
        )
        if coarse_reapproach:
            self.backend.set_contact_wrist_damping(False)
            probe = dual_probe_frame(
                self.backend.model, self.backend.data
            ).center
            self.backend.task_target_rotation = (
                probe.rotation_world_from_frame.copy()
            )
            self.backend.task_rotation_hold_enabled = True
            self.backend.contact_wrist_stabilization_enabled = False
            self.backend.contact_wrist_reference_rad = None
        self.search_states.append(
            PulseSearchState.VERIFY_CONTACT_RELEASE.value
        )
        self.search_states.append(
            PulseSearchState.TRANSLATE_ABOVE_SURFACE.value
        )
        translation_target_xy = target_xy.copy()
        if coarse_reapproach:
            axial_precompensation = float(
                self.task.get(
                    "pulse_search_coarse_landing_axial_precompensation_m",
                    0.0,
                )
            )
            translation_target_xy += (
                axial_precompensation * self.patient_axis_xy
            )
            signed_lateral_move = float(
                np.dot(
                    target_xy - start[:2],
                    self.patient_lateral_xy,
                )
            )
            lateral_precompensation = float(
                self.task.get(
                    (
                        "pulse_search_coarse_landing_lateral_precompensation_positive_m"
                        if signed_lateral_move > 0.0
                        else "pulse_search_coarse_landing_lateral_precompensation_negative_m"
                    ),
                    0.0,
                )
            )
            if abs(signed_lateral_move) > 1e-12:
                translation_target_xy -= (
                    np.sign(signed_lateral_move)
                    * lateral_precompensation
                    * self.patient_lateral_xy
                )
            if self.recovery_events and self.recovery_events[-1].get(
                "event"
            ) == "whole_arm_coarse_reapproach":
                self.recovery_events[-1]["translation_target_xy_m"] = (
                    translation_target_xy.tolist()
                )
                self.recovery_events[-1][
                    "axial_landing_precompensation_m"
                ] = axial_precompensation
                self.recovery_events[-1][
                    "lateral_landing_precompensation_m"
                ] = lateral_precompensation
        for _ in range(80 if coarse_reapproach else 40):
            current = dual_probe_frame(
                self.backend.model, self.backend.data
            ).center.position_m
            error = translation_target_xy - current[:2]
            if float(np.linalg.norm(error)) <= float(
                self.task["pulse_search_translation_tolerance_m"]
            ):
                break
            delta_xy = np.clip(error, -0.00060, 0.00060)
            observation, _ = self._step(
                PulseAction(
                    np.asarray([delta_xy[0], delta_xy[1], 0.0]),
                    np.zeros(3),
                    phase_command=int(TaskPhase.PULSE_SEARCH),
                ),
                "search_translate",
            )
            if self._diagnostic_contact(observation).any():
                raise RuntimeError(
                    "pulse_search_contact_during_translation"
                )
        else:
            raise RuntimeError("pulse_search_translation_timeout")
        if coarse_reapproach:
            self.backend.contact_wrist_stabilization_enabled = (
                original_wrist_stabilization
            )
            current_arm = self.backend.kinematics.positions(
                self.backend.data
            )
            self.backend.contact_wrist_reference_rad = (
                current_arm[-3:].copy()
            )
            self.backend.set_contact_wrist_damping(True)
        self.search_states.append(
            PulseSearchState.VERIFY_TARGET_REACHED.value
        )
        if free_space_recenter and not coarse_reapproach:
            # Recenter only while clear of the patient.  Contact establishment
            # must retain its small compliant wrist adjustment so all active
            # fingertips can seat on the curved skin.
            self.backend.task_rotation_hold_enabled = False
        self.search_states.append(
            PulseSearchState.DESCEND_TO_NEW_POINT.value
        )
        first_contact = None
        post_first_contact_limit = int(
            self.task.get(
                "pulse_search_coarse_post_contact_descent_steps", 12
            )
            if coarse_reapproach
            else 2
        )
        first_contact_force_limit_n = float(
            self.task.get(
                "pulse_search_coarse_first_contact_force_limit_n", 1.5
            )
        )
        for descent_index in range(
            int(self.task["pulse_search_maximum_descent_steps"])
        ):
            observation, info = self._step(
                PulseAction(
                    np.asarray(
                        [
                            0.0,
                            0.0,
                            -float(self.task["pulse_search_descent_step_m"]),
                        ]
                    ),
                    np.zeros(3),
                    phase_command=int(TaskPhase.PULSE_SEARCH),
                ),
                "search_descend",
            )
            contacts = self._diagnostic_contact(observation)
            if contacts.any() and first_contact is None:
                first_contact = descent_index
            if contacts.all():
                break
            if (
                first_contact is not None
                and (
                    descent_index - first_contact
                    >= post_first_contact_limit
                    or (
                        coarse_reapproach
                        and float(
                            np.max(
                                self._diagnostic_force(info)
                            )
                        )
                        >= first_contact_force_limit_n
                    )
                )
            ):
                break
        else:
            # At the edge of the curved mesh the IK target can settle a few
            # millimetres above the skin even though the original contact
            # point was valid. Reacquire once at the last known contact point
            # before rejecting the episode; this produces a useful recovery
            # transition for both the expert and DAgger.
            self.recovery_events.append(
                {
                    "event": "contact_reacquisition",
                    "mode": "return_to_last_contact",
                    "time_s": float(self.backend.data.time),
                    "target_m": start[:2].tolist(),
                }
            )
            self.search_states.append(PulseSearchState.RECENTER_SEARCH.value)
            for _ in range(24):
                current = dual_probe_frame(
                    self.backend.model, self.backend.data
                ).center.position_m
                error = start[:2] - current[:2]
                if float(np.linalg.norm(error)) <= 0.0005:
                    break
                delta_xy = np.clip(error, -0.0006, 0.0006)
                observation, info = self._step(
                    PulseAction(
                        np.asarray([delta_xy[0], delta_xy[1], 0.0]),
                        np.zeros(3),
                        phase_command=int(TaskPhase.PULSE_SEARCH),
                    ),
                    "search_reacquire_translate",
                )
                if self._diagnostic_contact(observation).any():
                    break
            for retry_index in range(36):
                observation, info = self._step(
                    PulseAction(
                        np.asarray(
                            [
                                0.0,
                                0.0,
                                -float(
                                    self.task[
                                        "pulse_search_descent_step_m"
                                    ]
                                ),
                            ]
                        ),
                        np.zeros(3),
                        phase_command=int(TaskPhase.PULSE_SEARCH),
                    ),
                    "search_reacquire_descend",
                )
                if self._diagnostic_contact(observation).any():
                    first_contact = retry_index
                    break
            if first_contact is None:
                raise RuntimeError("pulse_search_descent_contact_timeout")
        if first_contact is None:
            raise RuntimeError("pulse_search_descent_contact_timeout")
        self.backend.task_rotation_hold_enabled = False
        recovered_search_landing = False
        if not self._diagnostic_contact(observation).all():
            recovery_iterations = (
                int(
                    self.task.get(
                        (
                            "pulse_search_coarse_dual_contact_recovery_iterations"
                            if coarse_reapproach
                            else "pulse_search_dual_contact_recovery_iterations"
                        ),
                        self.task["dual_contact_recovery_iterations"],
                    )
                )
            )
            recovery_force_limit = (
                float(
                    self.task.get(
                        (
                            "pulse_search_coarse_dual_contact_recovery_force_limit_n"
                            if coarse_reapproach
                            else "pulse_search_dual_contact_recovery_force_limit_n"
                        ),
                        self.task["pulse_search_maximum_sampling_force_n"],
                    )
                )
            )
            recovery_minimum_imbalance = (
                float(
                    self.task.get(
                        (
                            "pulse_search_coarse_dual_contact_minimum_tilt_imbalance_n"
                            if coarse_reapproach
                            else "pulse_search_dual_contact_minimum_tilt_imbalance_n"
                        ),
                        0.0,
                    )
                )
            )
            recovery_descent_step_m = float(
                self.task.get(
                    (
                        "pulse_search_coarse_dual_contact_recovery_descent_step_m"
                        if coarse_reapproach
                        else "pulse_search_dual_contact_recovery_descent_step_m"
                    ),
                    self.task.get(
                        "dual_contact_recovery_descent_step_m", 0.00005
                    ),
                )
            )
            observation, info = self._recover_dual_contact(
                observation,
                info,
                phase=TaskPhase.PULSE_SEARCH,
                maximum_iterations=recovery_iterations,
                maximum_force_n=recovery_force_limit,
                minimum_single_contact_imbalance_n=(
                    recovery_minimum_imbalance
                ),
                descent_step_m=recovery_descent_step_m,
            )
            recovered_search_landing = True
        if (
            recovered_search_landing
            and _search_landing_pressure_handoff_hold_enabled(self.task)
        ):
            # The first pressure command after a search landing needs the
            # same seated joint target as a later contact-loss recovery.
            # Otherwise the pressure IK can resume the stale aerial target
            # and lift an already recovered three-pad contact off the skin.
            self.backend.arm_pressure_recovery_rotation_hold()
            self.recovery_events.append(
                {
                    "event": "search_landing_pressure_handoff_hold",
                    "mode": "observable_recovered_contact_target",
                    "contacts": self._diagnostic_contact(
                        observation
                    ).astype(bool).tolist(),
                    "forces_n": self._diagnostic_force(info).astype(
                        float
                    ).tolist(),
                    "truth_in_observation": False,
                    "time_s": float(self.backend.data.time),
                }
            )
        self.search_states.append(
            PulseSearchState.ESTABLISH_DUAL_CONTACT.value
        )
        self.search_states.append(PulseSearchState.REGULATE_PRESSURE.value)
        self._regulate_pressure(TaskPhase.PULSE_SEARCH)
        reached_position = dual_probe_frame(
            self.backend.model, self.backend.data
        ).center.position_m.copy()
        if (
            self.search_segment_plans
            and self.search_segment_plans[-1]["to_point_index"]
            == int(point_index)
        ):
            plan = self.search_segment_plans[-1]
            executed_delta = reached_position[:2] - np.asarray(
                plan["sample_position_xy_m"], dtype=float
            )
            plan["executed_landing_xy_m"] = reached_position[:2].tolist()
            plan["executed_landing_delta_xy_m"] = executed_delta.tolist()
            plan["executed_signed_lateral_m"] = float(
                np.dot(executed_delta, self.patient_lateral_xy)
            )
            plan["landing_target_error_xy_m"] = (
                reached_position[:2]
                - np.asarray(plan["planned_target_xy_m"], dtype=float)
            ).tolist()
        source_policy_record = self._search_segment_policy_by_point.get(
            int(point_index) - 1
        )
        if (
            not execution_internal_correction
            and
            source_policy_record is not None
            and source_policy_record.get("policy_target_executed", False)
        ):
            source_position = np.asarray(
                self.search_samples[int(point_index) - 1]["position_m"],
                dtype=float,
            )
            executed_delta = reached_position[:2] - source_position[:2]
            source_policy_record["executed_landing_xy_m"] = (
                reached_position[:2].tolist()
            )
            source_policy_record["executed_landing_delta_xy_m"] = (
                executed_delta.tolist()
            )
            source_policy_record["executed_signed_lateral_m"] = float(
                np.dot(executed_delta, self.patient_lateral_xy)
            )
            source_policy_record["landing_target_error_xy_m"] = (
                reached_position[:2]
                - np.asarray(
                    source_policy_record["execution_target_xy_m"], dtype=float
                )
            ).tolist()
        self.events.append(
            {
                "event": "search_point_reached",
                "point_index": int(point_index),
                "time_s": float(self.backend.data.time),
                "position_m": reached_position.tolist(),
                "execution_internal_correction": bool(
                    execution_internal_correction
                ),
            }
        )
        self._active_search_recovery_mode = None
        return reached_position

    def _thumb_escape_direction(self) -> np.ndarray:
        """Return a horizontal direction away from the forearm centerline."""
        if not bool(
            self.task.get("pulse_search_lift_escape_enabled", False)
        ):
            return np.zeros(2, dtype=float)
        side = str(self.config["hand"].get("diagnosis_side", "right")).lower()
        thumb_name = "L_thumb_distal" if side == "left" else "R_thumb_distal"
        try:
            thumb = self.backend.data.xpos[
                self.backend.model.body(thumb_name).id
            ][:2]
        except (KeyError, ValueError):
            return np.zeros(2, dtype=float)
        forearm = self.config["scene"]["patient_forearm"]
        start = np.asarray(forearm["from_m"], dtype=float)[:2]
        end = np.asarray(forearm["to_m"], dtype=float)[:2]
        axis = end - start
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            return np.zeros(2, dtype=float)
        axis /= norm
        closest = start + axis * float(np.dot(thumb - start, axis))
        radial = thumb - closest
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= 1e-9 or radial_norm > float(
            self.task.get("pulse_search_lift_escape_trigger_m", 0.080)
        ):
            return np.zeros(2, dtype=float)
        return radial / radial_norm

    def _schedule_quality_recovery_target(
        self, origin: np.ndarray, step_m: float, radius_m: float
    ) -> np.ndarray | None:
        """Choose an unvisited lateral probe point after a weak sample."""
        if not bool(
            self.task.get("pulse_search_quality_recovery_enabled", False)
        ):
            return None
        lateral = np.asarray(self.patient_lateral_xy, dtype=float)
        lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
        axial = np.asarray(self.patient_axis_xy, dtype=float)
        axial /= max(float(np.linalg.norm(axial)), 1e-12)
        axes = (
            (lateral, axial)
            if bool(self.task.get("pulse_search_two_dimensional_enabled", False))
            else (lateral,)
        )
        base_step = float(
            np.clip(
                step_m
                * float(
                    self.task.get(
                        "pulse_search_quality_recovery_step_scale", 0.75
                    )
                ),
                float(self.task["pulse_search_quality_minimum_step_m"]),
                float(self.task["pulse_search_quality_maximum_step_m"]),
            )
        )
        maximum_radius = min(
            float(radius_m),
            float(
                self.task.get(
                    "pulse_search_quality_recovery_max_radius_m", radius_m
                )
            ),
        )
        existing = [
            np.asarray(sample["position_m"], dtype=float)[:2]
            for sample in self.search_samples
        ]
        existing.extend(
            target.copy() for target in self._quality_recovery_targets_xy
        )
        # Alternate sides and expand one step at a time.  This is bounded,
        # deterministic, and keeps the three-stage motion contract intact.
        for multiplier in range(1, 8):
            magnitude = base_step * multiplier
            if magnitude > maximum_radius + 1e-9:
                break
            for axis in axes:
                for sign in (1.0, -1.0):
                    candidate = origin[:2] + sign * magnitude * axis
                    if any(
                        np.linalg.norm(candidate - previous) <= 0.00010
                        for previous in existing
                    ):
                        continue
                    self._quality_recovery_index += 1
                    self._quality_recovery_targets_xy.append(candidate.copy())
                    self.recovery_events.append(
                        {
                            "event": "pulse_search_quality_recovery",
                            "mode": (
                                "bounded_planar_probe"
                                if len(axes) == 2
                                else "bounded_lateral_probe"
                            ),
                            "index": self._quality_recovery_index,
                            "offset_m": (candidate - origin[:2]).tolist(),
                            "time_s": float(self.backend.data.time),
                        }
                    )
                    self._pending_search_recovery_mode = (
                        "pulse_search_quality_recovery"
                    )
                    return np.asarray([candidate[0], candidate[1]], dtype=float)
        return None

    def _directional_search(
        self, *, stop_on_signal_quality: bool = False
    ) -> tuple[int, str]:
        prefix = (
            "pulse_search_quality"
            if stop_on_signal_quality
            else "pulse_search_directional"
        )
        maximum_points = int(self.task[f"{prefix}_maximum_points"])
        maximum_attempts = int(self.task[f"{prefix}_maximum_attempts"])
        minimum_points = int(
            self.task["pulse_search_directional_minimum_points"]
        )
        minimum_step = float(self.task[f"{prefix}_minimum_step_m"])
        maximum_step = float(self.task[f"{prefix}_maximum_step_m"])
        convergence = float(
            self.task["pulse_search_directional_convergence_bias_m"]
        )
        search_radius = float(
            self.task.get("pulse_search_quality_radius_m", np.inf)
        )
        origin = dual_probe_frame(
            self.backend.model, self.backend.data
        ).center.position_m.copy()
        self._search_origin_xy = origin[:2].copy()
        target = origin.copy()
        travel_direction = None
        direction_reversed = False
        non_improving_count = 0
        best_score = float("-inf")
        best_centering_error_mm = float("inf")
        last_step = maximum_step
        stop_reason = "maximum_points"
        attempts = 0
        failed_landing_recoveries = 0

        def bounded_subtarget(value: np.ndarray) -> np.ndarray:
            """Turn a distant logical recovery target into one safe segment.

            Recovery candidates are defined around the first-contact origin,
            while supervision is defined between consecutive sampled points.
            Once the search has moved away from the origin those are not the
            same distance.  Emit an explicit intermediate target instead of
            hiding an out-of-contract label with dataset-side clipping.
            """
            bounded = np.asarray(value, dtype=float).copy()
            if not self.search_samples:
                return bounded
            current = np.asarray(
                self.search_samples[-1]["position_m"], dtype=float
            )
            delta = bounded[:2] - current[:2]
            distance = float(np.linalg.norm(delta))
            if distance <= maximum_step + 1e-9:
                return bounded
            bounded[:2] = current[:2] + delta * (maximum_step / distance)
            self.recovery_events.append(
                {
                    "event": "bounded_search_subtarget",
                    "mode": "maximum_segment_step",
                    "requested_target_xy_m": np.asarray(value)[:2].tolist(),
                    "bounded_target_xy_m": bounded[:2].tolist(),
                    "maximum_step_m": maximum_step,
                    "time_s": float(self.backend.data.time),
                }
            )
            self._pending_search_recovery_mode = "bounded_search_subtarget"
            return bounded

        def anchor_patient_axis(value: np.ndarray) -> np.ndarray:
            """Keep each planned search segment purely lateral.

            Do not pull every landing back toward the first-contact axial
            coordinate. That turns a lateral probe into a diagonal drag when
            compliant contact has a small axial offset. The pressure hold now
            suppresses cumulative creep, so preserving the current sample's
            axial coordinate is both safer and faithful to the label contract.
            """
            anchored = np.asarray(value, dtype=float).copy()
            if bool(
                self.task.get("pulse_search_two_dimensional_enabled", False)
            ):
                return anchored
            reference = (
                np.asarray(self.search_samples[-1]["position_m"], dtype=float)
                if self.search_samples
                else origin
            )
            axial_error = float(
                np.dot(anchored[:2] - reference[:2], self.patient_axis_xy)
            )
            anchored[:2] -= axial_error * self.patient_axis_xy
            return anchored

        def execute_with_landing_guard(
            logical_target: np.ndarray, point_index: int
        ) -> np.ndarray:
            """Reach one labelled target and correct execution error safely.

            The correction reads only commanded and measured probe positions.
            It is an execution-internal high-clearance move, so it neither
            changes the expert segment label nor creates an extra label.
            """

            desired_xy = np.asarray(logical_target, dtype=float)[:2].copy()
            reached = self._three_stage_move(desired_xy, point_index)
            enabled = bool(
                self.task.get(
                    "pulse_search_landing_correction_enabled", False
                )
            )
            if not enabled:
                return reached
            tolerance = float(
                self.task.get(
                    "pulse_search_landing_correction_tolerance_m", 0.003
                )
            )
            radius_tolerance = float(
                self.task.get(
                    "pulse_search_executed_radius_tolerance_m", 0.0005
                )
            )
            attempts = int(
                self.task.get(
                    "pulse_search_landing_correction_maximum_attempts", 0
                )
            )
            gain = float(
                self.task.get("pulse_search_landing_correction_gain", 1.0)
            )
            maximum_compensation = float(
                self.task.get(
                    "pulse_search_landing_correction_maximum_m", 0.006
                )
            )
            trusted_error_maximum = float(
                self.task.get(
                    "pulse_search_landing_trusted_error_maximum_m",
                    maximum_compensation,
                )
            )
            if (
                not np.isfinite(trusted_error_maximum)
                or trusted_error_maximum <= 0.0
            ):
                raise ValueError(
                    "pulse-search trusted landing error must be positive"
                )
            execution_target = desired_xy.copy()
            for attempt in range(attempts + 1):
                error_xy = reached[:2] - desired_xy
                error_norm = float(np.linalg.norm(error_xy))
                lateral_error = abs(
                    float(np.dot(error_xy, self.patient_lateral_xy))
                )
                executed_radius = float(
                    np.linalg.norm(reached[:2] - origin[:2])
                )
                if (
                    error_norm > trusted_error_maximum
                    or executed_radius
                    > search_radius + trusted_error_maximum
                ):
                    self.recovery_events.append(
                        {
                            "event": "pulse_search_untrusted_landing",
                            "mode": "reject_before_landing_correction",
                            "point_index": int(point_index),
                            "desired_target_xy_m": desired_xy.tolist(),
                            "measured_landing_xy_m": reached[:2].tolist(),
                            "landing_error_xy_m": error_xy.tolist(),
                            "landing_error_norm_m": error_norm,
                            "executed_radius_m": executed_radius,
                            "trusted_error_maximum_m": (
                                trusted_error_maximum
                            ),
                            "truth_in_observation": False,
                            "time_s": float(self.backend.data.time),
                        }
                    )
                    raise RuntimeError(
                        "pulse_search_untrusted_landing: "
                        f"xy_error_mm={1000.0 * error_norm:.3f}, "
                        f"radius_mm={1000.0 * executed_radius:.3f}"
                    )
                if (
                    lateral_error <= tolerance
                    and executed_radius <= search_radius + radius_tolerance
                ):
                    return reached
                if attempt >= attempts:
                    raise RuntimeError(
                        "pulse_search_landing_accuracy_exceeded: "
                        f"lateral_error_mm={1000.0 * lateral_error:.3f}, "
                        f"xy_error_mm={1000.0 * error_norm:.3f}, "
                        f"radius_mm={1000.0 * executed_radius:.3f}"
                    )
                compensation = -gain * error_xy
                compensation_norm = float(np.linalg.norm(compensation))
                if compensation_norm > maximum_compensation:
                    compensation *= maximum_compensation / compensation_norm
                # Iterate from the previous aerial command. Re-anchoring every
                # retry at desired_xy makes a systematic seating bias oscillate
                # instead of converge.
                execution_target = execution_target + compensation
                self.recovery_events.append(
                    {
                        "event": "pulse_search_landing_correction",
                        "mode": "measured_execution_error",
                        "point_index": int(point_index),
                        "attempt": int(attempt + 1),
                        "desired_target_xy_m": desired_xy.tolist(),
                        "previous_landing_xy_m": reached[:2].tolist(),
                        "previous_error_xy_m": error_xy.tolist(),
                        "previous_radius_m": executed_radius,
                        "compensated_target_xy_m": (
                            execution_target.tolist()
                        ),
                        "truth_in_observation": False,
                        "time_s": float(self.backend.data.time),
                    }
                )
                self._pending_search_recovery_mode = (
                    "landing_error_correction"
                )
                reached = self._three_stage_move(
                    execution_target,
                    point_index,
                    execution_internal_correction=True,
                )
            raise AssertionError("landing correction loop did not terminate")

        while (
            len(self.search_samples) < maximum_points
            and attempts < maximum_attempts
        ):
            attempts += 1
            if self.search_samples:
                target = bounded_subtarget(target)
                try:
                    execute_with_landing_guard(
                        target[:2], len(self.search_samples)
                    )
                except RuntimeError as error:
                    maximum_failed_landing_recoveries = int(
                        self.task.get(
                            "pulse_search_failed_landing_maximum_recoveries",
                            0,
                        )
                    )
                    preview_rejected = str(error).startswith(
                        "joint_path_preview_forbidden during "
                    )
                    recoverable = bool(
                        stop_on_signal_quality
                        and self.task.get(
                            "pulse_search_failed_landing_recovery_enabled",
                            False,
                        )
                        and (
                            str(error) == "dual_contact_establishment_timeout"
                            or preview_rejected
                            or str(error).startswith(
                                "pulse_search_untrusted_landing:"
                            )
                        )
                        and failed_landing_recoveries
                        < maximum_failed_landing_recoveries
                    )
                    if not recoverable:
                        raise
                    failed_landing_recoveries += 1
                    source_policy_record = self._search_segment_policy_by_point.get(
                        int(len(self.search_samples) - 1)
                    )
                    if source_policy_record is not None and preview_rejected:
                        source_policy_record.update(
                            {
                                "policy_target_execution_completed": False,
                                "execution_rejection_reason": (
                                    "joint_path_preview_forbidden"
                                ),
                            }
                        )
                    failed_target_xy = target[:2].copy()
                    last_valid = np.asarray(
                        self.search_samples[-1]["position_m"], dtype=float
                    )
                    last_valid_point_index = len(self.search_samples) - 1
                    return_command_xy = last_valid[:2].copy()
                    for plan in reversed(self.search_segment_plans):
                        if (
                            int(plan.get("to_point_index", -1))
                            == last_valid_point_index
                            and plan.get("execution_target_xy_m") is not None
                        ):
                            return_command_xy = np.asarray(
                                plan["execution_target_xy_m"], dtype=float
                            ).reshape(2)
                            break
                    self.recovery_events.append(
                        {
                            "event": "pulse_search_failed_landing_recovery",
                            "mode": "return_to_last_valid_contact",
                            "recovery_index": failed_landing_recoveries,
                            "failed_target_xy_m": failed_target_xy.tolist(),
                            "last_valid_xy_m": last_valid[:2].tolist(),
                            "return_command_xy_m": (
                                return_command_xy.tolist()
                            ),
                            "failure": str(error),
                            "truth_in_observation": False,
                            "time_s": float(self.backend.data.time),
                        }
                    )
                    self._pending_search_recovery_mode = (
                        "failed_landing_return_to_last_valid"
                    )
                    self._three_stage_move(
                        return_command_xy,
                        len(self.search_samples),
                        execution_internal_correction=True,
                    )
                    recovery_target = self._schedule_quality_recovery_target(
                        origin, last_step, search_radius
                    )
                    if recovery_target is None:
                        raise
                    target = np.asarray(
                        [recovery_target[0], recovery_target[1], origin[2]],
                        dtype=float,
                    )
                    target = anchor_patient_axis(target)
                    continue
            record = self._sample_current_point(len(self.search_samples))
            if not record["settled"] or record["pulse_score_kpa"] <= 0.0:
                stop_reason = "invalid_contact_sample"
                break
            policy_record = self._query_search_segment_policy(record)
            if (
                stop_on_signal_quality
                and record["signal_quality"]["accepted"]
            ):
                if policy_record is not None:
                    model_only = (
                        self.search_segment_policy_mode
                        == "execute_model_only"
                    )
                    policy_record.update(
                        {
                            "expert_stop": None if model_only else True,
                            "expert_signed_displacement_m": (
                                None if model_only else 0.0
                            ),
                            "signed_error_mm": (
                                None
                                if model_only
                                else float(
                                    1000.0
                                    * policy_record[
                                        "signed_displacement_m"
                                    ]
                                )
                            ),
                            "direction_agrees_with_expert": None,
                            "policy_target_executed": False,
                            "execution_rejection_reason": None,
                            "stop_reference": (
                                "deterministic_signal_quality_gate"
                            ),
                        }
                    )
                self._accepted_quality_window = self._last_sample_window
                self._last_search_stop_reason = "signal_quality_satisfied"
                return len(self.search_samples) - 1, "signal_quality_satisfied"
            if self.search_segment_policy_mode == "execute_model_only":
                if policy_record is None:
                    raise RuntimeError("model-only search has no segment proposal")
                policy_record.update(
                    {
                        "expert_stop": None,
                        "expert_signed_displacement_m": None,
                        "signed_error_mm": None,
                        "direction_agrees_with_expert": None,
                        "stop_reference": "deterministic_signal_quality_gate",
                    }
                )
                if policy_record["stop"]:
                    policy_record["execution_rejection_reason"] = (
                        "weak_signal_stop_vetoed"
                    )
                    raise RuntimeError(
                        "model requested stop before signal quality was accepted"
                    )
                model_delta_uv = np.asarray(
                    policy_record["delta_uv_m"], dtype=float
                )
                model_delta_xy = _patient_delta_uv_to_world_xy(
                    model_delta_uv,
                    patient_lateral_xy=self.patient_lateral_xy,
                    patient_axis_xy=self.patient_axis_xy,
                )
                minimum_model_step = float(
                    self.task.get(
                        "pulse_search_model_minimum_segment_m", 0.00025
                    )
                )
                if np.linalg.norm(model_delta_uv) < minimum_model_step:
                    policy_record["execution_rejection_reason"] = (
                        "policy_segment_below_minimum"
                    )
                    raise RuntimeError("model search segment is too small")
                target = np.asarray(record["position_m"], dtype=float)
                target[:2] += model_delta_xy
                if policy_record["segment_contract"] == "patient_lateral_1d":
                    target = anchor_patient_axis(target)
                requested_target_xy = target[:2].copy()
                landing_radius_margin = float(
                    self.task.get(
                        "pulse_search_executed_radius_tolerance_m", 0.0005
                    )
                )
                commanded_search_radius = max(
                    minimum_model_step,
                    search_radius - landing_radius_margin,
                )
                target[:2], radius_limited = clip_target_to_search_radius(
                    origin[:2], target[:2], commanded_search_radius
                )
                executed_signed_step = float(
                    np.dot(
                        target[:2]
                        - np.asarray(record["position_m"], dtype=float)[:2],
                        self.patient_lateral_xy,
                    )
                )
                executed_axial_step = float(
                    np.dot(
                        target[:2]
                        - np.asarray(record["position_m"], dtype=float)[:2],
                        self.patient_axis_xy,
                    )
                )
                executed_delta_norm = float(
                    np.hypot(executed_signed_step, executed_axial_step)
                )
                policy_record["requested_execution_target_xy_m"] = (
                    requested_target_xy.tolist()
                )
                policy_record["search_radius_limited"] = bool(radius_limited)
                policy_record["execution_target_radius_m"] = float(
                    np.linalg.norm(target[:2] - origin[:2])
                )
                policy_record["commanded_search_radius_limit_m"] = float(
                    commanded_search_radius
                )
                policy_record["landing_radius_margin_m"] = float(
                    landing_radius_margin
                )
                policy_record["executed_signed_displacement_m"] = (
                    executed_signed_step
                )
                policy_record["executed_delta_uv_m"] = [
                    executed_signed_step,
                    executed_axial_step,
                ]
                if radius_limited:
                    policy_record["safety_modification_reason"] = (
                        "search_radius_clipped"
                    )
                if executed_delta_norm < minimum_model_step:
                    policy_record["execution_rejection_reason"] = (
                        "search_radius_exhausted"
                    )
                    raise RuntimeError(
                        "model search radius exhausted before signal quality"
                    )
                policy_record["policy_target_executed"] = True
                policy_record["execution_target_xy_m"] = target[:2].tolist()
                self._pending_search_recovery_mode = "model_only_segment"
                continue
            self.search_expert_target_queries += 1
            score = float(record["pulse_score_kpa"])
            direction_xy = np.asarray(
                record["pulse_direction_xy"], dtype=float
            )
            rejection_reasons = list(
                record["signal_quality"].get("rejection_reasons", ())
            )
            if self.privileged_search_assist and (
                (
                    self.privileged_search_assist_after_student_execution
                    and self.search_segment_policy_executions > 0
                )
                or _use_privileged_search_fallback(
                    self.task, rejection_reasons
                )
            ):
                # Training-only oracle label denoising.  The policy-visible
                # observation remains unchanged; only the expert's search
                # target uses simulator truth.  Reference the actual dual-pad
                # probe center rather than the noisy signal-weighted taxel
                # center.  The latter moves as the pulse waveform/noise moves
                # and produced direction reversals in matched A/B sweeps.
                self.privileged_state_reads += 1
                pulse_truth = np.asarray(
                    self.env.get_privileged_state()["pulse_center_m"],
                    dtype=float,
                )
                probe_center = np.asarray(record["position_m"], dtype=float)
                direction_xy = _privileged_search_direction_xy(
                    pulse_truth[:2],
                    probe_center[:2],
                    patient_lateral_xy=self.patient_lateral_xy,
                    two_dimensional=bool(
                        self.task.get(
                            "pulse_search_two_dimensional_enabled", False
                        )
                    ),
                )
                record["expert_guidance"] = "privileged_pulse_center"
                record["expert_guidance_reference"] = (
                    "dual_probe_center"
                )
                record["expert_guidance_direction_xy"] = (
                    direction_xy.tolist()
                )
            elif not pulse_spatial_direction_is_reliable(
                record["signal_quality"]
            ):
                record["expert_guidance"] = (
                    "observable_previous_valid_direction"
                )
                record["expert_guidance_invalid_reasons"] = list(
                    record["signal_quality"].get(
                        "rejection_reasons", ()
                    )
                )
                if travel_direction is None:
                    # A transient spike or short-window spectral miss can
                    # invalidate the very first quality window. There is no
                    # previous direction to reuse yet, but terminating here
                    # makes one observable outlier fatal. Bootstrap with one
                    # minimum-size probe on the fixed patient-lateral axis.
                    # This uses neither pulse truth nor the rejected spatial
                    # gradient; subsequent valid tactile windows regain full
                    # directional authority.
                    travel_direction = self.patient_lateral_xy.copy()
                    direction_norm = float(np.linalg.norm(travel_direction))
                    if direction_norm <= 1e-12:
                        stop_reason = "patient_lateral_axis_unavailable"
                        break
                    travel_direction /= direction_norm
                    record["expert_guidance"] = (
                        "observable_deterministic_initial_probe"
                    )
                step = float(
                    minimum_step
                    if record["expert_guidance"]
                    == "observable_deterministic_initial_probe"
                    else np.clip(last_step, minimum_step, maximum_step)
                )
                target = np.asarray(record["position_m"], dtype=float)
                target[:2] += step * travel_direction
                target = anchor_patient_axis(target)
                offset = target[:2] - origin[:2]
                if np.linalg.norm(offset) > search_radius:
                    target[:2] = origin[:2] + (
                        search_radius * offset / np.linalg.norm(offset)
                    )
                self.recovery_events.append(
                    {
                        "event": (
                            "pulse_search_temporal_initial_probe"
                            if record["expert_guidance"]
                            == "observable_deterministic_initial_probe"
                            else "pulse_search_temporal_outlier_ignored"
                        ),
                        "point_index": int(record["point_index"]),
                        "rejection_reasons": list(
                            record["signal_quality"].get(
                                "rejection_reasons", ()
                            )
                        ),
                        "reused_direction_xy": travel_direction.tolist(),
                        "step_m": step,
                        "truth_in_observation": False,
                        "time_s": float(self.backend.data.time),
                    }
                )
                self._pending_search_recovery_mode = str(
                    record["expert_guidance"]
                )
                continue
            direction_magnitude = float(np.linalg.norm(direction_xy))
            valid_count = len(self.search_samples)
            coverage = record["signal_quality"].get("tactile_coverage", {})
            per_finger_offsets = np.asarray(
                coverage.get(
                    "per_finger_weighted_centroid_offset_mm", ()
                ),
                dtype=float,
            )
            spatial_centering_active = bool(
                coverage.get("enabled", False)
                and not coverage.get("accepted", False)
                and per_finger_offsets.shape
                == (len(self.diagnostic_finger_names),)
                and np.isfinite(per_finger_offsets).all()
            )
            if spatial_centering_active:
                # Unequal per-finger amplitude is valid. While any active pad
                # is off-centre, judge progress by the worst individual pad
                # centroid rather than aggregate pulse strength; otherwise a
                # strong finger can reverse a geometrically improving search.
                centering_error_mm = float(np.max(per_finger_offsets))
                improved = bool(
                    centering_error_mm < best_centering_error_mm - 0.02
                )
                best_centering_error_mm = min(
                    best_centering_error_mm, centering_error_mm
                )
                record["search_progress_metric"] = (
                    "maximum_per_finger_centroid_offset_mm"
                )
                record["search_progress_value"] = centering_error_mm
            else:
                improved = bool(score > best_score * 1.01)
                record["search_progress_metric"] = "pulse_score_kpa"
                record["search_progress_value"] = score
            non_improving_count = (
                0 if improved else non_improving_count + 1
            )
            # A single pulse-amplitude dip is common under tactile noise and
            # should not override a still-consistent per-taxel direction.
            # Require two consecutive non-improving points before bracketing
            # or reversing the travel direction.
            if valid_count > 1 and non_improving_count >= 2:
                if not direction_reversed and travel_direction is not None:
                    travel_direction = -travel_direction
                    direction_reversed = True
                    non_improving_count = 0
                    last_step = max(minimum_step, 0.5 * last_step)
                    target = origin.copy()
                    target[:2] += last_step * travel_direction
                    target = anchor_patient_axis(target)
                    offset = target[:2] - origin[:2]
                    if np.linalg.norm(offset) > search_radius:
                        target[:2] = origin[:2] + (
                            search_radius * offset / np.linalg.norm(offset)
                        )
                    continue
                if valid_count >= minimum_points:
                    recovery_target = (
                        self._schedule_quality_recovery_target(
                            origin, last_step, search_radius
                        )
                        if stop_on_signal_quality
                        else None
                    )
                    if recovery_target is not None:
                        target = np.asarray(
                            [recovery_target[0], recovery_target[1], origin[2]]
                        )
                        target = anchor_patient_axis(target)
                        continue
                    stop_reason = "maximum_bracketed"
                    break
            best_score = max(best_score, score)
            if (
                valid_count >= minimum_points
                and direction_magnitude <= convergence
            ):
                recovery_target = (
                    self._schedule_quality_recovery_target(
                        origin, last_step, search_radius
                    )
                    if stop_on_signal_quality
                    else None
                )
                if recovery_target is not None:
                    target = np.asarray(
                        [recovery_target[0], recovery_target[1], origin[2]]
                    )
                    target = anchor_patient_axis(target)
                    continue
                stop_reason = "taxel_gradient_converged"
                break
            if direction_magnitude <= 1e-9:
                stop_reason = "no_spatial_gradient"
                break
            direction_unit = direction_xy / direction_magnitude
            directional_maximum_step = _direction_limited_search_maximum_step(
                self.task,
                patient_axis_xy=self.patient_axis_xy,
                direction_unit_xy=direction_unit,
                default_maximum_step_m=maximum_step,
            )
            step = float(
                np.clip(
                    0.65 * direction_magnitude,
                    minimum_step,
                    directional_maximum_step,
                )
            )
            if travel_direction is None:
                travel_direction = direction_unit
            elif np.dot(direction_unit, travel_direction) <= 0.0:
                if valid_count >= minimum_points:
                    recovery_target = (
                        self._schedule_quality_recovery_target(
                            origin, last_step, search_radius
                        )
                        if stop_on_signal_quality
                        else None
                    )
                    if recovery_target is not None:
                        target = np.asarray(
                            [recovery_target[0], recovery_target[1], origin[2]]
                        )
                        target = anchor_patient_axis(target)
                        continue
                    stop_reason = "gradient_reversal"
                    break
                travel_direction = direction_unit
            last_step = step
            target = np.asarray(record["position_m"], dtype=float)
            target[:2] += step * travel_direction
            target = anchor_patient_axis(target)
            offset = target[:2] - origin[:2]
            if np.linalg.norm(offset) > search_radius:
                target[:2] = origin[:2] + (
                    search_radius * offset / np.linalg.norm(offset)
                )

        if stop_on_signal_quality:
            self._last_search_stop_reason = stop_reason
            raise RuntimeError(
                "signal_quality_not_satisfied: " + stop_reason
            )

        scores = [sample["pulse_score_kpa"] for sample in self.search_samples]
        best_index = best_finite_score_index(scores)
        if best_index is None:
            raise RuntimeError("directional_search_has_no_valid_sample")
        current = dual_probe_frame(
            self.backend.model, self.backend.data
        ).center.position_m
        selected = np.asarray(
            self.search_samples[best_index]["position_m"], dtype=float
        )
        self.search_states.append(PulseSearchState.SELECT_BEST_POINT.value)
        if np.linalg.norm(current[:2] - selected[:2]) > 0.00010:
            self.search_states.append(PulseSearchState.RECENTER_SEARCH.value)
            self._three_stage_move(selected[:2], len(self.search_samples))
        return best_index, stop_reason

    def _capture(
        self, initial_window: dict[str, np.ndarray] | None = None
    ) -> dict:
        self.search_states.append(PulseSearchState.FINAL_HOLD.value)
        tactile = (
            []
            if initial_window is None
            else list(np.asarray(initial_window["tactile"]).copy())
        )
        raw = (
            []
            if initial_window is None
            else list(np.asarray(initial_window["raw"]).copy())
        )
        force = (
            []
            if initial_window is None
            else list(np.asarray(initial_window["force"]).copy())
        )
        contact = (
            []
            if initial_window is None
            else list(np.asarray(initial_window["contact"]).copy())
        )
        capture_steps = int(self.task.get("signal_capture_steps", 120))
        reused_steps = min(len(tactile), capture_steps)
        if reused_steps:
            tactile = tactile[-reused_steps:]
            raw = raw[-reused_steps:]
            force = force[-reused_steps:]
            contact = contact[-reused_steps:]
        new_capture_steps = capture_steps - reused_steps
        use_policy_tactile = bool(
            self.task.get("policy_tactile_bandpass_enabled", False)
        )
        for _ in range(new_capture_steps):
            observation, info = self._step(
                self._phase_action(
                    TaskPhase.SIGNAL_CAPTURE,
                    pressure_kpa=self.target_pressure_kpa,
                ),
                "signal_capture",
            )
            diagnostic = self._diagnostic_arrays()
            tactile.append(
                (
                    diagnostic["pressure"].astype(float, copy=True)
                    if use_policy_tactile
                    else diagnostic["dynamic"].astype(float, copy=True)
                )
            )
            raw.append(diagnostic["raw"].copy())
            force.append(diagnostic["force"].copy())
            contact.append(diagnostic["contact"].copy())
        sample_rate_hz = 1.0 / (
            self.backend.physics_steps_per_policy
            * float(self.backend.model.opt.timestep)
        )
        tactile_array = np.asarray(tactile, dtype=float)
        tactile_for_filter, despike_replacements = (
            self._despike_policy_tactile_window(tactile_array)
            if use_policy_tactile
            else (tactile_array, 0)
        )
        dynamic_array = (
            self._pulse_band_component(
                tactile_for_filter,
                sample_rate_hz,
                float(
                    self.task.get(
                        "policy_tactile_filter_minimum_frequency_hz",
                        self.quality_thresholds.minimum_frequency_hz,
                    )
                ),
                float(
                    self.task.get(
                        "policy_tactile_filter_maximum_frequency_hz",
                        self.quality_thresholds.maximum_frequency_hz,
                    )
                ),
            )
            if use_policy_tactile
            else np.asarray(tactile)
        )
        if use_policy_tactile:
            dynamic_array *= float(
                self.task.get(
                    "policy_tactile_bandpass_calibration_gain", 1.0
                )
            )
        channel_amplitude = self._channel_amplitude(dynamic_array)
        signal = dynamic_array.mean(axis=(1, 2))
        sample_time = np.arange(len(signal), dtype=float) / 20.0
        detrended = signal - np.polyval(
            np.polyfit(sample_time, signal, 1), sample_time
        )
        spectrum = np.abs(np.fft.rfft(detrended))
        spectrum[0] = 0.0
        frequencies = np.fft.rfftfreq(len(detrended), d=0.05)
        return {
            "steps": capture_steps,
            "reused_search_samples": reused_steps,
            "new_capture_steps": new_capture_steps,
            "tactile_despike_replacements": int(despike_replacements),
            "duration_s": capture_steps
            / (1.0 / (self.backend.physics_steps_per_policy * self.backend.model.opt.timestep)),
            "contact_ratio": float(np.asarray(contact).all(axis=1).mean()),
            "mean_force_n": np.asarray(force).mean(axis=0).tolist(),
            "channel_amplitude_kpa": channel_amplitude.tolist(),
            "maximum_channel_amplitude_kpa": float(
                channel_amplitude.max()
            ),
            "mean_dynamic_pulse_peak_to_peak_kpa": float(
                np.ptp(detrended)
            ),
            "dominant_frequency_hz": float(
                frequencies[int(np.argmax(spectrum))]
            ),
            "raw_min": int(np.asarray(raw).min()),
            "raw_max": int(np.asarray(raw).max()),
        }

    def _retreat(self) -> None:
        self.backend.task_rotation_hold_enabled = False
        self.backend.contact_wrist_stabilization_enabled = bool(
            self.task.get("contact_wrist_stabilization_enabled", False)
        )
        unload_steps = int(
            self.task.get("retreat_pressure_unload_steps", 0)
        )
        has_contact = bool(
            self.backend.last_tactile is not None
            and self.backend.diagnostic_contact().any()
        )
        if (
            unload_steps > 0
            and has_contact
            and self.backend.pressure_command_force_n > 0.0
        ):
            target_total = len(self.diagnostic_finger_names) * float(
                self.task["pressure_target_force_n_per_finger"]
            )
            start_fraction = float(
                np.clip(
                    self.backend.pressure_command_force_n
                    / max(target_total, 1e-9),
                    0.0,
                    1.0,
                )
            )
            for index in range(unload_steps):
                fraction = start_fraction * (
                    1.0 - (index + 1) / (unload_steps + 1)
                )
                self._step(
                    self._phase_action(
                        TaskPhase.RETREAT,
                        pressure_kpa=self.target_pressure_kpa * fraction,
                    ),
                    "retreat_pressure_unload",
                )
        self._step(
            self._phase_action(TaskPhase.RETREAT, pressure_kpa=0.0),
            "retreat_pressure_release",
        )
        self.backend.phase = TaskPhase.RETREAT
        escape_direction = self._thumb_escape_direction()
        escape_steps = int(
            self.task.get("pulse_search_lift_escape_steps", 0)
        )
        soft_lift_steps = int(
            self.task.get("retreat_soft_lift_steps", 0)
        )
        for lift_index in range(
            int(self.task.get("retreat_lift_steps", 14))
        ):
            escape_step = float(
                self.task.get("pulse_search_lift_escape_step_m", 0.0)
            )
            escape_xy = (
                escape_step * escape_direction
                if lift_index < escape_steps
                else np.zeros(2, dtype=float)
            )
            self._step(
                PulseAction(
                    np.asarray(
                        [
                            escape_xy[0],
                            escape_xy[1],
                            (
                                float(
                                    self.task.get(
                                        "retreat_soft_lift_step_m", 0.00025
                                    )
                                )
                                if lift_index < soft_lift_steps
                                else float(
                                    self.task.get(
                                        "retreat_lift_step_m", 0.00075
                                    )
                                )
                            ),
                        ]
                    ),
                    np.zeros(3),
                    target_pressure_kpa=0.0,
                    phase_command=int(TaskPhase.RETREAT),
                ),
                "retreat_lift",
            )
        self.backend.set_contact_wrist_damping(False)
        if bool(self.task.get("retreat_via_precontact", True)):
            self._move_named_pose(
                "precontact_pose_rad", TaskPhase.RETREAT, 1.0
            )
        self._move_safe_waypoints(TaskPhase.RETREAT, reverse=True)

    def prepare_search_contact(self, seed: int = 0) -> None:
        """Run the shared task prefix and stop immediately before search."""

        observation, info = self.env.reset(seed=seed)
        self._record(observation, info, "reset")
        self._form_diagnosis_hand_pose()
        self._move_safe_waypoints(TaskPhase.COARSE_APPROACH)
        if bool(
            self.task.get("m5_cartesian_fine_approach_enabled", False)
        ):
            self._fine_approach_to_precontact_pose()
            self.backend.phase = TaskPhase.PRE_CONTACT
            self.backend.previous_action.fill(0.0)
            self._sync()
        elif bool(self.task.get("m5_joint_waypoint_shadow_enabled", False)):
            self._move_named_pose(
                "precontact_pose_rad",
                TaskPhase.FINE_APPROACH,
                float(self.task.get("m5_joint_waypoint_duration_s", 40.0)),
                m5_joint_waypoint_shadow=True,
            )
            self.backend.phase = TaskPhase.PRE_CONTACT
            self.backend.previous_action.fill(0.0)
            self._sync()
        elif bool(
            self.task.get("m5_joint_waypoint_protected_enabled", False)
        ):
            self._move_named_pose_path_controlled(
                "precontact_pose_rad",
                TaskPhase.FINE_APPROACH,
                float(self.task.get("m5_joint_waypoint_duration_s", 40.0)),
            )
            self.backend.phase = TaskPhase.PRE_CONTACT
            self.backend.previous_action.fill(0.0)
            self._sync()
        else:
            self._move_named_pose(
                "precontact_pose_rad", TaskPhase.PRE_CONTACT, 2.0
            )
        if self.backend.contact_wrist_stabilization_enabled:
            settled_pose = {
                name: float(
                    self.backend.data.qpos[
                        self.backend.model.jnt_qposadr[
                            self.backend.model.joint(name).id
                        ]
                    ]
                )
                for name in self.backend.arm_joint_names
            }
            self.backend.set_contact_wrist_damping(True)
            self.backend.move_to_joint_pose(
                settled_pose,
                duration_s=0.15,
                settle_s=float(self.task.get("precontact_settle_s", 1.2)),
                on_control_step=lambda: self._record_waypoint(
                    "precontact_damping_settle"
                ),
            )
            self._sync()
        self._apply_diagnostic_precontact_offset()
        if bool(self.task.get("contact_task_rotation_hold_enabled", False)):
            settled_probe = dual_probe_frame(
                self.backend.model, self.backend.data
            ).center
            self.backend.task_target_rotation = (
                settled_probe.rotation_world_from_frame.copy()
            )
            self.backend.task_rotation_hold_enabled = True
        self._approach_to_dual_contact()
        self._regulate_pressure()
        if bool(
            self.task.get(
                "pulse_search_free_space_rotation_recenter_enabled", False
            )
        ):
            self._search_free_space_rotation_reference = dual_probe_frame(
                self.backend.model, self.backend.data
            ).center.rotation_world_from_frame.copy()
        if bool(
            self.task.get("pulse_search_task_rotation_hold_enabled", False)
        ):
            settled_probe = dual_probe_frame(
                self.backend.model, self.backend.data
            ).center
            self.backend.task_target_rotation = (
                settled_probe.rotation_world_from_frame.copy()
            )
            self.backend.task_rotation_hold_enabled = True

    def run(self, seed: int = 0) -> ExpertRun:
        failure = None
        best_index = None
        stop_reason = None
        capture = None
        try:
            self.prepare_search_contact(seed=seed)
            if self.search_strategy == "signal_quality_early_stop":
                best_index, stop_reason = self._directional_search(
                    stop_on_signal_quality=True
                )
            else:
                best_index, stop_reason = self._directional_search()
            capture = self._capture(
                self._accepted_quality_window
                if self.search_strategy == "signal_quality_early_stop"
                else None
            )
            self._retreat()
            self.backend.phase = TaskPhase.SUCCESS
            final_observation, final_info = self._sync()
            self._record(final_observation, final_info, "success")
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            self.backend.phase = TaskPhase.FAILURE
            final_observation, final_info = self._sync()
            self._record(final_observation, final_info, "failure")
            try:
                self._retreat()
            except Exception as retreat_error:
                failure += (
                    f"; retreat_failed={type(retreat_error).__name__}: "
                    f"{retreat_error}"
                )
            self.backend.phase = TaskPhase.FAILURE

        selected_position = (
            None
            if best_index is None
            else np.asarray(
                self.search_samples[best_index]["position_m"], dtype=float
            )
        )
        # Evaluation only. This value was not read until all expert decisions
        # and the retreat had completed.
        self.posthoc_privileged_state_reads += 1
        pulse_truth = np.asarray(
            self.env.get_privileged_state()["pulse_center_m"], dtype=float
        )
        selected_sensing_position = (
            None
            if best_index is None
            else np.asarray(
                self.search_samples[best_index][
                    "pulse_weighted_taxel_center_m"
                ],
                dtype=float,
            )
        )
        lateral_error_mm = (
            None
            if selected_sensing_position is None
            else 1000.0
            * abs(
                float(
                    np.dot(
                        selected_sensing_position[:2] - pulse_truth[:2],
                        self.patient_lateral_xy,
                    )
                )
            )
        )
        positions = [
            sample["position_m"] for sample in self.search_samples
        ]
        selected_quality = (
            None
            if best_index is None
            else self.search_samples[best_index]["signal_quality"]
        )
        search_required = len(self.search_samples) > 1
        result = {
            "success": failure is None,
            "failure": failure,
            "simulator": "mujoco",
            "search_strategy": self.search_strategy,
            "signal_quality_schema_version": PULSE_SIGNAL_QUALITY_SCHEMA_VERSION,
            "signal_quality_thresholds": self.quality_thresholds.as_dict(),
            "selected_signal_quality": selected_quality,
            "signal_quality_accepted": bool(
                selected_quality is not None and selected_quality["accepted"]
            ),
            "uses_privileged_pulse_guidance": self.privileged_search_assist,
            "fixed_trajectory_replay": False,
            "control_authority": self.control_authority,
            "policy_mode": (
                PURE_POLICY_MODE
                if self.control_authority
                == CONTROL_AUTHORITY_DEPLOYABLE_POLICY
                else "expert_supervised"
            ),
            "truth_in_observation": False,
            "pressure_target_kpa": float(self.target_pressure_kpa),
            "search_samples": self.search_samples,
            "search_segment_plans": self.search_segment_plans,
            "search_segment_policy_records": self.search_segment_policy_records,
            "search_segment_policy_execution_budget": (
                None
                if self.search_segment_policy_execution_budget is None
                else int(self.search_segment_policy_execution_budget)
            ),
            "search_segment_policy_executions": int(
                self.search_segment_policy_executions
            ),
            "search_segment_branch_ready_point_index": (
                None
                if self.search_segment_branch_ready_point_index is None
                else int(self.search_segment_branch_ready_point_index)
            ),
            "expert_action_queries": int(self.expert_action_queries),
            # This exact field is the deployment/control-path audit.  Posthoc
            # truth reads used only to score a finished simulation are exposed
            # separately and never enter policy observations or decisions.
            "privileged_state_reads": int(self.privileged_state_reads),
            "posthoc_privileged_state_reads": int(
                self.posthoc_privileged_state_reads
            ),
            "expert_direction_queries": int(
                self.search_expert_target_queries
            ),
            "search_expert_target_queries": int(
                self.search_expert_target_queries
            ),
            "search_states": self.search_states,
            "three_stage_search_required": search_required,
            "three_stage_search_success": (
                validate_three_stage_transitions(self.search_states)
                if search_required
                else True
            ),
            "unique_search_positions": (
                unique_position_count(positions, threshold_m=0.00010)
                if positions
                else 0
            ),
            "selected_sample_index": best_index,
            "search_stop_reason": stop_reason or self._last_search_stop_reason,
            "selected_position_m": (
                None
                if selected_position is None
                else selected_position.tolist()
            ),
            "selected_sensing_position_m": (
                None
                if selected_sensing_position is None
                else selected_sensing_position.tolist()
            ),
            "evaluation_lateral_error_mm": lateral_error_mm,
            "capture": capture,
            "events": self.events,
            "recovery_events": self.recovery_events,
            "simulation_time_s": float(self.backend.data.time),
            "final_phase": self.backend.phase.name.lower(),
            "maximum_force_n_per_finger": (
                np.max(
                    np.asarray(self.telemetry["force_n"], dtype=float),
                    axis=0,
                ).tolist()
            ),
            "maximum_channel_pressure_kpa": float(
                np.max(
                    np.asarray(
                        self.telemetry["tactile_kpa"], dtype=float
                    )
                )
            ),
            "forbidden_contact_events": int(
                self.backend.runtime_forbidden_contact_events
            ),
        }
        telemetry = {
            key: np.asarray(value) for key, value in self.telemetry.items()
        }
        return ExpertRun(
            success=failure is None,
            result=result,
            tactile_time_s=telemetry["time_s"],
            tactile_kpa=telemetry["tactile_kpa"],
            tactile_dynamic_kpa=telemetry["tactile_dynamic_kpa"],
            tactile_raw=telemetry["tactile_raw"],
            tactile_raw_delta=telemetry["tactile_raw_delta"],
            force_n=telemetry["force_n"],
            contact=telemetry["contact"],
            probe_position_m=telemetry["probe_position_m"],
            phase=telemetry["phase"],
            tactile_full_56_kpa=telemetry["tactile_full_56_kpa"],
            tactile_full_56_dynamic_kpa=telemetry[
                "tactile_full_56_dynamic_kpa"
            ],
            tactile_full_56_raw=telemetry["tactile_full_56_raw"],
            full_hand_force_n=telemetry["full_hand_force_n"],
            full_hand_contact=telemetry["full_hand_contact"],
        )
