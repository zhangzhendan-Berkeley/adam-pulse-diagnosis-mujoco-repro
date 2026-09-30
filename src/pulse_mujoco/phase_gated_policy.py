from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import PulseAction
from .environment import PulseDiagnosisEnv
from .observation import PulseObservation
from .task import TaskPhase


def observable_diagnostic_contact(
    observation_contact: np.ndarray,
    backend_info: dict,
) -> np.ndarray:
    """Return the deployable contact state for every active diagnostic pad.

    Legacy observations expose the original index/middle pair.  The custom
    hand backend additionally publishes the full-hand tactile contact bits and
    the configured diagnostic-finger names in ``backend_info``.  Selecting
    those bits is sensor-only bookkeeping, not simulator truth, and prevents
    a three-finger task from silently applying a two-finger safety contract.
    """

    fallback = np.asarray(observation_contact, dtype=bool)
    if fallback.ndim != 1 or not 2 <= fallback.size <= 5:
        raise ValueError("observation contact state must contain 2--5 fingers")
    full = backend_info.get("tactile_full_56")
    diagnostic = backend_info.get("diagnostic_finger_names")
    if full is None or diagnostic is None:
        return fallback.copy()
    finger_names = tuple(str(name) for name in full.get("finger_names", ()))
    full_contact = np.asarray(full.get("contact", ()), dtype=bool)
    diagnostic_names = tuple(str(name) for name in diagnostic)
    if (
        full_contact.ndim != 1
        or len(finger_names) != full_contact.size
        or not 2 <= len(diagnostic_names) <= 5
        or any(name not in finger_names for name in diagnostic_names)
    ):
        raise ValueError("invalid observable full-hand contact contract")
    index = {name: position for position, name in enumerate(finger_names)}
    return np.asarray(
        [full_contact[index[name]] for name in diagnostic_names], dtype=bool
    )


def review_search_segment_direction(
    policy_signed_displacement_m: float,
    expert_delta_xy_m: np.ndarray,
    patient_lateral_xy: np.ndarray,
    *,
    minimum_direction_cosine: float = float(np.cos(np.deg2rad(30.0))),
) -> tuple[bool, str | None, float | None]:
    """Review a full model segment without replacing its magnitude.

    Level 1 uses the expert only as a direction safety reviewer.  Once
    accepted, the model's own bounded displacement is executed unchanged.
    """

    signed = float(policy_signed_displacement_m)
    expert = np.asarray(expert_delta_xy_m, dtype=float)
    lateral = np.asarray(patient_lateral_xy, dtype=float)
    if expert.shape != (2,) or lateral.shape != (2,):
        raise ValueError("expert delta and patient lateral axis must have shape (2,)")
    if not np.isfinite(signed) or not np.isfinite(expert).all() or not np.isfinite(
        lateral
    ).all():
        raise ValueError("segment direction review inputs must be finite")
    if not -1.0 <= minimum_direction_cosine <= 1.0:
        raise ValueError("minimum_direction_cosine must be in [-1, 1]")
    lateral_norm = float(np.linalg.norm(lateral))
    expert_norm = float(np.linalg.norm(expert))
    if lateral_norm <= 1e-12:
        raise ValueError("patient lateral axis must be non-zero")
    if abs(signed) <= 1e-12:
        return False, "policy_segment_direction_unavailable", None
    if expert_norm <= 1e-12:
        return False, "expert_segment_direction_unavailable", None
    policy = signed * lateral / lateral_norm
    cosine = float(np.dot(policy, expert) / (np.linalg.norm(policy) * expert_norm))
    if cosine < minimum_direction_cosine:
        return False, "segment_direction_disagrees_with_expert", cosine
    return True, None, cosine


def project_search_translation(
    policy_continuous_action: np.ndarray,
    expert_action: PulseAction,
    *,
    minimum_direction_cosine: float = float(np.cos(np.deg2rad(30.0))),
    policy_direction_blend: float = 0.20,
) -> tuple[np.ndarray | None, str | None]:
    """Build a bounded learned direction correction for lifted search.

    Search motion is scheduled by the expert state machine for a fixed number
    of control ticks.  Replacing an expert tick with a smaller learned step
    therefore under-runs the planned target and accumulates position error.
    This projection keeps the expert's horizontal step length while allowing
    the learned policy to contribute a small, bounded direction correction.
    All vertical, rotational, and pressure channels remain disabled.

    Returns ``(None, reason)`` when the learned direction should not be
    admitted.  The caller can then fall back to the expert action.
    """
    proposal = np.asarray(policy_continuous_action, dtype=np.float32)
    if proposal.shape != (7,):
        raise ValueError(
            "policy continuous action must have shape (7,), got "
            f"{proposal.shape}"
        )
    if not 0.0 <= policy_direction_blend <= 1.0:
        raise ValueError("policy_direction_blend must be in [0, 1]")
    if not -1.0 <= minimum_direction_cosine <= 1.0:
        raise ValueError("minimum_direction_cosine must be in [-1, 1]")

    expert_xy = np.asarray(expert_action.delta_tcp_xyz_m[:2], dtype=np.float32)
    policy_xy = proposal[:2]
    expert_norm = float(np.linalg.norm(expert_xy))
    policy_norm = float(np.linalg.norm(policy_xy))
    if expert_norm <= 1e-9:
        return None, "search_expert_direction_unavailable"
    if policy_norm <= 1e-9:
        return None, "search_policy_direction_unavailable"

    expert_unit = expert_xy / expert_norm
    policy_unit = policy_xy / policy_norm
    cosine = float(np.dot(expert_unit, policy_unit))
    if cosine < minimum_direction_cosine:
        return None, "search_direction_disagrees_with_expert"

    blended_unit = (
        (1.0 - policy_direction_blend) * expert_unit
        + policy_direction_blend * policy_unit
    )
    blended_norm = float(np.linalg.norm(blended_unit))
    if blended_norm <= 1e-9:
        return None, "search_blended_direction_unavailable"

    projected = np.zeros(7, dtype=np.float32)
    projected[:2] = blended_unit / blended_norm * expert_norm
    return projected, None


def project_joint_waypoint_progress(
    policy_continuous_action: np.ndarray,
    expert_action: PulseAction,
    *,
    minimum_direction_cosine: float = float(np.cos(np.deg2rad(30.0))),
    minimum_progress_scale: float = 0.50,
    maximum_progress_scale: float = 1.00,
    translation_scale_m: float = 0.00075,
    rotation_scale_rad: float = 0.02,
    verified_path_owns_direction: bool = False,
) -> tuple[np.ndarray | None, str | None, float | None]:
    """Project a policy proposal onto a proven joint-waypoint tangent.

    The first protected M5 gate may change only progress along the verified
    collision-free joint path. Translation and rotation are normalized by
    their unchanged safety limits before direction and magnitude are
    compared. The policy can slow the path but cannot speed it up or move it
    laterally; pressure and discrete commands remain executor owned.

    ``verified_path_owns_direction`` is used by the slow joint-space M5
    executor. Its measured TCP increments alternate sign because the joint PD
    loop rings around each slowly changing target. The model is trained to
    predict that *next physical increment*, while the collision-free joint
    path already fixes the commanded direction. Comparing the prediction to
    either the geometric tangent or the previous physical increment therefore
    rejects a correct temporal prediction. In this mode the proposal is an
    authorization signal: it must remain inside the unchanged per-channel
    action envelope, while the verified path supplies direction and a
    deterministic unit progress tick.
    """
    proposal = np.asarray(policy_continuous_action, dtype=np.float64)
    expert = expert_action.as_array().astype(np.float64)
    if proposal.shape != (7,) or not np.isfinite(proposal).all():
        return None, "joint_waypoint_policy_action_invalid", None
    if not -1.0 <= minimum_direction_cosine <= 1.0:
        raise ValueError("minimum_direction_cosine must be in [-1, 1]")
    if not 0.0 < minimum_progress_scale <= maximum_progress_scale <= 1.0:
        raise ValueError("joint waypoint progress scales must satisfy 0 < min <= max <= 1")
    if translation_scale_m <= 0.0 or rotation_scale_rad <= 0.0:
        raise ValueError("joint waypoint normalization scales must be positive")

    scale = np.asarray(
        [translation_scale_m] * 3 + [rotation_scale_rad] * 3,
        dtype=np.float64,
    )
    policy_normalized = proposal[:6] / scale
    if verified_path_owns_direction:
        if np.any(np.abs(policy_normalized) > 1.0 + 1e-6):
            return None, "joint_waypoint_policy_action_outside_envelope", None
        if float(np.linalg.norm(policy_normalized)) <= 1e-9:
            return None, "joint_waypoint_policy_progress_unavailable", None
        projected = proposal.copy()
        projected[6] = 0.0
        return projected.astype(np.float32), None, maximum_progress_scale

    expert_normalized = expert[:6] / scale
    policy_norm = float(np.linalg.norm(policy_normalized))
    expert_norm = float(np.linalg.norm(expert_normalized))
    if expert_norm <= 1e-9:
        return None, "joint_waypoint_expert_progress_unavailable", None
    if policy_norm <= 1e-9:
        return None, "joint_waypoint_policy_progress_unavailable", None
    cosine = float(
        np.dot(policy_normalized, expert_normalized)
        / (policy_norm * expert_norm)
    )
    if cosine < minimum_direction_cosine:
        return None, "joint_waypoint_direction_disagrees_with_expert", None

    raw_progress = float(
        np.dot(policy_normalized, expert_normalized)
        / np.dot(expert_normalized, expert_normalized)
    )
    progress = float(
        np.clip(raw_progress, minimum_progress_scale, maximum_progress_scale)
    )
    projected = expert.copy()
    projected[:6] *= progress
    projected[6] = expert[6]
    return projected.astype(np.float32), None, progress


def project_contact_establishment_action(
    policy_continuous_action: np.ndarray,
    contact: np.ndarray,
    *,
    maximum_approach_step_m: float = 0.00020,
    maximum_single_contact_step_m: float = 0.00005,
    maximum_balance_rotation_step_rad: float = 0.0008,
) -> tuple[np.ndarray | None, str | None]:
    """Project a proposal into the observable dual-contact safety envelope.

    Before first contact, only a bounded vertical descent is meaningful.  Once
    one pad touches, the remaining pad may be seated with a smaller descent
    and bounded posture correction.  Horizontal motion and pressure remain
    disabled throughout contact establishment.  This projection uses only
    current sensor contact state; it neither consults an expert action nor
    pulse-location truth.
    """

    proposal = np.asarray(policy_continuous_action, dtype=np.float64)
    contacts = np.asarray(contact, dtype=bool)
    if proposal.shape != (7,) or not np.isfinite(proposal).all():
        return None, "contact_policy_action_invalid"
    if contacts.ndim != 1 or not 2 <= contacts.size <= 5:
        raise ValueError("contact state must contain 2--5 fingers")
    if maximum_approach_step_m <= 0.0 or maximum_single_contact_step_m <= 0.0:
        raise ValueError("contact descent limits must be positive")
    if maximum_balance_rotation_step_rad <= 0.0:
        raise ValueError("contact rotation limit must be positive")

    projected = np.zeros(7, dtype=np.float64)
    contact_count = int(np.count_nonzero(contacts))
    if contact_count == contacts.size:
        return projected.astype(np.float32), None
    if proposal[2] < -1e-9:
        descent_limit = (
            maximum_approach_step_m
            if contact_count == 0
            else maximum_single_contact_step_m
        )
        projected[2] = float(np.clip(proposal[2], -descent_limit, 0.0))
    # A non-descending z proposal is held at zero.  Once exactly one pad is
    # in contact, however, its bounded posture correction remains useful for
    # seating the second pad and must not be discarded with the z component.
    # This is still a projection of the model proposal; no expert direction
    # or simulator truth is injected.
    if 0 < contact_count < contacts.size:
        rotation = proposal[3:6]
        rotation_norm = float(np.linalg.norm(rotation))
        if rotation_norm > maximum_balance_rotation_step_rad:
            rotation = (
                rotation
                * maximum_balance_rotation_step_rad
                / rotation_norm
            )
        projected[3:6] = rotation
    return projected.astype(np.float32), None


def project_pressure_regulation_action(
    policy_continuous_action: np.ndarray,
    contact: np.ndarray,
    previous_action: np.ndarray,
    *,
    maximum_pressure_kpa: float = 127.5,
    maximum_increase_kpa_per_step: float = 4.0,
    maximum_decrease_kpa_per_step: float = 8.0,
) -> tuple[np.ndarray | None, str | None]:
    """Lock pose and rate-limit an observable pressure target.

    The learned specialist owns only the absolute pressure channel. Cartesian
    motion, wrist rotation and phase transitions remain deterministic. The
    limiter uses the last executed command and current contact state. A
    single-contact transient holds rather than increases load; neither expert
    action nor pulse-center truth enters this projection.
    """
    proposal = np.asarray(policy_continuous_action, dtype=np.float64)
    contacts = np.asarray(contact, dtype=bool)
    previous = np.asarray(previous_action, dtype=np.float64)
    if proposal.shape != (7,) or not np.isfinite(proposal).all():
        return None, "pressure_policy_action_invalid"
    if previous.shape != (7,) or not np.isfinite(previous).all():
        return None, "pressure_previous_action_invalid"
    if contacts.ndim != 1 or not 2 <= contacts.size <= 5:
        raise ValueError("contact state must contain 2--5 fingers")
    if maximum_pressure_kpa <= 0.0:
        raise ValueError("maximum pressure must be positive")
    if maximum_increase_kpa_per_step <= 0.0:
        raise ValueError("pressure increase rate must be positive")
    if maximum_decrease_kpa_per_step <= 0.0:
        raise ValueError("pressure decrease rate must be positive")

    desired = float(np.clip(proposal[6], 0.0, maximum_pressure_kpa))
    prior = float(np.clip(previous[6], 0.0, maximum_pressure_kpa))
    contact_count = int(np.count_nonzero(contacts))
    if contact_count == 0:
        # Preserve the latched contact target through a brief sensor blink.
        # With no measured contact this applies no physical load; clearing the
        # command here would release the backend latch and make recovery
        # impossible without silently returning control to the expert.
        projected = np.zeros(7, dtype=np.float32)
        projected[6] = prior
        return projected, "pressure_no_contact_hold"
    if contact_count < contacts.size:
        # A single pad can momentarily blink at the force threshold during
        # compliance settling. Hold, rather than increase, the last safe
        # pressure until both pads are observable again.
        projected = np.zeros(7, dtype=np.float32)
        projected[6] = prior
        return projected, "pressure_single_contact_hold"
    target = prior + float(
        np.clip(
            desired - prior,
            -maximum_decrease_kpa_per_step,
            maximum_increase_kpa_per_step,
        )
    )
    projected = np.zeros(7, dtype=np.float32)
    projected[6] = target
    return projected, None


def pressure_regulation_effective_uncertainty(
    projected_action: np.ndarray,
    previous_action: np.ndarray,
    *,
    maximum_pressure_kpa: float = 127.5,
) -> float:
    """Measure uncertainty only over the pressure action that can execute.

    Pressure regulation hard-locks the six Cartesian pose channels to zero.
    Their raw temporal variation remains useful diagnostic evidence, but it
    cannot change the executed command and therefore must not trigger a false
    expert takeover.  The remaining pressure change is normalized by the
    observable sensor range, matching the normalized temporal uncertainty used
    by the direct-policy runtime.
    """
    projected = np.asarray(projected_action, dtype=np.float64)
    previous = np.asarray(previous_action, dtype=np.float64)
    if projected.shape != (7,) or not np.isfinite(projected).all():
        raise ValueError("projected pressure action must be finite shape (7,)")
    if previous.shape != (7,) or not np.isfinite(previous).all():
        raise ValueError("previous pressure action must be finite shape (7,)")
    if maximum_pressure_kpa <= 0.0:
        raise ValueError("maximum pressure must be positive")
    return abs(float(projected[6]) - float(previous[6])) / float(
        maximum_pressure_kpa
    )


def project_stable_capture_action(
    policy_continuous_action: np.ndarray,
    contact: np.ndarray,
    previous_action: np.ndarray,
    *,
    maximum_pressure_kpa: float = 127.5,
) -> tuple[np.ndarray | None, str | None]:
    """Project capture control to a motionless, constant-pressure hold.

    Signal capture starts only after the observable signal-quality gate has
    accepted a dual-contact window. The model may propose an action, but the
    deployable capture contract permits no pose motion or pressure drift. A
    contact loss therefore fails closed instead of being hidden by the hold.
    """
    proposal = np.asarray(policy_continuous_action, dtype=np.float64)
    contacts = np.asarray(contact, dtype=bool)
    previous = np.asarray(previous_action, dtype=np.float64)
    if proposal.shape != (7,) or not np.isfinite(proposal).all():
        return None, "stable_capture_policy_action_invalid"
    if previous.shape != (7,) or not np.isfinite(previous).all():
        return None, "stable_capture_previous_action_invalid"
    if contacts.ndim != 1 or not 2 <= contacts.size <= 5:
        raise ValueError("contact state must contain 2--5 fingers")
    if maximum_pressure_kpa <= 0.0:
        raise ValueError("maximum pressure must be positive")
    if not bool(np.all(contacts)):
        return None, "stable_capture_dual_contact_required"
    prior = float(previous[6])
    if not 0.0 < prior <= maximum_pressure_kpa:
        return None, "stable_capture_pressure_hold_invalid"
    projected = np.zeros(7, dtype=np.float32)
    projected[6] = prior
    return projected, None


@dataclass(frozen=True)
class PhaseGatedPolicyConfig:
    """Admission limits for the first learned-policy closed-loop stage."""

    allowed_phases: tuple[TaskPhase, ...] = (TaskPhase.FINE_APPROACH,)
    maximum_uncertainty: float = 0.22
    require_vision: bool = True
    takeover_on_safety_modification: bool = True
    fine_approach_requires_no_contact: bool = True
    no_contact_required_phases: tuple[TaskPhase, ...] = (
        TaskPhase.FINE_APPROACH,
    )

    def __post_init__(self) -> None:
        phases = tuple(TaskPhase(int(phase)) for phase in self.allowed_phases)
        if not phases:
            raise ValueError("allowed_phases must not be empty")
        if not np.isfinite(self.maximum_uncertainty):
            raise ValueError("maximum_uncertainty must be finite")
        if self.maximum_uncertainty < 0.0:
            raise ValueError("maximum_uncertainty must be non-negative")
        object.__setattr__(self, "allowed_phases", phases)
        no_contact_phases = tuple(
            TaskPhase(int(phase)) for phase in self.no_contact_required_phases
        )
        object.__setattr__(self, "no_contact_required_phases", no_contact_phases)


@dataclass(frozen=True)
class PolicyControlDecision:
    """Auditable selection between a learned continuous action and expert."""

    action: PulseAction
    policy_proposed_action: np.ndarray | None
    source: str
    expert_intervened: bool
    intervention_reasons: tuple[str, ...]
    task_phase: TaskPhase
    policy_uncertainty: float | None
    policy_safety_modified: bool
    policy_safety_reasons: tuple[str, ...]

    @property
    def policy_action_executed(self) -> bool:
        return self.source == "policy"

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "policy_action_executed": self.policy_action_executed,
            "expert_intervened": self.expert_intervened,
            "intervention_reasons": list(self.intervention_reasons),
            "task_phase": self.task_phase.name.lower(),
            "policy_uncertainty": self.policy_uncertainty,
            "policy_proposed_action": (
                None
                if self.policy_proposed_action is None
                else self.policy_proposed_action.tolist()
            ),
            "selected_action": self.action.as_array().tolist(),
            "policy_safety_modified": self.policy_safety_modified,
            "policy_safety_reasons": list(self.policy_safety_reasons),
        }


class PhaseGatedPolicyController:
    """Use a learned action only inside a bounded, expert-owned phase.

    The World-VLA direct head controls seven continuous channels.  Hand shape
    and state-machine transitions are kept from the expert command so a model
    cannot bypass the diagnosis-pose or contact safety logic.
    """

    def __init__(self, config: PhaseGatedPolicyConfig | None = None):
        self.config = config or PhaseGatedPolicyConfig()
        self.policy_steps = 0
        self.expert_steps = 0
        self.policy_steps_by_phase: dict[str, int] = {}
        self.expert_steps_by_phase: dict[str, int] = {}
        self.intervention_counts: dict[str, int] = {}
        self.intervention_counts_by_phase: dict[str, dict[str, int]] = {}

    def reset(self) -> None:
        self.policy_steps = 0
        self.expert_steps = 0
        self.policy_steps_by_phase.clear()
        self.expert_steps_by_phase.clear()
        self.intervention_counts.clear()
        self.intervention_counts_by_phase.clear()

    def record_expert_intervention(
        self, phase: TaskPhase | int, reasons: tuple[str, ...] | list[str]
    ) -> None:
        phase_name = TaskPhase(int(phase)).name.lower()
        unique_reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
        self.expert_steps += 1
        self.expert_steps_by_phase[phase_name] = (
            self.expert_steps_by_phase.get(phase_name, 0) + 1
        )
        phase_interventions = self.intervention_counts_by_phase.setdefault(
            phase_name, {}
        )
        for reason in unique_reasons:
            self.intervention_counts[reason] = (
                self.intervention_counts.get(reason, 0) + 1
            )
            phase_interventions[reason] = (
                phase_interventions.get(reason, 0) + 1
            )

    @staticmethod
    def _policy_action(
        continuous_action: np.ndarray, expert_action: PulseAction
    ) -> PulseAction:
        value = np.asarray(continuous_action, dtype=np.float32)
        if value.shape != (7,):
            raise ValueError(
                "policy continuous action must have shape (7,), got "
                f"{value.shape}"
            )
        return PulseAction(
            value[:3],
            value[3:6],
            target_pressure_kpa=float(value[6]),
            hand_mode=expert_action.hand_mode,
            phase_command=expert_action.phase_command,
        )

    def decide(
        self,
        *,
        observation: PulseObservation,
        backend_info: dict,
        expert_action: PulseAction,
        policy_continuous_action: np.ndarray | None,
        policy_uncertainty: float | None,
        env: PulseDiagnosisEnv,
    ) -> PolicyControlDecision:
        phase = TaskPhase(int(observation.task_phase))
        reasons: list[str] = []
        policy_safe_modified = False
        policy_safety_reasons: tuple[str, ...] = ()
        policy_action = None

        if phase not in self.config.allowed_phases:
            reasons.append("phase_not_enabled")
        if policy_continuous_action is None:
            reasons.append("policy_not_ready")
        if self.config.require_vision and (
            observation.rgb is None or observation.depth_m is None
        ):
            reasons.append("vision_unavailable")
        if policy_uncertainty is None or not np.isfinite(policy_uncertainty):
            reasons.append("uncertainty_unavailable")
        elif policy_uncertainty > self.config.maximum_uncertainty:
            reasons.append("uncertainty_above_threshold")
        no_contact_required = bool(
            self.config.fine_approach_requires_no_contact
            and phase in self.config.no_contact_required_phases
        )
        if no_contact_required and bool(np.any(observation.contact)):
            reasons.append("contact_requires_expert")

        if policy_continuous_action is not None:
            policy_action = self._policy_action(
                policy_continuous_action, expert_action
            )
            safe_preview, preview = env.safety.filter_action(
                observation, policy_action, backend_info
            )
            policy_safe_modified = bool(preview["modified"])
            policy_safety_reasons = tuple(preview["reasons"])
            if (
                self.config.takeover_on_safety_modification
                and policy_safe_modified
            ):
                reasons.append("policy_safety_modified")
            else:
                policy_action = safe_preview

        unique_reasons = tuple(dict.fromkeys(reasons))
        phase_name = phase.name.lower()
        if unique_reasons:
            selected = expert_action
            source = "expert"
            self.record_expert_intervention(phase, list(unique_reasons))
        else:
            if policy_action is None:
                raise RuntimeError("admitted policy decision has no action")
            selected = policy_action
            source = "policy"
            self.policy_steps += 1
            self.policy_steps_by_phase[phase_name] = (
                self.policy_steps_by_phase.get(phase_name, 0) + 1
            )

        return PolicyControlDecision(
            action=selected,
            policy_proposed_action=(
                None
                if policy_continuous_action is None
                else np.asarray(policy_continuous_action, dtype=np.float32).copy()
            ),
            source=source,
            expert_intervened=source == "expert",
            intervention_reasons=unique_reasons,
            task_phase=phase,
            policy_uncertainty=(
                None
                if policy_uncertainty is None
                else float(policy_uncertainty)
            ),
            policy_safety_modified=policy_safe_modified,
            policy_safety_reasons=policy_safety_reasons,
        )

    def step(
        self,
        *,
        env: PulseDiagnosisEnv,
        expert_action: PulseAction,
        policy_continuous_action: np.ndarray | None,
        policy_uncertainty: float | None,
    ) -> tuple[dict, float, bool, bool, dict]:
        if env.observation is None:
            raise RuntimeError("environment must be reset before policy step")
        decision = self.decide(
            observation=env.observation,
            backend_info=env.backend_info,
            expert_action=expert_action,
            policy_continuous_action=policy_continuous_action,
            policy_uncertainty=policy_uncertainty,
            env=env,
        )
        observation, reward, terminated, truncated, info = env.step(
            decision.action
        )
        info["policy_control"] = decision.as_dict()
        info["policy_uncertainty"] = decision.policy_uncertainty
        info["expert_intervened"] = decision.expert_intervened
        info["policy_action_executed"] = decision.policy_action_executed
        return observation, reward, terminated, truncated, info

    def summary(self) -> dict:
        total = self.policy_steps + self.expert_steps
        audited_phase_names = sorted(
            {
                *(phase.name.lower() for phase in self.config.allowed_phases),
                *self.policy_steps_by_phase,
                *self.expert_steps_by_phase,
            }
        )
        return {
            "policy_steps": self.policy_steps,
            "expert_steps": self.expert_steps,
            "total_steps": total,
            "policy_execution_fraction": (
                float(self.policy_steps / total) if total else None
            ),
            "expert_intervention_fraction": (
                float(self.expert_steps / total) if total else None
            ),
            # Emit explicit zeros for every enabled phase.  A downstream
            # fail-closed audit must be able to distinguish a verified zero
            # from an omitted counter.
            "policy_steps_by_phase": {
                phase: int(self.policy_steps_by_phase.get(phase, 0))
                for phase in audited_phase_names
            },
            "expert_steps_by_phase": {
                phase: int(self.expert_steps_by_phase.get(phase, 0))
                for phase in audited_phase_names
            },
            "intervention_counts": dict(self.intervention_counts),
            "intervention_counts_by_phase": {
                phase: dict(counts)
                for phase, counts in self.intervention_counts_by_phase.items()
            },
            "allowed_phases": [
                phase.name.lower() for phase in self.config.allowed_phases
            ],
            "maximum_uncertainty": self.config.maximum_uncertainty,
            "no_contact_required_phases": [
                phase.name.lower()
                for phase in self.config.no_contact_required_phases
            ],
        }
