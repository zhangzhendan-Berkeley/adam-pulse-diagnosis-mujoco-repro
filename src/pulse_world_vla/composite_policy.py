from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np

from pulse_mujoco.actions import PulseAction
from pulse_mujoco.environment import PulseDiagnosisEnv
from pulse_mujoco.frames import dual_probe_frame
from pulse_mujoco.m5_transition_overlay import project_m5_action_to_safety
from pulse_mujoco.observation import PulseObservation
from pulse_mujoco.phase_gated_policy import (
    observable_diagnostic_contact,
    pressure_regulation_effective_uncertainty,
    project_contact_establishment_action,
    project_joint_waypoint_progress,
    project_pressure_regulation_action,
    project_stable_capture_action,
)
from pulse_mujoco.task import TaskPhase

from .direct_policy import DirectPolicyRuntime
from .contact_balance import observable_force_balance_rotation


M5_SPECIALIST_SCOPES = (
    "fine_approach",
    "contact_establishment",
    "pressure_regulation",
    "stable_capture",
)

OPTIONAL_SPECIALIST_SCOPES = ("search_contact_recovery",)


@dataclass(frozen=True)
class LayeredSearchRecoveryState:
    preload_latched: bool = False
    post_latch_reacquire_active: bool = False
    reacquire_step_count: int = 0
    reacquire_rotation_budget_used_rad: float = 0.0


def layered_search_recovery_projection(
    projected_action: np.ndarray,
    contact: np.ndarray,
    force_n: np.ndarray,
    *,
    state: LayeredSearchRecoveryState,
    maximum_preload_step_m: float,
    maximum_reacquire_descent_step_m: float,
    minimum_seated_force_n: float,
    maximum_reacquire_steps: int,
    maximum_reacquire_rotation_budget_rad: float,
) -> tuple[np.ndarray, LayeredSearchRecoveryState, str]:
    """Project search recovery through a bounded observable contact state."""

    action = np.asarray(projected_action, dtype=np.float32)
    mask = np.asarray(contact, dtype=bool)
    force = np.asarray(force_n, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("search recovery action must be finite 7-D")
    if mask.ndim != 1 or not 2 <= mask.size <= 5 or force.shape != mask.shape:
        raise ValueError("search recovery contact and force shapes must match")
    if not np.isfinite(force).all():
        raise ValueError("search recovery force must be finite")
    if (
        maximum_preload_step_m <= 0.0
        or maximum_reacquire_descent_step_m <= 0.0
        or maximum_reacquire_descent_step_m > maximum_preload_step_m
        or minimum_seated_force_n < 0.0
        or maximum_reacquire_steps <= 0
        or maximum_reacquire_rotation_budget_rad <= 0.0
    ):
        raise ValueError("search recovery preload bounds must be valid")
    if (
        state.reacquire_step_count < 0
        or state.reacquire_rotation_budget_used_rad < 0.0
        or not np.isfinite(state.reacquire_rotation_budget_used_rad)
    ):
        raise ValueError("search recovery state must be finite and non-negative")

    latched = bool(state.preload_latched or mask.all())
    if not latched:
        return action.copy(), state, f"acquire_{int(mask.sum())}_of_3"

    if not mask.all():
        step_count = state.reacquire_step_count + 1
        remaining_rotation = max(
            0.0,
            float(maximum_reacquire_rotation_budget_rad)
            - state.reacquire_rotation_budget_used_rad,
        )
        requested_rotation = action[3:6].astype(np.float64)
        requested_norm = float(np.linalg.norm(requested_rotation))
        rotation = np.zeros(3, dtype=np.float32)
        budget_available = (
            step_count <= int(maximum_reacquire_steps)
            and remaining_rotation > 1e-12
        )
        if requested_norm > 1e-12 and budget_available:
            applied_norm = min(requested_norm, remaining_rotation)
            rotation = np.asarray(
                requested_rotation * (applied_norm / requested_norm),
                dtype=np.float32,
            )
        else:
            applied_norm = 0.0
        result = np.zeros(7, dtype=np.float32)
        # A two-of-three mask is already geometrically constrained enough to
        # use the validated 50 um preload stroke. Zero/one contact retains the
        # gentler 25 um reacquire stroke while observable balance acts.
        reacquire_descent_step = (
            maximum_preload_step_m
            if int(mask.sum()) >= 2
            else maximum_reacquire_descent_step_m
        )
        result[2] = -float(reacquire_descent_step)
        result[3:6] = rotation
        updated = LayeredSearchRecoveryState(
            preload_latched=True,
            post_latch_reacquire_active=True,
            reacquire_step_count=step_count,
            reacquire_rotation_budget_used_rad=(
                state.reacquire_rotation_budget_used_rad + applied_norm
            ),
        )
        layer = (
            "post_latch_reacquire"
            if budget_available
            else "post_latch_reacquire_budget_exhausted"
        )
        return result, updated, layer

    result = np.zeros(7, dtype=np.float32)
    updated = LayeredSearchRecoveryState(
        preload_latched=True,
        post_latch_reacquire_active=False,
        reacquire_step_count=state.reacquire_step_count,
        reacquire_rotation_budget_used_rad=(
            state.reacquire_rotation_budget_used_rad
        ),
    )
    if not np.all(force >= float(minimum_seated_force_n)):
        result[2] = -float(maximum_preload_step_m)
        return result, updated, "preload_submillimeter"
    return result, updated, "seated_hold"


def deployable_specialist_scope(
    label: str,
    phase: TaskPhase | int,
    *,
    search_contact_recovery_enabled: bool = False,
) -> str | None:
    """Route only labels and phases covered by a frozen M5 specialist."""

    current = TaskPhase(int(phase))
    if label == "m5_joint_waypoint_protected" and current == TaskPhase.FINE_APPROACH:
        return "fine_approach"
    if (
        label in {"contact_approach", "single_finger_balance"}
        and current == TaskPhase.CONTACT_ESTABLISHMENT
    ):
        return "contact_establishment"
    if label == "pressure_regulation" and current == TaskPhase.PRESSURE_REGULATION:
        return "pressure_regulation"
    if label == "signal_capture" and current == TaskPhase.SIGNAL_CAPTURE:
        return "stable_capture"
    if (
        search_contact_recovery_enabled
        and label == "single_finger_balance"
        and current == TaskPhase.PULSE_SEARCH
    ):
        return "search_contact_recovery"
    return None


def _observation(value: object) -> PulseObservation:
    if isinstance(value, PulseObservation):
        return value
    item = value if isinstance(value, dict) else value.policy_dict()
    return PulseObservation(
        timestamp_s=float(item["timestamp_s"]),
        rgb=item.get("rgb"),
        depth_m=item.get("depth_m"),
        joint_position_rad=item["joint_position_rad"],
        joint_velocity_rad_s=item["joint_velocity_rad_s"],
        tcp_pose=item["tcp_pose"],
        tcp_velocity=item["tcp_velocity"],
        proximity_m=item["proximity_m"],
        proximity_valid=item["proximity_valid"],
        tactile_kpa=item["tactile_kpa"],
        tactile_raw=item["tactile_raw"],
        tactile_raw_delta=item["tactile_raw_delta"],
        contact=item["contact"],
        task_phase=int(item["task_phase"]),
        previous_action=item["previous_action"],
    )


class CompositePolicyActionSelector:
    """Expert-free action selector for the four frozen M5 specialists.

    Unknown labels stay with the deterministic task/safety executor. A known
    specialist label must execute a valid model projection or fail closed; it
    never falls back to an expert action. The input action contributes only the
    executor-owned discrete hand/phase command and is not used as a continuous
    target or supervisory label.
    """

    consumes_expert_action = False

    def __init__(
        self,
        *,
        env: PulseDiagnosisEnv,
        checkpoints: Mapping[str, Path | str] | None = None,
        runtimes: Mapping[str, DirectPolicyRuntime] | None = None,
        device: str = "cuda",
        maximum_uncertainty: float = 0.22,
        search_contact_recovery_maximum_balance_rotation_step_rad: float = 4e-5,
        search_contact_recovery_maximum_single_contact_step_m: float = 5e-5,
        search_contact_recovery_observable_balance: bool = False,
        search_contact_recovery_minimum_seated_force_n: float = 0.10,
        search_contact_recovery_maximum_reacquire_steps: int = 40,
        search_contact_recovery_maximum_rotation_budget_rad: float = 1.6e-3,
    ) -> None:
        if (checkpoints is None) == (runtimes is None):
            raise ValueError("provide exactly one of checkpoints or runtimes")
        runtime_map = (
            {
                scope: DirectPolicyRuntime(
                    Path(path), device=device, control_substeps=1
                )
                for scope, path in dict(checkpoints or {}).items()
            }
            if runtimes is None
            else dict(runtimes)
        )
        required_scopes = set(M5_SPECIALIST_SCOPES)
        allowed_scopes = required_scopes | set(OPTIONAL_SPECIALIST_SCOPES)
        missing = sorted(required_scopes - set(runtime_map))
        extra = sorted(set(runtime_map) - allowed_scopes)
        if missing or extra:
            raise ValueError(
                "composite runtime scope mismatch: "
                f"missing={missing}, extra={extra}"
            )
        if maximum_uncertainty < 0.0 or not np.isfinite(maximum_uncertainty):
            raise ValueError("maximum_uncertainty must be finite and non-negative")
        if (
            search_contact_recovery_maximum_balance_rotation_step_rad <= 0.0
            or not np.isfinite(
                search_contact_recovery_maximum_balance_rotation_step_rad
            )
        ):
            raise ValueError(
                "search recovery rotation limit must be finite and positive"
            )
        if (
            search_contact_recovery_maximum_single_contact_step_m <= 0.0
            or not np.isfinite(
                search_contact_recovery_maximum_single_contact_step_m
            )
        ):
            raise ValueError(
                "search recovery partial-contact descent limit must be "
                "finite and positive"
            )
        self.env = env
        self.runtimes = runtime_map
        self.search_contact_recovery_enabled = (
            "search_contact_recovery" in runtime_map
        )
        self.maximum_uncertainty = float(maximum_uncertainty)
        self.search_contact_recovery_maximum_balance_rotation_step_rad = float(
            search_contact_recovery_maximum_balance_rotation_step_rad
        )
        self.search_contact_recovery_maximum_single_contact_step_m = float(
            search_contact_recovery_maximum_single_contact_step_m
        )
        self.search_contact_recovery_observable_balance = bool(
            search_contact_recovery_observable_balance
        )
        if search_contact_recovery_minimum_seated_force_n < 0.0:
            raise ValueError("search recovery seated force must be non-negative")
        self.search_contact_recovery_minimum_seated_force_n = float(
            search_contact_recovery_minimum_seated_force_n
        )
        if search_contact_recovery_maximum_reacquire_steps <= 0:
            raise ValueError("search recovery reacquire steps must be positive")
        if search_contact_recovery_maximum_rotation_budget_rad <= 0.0:
            raise ValueError("search recovery rotation budget must be positive")
        self.search_contact_recovery_maximum_reacquire_steps = int(
            search_contact_recovery_maximum_reacquire_steps
        )
        self.search_contact_recovery_maximum_rotation_budget_rad = float(
            search_contact_recovery_maximum_rotation_budget_rad
        )
        self.search_contact_recovery_state = LayeredSearchRecoveryState()
        self.frame = 0
        self.policy_steps = 0
        self.deterministic_steps = 0
        self.safety_modifications = 0
        self.policy_steps_by_scope: Counter[str] = Counter()
        self.deterministic_steps_by_label: Counter[str] = Counter()

    @property
    def observation_horizon(self) -> int:
        return int(self.runtimes["fine_approach"].observation_horizon)

    def reset_histories(self) -> None:
        """Clear episode-local temporal state without reloading checkpoints."""

        for runtime in {id(item): item for item in self.runtimes.values()}.values():
            runtime.reset()
        self.frame = 0
        self.search_contact_recovery_state = LayeredSearchRecoveryState()

    def prime_observation(self, observation: object) -> None:
        item = _observation(observation)
        primed = replace(
            item,
            previous_action=project_m5_action_to_safety(item.previous_action),
        )
        for runtime in {id(item): item for item in self.runtimes.values()}.values():
            runtime.append_observation(primed)

    @staticmethod
    def _action_from_projection(
        projected: np.ndarray,
        deterministic_action: PulseAction,
    ) -> PulseAction:
        value = np.asarray(projected, dtype=np.float32)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise RuntimeError("composite policy projection is not finite 7-D")
        return PulseAction(
            value[:3],
            value[3:6],
            target_pressure_kpa=float(value[6]),
            hand_mode=deterministic_action.hand_mode,
            phase_command=deterministic_action.phase_command,
        )

    def _deterministic(
        self,
        action: PulseAction,
        label: str,
    ) -> tuple[PulseAction, dict]:
        self.deterministic_steps += 1
        self.deterministic_steps_by_label[str(label)] += 1
        return action, {
            "source": "deterministic_safety",
            "policy_action_executed": False,
            "expert_intervened": False,
            "expert_action_consumed": False,
            "intervention_reasons": [],
            "deterministic_label": str(label),
            "selected_action": action.as_array().tolist(),
        }

    def __call__(
        self,
        observation: object,
        backend_info: dict,
        deterministic_action: PulseAction,
        label: str,
    ) -> tuple[PulseAction, dict]:
        item = _observation(observation)
        self.frame += 1
        scope = deployable_specialist_scope(
            label,
            item.task_phase,
            search_contact_recovery_enabled=(
                self.search_contact_recovery_enabled
            ),
        )
        if scope != "search_contact_recovery":
            self.search_contact_recovery_state = LayeredSearchRecoveryState()
        runtime_observation = (
            replace(
                item,
                previous_action=project_m5_action_to_safety(
                    item.previous_action
                ),
            )
            if scope == "fine_approach"
            else item
        )
        for runtime in {id(item): item for item in self.runtimes.values()}.values():
            runtime.append_observation(
                runtime_observation, backend_info=backend_info
            )

        if scope is None:
            return self._deterministic(deterministic_action, label)
        runtime = self.runtimes[scope]
        if not runtime.ready:
            raise RuntimeError(f"{scope} policy history is not ready")
        prediction = runtime.predict_shadow(
            frame=self.frame,
            timestamp_s=item.timestamp_s,
            task_phase=int(item.task_phase),
            safety_observation=item,
            safety_supervisor=self.env.safety,
            backend_info=backend_info,
        )
        raw = prediction.control_tick_action.copy()
        progress_scale = None
        projection_reason = None
        if scope == "fine_approach":
            projected, projection_reason, progress_scale = (
                project_joint_waypoint_progress(
                    raw,
                    PulseAction.from_array(
                        project_m5_action_to_safety(item.previous_action)
                    ),
                    verified_path_owns_direction=True,
                )
            )
            uncertainty = prediction.policy_uncertainty
        elif scope == "contact_establishment":
            contact_state = observable_diagnostic_contact(
                item.contact, backend_info
            )
            projected, projection_reason = project_contact_establishment_action(
                raw,
                contact_state,
            )
            uncertainty = prediction.policy_uncertainty
        elif scope == "search_contact_recovery":
            contact_state = observable_diagnostic_contact(
                item.contact, backend_info
            )
            observable_balance_rotation = None
            if self.search_contact_recovery_observable_balance:
                names = tuple(
                    str(name)
                    for name in backend_info.get(
                        "diagnostic_finger_names", ()
                    )
                )
                if names != ("index", "middle", "ring"):
                    raise RuntimeError(
                        "observable search balance requires index/middle/ring"
                    )
                full_tactile = backend_info.get("tactile_full_56", {})
                full_force = np.asarray(
                    full_tactile.get("total_normal_force_n", ()), dtype=float
                )
                if full_force.shape != (5,):
                    raise RuntimeError(
                        "observable search balance requires five finger forces"
                    )
                centers = np.asarray(
                    [
                        self.env.backend.data.site_xpos[
                            self.env.backend.model.site(
                                f"handle_l1_{finger}_tactile_site"
                            ).id
                        ]
                        for finger in names
                    ],
                    dtype=float,
                )
                probe_rotation = dual_probe_frame(
                    self.env.backend.model, self.env.backend.data
                ).center.rotation_world_from_frame
                observable_balance_rotation = observable_force_balance_rotation(
                    contact_state,
                    full_force[1:4],
                    centers,
                    probe_rotation,
                    maximum_step_rad=(
                        self.search_contact_recovery_maximum_balance_rotation_step_rad
                    ),
                )
                raw[3:6] = observable_balance_rotation
            projected, projection_reason = project_contact_establishment_action(
                raw,
                contact_state,
                maximum_single_contact_step_m=(
                    self.search_contact_recovery_maximum_single_contact_step_m
                ),
                maximum_balance_rotation_step_rad=(
                    self.search_contact_recovery_maximum_balance_rotation_step_rad
                ),
            )
            if projected is None:
                raise RuntimeError(projection_reason or "contact_policy_action_invalid")
            diagnostic_force = np.asarray(
                full_force[1:4]
                if self.search_contact_recovery_observable_balance
                else backend_info.get(
                    "normal_force_n", np.zeros(contact_state.size)
                ),
                dtype=float,
            )
            projected, self.search_contact_recovery_state, layer = (
                layered_search_recovery_projection(
                    projected,
                    contact_state,
                    diagnostic_force,
                    state=self.search_contact_recovery_state,
                    maximum_preload_step_m=(
                        self.search_contact_recovery_maximum_single_contact_step_m
                    ),
                    maximum_reacquire_descent_step_m=(
                        0.5
                        * self.search_contact_recovery_maximum_single_contact_step_m
                    ),
                    minimum_seated_force_n=(
                        self.search_contact_recovery_minimum_seated_force_n
                    ),
                    maximum_reacquire_steps=(
                        self.search_contact_recovery_maximum_reacquire_steps
                    ),
                    maximum_reacquire_rotation_budget_rad=(
                        self.search_contact_recovery_maximum_rotation_budget_rad
                    ),
                )
            )
            uncertainty = prediction.policy_uncertainty
        elif scope == "pressure_regulation":
            contact_state = observable_diagnostic_contact(
                item.contact, backend_info
            )
            projected, projection_reason = project_pressure_regulation_action(
                raw,
                contact_state,
                item.previous_action,
            )
            uncertainty = (
                float("inf")
                if projected is None
                else pressure_regulation_effective_uncertainty(
                    projected,
                    item.previous_action,
                )
            )
        else:
            contact_state = observable_diagnostic_contact(
                item.contact, backend_info
            )
            projected, projection_reason = project_stable_capture_action(
                raw,
                contact_state,
                item.previous_action,
            )
            uncertainty = 0.0

        if projected is None:
            raise RuntimeError(
                f"{scope} policy projection rejected: {projection_reason}"
            )
        if not np.isfinite(uncertainty) or uncertainty > self.maximum_uncertainty:
            raise RuntimeError(
                f"{scope} policy uncertainty {uncertainty:.6f} exceeds "
                f"{self.maximum_uncertainty:.6f}"
            )
        proposed = self._action_from_projection(projected, deterministic_action)
        selected, safety = self.env.safety.filter_action(
            item,
            proposed,
            backend_info,
        )
        if safety["modified"]:
            self.safety_modifications += 1
        self.policy_steps += 1
        self.policy_steps_by_scope[scope] += 1
        control = {
            "source": "policy",
            "policy_action_executed": True,
            "expert_intervened": False,
            "expert_action_consumed": False,
            "intervention_reasons": [],
            "specialist_scope": scope,
            "policy_uncertainty": float(uncertainty),
            "raw_policy_uncertainty": float(prediction.policy_uncertainty),
            "policy_proposed_action": raw.tolist(),
            "selected_action": selected.as_array().tolist(),
            "policy_safety_modified": bool(safety["modified"]),
            "policy_safety_reasons": list(safety["reasons"]),
            "projection_reason": projection_reason,
            "policy_route": str(getattr(runtime, "last_route", scope)),
        }
        if progress_scale is not None:
            control["joint_waypoint_progress_scale"] = float(progress_scale)
            control["joint_waypoint_path_projected"] = True
        if scope == "search_contact_recovery":
            control["observable_contact_state"] = contact_state.tolist()
            control["contact_mask_recovery_layer"] = layer
            recovery_state = self.search_contact_recovery_state
            control["preload_latched"] = bool(recovery_state.preload_latched)
            control["post_latch_reacquire_active"] = bool(
                recovery_state.post_latch_reacquire_active
            )
            control["reacquire_step_count"] = int(
                recovery_state.reacquire_step_count
            )
            control["reacquire_rotation_budget_used_rad"] = float(
                recovery_state.reacquire_rotation_budget_used_rad
            )
            control["descent_amplitude_m"] = float(abs(projected[2]))
            control["descent_decision"] = (
                "hold" if abs(float(projected[2])) <= 1e-12 else "descend"
            )
            control["maximum_balance_rotation_step_rad"] = float(
                self.search_contact_recovery_maximum_balance_rotation_step_rad
            )
            contact_descent = getattr(runtime, "last_contact_descent", None)
            if contact_descent is not None:
                control["contact_descent"] = dict(contact_descent)
            contact_balance = getattr(runtime, "last_contact_balance", None)
            if contact_balance is not None:
                control["contact_balance"] = dict(contact_balance)
            if observable_balance_rotation is not None:
                control["observable_force_balance_rotation_rad"] = (
                    observable_balance_rotation.tolist()
                )
        return selected, control

    def summary(self) -> dict:
        return {
            "schema_version": "adam_pulse_composite_policy_runtime_v1",
            "control_authority": "deployable_policy",
            "expert_action_consumed": False,
            "policy_steps": int(self.policy_steps),
            "deterministic_steps": int(self.deterministic_steps),
            "policy_steps_by_scope": dict(self.policy_steps_by_scope),
            "deterministic_steps_by_label": dict(
                self.deterministic_steps_by_label
            ),
            "safety_modifications": int(self.safety_modifications),
            "truth_in_observation": False,
            "search_contact_recovery_enabled": bool(
                self.search_contact_recovery_enabled
            ),
            "search_contact_recovery_maximum_balance_rotation_step_rad": float(
                self.search_contact_recovery_maximum_balance_rotation_step_rad
            ),
            "search_contact_recovery_maximum_single_contact_step_m": float(
                self.search_contact_recovery_maximum_single_contact_step_m
            ),
            "search_contact_recovery_observable_balance": bool(
                self.search_contact_recovery_observable_balance
            ),
            "search_contact_recovery_maximum_reacquire_steps": int(
                self.search_contact_recovery_maximum_reacquire_steps
            ),
            "search_contact_recovery_maximum_rotation_budget_rad": float(
                self.search_contact_recovery_maximum_rotation_budget_rad
            ),
            "search_contact_recovery_final_state": {
                "preload_latched": bool(
                    self.search_contact_recovery_state.preload_latched
                ),
                "post_latch_reacquire_active": bool(
                    self.search_contact_recovery_state.post_latch_reacquire_active
                ),
                "reacquire_step_count": int(
                    self.search_contact_recovery_state.reacquire_step_count
                ),
                "reacquire_rotation_budget_used_rad": float(
                    self.search_contact_recovery_state.reacquire_rotation_budget_used_rad
                ),
            },
        }
