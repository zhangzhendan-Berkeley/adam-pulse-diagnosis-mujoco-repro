from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np

from pulse_mujoco import (
    CollectionScenario,
    build_stage3_collection_scenario,
    MujocoEpisodeRecorder,
    MujocoPulseDiagnosisBackend,
    MujocoPulseExpert,
    PhaseGatedPolicyConfig,
    PhaseGatedPolicyController,
    PulseAction,
    PulseDiagnosisEnv,
    PulseObservation,
    SafetySupervisor,
    TaskPhase,
    observable_diagnostic_contact,
    apply_collection_scenario,
    collection_acceptance,
    collection_perturbation_scheduler,
    validate_episode,
    project_m5_action_to_safety,
    project_joint_waypoint_progress,
    project_contact_establishment_action,
    pressure_regulation_effective_uncertainty,
    project_pressure_regulation_action,
    project_stable_capture_action,
    project_search_translation,
)
from pulse_mujoco.search_metadata import compact_search_episode_metadata
from pulse_world_vla.cuda import require_cuda
from pulse_world_vla.direct_policy import DirectPolicyRuntime, ShadowRecorder
from pulse_world_vla.unified_policy_runtime import UnifiedPolicyRuntime
from pulse_world_vla.search_segment_runtime import (
    SearchSegmentShadowHook,
    load_search_segment_runtime,
)
from pulse_mujoco.perturbations import (
    PerturbationEvent,
    PerturbationKind,
    PerturbationScheduler,
)


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "安全接近患者腕部并采集满足质量标准的脉搏信号"
PURE_EXPERT_COLLECTION_MODE = "pure_expert_segment_collection"
M5_SPECIALIST_SCOPES = (
    "fine_approach",
    "contact_establishment",
    "pressure_regulation",
    "stable_capture",
)


class PressurePhaseMixSampler:
    """Keep one DAgger source for a contiguous pressure-regulation phase.

    Per-step Bernoulli mixing is retained for the other M5 specialists. During
    pressure regulation it can alternate two otherwise valid absolute pressure
    targets and repeatedly reset the expert's consecutive stability window.
    Sampling once for the contiguous pressure phase evaluates either controller
    coherently without weakening the physical completion criteria.
    """

    def __init__(self, probability: float, rng: np.random.Generator):
        self.probability = float(np.clip(probability, 0.0, 1.0))
        self.rng = rng
        self._active_key: tuple[str, str] | None = None
        self._latched_policy: bool | None = None

    def observe(self, key: tuple[str, str] | None) -> None:
        if key != self._active_key:
            self._active_key = key
            self._latched_policy = None

    def select(self, key: tuple[str, str] | None) -> tuple[bool, bool]:
        self.observe(key)
        if key is None:
            return bool(self.rng.random() <= self.probability), False
        if self._latched_policy is None:
            self._latched_policy = bool(
                self.rng.random() <= self.probability
            )
        return self._latched_policy, True


def _specialist_scope_for_label(
    policy_scope: str, label: str
) -> str | None:
    """Resolve one expert-labelled control tick to its frozen specialist.

    Composite routing is deliberately label-exact.  An unknown label remains
    expert controlled during M5.4 instead of being guessed from a nearby task
    phase and silently broadening a specialist's authority.
    """
    if policy_scope != "composite":
        return policy_scope
    if label == "m5_joint_waypoint_protected":
        return "fine_approach"
    if label in {"contact_approach", "single_finger_balance"}:
        return "contact_establishment"
    if label == "pressure_regulation":
        return "pressure_regulation"
    if label == "signal_capture":
        return "stable_capture"
    return None


def _expert_source(privileged_search_assist: bool) -> str:
    return (
        "privileged_expert"
        if privileged_search_assist
        else "observable_expert"
    )


def _policy_collection_contract(
    *,
    disable_policy_shadow: bool,
    policy_mix: float,
    checkpoint: Path,
    privileged_search_assist: bool,
    specialist_checkpoints: dict[str, Path] | None = None,
    policy_runtime: str = "direct",
    unified_scope: str | None = None,
    unified_checkpoint: Path | None = None,
    policy_execution_budget_steps: int | None = None,
    policy_execution_budget_scope: str | None = None,
) -> dict:
    """Return truthful collection-level policy provenance.

    The pure-expert branch deliberately returns before resolving the
    checkpoint.  Besides keeping its metadata honest, this lets pure label
    collection run on machines that do not have a policy artifact at all.
    """
    if disable_policy_shadow:
        return {
            "mode": PURE_EXPERT_COLLECTION_MODE,
            "source": _expert_source(privileged_search_assist),
            "policy_mix_probability": 0.0,
            "policy_shadow_disabled": True,
        }
    contract = {
        "mode": (
            "teacher_shadow_only"
            if float(policy_mix) <= 0.0
            else "phase_gated_dagger"
        ),
        "policy_mix_probability": float(policy_mix),
        "policy_mix_granularity": "per_step_with_pressure_phase_latch",
        "policy_shadow_disabled": False,
        "policy_runtime": str(policy_runtime),
        "policy_execution_budget_steps": policy_execution_budget_steps,
        "policy_execution_budget_scope": policy_execution_budget_scope,
    }
    if specialist_checkpoints:
        contract["specialist_checkpoints"] = {
            scope: str(path.resolve())
            for scope, path in sorted(specialist_checkpoints.items())
        }
        if unified_scope is not None:
            if unified_checkpoint is None:
                raise ValueError("unified scope requires a unified checkpoint")
            contract["unified_runtime_override"] = {
                "scope": str(unified_scope),
                "checkpoint": str(unified_checkpoint.resolve()),
            }
    else:
        contract["checkpoint"] = str(checkpoint.resolve())
    return contract


def _validate_policy_checkpoint(
    checkpoint: Path, *, disable_policy_shadow: bool
) -> None:
    if disable_policy_shadow:
        return
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)


def _episode_policy_metadata(
    collection_contract: dict, *, dagger_shadow: dict | None
) -> dict:
    metadata = {
        "mode": collection_contract["mode"],
        "policy_mix_probability": collection_contract[
            "policy_mix_probability"
        ],
        "policy_shadow_disabled": collection_contract[
            "policy_shadow_disabled"
        ],
        "dagger_shadow": dagger_shadow,
    }
    if "source" in collection_contract:
        metadata["source"] = collection_contract["source"]
    if "checkpoint" in collection_contract:
        metadata["checkpoint"] = collection_contract["checkpoint"]
    if "specialist_checkpoints" in collection_contract:
        metadata["specialist_checkpoints"] = dict(
            collection_contract["specialist_checkpoints"]
        )
    if "policy_mix_granularity" in collection_contract:
        metadata["policy_mix_granularity"] = collection_contract[
            "policy_mix_granularity"
        ]
    if "policy_runtime" in collection_contract:
        metadata["policy_runtime"] = collection_contract["policy_runtime"]
    for name in (
        "unified_runtime_override",
        "policy_execution_budget_steps",
        "policy_execution_budget_scope",
    ):
        if name in collection_contract:
            metadata[name] = collection_contract[name]
    return metadata


def _search_episode_metadata(result: dict) -> dict:
    return compact_search_episode_metadata(result)


def _maybe_create_dagger_hook(
    *, disable_policy_shadow: bool, **hook_kwargs
) -> "DAggerEpisodeHook | None":
    if disable_policy_shadow:
        return None
    return DAggerEpisodeHook(**hook_kwargs)


def _runtime_device(device: str) -> str:
    if device == "cpu":
        raise ValueError(
            "policy DAgger collection refuses CPU execution; restore CUDA visibility"
        )
    require_cuda(purpose="Stage-3 policy DAgger collection")
    return "cuda"


def _build_policy_runtime(
    checkpoint: Path,
    *,
    policy_runtime: str,
    device: str,
):
    runtime_device = _runtime_device(device)
    if policy_runtime == "direct":
        return DirectPolicyRuntime(
            checkpoint, device=runtime_device, control_substeps=1
        )
    if policy_runtime == "unified":
        return UnifiedPolicyRuntime(
            checkpoint, device=runtime_device, control_substeps=1
        )
    raise ValueError(f"unsupported policy runtime: {policy_runtime}")


def _scenario(
    index: int, seed: int, profile: str = "random_recovery"
) -> CollectionScenario:
    return build_stage3_collection_scenario(index, seed, profile)


def _observation(value: object) -> PulseObservation:
    """Convert the expert's policy dict to the runtime observation contract."""
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


class DAggerEpisodeHook:
    def __init__(
        self,
        *,
        checkpoint: Path | None,
        specialist_checkpoints: dict[str, Path] | None = None,
        shadow_path: Path,
        env: PulseDiagnosisEnv,
        recorder: MujocoEpisodeRecorder,
        device: str,
        policy_mix: float,
        seed: int,
        policy_scope: str,
        search_direction_blend: float,
        policy_runtime: str = "direct",
        unified_scope: str | None = None,
        unified_checkpoint: Path | None = None,
        policy_execution_budget_steps: int | None = None,
        policy_execution_budget_scope: str | None = None,
    ):
        self.env = env
        self.recorder = recorder
        self.shadow_path = shadow_path
        if policy_scope == "composite":
            if policy_runtime == "unified" and unified_scope is None:
                raise ValueError("composite unified runtime requires unified_scope")
            if unified_scope is not None and unified_checkpoint is None:
                raise ValueError("unified_scope requires unified_checkpoint")
        elif unified_scope is not None or unified_checkpoint is not None:
            raise ValueError("unified scope override requires composite policy")
        checkpoint_map = dict(specialist_checkpoints or {})
        if policy_scope == "composite":
            missing = sorted(set(M5_SPECIALIST_SCOPES) - set(checkpoint_map))
            extra = sorted(set(checkpoint_map) - set(M5_SPECIALIST_SCOPES))
            if missing or extra:
                raise ValueError(
                    "composite specialist checkpoint mismatch: "
                    f"missing={missing}, extra={extra}"
                )
        else:
            if checkpoint is None:
                raise ValueError("single-scope policy requires a checkpoint")
            checkpoint_map = {policy_scope: checkpoint}
        self.runtimes = {}
        for scope, path in checkpoint_map.items():
            runtime_kind = (
                "unified" if scope == unified_scope else "direct"
            ) if policy_scope == "composite" else policy_runtime
            runtime_checkpoint = (
                unified_checkpoint if scope == unified_scope else path
            )
            assert runtime_checkpoint is not None
            self.runtimes[scope] = _build_policy_runtime(
                runtime_checkpoint,
                policy_runtime=runtime_kind,
                device=device,
            )
        self.policy_runtime = str(policy_runtime)
        self.unified_scope = unified_scope
        self.unified_checkpoint = (
            None
            if unified_checkpoint is None
            else str(unified_checkpoint.resolve())
        )
        self.policy_execution_budget_steps = policy_execution_budget_steps
        self.policy_execution_budget_scope = policy_execution_budget_scope
        self.specialist_checkpoints = {
            scope: str(path.resolve()) for scope, path in checkpoint_map.items()
        }
        # Retain the existing attribute for the single-scope history-primer
        # contract and expose the fine-approach horizon for composite routing.
        self.runtime = self.runtimes.get(
            "fine_approach", next(iter(self.runtimes.values()))
        )
        self.shadow = ShadowRecorder(shadow_path)
        self.safety = SafetySupervisor()
        if policy_scope == "search_no_contact":
            allowed_phases = (TaskPhase.PULSE_SEARCH,)
        elif policy_scope == "fine_approach":
            allowed_phases = (TaskPhase.FINE_APPROACH,)
        elif policy_scope == "contact_establishment":
            allowed_phases = (TaskPhase.CONTACT_ESTABLISHMENT,)
        elif policy_scope == "pressure_regulation":
            allowed_phases = (TaskPhase.PRESSURE_REGULATION,)
        elif policy_scope == "stable_capture":
            allowed_phases = (TaskPhase.SIGNAL_CAPTURE,)
        elif policy_scope == "composite":
            allowed_phases = (
                TaskPhase.FINE_APPROACH,
                TaskPhase.CONTACT_ESTABLISHMENT,
                TaskPhase.PRESSURE_REGULATION,
                TaskPhase.SIGNAL_CAPTURE,
            )
        elif policy_scope == "pre_contact":
            allowed_phases = (
                TaskPhase.PRE_CONTACT,
                TaskPhase.CONTACT_ESTABLISHMENT,
            )
        else:
            raise ValueError(f"unsupported policy scope: {policy_scope}")
        self.policy_scope = policy_scope
        no_contact_required_phases = (
            (TaskPhase.FINE_APPROACH,)
            if policy_scope == "composite"
            else (
                ()
                if policy_scope in {
                    "contact_establishment",
                    "pressure_regulation",
                    "stable_capture",
                }
                else allowed_phases
            )
        )
        self.controller = PhaseGatedPolicyController(
            PhaseGatedPolicyConfig(
                allowed_phases=allowed_phases,
                no_contact_required_phases=no_contact_required_phases,
                maximum_uncertainty=0.22,
            )
        )
        self.policy_mix = float(np.clip(policy_mix, 0.0, 1.0))
        self.search_direction_blend = float(
            np.clip(search_direction_blend, 0.0, 1.0)
        )
        self.rng = np.random.default_rng(seed)
        self.mix_sampler = PressurePhaseMixSampler(
            self.policy_mix,
            self.rng,
        )
        self.frame = 0
        self.policy_action_frames = 0
        self.policy_action_frames_by_scope: dict[str, int] = {}
        self.pending_expert_label: np.ndarray | None = None
        self.pending_task_label: str | None = None

    def prime_observation(self, observation: object) -> None:
        """Fill temporal history during a zero-motion pre-phase hold."""
        if self.shadow.pending is not None:
            raise RuntimeError("cannot prime history with a pending prediction")
        item = _observation(observation)
        primed = replace(
            item,
            previous_action=project_m5_action_to_safety(
                item.previous_action
            ),
        )
        for runtime in self.runtimes.values():
            runtime.append_observation(primed)

    def select_action(
        self,
        observation: PulseObservation,
        backend_info: dict,
        expert_action,
        label: str,
    ):
        if self.shadow.pending is not None and self.pending_expert_label is None:
            self.shadow.discard_pending("fine_approach_phase_boundary")
            self.pending_task_label = None
        if self.shadow.pending is not None and self.pending_expert_label is not None:
            # A forbidden-contact exception can skip the normal on-step
            # callback.  Align that pending prediction with the expert label
            # from the same decision before an emergency-retreat action is
            # requested.
            self.shadow.observe_expert_action(
                self.pending_expert_label,
                expert_label=self.pending_task_label,
            )
            self.pending_expert_label = None
            self.pending_task_label = None
        self.frame += 1
        expert_label_action, _ = self.env.safety.filter_action(
            observation, expert_action, backend_info
        )
        control = {
            "source": "expert",
            "expert_intervened": True,
            "intervention_reasons": ["dagger_teacher_sample"],
            "task_phase": TaskPhase(int(observation.task_phase)).name.lower(),
            "policy_action_executed": False,
            "expert_label_action": expert_label_action.as_array().tolist(),
        }
        if observation.rgb is None or observation.depth_m is None:
            control["intervention_reasons"] = ["vision_unavailable"]
            return expert_action, control
        selected_scope = _specialist_scope_for_label(
            self.policy_scope, label
        )
        pressure_mix_key = (
            ("pressure_regulation", "pressure_regulation")
            if selected_scope == "pressure_regulation"
            and label == "pressure_regulation"
            else None
        )
        self.mix_sampler.observe(pressure_mix_key)
        runtime_observation = observation
        if (
            selected_scope == "fine_approach"
            and label == "m5_joint_waypoint_protected"
        ):
            runtime_observation = replace(
                observation,
                previous_action=project_m5_action_to_safety(
                    observation.previous_action
                ),
            )
        for runtime in self.runtimes.values():
            runtime.append_observation(
                runtime_observation, backend_info=backend_info
            )
        if selected_scope is None:
            self.controller.record_expert_intervention(
                TaskPhase(int(observation.task_phase)),
                ["composite_label_not_enabled"],
            )
            control["intervention_reasons"] = [
                "composite_label_not_enabled"
            ]
            return expert_action, control
        runtime = self.runtimes[selected_scope]
        control["specialist_scope"] = selected_scope
        if not runtime.ready:
            control["intervention_reasons"] = ["policy_not_ready"]
            return expert_action, control
        prediction = runtime.predict_shadow(
            frame=self.frame,
            timestamp_s=observation.timestamp_s,
            task_phase=int(observation.task_phase),
            safety_observation=observation,
            safety_supervisor=self.safety,
            backend_info=backend_info,
        )
        self.shadow.set_prediction(prediction)
        self.pending_expert_label = expert_label_action.as_array().copy()
        self.pending_task_label = str(label)
        policy_continuous_action = prediction.control_tick_action
        if selected_scope == "search_no_contact":
            if label != "search_translate":
                control.update(
                    {
                        "policy_uncertainty": prediction.policy_uncertainty,
                        "intervention_reasons": [
                            "search_label_not_enabled"
                        ],
                        "policy_proposed_action": (
                            policy_continuous_action.tolist()
                        ),
                    }
                )
                return expert_action, control
            # During the lifted translation segment the learned specialist
            # controls only horizontal displacement.  Height, wrist rotation
            # and pressure remain exactly on the expert state-machine
            # contract, preventing accidental descent/contact mid-translation.
            projected, projection_reason = project_search_translation(
                policy_continuous_action,
                expert_action,
                policy_direction_blend=self.search_direction_blend,
            )
            if projected is None:
                control.update(
                    {
                        "policy_uncertainty": prediction.policy_uncertainty,
                        "intervention_reasons": [projection_reason],
                        "policy_proposed_action": (
                            policy_continuous_action.tolist()
                        ),
                    }
                )
                return expert_action, control
            policy_continuous_action = projected
        execute_policy, pressure_phase_latched = self.mix_sampler.select(
            pressure_mix_key
        )
        if (
            execute_policy
            and self.policy_execution_budget_steps is not None
            and selected_scope == self.policy_execution_budget_scope
            and self.policy_action_frames_by_scope.get(selected_scope, 0)
            >= self.policy_execution_budget_steps
        ):
            execute_policy = False
            self.controller.record_expert_intervention(
                TaskPhase(int(observation.task_phase)),
                ["dagger_policy_execution_budget_exhausted"],
            )
            control.update(
                {
                    "policy_uncertainty": prediction.policy_uncertainty,
                    "intervention_reasons": [
                        "dagger_policy_execution_budget_exhausted"
                    ],
                    "policy_proposed_action": policy_continuous_action.tolist(),
                    "policy_execution_budget_steps": (
                        self.policy_execution_budget_steps
                    ),
                    "policy_execution_budget_scope": (
                        self.policy_execution_budget_scope
                    ),
                }
            )
            return expert_action, control
        if not execute_policy:
            self.controller.record_expert_intervention(
                TaskPhase(int(observation.task_phase)),
                ["dagger_teacher_mix"],
            )
            control.update(
                {
                    "policy_uncertainty": prediction.policy_uncertainty,
                    "intervention_reasons": ["dagger_teacher_mix"],
                    "policy_proposed_action": policy_continuous_action.tolist(),
                    "dagger_mix_granularity": (
                        "pressure_phase"
                        if pressure_phase_latched
                        else "control_step"
                    ),
                }
            )
            return expert_action, control
        raw_policy_continuous_action = policy_continuous_action.copy()
        decision_uncertainty = prediction.policy_uncertainty
        joint_waypoint_progress_scale = None
        contact_establishment_projected = False
        pressure_regulation_projected = False
        stable_capture_projected = False
        if selected_scope == "fine_approach":
            if label != "m5_joint_waypoint_protected":
                self.controller.record_expert_intervention(
                    TaskPhase(int(observation.task_phase)),
                    ["fine_approach_label_not_enabled"],
                )
                control.update(
                    {
                        "policy_uncertainty": prediction.policy_uncertainty,
                        "intervention_reasons": [
                            "fine_approach_label_not_enabled"
                        ],
                        "policy_proposed_action": (
                            raw_policy_continuous_action.tolist()
                        ),
                    }
                )
                return expert_action, control
            (
                projected,
                projection_reason,
                joint_waypoint_progress_scale,
            ) = project_joint_waypoint_progress(
                raw_policy_continuous_action,
                PulseAction.from_array(
                    project_m5_action_to_safety(
                        observation.previous_action
                    )
                ),
                verified_path_owns_direction=True,
            )
            if projected is None:
                self.controller.record_expert_intervention(
                    TaskPhase(int(observation.task_phase)),
                    [str(projection_reason)],
                )
                control.update(
                    {
                        "policy_uncertainty": prediction.policy_uncertainty,
                        "intervention_reasons": [projection_reason],
                        "policy_proposed_action": (
                            raw_policy_continuous_action.tolist()
                        ),
                    }
                )
                return expert_action, control
            policy_continuous_action = projected
        elif selected_scope == "contact_establishment":
            if label not in {"contact_approach", "single_finger_balance"}:
                commanded_phase = (
                    TaskPhase(int(expert_action.phase_command))
                    if expert_action.phase_command is not None
                    else TaskPhase(int(observation.task_phase))
                )
                self.controller.record_expert_intervention(
                    commanded_phase,
                    ["contact_label_not_enabled"],
                )
                control.update(
                    {
                        "policy_uncertainty": prediction.policy_uncertainty,
                        "intervention_reasons": ["contact_label_not_enabled"],
                        "policy_proposed_action": (
                            raw_policy_continuous_action.tolist()
                        ),
                    }
                )
                return expert_action, control
            contact_state = observable_diagnostic_contact(
                observation.contact, backend_info
            )
            projected, projection_reason = (
                project_contact_establishment_action(
                    raw_policy_continuous_action,
                    contact_state,
                )
            )
            if projected is None:
                self.controller.record_expert_intervention(
                    TaskPhase(int(observation.task_phase)),
                    [str(projection_reason)],
                )
                control.update(
                    {
                        "policy_uncertainty": prediction.policy_uncertainty,
                        "intervention_reasons": [projection_reason],
                        "policy_proposed_action": (
                            raw_policy_continuous_action.tolist()
                        ),
                    }
                )
                return expert_action, control
            policy_continuous_action = projected
            contact_establishment_projected = True
        elif selected_scope == "pressure_regulation":
            if label != "pressure_regulation":
                commanded_phase = (
                    TaskPhase(int(expert_action.phase_command))
                    if expert_action.phase_command is not None
                    else TaskPhase(int(observation.task_phase))
                )
                self.controller.record_expert_intervention(
                    commanded_phase,
                    ["pressure_label_not_enabled"],
                )
                control.update(
                    {
                        "policy_uncertainty": prediction.policy_uncertainty,
                        "intervention_reasons": ["pressure_label_not_enabled"],
                        "policy_proposed_action": (
                            raw_policy_continuous_action.tolist()
                        ),
                    }
                )
                return expert_action, control
            pressure_contact_state = observable_diagnostic_contact(
                observation.contact, backend_info
            )
            projected, projection_reason = project_pressure_regulation_action(
                raw_policy_continuous_action,
                pressure_contact_state,
                observation.previous_action,
            )
            if projected is None:
                self.controller.record_expert_intervention(
                    TaskPhase(int(observation.task_phase)),
                    [str(projection_reason)],
                )
                control.update(
                    {
                        "policy_uncertainty": prediction.policy_uncertainty,
                        "intervention_reasons": [projection_reason],
                        "policy_proposed_action": (
                            raw_policy_continuous_action.tolist()
                        ),
                    }
                )
                return expert_action, control
            policy_continuous_action = projected
            pressure_regulation_projected = True
            decision_uncertainty = pressure_regulation_effective_uncertainty(
                policy_continuous_action,
                observation.previous_action,
            )
        elif selected_scope == "stable_capture":
            if label != "signal_capture":
                commanded_phase = (
                    TaskPhase(int(expert_action.phase_command))
                    if expert_action.phase_command is not None
                    else TaskPhase(int(observation.task_phase))
                )
                self.controller.record_expert_intervention(
                    commanded_phase,
                    ["stable_capture_label_not_enabled"],
                )
                control.update(
                    {
                        "policy_uncertainty": prediction.policy_uncertainty,
                        "intervention_reasons": [
                            "stable_capture_label_not_enabled"
                        ],
                        "policy_proposed_action": (
                            raw_policy_continuous_action.tolist()
                        ),
                    }
                )
                return expert_action, control
            stable_contact_state = observable_diagnostic_contact(
                observation.contact, backend_info
            )
            projected, projection_reason = project_stable_capture_action(
                raw_policy_continuous_action,
                stable_contact_state,
                observation.previous_action,
            )
            if projected is None:
                self.controller.record_expert_intervention(
                    TaskPhase(int(observation.task_phase)),
                    [str(projection_reason)],
                )
                control.update(
                    {
                        "policy_uncertainty": prediction.policy_uncertainty,
                        "intervention_reasons": [projection_reason],
                        "policy_proposed_action": (
                            raw_policy_continuous_action.tolist()
                        ),
                    }
                )
                return expert_action, control
            policy_continuous_action = projected
            stable_capture_projected = True
            decision_uncertainty = 0.0
        decision = self.controller.decide(
            observation=observation,
            backend_info=backend_info,
            expert_action=expert_action,
            policy_continuous_action=policy_continuous_action,
            policy_uncertainty=decision_uncertainty,
            env=self.env,
        )
        if decision.policy_action_executed:
            self.policy_action_frames += 1
            self.policy_action_frames_by_scope[selected_scope] = (
                self.policy_action_frames_by_scope.get(selected_scope, 0) + 1
            )
        decision_record = decision.as_dict()
        decision_record["specialist_scope"] = selected_scope
        decision_record["dagger_mix_granularity"] = (
            "pressure_phase"
            if pressure_phase_latched
            else "control_step"
        )
        if joint_waypoint_progress_scale is not None:
            decision_record.update(
                {
                    "raw_policy_proposed_action": (
                        raw_policy_continuous_action.tolist()
                    ),
                    "joint_waypoint_progress_scale": float(
                        joint_waypoint_progress_scale
                    ),
                    "joint_waypoint_path_projected": True,
                }
            )
        if contact_establishment_projected:
            decision_record.update(
                {
                    "raw_policy_proposed_action": (
                        raw_policy_continuous_action.tolist()
                    ),
                    "contact_establishment_projected": True,
                    "contact_projection_uses_expert_action": False,
                    "contact_non_descent_suppressed": bool(
                        raw_policy_continuous_action[2] >= -1e-9
                        and not np.all(contact_state)
                    ),
                }
            )
        if pressure_regulation_projected:
            decision_record.update(
                {
                    "raw_policy_proposed_action": (
                        raw_policy_continuous_action.tolist()
                    ),
                    "pressure_regulation_projected": True,
                    "pressure_projection_uses_expert_action": False,
                    "pressure_pose_locked": True,
                    "pressure_rate_limited": bool(
                        abs(
                            float(policy_continuous_action[6])
                            - float(raw_policy_continuous_action[6])
                        )
                        > 1e-6
                    ),
                    "pressure_contact_guard": projection_reason,
                    "raw_policy_uncertainty": prediction.policy_uncertainty,
                    "effective_policy_uncertainty": decision_uncertainty,
                }
            )
        if stable_capture_projected:
            decision_record.update(
                {
                    "raw_policy_proposed_action": (
                        raw_policy_continuous_action.tolist()
                    ),
                    "stable_capture_projected": True,
                    "stable_capture_projection_uses_expert_action": False,
                    "stable_capture_pose_locked": True,
                    "stable_capture_pressure_held": True,
                    "raw_policy_uncertainty": prediction.policy_uncertainty,
                    "effective_policy_uncertainty": decision_uncertainty,
                }
            )
        return decision.action, decision_record

    def __call__(self, backend, observation, info, label):
        m5_shadow = bool(info.get("m5_joint_waypoint_shadow", False))
        m5_observation = None
        m5_safe_label = None
        if m5_shadow:
            m5_observation = _observation(observation)
            if "executed_action" in info:
                m5_safe_label = project_m5_action_to_safety(
                    np.asarray(info["executed_action"], dtype=np.float32)
                )
                # The shadow input history must describe the command that a
                # deployable policy could have executed, not the raw TCP
                # displacement produced by the joint PD waypoint.
                m5_observation = replace(
                    m5_observation,
                    previous_action=m5_safe_label.copy(),
                )
        if "executed_action" in info:
            self.shadow.observe_expert_action(
                m5_safe_label
                if m5_safe_label is not None
                else info["executed_action"],
                expert_label=self.pending_task_label,
            )
            self.pending_expert_label = None
            self.pending_task_label = None
        if m5_shadow:
            observation_object = (
                m5_observation
                if m5_observation is not None
                else _observation(observation)
            )
            self.runtime.append_observation(
                observation_object, backend_info=info
            )
            if self.runtime.ready:
                prediction = self.runtime.predict_shadow(
                    frame=self.frame,
                    timestamp_s=observation_object.timestamp_s,
                    task_phase=int(observation_object.task_phase),
                    safety_observation=observation_object,
                    safety_supervisor=self.safety,
                    backend_info=info,
                )
                self.shadow.set_prediction(prediction)
            self.frame += 1
        self.recorder(backend, observation, info, label)

    def close(self, action_scale) -> dict:
        summary = self.shadow.close(action_scale)
        summary["mode"] = (
            "teacher_shadow_only"
            if self.policy_mix <= 0.0
            else "phase_gated_dagger"
        )
        summary["policy_action_executed"] = self.policy_action_frames > 0
        summary["policy_mix_probability"] = self.policy_mix
        summary["policy_scope"] = self.policy_scope
        summary["policy_runtime"] = self.policy_runtime
        summary["unified_scope"] = self.unified_scope
        summary["unified_checkpoint"] = self.unified_checkpoint
        summary["policy_execution_budget_steps"] = (
            self.policy_execution_budget_steps
        )
        summary["policy_execution_budget_scope"] = (
            self.policy_execution_budget_scope
        )
        summary["policy_action_frames_by_scope"] = dict(
            sorted(self.policy_action_frames_by_scope.items())
        )
        summary["specialist_checkpoints"] = dict(
            self.specialist_checkpoints
        )
        summary["search_direction_blend"] = self.search_direction_blend
        summary["policy_action_frames"] = self.policy_action_frames
        summary["controller"] = self.controller.summary()
        self.shadow_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect protected Stage-3 teacher-shadow/DAgger episodes."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/training_stage2/adam_pro_right_world_vla_stage2_v1.pt",
    )
    parser.add_argument(
        "--fine-approach-checkpoint",
        type=Path,
        help="frozen M5 fine-approach specialist for composite routing",
    )
    parser.add_argument(
        "--contact-establishment-checkpoint",
        type=Path,
        help="frozen M5 contact-establishment specialist for composite routing",
    )
    parser.add_argument(
        "--pressure-regulation-checkpoint",
        type=Path,
        help="frozen M5 pressure-regulation specialist for composite routing",
    )
    parser.add_argument(
        "--stable-capture-checkpoint",
        type=Path,
        help="frozen M5 stable-capture specialist for composite routing",
    )
    parser.add_argument(
        "--policy-scope",
        choices=(
            "fine_approach",
            "contact_establishment",
            "pressure_regulation",
            "stable_capture",
            "composite",
            "pre_contact",
            "search_no_contact",
        ),
        default="pre_contact",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--alignment",
        type=Path,
        default=ROOT / "configs/alignment_pro.yaml",
        help="versioned MuJoCo alignment/embodiment configuration",
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="first deterministic scenario/episode index (for safe supplements)",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--policy-mix",
        type=float,
        default=0.0,
        help="fraction of admitted actions allowed to execute; 0 keeps teacher-shadow only",
    )
    parser.add_argument(
        "--search-direction-blend",
        type=float,
        default=0.20,
        help="fraction of the admitted search direction taken from the policy",
    )
    parser.add_argument(
        "--device", choices=("cuda",), default="cuda",
        help="CUDA is mandatory whenever a learned policy is instantiated.",
    )
    parser.add_argument(
        "--policy-runtime",
        choices=("direct", "unified"),
        default="direct",
        help=(
            "checkpoint adapter used for DAgger shadow inference; unified "
            "supports full56 benchmark/JEPA checkpoints for one policy scope"
        ),
    )
    parser.add_argument(
        "--unified-scope",
        choices=M5_SPECIALIST_SCOPES,
        help=(
            "for composite DAgger, replace this one Direct specialist with "
            "--checkpoint loaded through the unified runtime"
        ),
    )
    parser.add_argument(
        "--policy-execution-budget-steps",
        type=int,
        help=(
            "maximum physically executed policy steps in the selected budget "
            "scope before deterministic expert recovery"
        ),
    )
    parser.add_argument(
        "--policy-execution-budget-scope",
        choices=M5_SPECIALIST_SCOPES,
        help="scope governed by --policy-execution-budget-steps",
    )
    parser.add_argument(
        "--disable-policy-shadow",
        action="store_true",
        help="collect pure expert segment labels without loading/inferencing a policy",
    )
    parser.add_argument(
        "--disable-proximity",
        action="store_true",
        help=(
            "model the current real Adam Pro hardware by marking all five "
            "simulated proximity channels invalid for the complete episode"
        ),
    )
    parser.add_argument("--capture-interval", type=int, default=2)
    parser.add_argument(
        "--rgbd-capture-interval",
        type=int,
        default=None,
        help=(
            "refresh RGB-D every N control observations; defaults to the "
            "episode capture interval"
        ),
    )
    parser.add_argument(
        "--m5-cartesian-fine-approach",
        action="store_true",
        help=(
            "replace only the final joint-space pre-contact transition with "
            "bounded Cartesian expert ticks for M5 data/shadow"
        ),
    )
    parser.add_argument(
        "--m5-joint-waypoint-shadow",
        action="store_true",
        help=(
            "shadow fine approach on the proven joint-space waypoint; the "
            "policy cannot execute in this mode"
        ),
    )
    parser.add_argument(
        "--m5-joint-waypoint-duration-s",
        type=float,
        help=(
            "override the slow joint-waypoint duration; useful for collecting "
            "labels that stay inside the unchanged SafetySupervisor step limit"
        ),
    )
    parser.add_argument(
        "--m5-joint-waypoint-protected",
        action="store_true",
        help=(
            "execute admitted fine-approach proposals only as bounded progress "
            "along the proven joint-space waypoint"
        ),
    )
    parser.add_argument(
        "--scenario-profile",
        choices=(
            "random_recovery",
            "contactfirst_collection_v1",
            "directional_recovery",
            "directional_recovery_balanced",
            "directional_recovery_blind_v1",
            "ch5_multifinger_balanced",
            "ch5_multifinger_train_v1",
            "ch5_multifinger_axial_balance_v1",
            "ch5_multifinger_positive_axial_v1",
            "ch5_multifinger_approach_axial_v1",
            "ch5_search_recovery_gate_v1",
            "ch5_multifinger_ood_v1",
            "model_only_ood_v1",
            "model_only_ood_feasible_v1",
        ),
        default="random_recovery",
    )
    parser.add_argument(
        "--expert-privileged-search-assist",
        action="store_true",
        help=(
            "training-only oracle direction for expert labels; simulator "
            "truth is never added to policy observations"
        ),
    )
    parser.add_argument(
        "--segment-checkpoint",
        type=Path,
        help="frozen ACT signal-quality segment checkpoint for M4 shadow/execute",
    )
    parser.add_argument(
        "--segment-policy-mode",
        choices=(
            "disabled",
            "shadow",
            "execute_reviewed",
            "execute",
            "execute_model_only",
        ),
        default="disabled",
    )
    parser.add_argument(
        "--segment-policy-execution-budget",
        type=int,
        default=None,
        help=(
            "maximum admitted student search segments physically executed in "
            "execute_reviewed mode; later student-visited states retain model "
            "proposals while the expert counterfactual executes"
        ),
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    m5_route_modes = sum(
        bool(value)
        for value in (
            args.m5_cartesian_fine_approach,
            args.m5_joint_waypoint_shadow,
            args.m5_joint_waypoint_protected,
        )
    )
    if m5_route_modes > 1:
        parser.error("M5 Cartesian, shadow and protected route modes are exclusive")
    if args.m5_joint_waypoint_shadow and args.policy_mix > 0.0:
        parser.error("joint-waypoint shadow is observation-only; policy-mix must be 0")
    if args.m5_joint_waypoint_shadow and args.policy_scope != "fine_approach":
        parser.error("joint-waypoint shadow requires --policy-scope fine_approach")
    if (
        args.m5_joint_waypoint_protected
        and args.policy_scope not in {"fine_approach", "composite"}
    ):
        parser.error(
            "protected joint waypoint requires --policy-scope "
            "fine_approach or composite"
        )
    if args.m5_joint_waypoint_protected and args.policy_mix <= 0.0:
        parser.error("protected joint waypoint requires positive --policy-mix")
    if (
        args.m5_joint_waypoint_duration_s is not None
        and args.m5_joint_waypoint_duration_s <= 0.0
    ):
        parser.error("--m5-joint-waypoint-duration-s must be positive")
    if args.segment_policy_mode != "disabled":
        if args.segment_checkpoint is None:
            parser.error("enabled segment policy requires --segment-checkpoint")
        if not args.segment_checkpoint.exists():
            parser.error(f"segment checkpoint does not exist: {args.segment_checkpoint}")
    elif args.segment_checkpoint is not None:
        parser.error("--segment-checkpoint requires shadow or execute mode")
    if args.segment_policy_execution_budget is not None:
        if args.segment_policy_execution_budget < 0:
            parser.error("--segment-policy-execution-budget must be non-negative")
        if args.segment_policy_mode != "execute_reviewed":
            parser.error(
                "--segment-policy-execution-budget requires "
                "--segment-policy-mode execute_reviewed"
            )
    if not args.alignment.exists():
        parser.error(f"alignment configuration does not exist: {args.alignment}")
    specialist_checkpoints = {
        scope: path
        for scope, path in {
            "fine_approach": args.fine_approach_checkpoint,
            "contact_establishment": args.contact_establishment_checkpoint,
            "pressure_regulation": args.pressure_regulation_checkpoint,
            "stable_capture": args.stable_capture_checkpoint,
        }.items()
        if path is not None
    }
    if args.policy_scope == "composite" and not args.disable_policy_shadow:
        if args.policy_runtime == "unified" and args.unified_scope is None:
            parser.error("composite unified runtime requires --unified-scope")
        missing = sorted(
            set(M5_SPECIALIST_SCOPES) - set(specialist_checkpoints)
        )
        if missing:
            parser.error(
                "composite policy requires all specialist checkpoints; "
                f"missing: {', '.join(missing)}"
            )
        for path in specialist_checkpoints.values():
            _validate_policy_checkpoint(path, disable_policy_shadow=False)
        if args.unified_scope is not None:
            _validate_policy_checkpoint(
                args.checkpoint, disable_policy_shadow=False
            )
    elif specialist_checkpoints:
        parser.error(
            "specialist checkpoint options require --policy-scope composite"
        )
    else:
        if args.unified_scope is not None:
            parser.error("--unified-scope requires --policy-scope composite")
        _validate_policy_checkpoint(
            args.checkpoint,
            disable_policy_shadow=args.disable_policy_shadow,
        )
    if args.policy_execution_budget_steps is not None:
        if args.policy_execution_budget_steps < 0:
            parser.error("policy execution budget must be non-negative")
        if args.policy_mix <= 0.0:
            parser.error("policy execution budget requires positive --policy-mix")
        if args.policy_execution_budget_scope is None:
            args.policy_execution_budget_scope = (
                args.unified_scope
                if args.policy_scope == "composite"
                else args.policy_scope
            )
    elif args.policy_execution_budget_scope is not None:
        parser.error(
            "--policy-execution-budget-scope requires "
            "--policy-execution-budget-steps"
        )
    collection_contract = _policy_collection_contract(
        disable_policy_shadow=args.disable_policy_shadow,
        policy_mix=args.policy_mix,
        checkpoint=args.checkpoint,
        privileged_search_assist=args.expert_privileged_search_assist,
        specialist_checkpoints=specialist_checkpoints,
        policy_runtime=args.policy_runtime,
        unified_scope=args.unified_scope,
        unified_checkpoint=(
            args.checkpoint if args.unified_scope is not None else None
        ),
        policy_execution_budget_steps=args.policy_execution_budget_steps,
        policy_execution_budget_scope=args.policy_execution_budget_scope,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(args.start_index, args.start_index + args.episodes):
        episode_id = f"{index:06d}"
        episode_dir = args.output / f"episode_{episode_id}"
        if episode_dir.exists():
            raise FileExistsError(episode_dir)
        failed_dir = args.output / "failed_runs"
        recorder = MujocoEpisodeRecorder(
            args.output,
            episode_id,
            INSTRUCTION,
            capture_interval=args.capture_interval,
            jpeg_quality=85,
            record_size=(320, 240),
        )
        shadow_path = episode_dir / "dagger_shadow.json"
        backend = None
        shadow_summary: dict | None = None
        segment_hook: SearchSegmentShadowHook | None = None
        try:
            scenario = _scenario(index, args.seed, args.scenario_profile)
            backend = MujocoPulseDiagnosisBackend(
                ROOT,
                config_path=args.alignment,
                include_rgbd=True,
                rgbd_capture_interval=(
                    args.capture_interval
                    if args.rgbd_capture_interval is None
                    else args.rgbd_capture_interval
                ),
                maximum_steps=9000,
            )
            backend.config["task"]["m5_cartesian_fine_approach_enabled"] = bool(
                args.m5_cartesian_fine_approach
            )
            backend.config["task"]["m5_joint_waypoint_shadow_enabled"] = bool(
                args.m5_joint_waypoint_shadow
            )
            backend.config["task"]["m5_joint_waypoint_protected_enabled"] = bool(
                args.m5_joint_waypoint_protected
            )
            if args.m5_joint_waypoint_duration_s is not None:
                backend.config["task"]["m5_joint_waypoint_duration_s"] = float(
                    args.m5_joint_waypoint_duration_s
                )
            configured_target_force_n = float(
                backend.config["task"]["pressure_target_force_n_per_finger"]
            )
            apply_collection_scenario(backend, scenario)
            if len(backend.diagnostic_finger_names) > 2:
                # The CH5 three-finger pose was frozen at a deliberately
                # asymmetric 4 N command. The legacy two-finger Stage-3
                # scenario target can prevent the ring finger from satisfying
                # the pressure balance gate before search even begins.
                backend.config["task"]["pressure_target_force_n_per_finger"] = (
                    configured_target_force_n
                )
            perturbations = collection_perturbation_scheduler(scenario)
            if args.disable_proximity:
                perturbations = PerturbationScheduler(
                    perturbations.events
                    + (
                        PerturbationEvent(
                            "proximity_hardware_absent",
                            PerturbationKind.PROXIMITY_DROPOUT,
                            0,
                            20_000,
                            {"channels": [0, 1, 2, 3, 4]},
                            recoverable=False,
                        ),
                    )
                )
            env = PulseDiagnosisEnv(
                backend,
                perturbation_scheduler=perturbations,
            )
            hook = _maybe_create_dagger_hook(
                disable_policy_shadow=args.disable_policy_shadow,
                checkpoint=args.checkpoint,
                specialist_checkpoints=specialist_checkpoints,
                shadow_path=shadow_path,
                env=env,
                recorder=recorder,
                device=args.device,
                policy_mix=args.policy_mix,
                seed=args.seed + index,
                policy_scope=args.policy_scope,
                search_direction_blend=args.search_direction_blend,
                policy_runtime=args.policy_runtime,
                unified_scope=args.unified_scope,
                unified_checkpoint=(
                    args.checkpoint if args.unified_scope is not None else None
                ),
                policy_execution_budget_steps=(
                    args.policy_execution_budget_steps
                ),
                policy_execution_budget_scope=(
                    args.policy_execution_budget_scope
                ),
            )
            step_callback = recorder if hook is None else hook
            if args.segment_policy_mode != "disabled":
                segment_hook = SearchSegmentShadowHook(
                    load_search_segment_runtime(
                        args.segment_checkpoint,
                        device=args.device,
                    ),
                    downstream=step_callback,
                )
                step_callback = segment_hook
            run = MujocoPulseExpert(
                env,
                on_step=step_callback,
                action_selector=(None if hook is None else hook.select_action),
                search_strategy="signal_quality_early_stop",
                privileged_search_assist=(
                    args.expert_privileged_search_assist
                ),
                search_segment_selector=(
                    None if segment_hook is None else segment_hook.propose
                ),
                search_segment_policy_mode=args.segment_policy_mode,
                search_segment_policy_execution_budget=(
                    args.segment_policy_execution_budget
                ),
                policy_history_primer=(
                    None
                    if not args.m5_joint_waypoint_protected or hook is None
                    else hook.prime_observation
                ),
                policy_history_warmup_steps=(
                    0
                    if not args.m5_joint_waypoint_protected or hook is None
                    # One horizon is sufficient to make the runtime ready,
                    # but the first slow-waypoint ticks still contain the
                    # deterministic joint-PD entry transient. Keep a second
                    # horizon under the verified path before policy takeover
                    # so the live context matches the settled teacher data.
                    else 2 * hook.runtime.observation_horizon
                ),
            ).run(seed=args.seed + index)
            if hook is not None:
                shadow_summary = hook.close(
                    backend.config["task"].get("action_scale", [
                        0.025, 0.042, 0.030, 0.112, 0.092, 0.086, 150.0
                    ])
                )
            recorder.close(
                run.result,
                extra={
                    "stage": "stage_3_dagger",
                    "alignment": str(args.alignment.resolve()),
                    "embodiment": (
                        "adam_pro_custom_right_hand_multifinger"
                        if "custom_right_hand_multifinger" in args.alignment.stem
                        else "adam_pro_official_right_hand"
                    ),
                    "configured_target_force_n_per_finger": (
                        configured_target_force_n
                    ),
                    "scenario": scenario.as_dict(),
                    "scenario_profile": args.scenario_profile,
                    "capture_interval_control_steps": int(
                        args.capture_interval
                    ),
                    "rgbd_capture_interval_control_steps": int(
                        args.capture_interval
                        if args.rgbd_capture_interval is None
                        else args.rgbd_capture_interval
                    ),
                    "proximity_hardware_present": not args.disable_proximity,
                    **_episode_policy_metadata(
                        collection_contract,
                        dagger_shadow=shadow_summary,
                    ),
                    "policy_scope": args.policy_scope,
                    "m5_cartesian_fine_approach": bool(
                        args.m5_cartesian_fine_approach
                    ),
                    "m5_joint_waypoint_shadow": bool(
                        args.m5_joint_waypoint_shadow
                    ),
                    "m5_joint_waypoint_protected": bool(
                        args.m5_joint_waypoint_protected
                    ),
                    "m5_joint_waypoint_duration_s": backend.config["task"].get(
                        "m5_joint_waypoint_duration_s"
                    ),
                    "search_direction_blend": float(
                        args.search_direction_blend
                    ),
                    "expert_privileged_search_assist": bool(
                        args.expert_privileged_search_assist
                    ),
                    "search_segment_policy": (
                        None
                        if segment_hook is None
                        else {
                            **segment_hook.summary(),
                            "mode": args.segment_policy_mode,
                            "execution_budget_segments": (
                                args.segment_policy_execution_budget
                            ),
                            "checkpoint": str(
                                args.segment_checkpoint.resolve()
                            ),
                        }
                    ),
                    **_search_episode_metadata(run.result),
                },
            )
        except Exception as error:
            run = None
            recorder_error = f"{type(error).__name__}: {error}"
            if recorder.arm_joint_dim is not None and not episode_dir.joinpath(
                "episode.json"
            ).exists():
                recorder.close(
                    {
                        "success": False,
                        "failure": recorder_error,
                        "signal_quality_accepted": False,
                    },
                    extra={
                        "stage": "stage_3_dagger",
                        **_episode_policy_metadata(
                            collection_contract,
                            dagger_shadow=None,
                        ),
                    },
                )
            else:
                recorder_error = recorder_error
        finally:
            if backend is not None:
                backend.close()
        if not episode_dir.joinpath("episode.json").exists():
            records.append(
                {"episode": episode_id, "accepted": False, "error": recorder_error}
            )
            continue
        validation = validate_episode(episode_dir)
        result = run.result if run is not None else {"success": False}
        accepted, reasons = collection_acceptance(result, validation)
        if not accepted:
            failed_dir.mkdir(exist_ok=True)
            destination = failed_dir / episode_dir.name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(episode_dir), str(destination))
            episode_name = str(destination.relative_to(args.output))
        else:
            episode_name = episode_dir.name
        shadow_metrics = shadow_summary or {}
        records.append(
            {
                "episode": episode_name,
                "accepted": bool(accepted),
                "reasons": reasons,
                "success": bool(result.get("success", False)),
                "signal_quality_accepted": bool(
                    result.get("signal_quality_accepted", False)
                ),
                "signal_quality": result.get("selected_signal_quality"),
                "signal_quality_thresholds": result.get(
                    "signal_quality_thresholds"
                ),
                "failure": result.get("failure"),
                "recovery_events": result.get("recovery_events", []),
                "policy_action_executed": bool(
                    shadow_metrics.get("policy_action_executed", False)
                ),
                "policy_action_frames": int(
                    shadow_metrics.get("policy_action_frames", 0)
                ),
                "controller": shadow_metrics.get("controller", {}),
                "forbidden_contact_events": int(
                    result.get("forbidden_contact_events", 0)
                ),
                "maximum_force_n": result.get(
                    "maximum_force_n_per_finger", []
                ),
                "maximum_force_n_per_finger": result.get(
                    "maximum_force_n_per_finger", []
                ),
                "safe_terminal": bool(
                    result.get("success", False)
                    or result.get("safe_terminal", False)
                ),
                "truth_in_observation": False,
                "search_positions": int(
                    result.get("unique_search_positions", 0)
                ),
                "evaluation_lateral_error_mm": result.get(
                    "evaluation_lateral_error_mm"
                ),
                "simulation_time_s": result.get("simulation_time_s"),
                "scenario": scenario.as_dict(),
                "proximity_hardware_present": not args.disable_proximity,
                "segment_policy": (
                    None
                    if segment_hook is None
                    else segment_hook.summary()
                ),
                "segment_policy_records": result.get(
                    "search_segment_policy_records", []
                ),
                "search_segment_policy_executions": int(
                    result.get("search_segment_policy_executions", 0)
                ),
                "expert_action_queries": int(
                    result.get("expert_action_queries", -1)
                ),
                "privileged_state_reads": int(
                    result.get("privileged_state_reads", -1)
                ),
                "posthoc_privileged_state_reads": int(
                    result.get("posthoc_privileged_state_reads", -1)
                ),
                "expert_direction_queries": int(
                    result.get("expert_direction_queries", -1)
                ),
                "search_expert_target_queries": int(
                    result.get("search_expert_target_queries", 0)
                ),
            }
        )
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    summary = {
        "schema_version": "adam_pro_stage3_dagger_collection_v1",
        "episodes": records,
        "episode_count": len(records),
        "accepted_count": sum(bool(item["accepted"]) for item in records),
        **collection_contract,
        "scenario_profile": args.scenario_profile,
        "capture_interval_control_steps": int(args.capture_interval),
        "rgbd_capture_interval_control_steps": int(
            args.capture_interval
            if args.rgbd_capture_interval is None
            else args.rgbd_capture_interval
        ),
        "policy_scope": args.policy_scope,
        "policy_runtime": args.policy_runtime,
        "unified_scope": args.unified_scope,
        "policy_execution_budget_steps": args.policy_execution_budget_steps,
        "policy_execution_budget_scope": args.policy_execution_budget_scope,
        "proximity_hardware_present": not args.disable_proximity,
        "m5_cartesian_fine_approach": bool(
            args.m5_cartesian_fine_approach
        ),
        "m5_joint_waypoint_shadow": bool(args.m5_joint_waypoint_shadow),
        "m5_joint_waypoint_protected": bool(
            args.m5_joint_waypoint_protected
        ),
        "m5_joint_waypoint_duration_s": (
            None
            if args.m5_joint_waypoint_duration_s is None
            else float(args.m5_joint_waypoint_duration_s)
        ),
        "search_direction_blend": float(args.search_direction_blend),
        "expert_privileged_search_assist": bool(
            args.expert_privileged_search_assist
        ),
        "segment_policy_mode": args.segment_policy_mode,
        "segment_policy_execution_budget": (
            args.segment_policy_execution_budget
        ),
        "segment_policy_execution_count": sum(
            int(item.get("search_segment_policy_executions", 0))
            for item in records
        ),
        "segment_checkpoint": (
            None
            if args.segment_checkpoint is None
            else str(args.segment_checkpoint.resolve())
        ),
        "segment_policy_proposals": sum(
            int((item.get("segment_policy") or {}).get("proposals", 0))
            for item in records
        ),
        "policy_action_episode_count": sum(
            bool(item.get("policy_action_executed", False))
            for item in records
        ),
        "policy_action_frame_count": sum(
            int(item.get("policy_action_frames", 0)) for item in records
        ),
        "expert_action_queries": sum(
            int(item.get("expert_action_queries", -1)) for item in records
        ),
        "privileged_state_reads": sum(
            int(item.get("privileged_state_reads", -1)) for item in records
        ),
        "posthoc_privileged_state_reads": sum(
            int(item.get("posthoc_privileged_state_reads", -1))
            for item in records
        ),
        "expert_direction_queries": sum(
            int(item.get("expert_direction_queries", -1))
            for item in records
        ),
        "forbidden_contact_events": sum(
            int(item.get("forbidden_contact_events", 0))
            for item in records
        ),
    }
    (args.output / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
