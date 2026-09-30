from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Any

import cv2
import numpy as np
import torch

from pulse_mujoco.observation import PulseObservation
from pulse_mujoco.pulse_signal_quality import (
    PulseSignalQualityThresholds,
    PulseTactileCoverageThresholds,
    pulse_diagnosis_quality_gate_from_metrics,
)
from pulse_mujoco.task import TaskPhase

from .cuda import require_cuda
from .search_segment_baselines import (
    ACTSegmentPolicy,
    feature_frame_dim,
    multimodal_feature_frames,
)
from .search_segment_2d import (
    ACTSearchSegment2DPolicy,
    COMPACT_SPATIAL_FEATURE_CONTRACT,
    CompactTactileSpatialPolicy,
    FEATURE_VERSION_LEGACY,
    HierarchicalSearchSegment2DPolicy,
    RidgeSearchSegment2DPolicy,
    compact_spatial_feature_vectors,
    multifinger_feature_frame_dim,
    multifinger_feature_frames,
    ridge_feature_vectors,
)


CHECKPOINT_SCHEMA_VERSION = "adam_pulse_act_signal_quality_gate_checkpoint_v2"
CHECKPOINT_2D_SCHEMA_VERSION = "adam_pulse_ch5_act_segment_2d_checkpoint_v1"
CHECKPOINT_2D_RIDGE_SCHEMA_VERSION = (
    "adam_pulse_ch5_ridge_segment_2d_checkpoint_v1"
)
CHECKPOINT_2D_HIERARCHICAL_SCHEMA_VERSION = (
    "adam_pulse_ch5_hierarchical_segment_2d_checkpoint_v1"
)
CHECKPOINT_2D_COMPACT_HIERARCHICAL_SCHEMA_VERSION = (
    "adam_pulse_ch5_compact_spatial_hierarchical_2d_checkpoint_v1"
)


def _conservative_current_quality_gate(
    quality_metrics: Mapping[str, Any],
    *,
    signal_thresholds: PulseSignalQualityThresholds,
    spatial_thresholds: PulseTactileCoverageThresholds | None,
) -> tuple[bool, tuple[str, ...]]:
    """Intersect a frozen checkpoint gate with the current task contract.

    New task revisions may add observable per-finger centering checks that an
    older checkpoint did not embed.  Such checks may veto stopping but never
    grant a stop rejected by the frozen gate.
    """
    coverage = quality_metrics.get("tactile_coverage")
    stop, reasons = pulse_diagnosis_quality_gate_from_metrics(
        quality_metrics,
        signal_thresholds=signal_thresholds,
        spatial_metrics=coverage,
        spatial_thresholds=spatial_thresholds,
    )
    if quality_metrics.get("accepted") is False:
        current_reasons = tuple(
            str(reason)
            for reason in quality_metrics.get("rejection_reasons", ())
        )
        reasons = tuple(dict.fromkeys((*reasons, *current_reasons)))
        stop = False
    return bool(stop), tuple(reasons)


def load_search_segment_runtime(
    checkpoint: Path | str,
    *,
    device: str | torch.device | None = None,
) -> "ACTSignalQualitySegmentRuntime | ACTSearchSegment2DRuntime":
    """Load the matching legacy 1-D or CH5 2-D runtime by schema."""

    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema = payload.get("schema_version")
    if schema == CHECKPOINT_SCHEMA_VERSION:
        return ACTSignalQualitySegmentRuntime(path, device=device)
    if schema in {
        CHECKPOINT_2D_SCHEMA_VERSION,
        CHECKPOINT_2D_RIDGE_SCHEMA_VERSION,
        CHECKPOINT_2D_HIERARCHICAL_SCHEMA_VERSION,
        CHECKPOINT_2D_COMPACT_HIERARCHICAL_SCHEMA_VERSION,
    }:
        return ACTSearchSegment2DRuntime(path, device=device)
    raise ValueError(f"unsupported search-segment checkpoint schema: {schema}")


@dataclass(frozen=True)
class SearchSegmentRuntimePrediction:
    """One bounded model proposal plus the independent hard stop gate."""

    stop: bool
    stop_rejection_reasons: tuple[str, ...]
    signed_displacement_m: float
    model_signed_displacement_m: float
    model_stop_probability: float
    maximum_displacement_m: float
    quality_window_available: bool


@dataclass(frozen=True)
class SearchSegment2DRuntimePrediction:
    """One bounded patient-frame ``[lateral, axial]`` proposal."""

    stop: bool
    stop_rejection_reasons: tuple[str, ...]
    delta_uv_m: tuple[float, float]
    model_delta_uv_m: tuple[float, float]
    maximum_displacement_m: float
    quality_window_available: bool
    route: str = "single_policy"
    coarse_model_delta_uv_m: tuple[float, float] | None = None
    tail_model_delta_uv_m: tuple[float, float] | None = None


class ACTSignalQualitySegmentRuntime:
    """Deployable search-segment runtime using policy-visible sensor input.

    ACT predicts the next bounded lateral search segment.  Search termination
    is deliberately not inferred from its four-frame stop head: a deterministic
    gate evaluates the completed current-point tactile/contact/force window.
    This mirrors the task success definition and prevents a model from stopping
    on a visually plausible but physiologically unusable signal.
    """

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        payload = torch.load(
            self.checkpoint, map_location="cpu", weights_only=False
        )
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported ACT signal-quality checkpoint")
        if payload.get("truth_in_observation") is not False:
            raise ValueError("checkpoint does not prove a truth-free observation")
        if int(payload.get("feature_frame_dim", -1)) != feature_frame_dim():
            raise ValueError("checkpoint feature contract differs from runtime")
        requested_device = torch.device("cuda" if device is None else device)
        self.device = (
            require_cuda(purpose="signal-quality segment runtime")
            if requested_device.type == "cuda"
            else requested_device
        )
        self.model = ACTSegmentPolicy(**payload["model_config"]).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        runtime = payload["runtime_config"]
        self.observation_horizon = int(runtime["observation_horizon"])
        # Formal M1 train/validation samples were recorded every four policy
        # steps.  Older v2 artifacts predate this explicit field, so retain
        # the audited stride as their backward-compatible default.
        self.observation_stride_policy_steps = int(
            runtime.get("observation_stride_policy_steps", 4)
        )
        if self.observation_stride_policy_steps < 1:
            raise ValueError("observation stride must be positive")
        self.tactile_size = tuple(int(value) for value in runtime["tactile_size"])
        self.tactile_scale_kpa = float(runtime["tactile_scale_kpa"])
        self.sensor_range_m = float(runtime["sensor_range_m"])
        self.state_dim = int(runtime["state_dim"])
        self.state_scale = float(runtime["state_scale"])
        self.action_scale = np.asarray(runtime["action_scale"], dtype=np.float32)
        self.maximum_displacement_m = float(payload["maximum_displacement_m"])
        if self.action_scale.shape != (7,):
            raise ValueError("segment runtime action scale must have shape (7,)")
        if self.observation_horizon != int(payload["model_config"]["maximum_horizon"]):
            raise ValueError("runtime and ACT history horizons differ")
        self.quality_thresholds = PulseSignalQualityThresholds.from_mapping(
            payload["stop_gate"]["thresholds"]
        )
        spatial_gate = payload["stop_gate"].get("spatial_gate", {})
        self.spatial_quality_thresholds = (
            PulseTactileCoverageThresholds.from_mapping(
                spatial_gate.get("thresholds")
            )
            if spatial_gate.get("enabled", False)
            else None
        )
        self.history: deque[dict[str, torch.Tensor]] = deque(
            maxlen=self.observation_horizon
        )

    def reset(self) -> None:
        self.history.clear()

    def _tactile(self, tactile_kpa: np.ndarray) -> torch.Tensor:
        tactile = np.asarray(tactile_kpa, dtype=np.float32)
        valid = bool(tactile.ndim == 2 and np.isfinite(tactile).all())
        if not valid:
            return torch.zeros(1, *self.tactile_size)
        resized = cv2.resize(
            tactile,
            self.tactile_size[::-1],
            interpolation=cv2.INTER_AREA,
        )
        return torch.from_numpy(
            np.clip(resized / self.tactile_scale_kpa, -1.0, 1.0)[None]
        ).float()

    def append(
        self,
        *,
        tactile_kpa,
        proximity_m,
        proximity_valid,
        robot_state,
        previous_action,
        task_phase: int,
    ) -> None:
        tactile = np.asarray(tactile_kpa, dtype=np.float32)
        tactile_valid = bool(tactile.ndim == 2 and np.isfinite(tactile).all())
        proximity = np.asarray(proximity_m, dtype=object)
        valid = np.asarray(proximity_valid, dtype=bool)
        state = np.asarray(robot_state, dtype=np.float32)
        previous = np.asarray(previous_action, dtype=np.float32)
        if proximity.shape != (5,) or valid.shape != (5,):
            raise ValueError("proximity and validity must have shape (5,)")
        if state.shape != (self.state_dim,):
            raise ValueError(f"robot_state must have shape ({self.state_dim},)")
        if previous.shape != (7,):
            raise ValueError("previous_action must have shape (7,)")
        filled = np.asarray(
            [
                self.sensor_range_m
                if value is None or not np.isfinite(float(value))
                else float(value)
                for value in proximity
            ],
            dtype=np.float32,
        )
        self.history.append(
            {
                "tactile": self._tactile(tactile),
                "tactile_valid": torch.tensor(tactile_valid),
                "proximity": torch.from_numpy(
                    np.clip(filled / self.sensor_range_m, 0.0, 1.0) * 2.0 - 1.0
                ).float(),
                "proximity_valid": torch.from_numpy(valid),
                "robot_state": torch.from_numpy(state / self.state_scale).float(),
                "action": torch.from_numpy(
                    np.clip(previous / self.action_scale, -1.0, 1.0)
                ).float(),
                "task_phase": torch.tensor(int(task_phase), dtype=torch.long),
            }
        )

    @staticmethod
    def robot_state_from_observation(observation: PulseObservation) -> np.ndarray:
        phase_scale = float(max(int(phase) for phase in TaskPhase))
        return np.concatenate(
            (
                np.asarray(observation.joint_position_rad, dtype=np.float32),
                np.asarray(observation.tcp_pose, dtype=np.float32),
                np.asarray(
                    [float(observation.task_phase) / phase_scale],
                    dtype=np.float32,
                ),
            )
        )

    def append_observation(self, observation: PulseObservation) -> None:
        state = self.robot_state_from_observation(observation)
        self.append(
            tactile_kpa=observation.tactile_kpa,
            proximity_m=observation.proximity_m,
            proximity_valid=observation.proximity_valid,
            robot_state=state,
            previous_action=observation.previous_action,
            task_phase=int(observation.task_phase),
        )

    def append_live(self, backend, observation: Mapping[str, Any]) -> None:
        """Append one recorder-style live frame (legacy 2-finger view)."""

        del backend
        phase_scale = float(max(int(phase) for phase in TaskPhase))
        state = np.concatenate(
            (
                np.asarray(observation["joint_position_rad"], dtype=np.float32),
                np.asarray(observation["tcp_pose"], dtype=np.float32),
                np.asarray(
                    [float(observation["task_phase"]) / phase_scale],
                    dtype=np.float32,
                ),
            )
        )
        self.append(
            tactile_kpa=observation["tactile_kpa"],
            proximity_m=observation["proximity_m"],
            proximity_valid=observation["proximity_valid"],
            robot_state=state,
            previous_action=observation["previous_action"],
            task_phase=int(observation["task_phase"]),
        )

    @property
    def ready(self) -> bool:
        return len(self.history) == self.observation_horizon

    def _batch(self) -> dict[str, torch.Tensor]:
        if not self.ready:
            raise RuntimeError(
                f"need {self.observation_horizon} frames, have {len(self.history)}"
            )
        return {
            key: torch.stack([frame[key] for frame in self.history])
            .unsqueeze(0)
            .to(self.device)
            for key in self.history[0]
        }

    @torch.no_grad()
    def predict(
        self,
        quality_metrics: Mapping[str, Any] | None = None,
        *,
        point_index: int | None = None,
    ) -> SearchSegmentRuntimePrediction:
        del point_index
        output = self.model(multimodal_feature_frames(self._batch()))
        model_signed = float(
            output.signed_displacement_normalized[0].cpu()
            * self.maximum_displacement_m
        )
        model_stop = float(output.stop_probability[0].cpu())
        if not np.isfinite(model_signed) or not np.isfinite(model_stop):
            raise RuntimeError("segment model produced non-finite output")
        if abs(model_signed) > self.maximum_displacement_m + 1e-9:
            raise RuntimeError("segment model exceeded its bounded output contract")
        if quality_metrics is None:
            stop = False
            rejection_reasons = ("quality_window_unavailable",)
        else:
            stop, rejection_reasons = _conservative_current_quality_gate(
                quality_metrics,
                signal_thresholds=self.quality_thresholds,
                spatial_thresholds=self.spatial_quality_thresholds,
            )
        return SearchSegmentRuntimePrediction(
            stop=bool(stop),
            stop_rejection_reasons=tuple(rejection_reasons),
            signed_displacement_m=(0.0 if stop else model_signed),
            model_signed_displacement_m=model_signed,
            model_stop_probability=model_stop,
            maximum_displacement_m=self.maximum_displacement_m,
            quality_window_available=quality_metrics is not None,
        )


class ACTSearchSegment2DRuntime:
    """CH5 three-finger 2-D search runtime without proximity dependency."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        schema = payload.get("schema_version")
        if schema not in {
            CHECKPOINT_2D_SCHEMA_VERSION,
            CHECKPOINT_2D_RIDGE_SCHEMA_VERSION,
            CHECKPOINT_2D_HIERARCHICAL_SCHEMA_VERSION,
            CHECKPOINT_2D_COMPACT_HIERARCHICAL_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported CH5 2-D segment checkpoint")
        if payload.get("truth_in_observation") is not False:
            raise ValueError("checkpoint does not prove a truth-free observation")
        self.ridge = schema == CHECKPOINT_2D_RIDGE_SCHEMA_VERSION
        self.compact_spatial_hierarchical = (
            schema == CHECKPOINT_2D_COMPACT_HIERARCHICAL_SCHEMA_VERSION
        )
        self.hierarchical = schema in {
            CHECKPOINT_2D_HIERARCHICAL_SCHEMA_VERSION,
            CHECKPOINT_2D_COMPACT_HIERARCHICAL_SCHEMA_VERSION,
        }
        self.spatial_rejection_tail_reasons: frozenset[str] = frozenset()
        self.spatial_rejection_tail_max_point_index: int | None = None
        self.minimum_spatial_coarse_cosine = -1.0
        model_config = dict(payload.get("model_config", {}))
        if self.hierarchical:
            model_config = dict(payload["coarse_policy"]["model_config"])
        self.include_proximity = bool(
            payload.get("include_proximity", model_config.get("include_proximity", False))
        )
        if bool(
            payload.get("proximity_ablation", {}).get(
                "included_in_policy_input", self.include_proximity
            )
        ) != self.include_proximity:
            raise ValueError("checkpoint proximity contract is inconsistent")
        if self.ridge:
            if payload.get("feature_contract") != "tactile_window_physiology_24d_v1":
                raise ValueError("unsupported CH5 ridge feature contract")
            if self.include_proximity:
                raise ValueError("CH5 ridge runtime forbids proximity input")
        else:
            self.feature_version = str(
                model_config.get("feature_version", FEATURE_VERSION_LEGACY)
            )
            expected_dim = multifinger_feature_frame_dim(
                include_proximity=self.include_proximity,
                feature_version=self.feature_version,
            )
            if int(model_config.get("feature_dim", -1)) != expected_dim:
                raise ValueError("checkpoint feature contract differs from CH5 runtime")
        requested_device = torch.device("cuda" if device is None else device)
        self.device = (
            require_cuda(purpose="CH5 search-segment runtime")
            if requested_device.type == "cuda"
            else requested_device
        )
        if self.ridge:
            self.model = RidgeSearchSegment2DPolicy(
                payload["feature_mean"], payload["feature_scale"], payload["weights"]
            ).to(self.device)
        elif self.compact_spatial_hierarchical:
            coarse_payload = payload["coarse_policy"]
            specialist_payload = payload["compact_spatial_policy"]

            def _build_compact_spatial_model(model_payload):
                if model_payload.get("feature_contract") != (
                    COMPACT_SPATIAL_FEATURE_CONTRACT
                ):
                    raise ValueError("unsupported compact spatial feature contract")
                if model_payload.get("truth_in_observation") is not False:
                    raise ValueError("compact spatial specialist is not truth-free")
                if model_payload.get("include_proximity") is not False:
                    raise ValueError("compact spatial specialist must exclude proximity")
                return CompactTactileSpatialPolicy(
                    feature_indices=model_payload["feature_indices"],
                    feature_mean=model_payload["feature_mean"],
                    feature_scale=model_payload["feature_scale"],
                    axis_class_weights=model_payload["axis_class_weights"],
                    step_radius_normalized=float(
                        model_payload["step_radius_normalized"]
                    ),
                ).to(self.device)

            coarse = ACTSearchSegment2DPolicy(**coarse_payload["model_config"])
            coarse.load_state_dict(coarse_payload["model"], strict=True)
            specialist = _build_compact_spatial_model(specialist_payload)
            firststate_payload = payload.get("compact_spatial_firststate_policy")
            late_payload = payload.get("compact_spatial_late_policy")
            self.compact_spatial_firststate_model = (
                _build_compact_spatial_model(firststate_payload)
                if firststate_payload is not None
                else None
            )
            self.compact_spatial_late_model = (
                _build_compact_spatial_model(late_payload)
                if late_payload is not None
                else specialist
            )
            routing = payload["routing"]
            self.minimum_spatial_coarse_cosine = float(
                routing.get("minimum_spatial_coarse_cosine", -1.0)
            )
            if not -1.0 <= self.minimum_spatial_coarse_cosine <= 1.0:
                raise ValueError("minimum spatial/coarse cosine must be in [-1,1]")
            self.spatial_rejection_tail_reasons = frozenset(
                str(value)
                for value in routing.get("spatial_rejection_tail_reasons", ())
            )
            if not self.spatial_rejection_tail_reasons:
                raise ValueError("compact spatial routing requires rejection reasons")
            spatial_max_point = routing.get(
                "spatial_rejection_tail_max_point_index"
            )
            spatial_min_point = routing.get(
                "spatial_rejection_tail_min_point_index", 0
            )
            self.spatial_rejection_tail_min_point_index = int(spatial_min_point)
            if self.spatial_rejection_tail_min_point_index < 0:
                raise ValueError(
                    "compact spatial minimum point limit must be non-negative"
                )
            self.spatial_rejection_tail_max_point_index = (
                None if spatial_max_point is None else int(spatial_max_point)
            )
            if (
                self.spatial_rejection_tail_max_point_index is not None
                and self.spatial_rejection_tail_max_point_index < 0
            ):
                raise ValueError(
                    "compact spatial point limit must be non-negative"
                )
            if (
                self.spatial_rejection_tail_max_point_index is not None
                and self.spatial_rejection_tail_min_point_index
                > self.spatial_rejection_tail_max_point_index
            ):
                raise ValueError("compact spatial point limits are inconsistent")
            self.coarse_model = coarse.to(self.device)
            self.compact_spatial_model = specialist
            self.coarse_observation_horizon = int(
                coarse_payload["model_config"]["maximum_horizon"]
            )
            self.model = None
        elif self.hierarchical:
            coarse_payload = payload["coarse_policy"]
            tail_payload = payload["tail_policy"]
            coarse = ACTSearchSegment2DPolicy(
                **coarse_payload["model_config"]
            )
            tail = ACTSearchSegment2DPolicy(**tail_payload["model_config"])
            coarse.load_state_dict(coarse_payload["model"], strict=True)
            tail.load_state_dict(tail_payload["model"], strict=True)
            routing = payload["routing"]
            self.spatial_rejection_tail_reasons = frozenset(
                str(value)
                for value in routing.get("spatial_rejection_tail_reasons", ())
            )
            spatial_max_point = routing.get(
                "spatial_rejection_tail_max_point_index"
            )
            self.spatial_rejection_tail_max_point_index = (
                None if spatial_max_point is None else int(spatial_max_point)
            )
            if (
                self.spatial_rejection_tail_max_point_index is not None
                and self.spatial_rejection_tail_max_point_index < 0
            ):
                raise ValueError(
                    "hierarchical spatial tail point limit must be non-negative"
                )
            maximum = float(payload["maximum_displacement_m"])
            self.model = HierarchicalSearchSegment2DPolicy(
                coarse,
                tail,
                unconditional_tail_point_index=int(
                    routing["unconditional_tail_point_index"]
                ),
                conditional_tail_point_index=int(
                    routing["conditional_tail_point_index"]
                ),
                conditional_coarse_norm_max_normalized=(
                    float(routing["conditional_coarse_norm_max_m"]) / maximum
                ),
                tail_magnitude_bias_normalized=(
                    float(routing["tail_magnitude_bias_m"]) / maximum
                ),
            ).to(self.device)
        else:
            self.model = ACTSearchSegment2DPolicy(**model_config).to(self.device)
            self.model.load_state_dict(payload["model"])
        if self.compact_spatial_hierarchical:
            self.coarse_model.eval()
            self.compact_spatial_model.eval()
        else:
            self.model.eval()
        runtime_config = dict(payload.get("runtime_config", {}))
        self.observation_horizon = int(
            runtime_config.get(
                "observation_horizon", model_config.get("maximum_horizon", -1)
            )
        )
        self.observation_stride_policy_steps = int(
            runtime_config.get("observation_stride_policy_steps", 2)
        )
        self.tactile_size = tuple(
            int(value) for value in runtime_config.get("tactile_size", (32, 32))
        )
        self.tactile_scale_kpa = float(
            runtime_config.get("tactile_scale_kpa", 320.0)
        )
        self.sensor_range_m = float(runtime_config.get("sensor_range_m", 0.03))
        self.state_dim = int(runtime_config.get("state_dim", 15))
        self.state_scale = float(runtime_config.get("state_scale", np.pi))
        self.action_scale = np.asarray(
            runtime_config.get(
                "action_scale", (0.025, 0.042, 0.030, 0.112, 0.092, 0.086, 150.0)
            ),
            dtype=np.float32,
        )
        if self.compact_spatial_hierarchical:
            if self.observation_horizon < self.coarse_observation_horizon:
                raise ValueError("compact runtime history is shorter than coarse history")
        elif not self.ridge and self.observation_horizon != int(model_config["maximum_horizon"]):
            raise ValueError("runtime and CH5 model history horizons differ")
        if self.observation_stride_policy_steps < 1:
            raise ValueError("observation stride must be positive")
        if self.action_scale.shape != (7,):
            raise ValueError("segment runtime action scale must have shape (7,)")
        self.maximum_displacement_m = float(payload["maximum_displacement_m"])
        stop_gate = payload.get("stop_gate")
        if stop_gate is None:
            dataset_index = Path(str(payload.get("dataset_index", "")))
            if not dataset_index.exists():
                raise ValueError(
                    "portable CH5 runtime checkpoint requires embedded stop_gate"
                )
            import json

            stop_gate = json.loads(dataset_index.read_text(encoding="utf-8"))[
                "stop_gate"
            ]
        self.quality_thresholds = PulseSignalQualityThresholds.from_mapping(
            stop_gate.get("thresholds")
        )
        spatial_gate = stop_gate.get("spatial_gate", {})
        self.spatial_quality_thresholds = (
            PulseTactileCoverageThresholds.from_mapping(
                spatial_gate.get("thresholds")
            )
            if spatial_gate.get("enabled", False)
            else None
        )
        self.history: deque[dict[str, torch.Tensor]] = deque(
            maxlen=self.observation_horizon
        )

    def reset(self) -> None:
        self.history.clear()

    def _tactile(self, tactile_kpa: np.ndarray) -> torch.Tensor:
        tactile = np.asarray(tactile_kpa, dtype=np.float32)
        if tactile.shape != (3, 10) or not np.isfinite(tactile).all():
            raise ValueError("CH5 active tactile view must have finite shape (3, 10)")
        resized = cv2.resize(
            tactile, self.tactile_size[::-1], interpolation=cv2.INTER_LINEAR
        )
        return torch.from_numpy(
            np.clip(resized / self.tactile_scale_kpa, -1.0, 1.0)[None]
        ).float()

    def append(
        self,
        *,
        tactile_active_kpa,
        proximity_m,
        proximity_valid,
        robot_state,
        previous_action,
        task_phase: int,
    ) -> None:
        state = np.asarray(robot_state, dtype=np.float32)
        previous = np.asarray(previous_action, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"robot_state must have shape ({self.state_dim},)")
        if previous.shape != (7,):
            raise ValueError("previous_action must have shape (7,)")
        frame = {
            "tactile": self._tactile(tactile_active_kpa),
            "tactile_valid": torch.tensor(True),
            "robot_state": torch.from_numpy(state / self.state_scale).float(),
            "action": torch.from_numpy(
                np.clip(previous / self.action_scale, -1.0, 1.0)
            ).float(),
            "task_phase": torch.tensor(int(task_phase), dtype=torch.long),
        }
        if self.include_proximity:
            proximity = np.asarray(proximity_m, dtype=object)
            valid = np.asarray(proximity_valid, dtype=bool)
            if proximity.shape != (5,) or valid.shape != (5,):
                raise ValueError("proximity and validity must have shape (5,)")
            filled = np.asarray(
                [
                    self.sensor_range_m
                    if value is None or not np.isfinite(float(value))
                    else float(value)
                    for value in proximity
                ],
                dtype=np.float32,
            )
            frame.update(
                {
                    "proximity": torch.from_numpy(
                        np.clip(filled / self.sensor_range_m, 0.0, 1.0) * 2.0
                        - 1.0
                    ).float(),
                    "proximity_valid": torch.from_numpy(valid),
                }
            )
        self.history.append(frame)

    @staticmethod
    def _active_tactile_from_backend(backend) -> np.ndarray:
        frame = getattr(backend, "last_tactile_full", None)
        if frame is None:
            raise RuntimeError("CH5 runtime requires the full-hand tactile frame")
        names = tuple(str(name) for name in backend.diagnostic_finger_names)
        if names != ("index", "middle", "ring"):
            raise RuntimeError(
                "CH5 2-D checkpoint requires active fingers index/middle/ring"
            )
        pressure = np.asarray(frame.pressure_kpa, dtype=np.float32)
        return np.stack([pressure[frame.finger_slice(name)] for name in names])

    def append_live(self, backend, observation: Mapping[str, Any]) -> None:
        phase_scale = float(max(int(phase) for phase in TaskPhase))
        state = np.concatenate(
            (
                np.asarray(observation["joint_position_rad"], dtype=np.float32),
                np.asarray(observation["tcp_pose"], dtype=np.float32),
                np.asarray(
                    [float(observation["task_phase"]) / phase_scale],
                    dtype=np.float32,
                ),
            )
        )
        self.append(
            tactile_active_kpa=self._active_tactile_from_backend(backend),
            proximity_m=observation.get("proximity_m", (None,) * 5),
            proximity_valid=observation.get(
                "proximity_valid", np.zeros(5, dtype=bool)
            ),
            robot_state=state,
            previous_action=observation["previous_action"],
            task_phase=int(observation["task_phase"]),
        )

    @property
    def ready(self) -> bool:
        return len(self.history) == self.observation_horizon

    def _batch(self) -> dict[str, torch.Tensor]:
        if not self.ready:
            raise RuntimeError(
                f"need {self.observation_horizon} frames, have {len(self.history)}"
            )
        return {
            key: torch.stack([frame[key] for frame in self.history])
            .unsqueeze(0)
            .to(self.device)
            for key in self.history[0]
        }

    @torch.no_grad()
    def predict(
        self,
        quality_metrics: Mapping[str, Any] | None = None,
        *,
        point_index: int | None = None,
    ) -> SearchSegment2DRuntimePrediction:
        batch = self._batch()
        if quality_metrics is None:
            stop = False
            rejection_reasons = ("quality_window_unavailable",)
        else:
            stop, rejection_reasons = _conservative_current_quality_gate(
                quality_metrics,
                signal_thresholds=self.quality_thresholds,
                spatial_thresholds=self.spatial_quality_thresholds,
            )
        if self.compact_spatial_hierarchical:
            if point_index is None or int(point_index) < 0:
                raise ValueError(
                    "compact hierarchical runtime requires a non-negative point_index"
                )
            coarse_batch = {
                key: value[:, -self.coarse_observation_horizon :]
                for key, value in batch.items()
            }
            coarse_features = multifinger_feature_frames(
                coarse_batch,
                include_proximity=self.include_proximity,
                feature_version=self.feature_version,
            )
            coarse_output = self.coarse_model(coarse_features)
            compact_features = compact_spatial_feature_vectors(batch)
            spatial_tail = bool(
                self.spatial_rejection_tail_reasons.intersection(rejection_reasons)
            )
            spatial_tail = spatial_tail and int(point_index) >= (
                self.spatial_rejection_tail_min_point_index
            )
            if (
                spatial_tail
                and self.spatial_rejection_tail_max_point_index is not None
            ):
                spatial_tail = int(point_index) <= self.spatial_rejection_tail_max_point_index
            selected_spatial_model = self.compact_spatial_model
            if spatial_tail and int(point_index) == 0 and self.compact_spatial_firststate_model is not None:
                selected_spatial_model = self.compact_spatial_firststate_model
            elif spatial_tail and self.compact_spatial_late_model is not None:
                selected_spatial_model = self.compact_spatial_late_model
            compact_output = selected_spatial_model(compact_features)
            output = None
        else:
            features = (
                ridge_feature_vectors(batch, include_proximity=False)
                if self.ridge
                else multifinger_feature_frames(
                    batch,
                    include_proximity=self.include_proximity,
                    feature_version=self.feature_version,
                )
            )
        if self.compact_spatial_hierarchical:
            pass
        elif self.hierarchical:
            if point_index is None or int(point_index) < 0:
                raise ValueError(
                    "hierarchical CH5 runtime requires a non-negative point_index"
                )
            output = self.model(
                features,
                torch.as_tensor([int(point_index)], device=self.device),
            )
        elif not self.compact_spatial_hierarchical:
            output = self.model(features)
        route = "single_policy"
        coarse_delta = None
        tail_delta = None
        if self.compact_spatial_hierarchical:
            if point_index is None or int(point_index) < 0:
                raise ValueError(
                    "compact hierarchical runtime requires a non-negative point_index"
                )
            coarse_array = (
                coarse_output.delta_uv_normalized[0].cpu().numpy().astype(float)
                * self.maximum_displacement_m
            )
            tail_array = (
                compact_output.delta_uv_normalized[0].cpu().numpy().astype(float)
                * self.maximum_displacement_m
            )
            coarse_delta = (float(coarse_array[0]), float(coarse_array[1]))
            tail_delta = (float(tail_array[0]), float(tail_array[1]))
            if spatial_tail and int(point_index) == 0 and self.compact_spatial_firststate_model is not None:
                route = "compact_spatial_firststate"
            else:
                route = "compact_spatial_rejection" if spatial_tail else "coarse"
            model_delta = tail_array if spatial_tail else coarse_array
            if spatial_tail and self.minimum_spatial_coarse_cosine > -1.0:
                coarse_norm = float(np.linalg.norm(coarse_array))
                tail_norm = float(np.linalg.norm(tail_array))
                cosine = (
                    float(np.dot(coarse_array, tail_array) / (coarse_norm * tail_norm))
                    if coarse_norm > 1e-12 and tail_norm > 1e-12
                    else -1.0
                )
                if cosine < self.minimum_spatial_coarse_cosine:
                    route = "coarse"
                    model_delta = coarse_array
        elif self.hierarchical:
            coarse_array = (
                output.coarse_delta_uv_normalized[0].cpu().numpy().astype(float)
                * self.maximum_displacement_m
            )
            tail_array = (
                output.tail_delta_uv_normalized[0].cpu().numpy().astype(float)
                * self.maximum_displacement_m
            )
            coarse_delta = (float(coarse_array[0]), float(coarse_array[1]))
            tail_delta = (float(tail_array[0]), float(tail_array[1]))
            spatial_tail = bool(
                self.spatial_rejection_tail_reasons.intersection(
                    rejection_reasons
                )
            )
            if (
                spatial_tail
                and self.spatial_rejection_tail_max_point_index is not None
            ):
                spatial_tail = int(point_index) <= (
                    self.spatial_rejection_tail_max_point_index
                )
            tail_route = bool(output.tail_route[0]) or spatial_tail
            route = (
                "tail_spatial_rejection"
                if spatial_tail
                else "tail_discrete" if tail_route else "coarse"
            )
            model_delta = tail_array if tail_route else coarse_array
        else:
            model_delta = (
                output.delta_uv_normalized[0].cpu().numpy().astype(float)
                * self.maximum_displacement_m
            )
        if not np.isfinite(model_delta).all():
            raise RuntimeError("CH5 segment model produced non-finite output")
        if np.linalg.norm(model_delta) > self.maximum_displacement_m + 1e-9:
            raise RuntimeError("CH5 segment model exceeded its bounded output contract")
        executed = np.zeros(2, dtype=float) if stop else model_delta
        return SearchSegment2DRuntimePrediction(
            stop=bool(stop),
            stop_rejection_reasons=tuple(rejection_reasons),
            delta_uv_m=(float(executed[0]), float(executed[1])),
            model_delta_uv_m=(float(model_delta[0]), float(model_delta[1])),
            maximum_displacement_m=self.maximum_displacement_m,
            quality_window_available=quality_metrics is not None,
            route=route,
            coarse_model_delta_uv_m=coarse_delta,
            tail_model_delta_uv_m=tail_delta,
        )


class SearchSegmentShadowHook:
    """Feed live policy-visible frames to a segment runtime and recorder."""

    def __init__(
        self,
        runtime: ACTSignalQualitySegmentRuntime | ACTSearchSegment2DRuntime,
        *,
        downstream=None,
    ) -> None:
        self.runtime = runtime
        self.downstream = downstream
        self.frames = 0
        self.callback_steps = 0
        self.proposals = 0

    def __call__(self, backend, observation: dict, info: dict, label: str) -> None:
        self.callback_steps += 1
        if (
            self.callback_steps
            % self.runtime.observation_stride_policy_steps
            == 0
        ):
            self.runtime.append_live(backend, observation)
            self.frames += 1
        if self.downstream is not None:
            self.downstream(backend, observation, info, label)

    def propose(self, sample: dict) -> dict:
        if not self.runtime.ready:
            raise RuntimeError("segment shadow runtime has insufficient live history")
        prediction = self.runtime.predict(
            sample.get("signal_quality"),
            point_index=sample.get("point_index"),
        )
        self.proposals += 1
        return asdict(prediction)

    def save_state(self) -> dict[str, Any]:
        """Capture live history and stride phase for deterministic branching."""

        return {
            "schema_version": "pulse_search_segment_shadow_state_v1",
            "observation_horizon": int(self.runtime.observation_horizon),
            "observation_stride_policy_steps": int(
                self.runtime.observation_stride_policy_steps
            ),
            "frames": int(self.frames),
            "callback_steps": int(self.callback_steps),
            "proposals": int(self.proposals),
            "history": [
                {
                    key: value.detach().cpu().clone()
                    for key, value in frame.items()
                }
                for frame in self.runtime.history
            ],
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`save_state` fail-closed."""

        if state.get("schema_version") != "pulse_search_segment_shadow_state_v1":
            raise ValueError("unsupported search segment shadow state")
        if int(state.get("observation_horizon", -1)) != int(
            self.runtime.observation_horizon
        ):
            raise ValueError("search segment shadow horizon differs")
        if int(state.get("observation_stride_policy_steps", -1)) != int(
            self.runtime.observation_stride_policy_steps
        ):
            raise ValueError("search segment shadow stride differs")
        history = state.get("history")
        if not isinstance(history, list) or len(history) > int(
            self.runtime.observation_horizon
        ):
            raise ValueError("invalid search segment shadow history")
        restored: list[dict[str, torch.Tensor]] = []
        for frame in history:
            if not isinstance(frame, Mapping) or not frame:
                raise ValueError("invalid search segment shadow frame")
            if not all(isinstance(value, torch.Tensor) for value in frame.values()):
                raise ValueError("search segment shadow frame must contain tensors")
            restored.append(
                {
                    str(key): value.detach().cpu().clone()
                    for key, value in frame.items()
                }
            )
        callback_steps = int(state.get("callback_steps", -1))
        frames = int(state.get("frames", -1))
        proposals = int(state.get("proposals", -1))
        if min(callback_steps, frames, proposals) < 0 or frames < len(restored):
            raise ValueError("invalid search segment shadow counters")
        self.runtime.history.clear()
        self.runtime.history.extend(restored)
        self.callback_steps = callback_steps
        self.frames = frames
        self.proposals = proposals

    def summary(self) -> dict:
        return {
            "schema_version": "pulse_search_segment_shadow_runtime_v1",
            "frames": int(self.frames),
            "callback_steps": int(self.callback_steps),
            "observation_stride_policy_steps": int(
                self.runtime.observation_stride_policy_steps
            ),
            "proposals": int(self.proposals),
            "truth_in_observation": False,
            "ready": bool(self.runtime.ready),
        }
