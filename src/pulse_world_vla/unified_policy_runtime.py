from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from pulse_mujoco.observation import PulseObservation
from pulse_mujoco.tactile_normalization import (
    FULL56_CHANNEL_COUNTS,
    FULL56_FINGER_NAMES,
    TactileNormalization,
)
from pulse_mujoco.task import TaskPhase

from .benchmark import PhaseRoutedPolicy, load_policy_checkpoint
from .cuda import require_cuda
from .contact_descent import (
    ContactDescentHead,
    compose_contact_descent_z,
    contact_descent_features_from_observation,
)
from .contact_balance import (
    ContactBalanceRotationHead,
    compose_contact_balance_rotation,
    contact_balance_features_from_observation,
)
from .direct_policy import ShadowPrediction, chunk_action_to_control_tick
from .world_policy_segment_adapter import WorldPolicySegmentAdapter


class UnifiedPolicyRuntime:
    """Online adapter for unified full56 benchmark checkpoints.

    The runtime accepts only policy-visible RGB-D, tactile, force, robot state,
    previous action and task phase.  Pulse truth, event labels and expert
    actions are not part of this boundary.
    """

    def __init__(
        self,
        default_checkpoint: Path | str,
        *,
        specialist_checkpoint: Path | str | None = None,
        specialist_phases: tuple[int | TaskPhase, ...] | None = None,
        device: str | torch.device | None = None,
        control_substeps: int = 1,
        image_size: tuple[int, int] = (64, 64),
        tactile_size: tuple[int, int] = (32, 32),
        sensor_range_m: float = 0.03,
        tactile_scale_kpa: float = 320.0,
        state_scale: float = float(np.pi),
        action_scale: tuple[float, ...] = (
            0.025,
            0.042,
            0.030,
            0.112,
            0.092,
            0.086,
            150.0,
        ),
        contact_descent_checkpoint: Path | str | None = None,
        contact_balance_rotation_checkpoint: Path | str | None = None,
    ) -> None:
        default, default_payload = load_policy_checkpoint(default_checkpoint)
        specialist = None
        specialist_payload = None
        if specialist_checkpoint is not None:
            specialist, specialist_payload = load_policy_checkpoint(
                specialist_checkpoint
            )
            model = PhaseRoutedPolicy(
                default,
                specialist,
                **(
                    {}
                    if specialist_phases is None
                    else {"specialist_phases": specialist_phases}
                ),
            )
        else:
            model = default
        requested_device = torch.device("cuda" if device is None else device)
        self.device = (
            require_cuda(purpose="unified-policy runtime")
            if requested_device.type == "cuda"
            else requested_device
        )
        self.model = model.to(self.device).eval()
        self.contact_descent_checkpoint = (
            None
            if contact_descent_checkpoint is None
            else str(Path(contact_descent_checkpoint).resolve())
        )
        self.contact_descent_head = None
        self.contact_descent_include_previous_action = False
        self.last_contact_descent: dict | None = None
        self.contact_balance_rotation_checkpoint = (
            None
            if contact_balance_rotation_checkpoint is None
            else str(Path(contact_balance_rotation_checkpoint).resolve())
        )
        self.contact_balance_rotation_head = None
        self.contact_balance_active_contact_counts: tuple[int, ...] = ()
        self.last_contact_balance: dict | None = None
        if contact_descent_checkpoint is not None:
            descent_payload = torch.load(
                contact_descent_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            if (
                descent_payload.get("schema_version")
                != "adam_pulse_contact_descent_head_v1"
            ):
                raise ValueError("unsupported contact-descent checkpoint")
            self.contact_descent_head = ContactDescentHead(
                **descent_payload["model_config"]
            )
            self.contact_descent_head.load_state_dict(
                descent_payload["model"], strict=True
            )
            self.contact_descent_head.to(self.device).eval()
            self.contact_descent_include_previous_action = bool(
                descent_payload["report"]["include_previous_action"]
            )
        if contact_balance_rotation_checkpoint is not None:
            balance_payload = torch.load(
                contact_balance_rotation_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            if (
                balance_payload.get("schema_version")
                != "adam_pulse_contact_balance_rotation_head_v1"
            ):
                raise ValueError("unsupported contact-balance checkpoint")
            self.contact_balance_rotation_head = ContactBalanceRotationHead(
                **balance_payload["model_config"]
            )
            self.contact_balance_rotation_head.load_state_dict(
                balance_payload["model"], strict=True
            )
            self.contact_balance_rotation_head.to(self.device).eval()
            self.contact_balance_active_contact_counts = tuple(
                int(value)
                for value in balance_payload["report"]["active_contact_counts"]
            )
        self.default_checkpoint = str(Path(default_checkpoint).resolve())
        self.specialist_checkpoint = (
            None
            if specialist_checkpoint is None
            else str(Path(specialist_checkpoint).resolve())
        )
        self.specialist_phases = (
            None
            if specialist_checkpoint is None
            else tuple(
                int(phase)
                for phase in (
                    (TaskPhase.PULSE_SEARCH, TaskPhase.RECENTER)
                    if specialist_phases is None
                    else specialist_phases
                )
            )
        )
        self.contract = default.contract
        if self.contract.tactile_view != "full56":
            raise ValueError("unified runtime requires a full56 checkpoint")
        self.normalization = TactileNormalization.from_dict(
            default_payload["normalization"]
        )
        if (
            self.normalization.finger_names != FULL56_FINGER_NAMES
            or self.normalization.channel_counts != FULL56_CHANNEL_COUNTS
        ):
            raise ValueError("unified checkpoint has an incompatible tactile topology")
        if specialist_payload is not None and (
            specialist_payload.get("normalization")
            != default_payload.get("normalization")
        ):
            raise ValueError("routed checkpoints must share tactile normalization")
        self.observation_horizon = int(self.contract.observation_horizon)
        self.action_horizon = int(self.contract.action_horizon)
        self.state_dim = int(self.contract.state_dim)
        self.image_size = tuple(int(value) for value in image_size)
        self.tactile_size = tuple(int(value) for value in tactile_size)
        self.sensor_range_m = float(sensor_range_m)
        self.tactile_scale_kpa = float(tactile_scale_kpa)
        self.state_scale = float(state_scale)
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        self.control_substeps = int(control_substeps)
        configured_image_size = default_payload["model_config"].get("image_size")
        if configured_image_size is not None and self.image_size != tuple(
            configured_image_size
        ):
            raise ValueError("runtime image size does not match checkpoint")
        configured_tactile_size = default_payload["model_config"].get(
            "tactile_size"
        )
        if configured_tactile_size is not None and self.tactile_size != tuple(
            configured_tactile_size
        ):
            raise ValueError("runtime tactile size does not match checkpoint")
        if (
            self.sensor_range_m <= 0
            or self.tactile_scale_kpa <= 0
            or self.state_scale <= 0
            or self.control_substeps < 1
        ):
            raise ValueError("runtime scales and control_substeps must be positive")
        if self.action_scale.shape != (self.contract.action_dim,) or np.any(
            self.action_scale <= 0
        ):
            raise ValueError("runtime action scale does not match checkpoint")
        self.history: deque[dict[str, torch.Tensor]] = deque(
            maxlen=self.observation_horizon
        )
        self.last_route = "default"

    def reset(self) -> None:
        self.history.clear()
        self.model.reset_episode()
        self.last_route = "default"
        self.last_contact_descent = None
        self.last_contact_balance = None

    def _vision(self, rgb, depth_m) -> torch.Tensor:
        color = np.asarray(rgb)
        if color.ndim != 3 or color.shape[-1] not in (3, 4):
            raise ValueError("RGB observation must have three or four channels")
        color = color[..., :3]
        if float(np.max(color)) <= 1.5:
            color = color * 255.0
        color = cv2.resize(
            np.clip(color, 0, 255).astype(np.uint8),
            self.image_size[::-1],
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        color = color / 127.5 - 1.0
        depth = cv2.resize(
            np.asarray(depth_m, dtype=np.float32),
            self.image_size[::-1],
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.nan_to_num(depth, nan=2.0, posinf=2.0, neginf=0.0)
        depth = np.clip(depth, 0.0, 2.0) - 1.0
        return torch.from_numpy(
            np.concatenate((color.transpose(2, 0, 1), depth[None]), axis=0)
        ).float()

    def robot_state_from_observation(
        self, observation: PulseObservation
    ) -> np.ndarray:
        phase_scale = float(max(int(phase) for phase in TaskPhase))
        state = np.concatenate(
            (
                np.asarray(observation.joint_position_rad, dtype=np.float32),
                np.asarray(observation.tcp_pose, dtype=np.float32),
                np.asarray(
                    [float(observation.task_phase) / phase_scale],
                    dtype=np.float32,
                ),
            )
        )
        if state.shape != (self.state_dim,):
            raise ValueError(
                f"runtime robot state has {state.shape}, expected ({self.state_dim},)"
            )
        return state

    @staticmethod
    def _full56(backend_info: dict | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if backend_info is None:
            return (
                np.zeros(56, dtype=np.float32),
                np.ones(56, dtype=bool),
                np.zeros(5, dtype=np.float32),
            )
        full = backend_info.get("tactile_full_56")
        if full is None:
            raise ValueError("full56 tactile runtime contract is unavailable")
        names = tuple(str(name) for name in full.get("finger_names", ()))
        counts = tuple(int(value) for value in full.get("channel_counts", ()))
        offsets = tuple(int(value) for value in full.get("channel_offsets", ()))
        pressure = np.asarray(full.get("pressure_kpa"), dtype=np.float32)
        force = np.asarray(full.get("total_normal_force_n"), dtype=np.float32)
        contact = np.asarray(full.get("contact"), dtype=bool)
        if (
            names != FULL56_FINGER_NAMES
            or counts != FULL56_CHANNEL_COUNTS
            or offsets != (0, 16, 26, 36, 46)
            or pressure.shape != (56,)
            or force.shape != (5,)
            or contact.shape != (5,)
            or not np.isfinite(pressure).all()
            or not np.isfinite(force).all()
        ):
            raise ValueError("invalid full56 tactile runtime payload")
        return pressure, np.ones(56, dtype=bool), force

    def _active_tactile_image(
        self, pressure: np.ndarray, backend_info: dict | None
    ) -> torch.Tensor:
        diagnostic = (
            ("index", "middle", "ring")
            if backend_info is None
            else tuple(
                str(name)
                for name in backend_info.get(
                    "diagnostic_finger_names", ("index", "middle", "ring")
                )
            )
        )
        if diagnostic != ("index", "middle", "ring"):
            raise ValueError("unified runtime expects index/middle/ring active tactile")
        rows = np.stack((pressure[16:26], pressure[26:36], pressure[36:46]))
        image = cv2.resize(
            rows,
            self.tactile_size[::-1],
            interpolation=cv2.INTER_LINEAR,
        )
        image = np.clip(image / self.tactile_scale_kpa, -1.0, 1.0)
        return torch.from_numpy(image[None].astype(np.float32))

    def append_observation(
        self,
        observation: PulseObservation,
        *,
        backend_info: dict | None = None,
    ) -> None:
        if observation.rgb is None or observation.depth_m is None:
            raise ValueError("unified policy requires RGB-D observations")
        raw_full56, full56_valid, normal_force = self._full56(backend_info)
        normalized_full56 = self.normalization.normalize(raw_full56)
        previous_full56 = (
            np.zeros(56, dtype=np.float32)
            if not self.history
            else self.history[-1]["tactile_full56"].numpy()
        )
        full56_delta = normalized_full56 - previous_full56
        proximity_raw = np.asarray(observation.proximity_m, dtype=np.float32)
        proximity_valid = np.asarray(observation.proximity_valid, dtype=bool)
        if proximity_raw.shape != (5,) or proximity_valid.shape != (5,):
            raise ValueError("proximity runtime contract must have five channels")
        proximity_filled = np.where(
            proximity_valid & np.isfinite(proximity_raw),
            proximity_raw,
            self.sensor_range_m,
        )
        normal_force_valid = (
            np.zeros(5, dtype=bool)
            if backend_info is None
            else np.asarray(
                backend_info["tactile_full_56"]["contact"], dtype=bool
            )
        )
        previous_action = np.asarray(
            observation.previous_action, dtype=np.float32
        )
        if previous_action.shape != (self.contract.action_dim,):
            raise ValueError("previous action does not match checkpoint")
        frame = {
            "vision": self._vision(observation.rgb, observation.depth_m),
            "tactile": self._active_tactile_image(raw_full56, backend_info),
            "tactile_valid": torch.tensor(True),
            "proximity": torch.from_numpy(
                np.clip(proximity_filled / self.sensor_range_m, 0.0, 1.0) * 2.0
                - 1.0
            ).float(),
            "proximity_valid": torch.from_numpy(proximity_valid.copy()),
            "robot_state": torch.from_numpy(
                self.robot_state_from_observation(observation) / self.state_scale
            ).float(),
            "action": torch.from_numpy(
                np.clip(previous_action / self.action_scale, -1.0, 1.0)
            ).float(),
            "task_phase": torch.tensor(
                int(observation.task_phase), dtype=torch.long
            ),
            "diagnostic_contact": torch.from_numpy(
                normal_force_valid[1:4].copy()
            ),
            "tactile_full56": torch.from_numpy(normalized_full56.copy()),
            "tactile_full56_delta": torch.from_numpy(full56_delta.copy()),
            "tactile_full56_valid": torch.from_numpy(full56_valid.copy()),
            "normal_force": torch.from_numpy(normal_force.copy()),
            "normal_force_valid": torch.from_numpy(normal_force_valid.copy()),
        }
        self.history.append(frame)

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

    def project_segment_delta(self, action7: np.ndarray) -> dict:
        """Project a policy action into a bounded 2-D search proposal."""
        return WorldPolicySegmentAdapter().project(action7).__dict__

    @torch.no_grad()
    def predict_chunk(self) -> np.ndarray:
        batch = self._batch()
        prediction = self.model.predict_action_chunk(
            batch, use_direct_action=True
        ).action_chunk[0]
        if isinstance(self.model, PhaseRoutedPolicy):
            self.last_route = (
                "specialist"
                if bool(self.model.route_mask(batch)[0])
                else "default"
            )
        else:
            self.last_route = "default"
        physical = prediction * torch.as_tensor(
            self.action_scale, device=prediction.device
        )[None]
        self.last_contact_descent = None
        self.last_contact_balance = None
        if self.contact_descent_head is not None:
            features = contact_descent_features_from_observation(
                batch,
                include_previous_action=(
                    self.contact_descent_include_previous_action
                ),
            )
            logits, magnitude = self.contact_descent_head(features)
            current_contact = batch["normal_force_valid"][:, -1, 1:4]
            incomplete = ~current_contact.all(dim=-1)
            composed = compose_contact_descent_z(
                physical[None, 0, 2], logits, magnitude, incomplete
            )
            physical = physical.clone()
            physical[0, 2] = composed[0]
            self.last_contact_descent = {
                "applied": bool(incomplete[0]),
                "observable_contact_state": current_contact[0]
                .detach()
                .cpu()
                .tolist(),
                "descent_probability": float(torch.sigmoid(logits[0])),
                "predicted_descent_magnitude_m": float(magnitude[0]),
                "baseline_z_m": float(
                    prediction[0, 2] * self.action_scale[2]
                ),
                "composed_z_m": float(physical[0, 2]),
            }
        if self.contact_balance_rotation_head is not None:
            balance_features = contact_balance_features_from_observation(batch)
            direction = self.contact_balance_rotation_head(balance_features)
            current_contact = batch["normal_force_valid"][:, -1, 1:4]
            contact_count = current_contact.sum(dim=-1)
            composed_rotation = compose_contact_balance_rotation(
                physical[None, 0, 3:6],
                direction,
                contact_count,
                active_contact_counts=(
                    self.contact_balance_active_contact_counts
                ),
            )
            physical = physical.clone()
            physical[0, 3:6] = composed_rotation[0]
            self.last_contact_balance = {
                "applied": bool(
                    int(contact_count[0])
                    in self.contact_balance_active_contact_counts
                ),
                "observable_contact_state": current_contact[0]
                .detach()
                .cpu()
                .tolist(),
                "predicted_rotation_direction": direction[0]
                .detach()
                .cpu()
                .tolist(),
                "baseline_rotation_rad": (
                    prediction[0, 3:6]
                    * torch.as_tensor(
                        self.action_scale[3:6], device=prediction.device
                    )
                )
                .detach()
                .cpu()
                .tolist(),
                "composed_rotation_rad": physical[0, 3:6]
                .detach()
                .cpu()
                .tolist(),
            }
        return physical.detach().cpu().numpy().astype(np.float32)

    def predict_shadow(
        self,
        *,
        frame: int,
        timestamp_s: float,
        task_phase: int,
        safety_observation: PulseObservation,
        safety_supervisor,
        backend_info: dict | None = None,
    ) -> ShadowPrediction:
        chunk = self.predict_chunk()
        first = chunk[0]
        control_tick = chunk_action_to_control_tick(first, self.control_substeps)
        from pulse_mujoco.actions import PulseAction

        safe, info = safety_supervisor.filter_action(
            safety_observation,
            PulseAction(
                control_tick[:3],
                control_tick[3:6],
                target_pressure_kpa=float(control_tick[6]),
            ),
            backend_info,
        )
        previous = self.history[-1]["action"].numpy()
        normalized_first = first / self.action_scale
        delta = np.abs(normalized_first - previous)
        uncertainty = float(
            np.max(delta[:6])
            if int(task_phase) == int(TaskPhase.PRESSURE_REGULATION)
            else np.max(delta)
        )
        return ShadowPrediction(
            frame=int(frame),
            timestamp_s=float(timestamp_s),
            task_phase=int(task_phase),
            chunk_action=chunk,
            control_tick_action=control_tick,
            safety_preview_action=safe.as_array(),
            safety_modified=bool(info["modified"]),
            safety_reasons=tuple(info["reasons"]),
            policy_uncertainty=uncertainty,
        )
