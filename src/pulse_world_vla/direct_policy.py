from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pulse_mujoco.actions import PulseAction
from pulse_mujoco.observation import PulseObservation
from pulse_mujoco.safety import SafetySupervisor
from pulse_mujoco.task import TaskPhase

from .cuda import require_cuda
from .model import PulseWorldVLA


@dataclass(frozen=True)
class ShadowPrediction:
    """One non-executing policy proposal aligned to the next recorded frame."""

    frame: int
    timestamp_s: float
    task_phase: int
    chunk_action: np.ndarray
    control_tick_action: np.ndarray
    safety_preview_action: np.ndarray
    safety_modified: bool
    safety_reasons: tuple[str, ...]
    policy_uncertainty: float


def chunk_action_to_control_tick(
    action: np.ndarray, control_substeps: int
) -> np.ndarray:
    """Convert a record-to-record 7D action to one control-tick command.

    Translation and rotation are accumulated deltas and are divided across
    substeps. Pressure is an absolute target and must not be divided.
    """
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (7,):
        raise ValueError(f"action must have shape (7,), got {value.shape}")
    if int(control_substeps) < 1:
        raise ValueError("control_substeps must be positive")
    result = value.copy()
    result[:6] /= int(control_substeps)
    return result


class DirectPolicyRuntime:
    """Stateful online preprocessor for a trained World-VLA direct head.

    It consumes only policy-visible signals and never accepts privileged pulse
    location or synthetic reference waveforms.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str | torch.device | None = None,
        control_substeps: int = 8,
    ):
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        config = payload["config"]
        self.model_config = dict(config["model"])
        self.data_config = dict(config["data"])
        requested_device = torch.device("cuda" if device is None else device)
        self.device = (
            require_cuda(purpose="direct-policy runtime")
            if requested_device.type == "cuda"
            else requested_device
        )
        self.model = PulseWorldVLA(**self.model_config).to(self.device)
        # EMA is the checkpoint selected by validation for the formal trainer.
        # Older smoke checkpoints only contain ``model`` and remain usable.
        self.model.load_state_dict(payload.get("ema") or payload["model"])
        self.model.eval()
        if self.model.direct_action_head is None:
            raise ValueError("checkpoint does not contain a direct action head")
        self.observation_horizon = int(
            self.model_config["observation_horizon"]
        )
        self.image_size = tuple(self.model_config["image_size"])
        self.tactile_size = tuple(self.model_config["tactile_size"])
        self.sensor_range_m = float(self.data_config["sensor_range_m"])
        self.state_scale = float(self.data_config["state_scale"])
        self.tactile_scale_kpa = float(
            self.data_config["tactile_scale_kpa"]
        )
        self.tactile_view = str(
            self.data_config.get("tactile_view", "diagnostic_2x10")
        )
        self.direct_numeric_tactile_fingers = int(
            self.model_config.get("direct_numeric_tactile_fingers", 2)
        )
        self.direct_numeric_contact_fingers = int(
            self.model_config.get("direct_numeric_contact_fingers", 0)
        )
        action_scale = self.data_config.get(
            "action_scale", self.data_config.get("action_scale_m")
        )
        if action_scale is None:
            raise ValueError("checkpoint data config has no 7-D action_scale")
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        if self.action_scale.shape != (7,):
            raise ValueError("direct policy requires a 7D action scale")
        self.control_substeps = int(control_substeps)
        if self.control_substeps < 1:
            raise ValueError("control_substeps must be positive")
        self.history: deque[dict[str, torch.Tensor]] = deque(
            maxlen=self.observation_horizon
        )

    def reset(self) -> None:
        self.history.clear()

    def _vision(self, rgb, depth_m) -> torch.Tensor:
        rgb = np.asarray(rgb)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if np.max(rgb) <= 1.5:
            rgb = rgb * 255.0
        rgb = cv2.resize(
            np.clip(rgb, 0, 255).astype(np.uint8),
            self.image_size[::-1],
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        rgb = rgb / 127.5 - 1.0
        depth = cv2.resize(
            np.asarray(depth_m, dtype=np.float32),
            self.image_size[::-1],
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.nan_to_num(
            depth, nan=2.0, posinf=2.0, neginf=0.0
        )
        depth = np.clip(depth, 0.0, 2.0).astype(np.float32) - 1.0
        return torch.from_numpy(
            np.concatenate((rgb.transpose(2, 0, 1), depth[None]), axis=0)
        )

    def _tactile(self, tactile_kpa, tactile_valid: bool) -> torch.Tensor:
        if not tactile_valid:
            return torch.zeros(1, *self.tactile_size)
        tactile = cv2.resize(
            np.asarray(tactile_kpa, dtype=np.float32),
            self.tactile_size[::-1],
            interpolation=cv2.INTER_AREA,
        )
        tactile = np.clip(
            tactile / self.tactile_scale_kpa, -1.0, 1.0
        )
        return torch.from_numpy(tactile[None]).float()

    def append(
        self,
        *,
        rgb,
        depth_m,
        tactile_kpa,
        tactile_valid: bool,
        proximity_m,
        proximity_valid,
        robot_state,
        previous_action,
        task_phase: int,
        diagnostic_contact=None,
    ) -> None:
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
        previous = np.asarray(previous_action, dtype=np.float32)
        state = np.asarray(robot_state, dtype=np.float32)
        if previous.shape != (7,):
            raise ValueError("previous_action must have shape (7,)")
        if state.shape != (int(self.model_config["state_dim"]),):
            raise ValueError(
                f"robot_state must have shape "
                f"({self.model_config['state_dim']},)"
            )
        frame = {
                "vision": self._vision(rgb, depth_m),
                "tactile": self._tactile(tactile_kpa, tactile_valid),
                "tactile_valid": torch.tensor(bool(tactile_valid)),
                "proximity": torch.from_numpy(
                    np.clip(filled / self.sensor_range_m, 0, 1) * 2 - 1
                ).float(),
                "proximity_valid": torch.from_numpy(valid),
                "robot_state": torch.from_numpy(
                    state / self.state_scale
                ).float(),
                "action": torch.from_numpy(
                    np.clip(previous / self.action_scale, -1, 1)
                ).float(),
                "task_phase": torch.tensor(int(task_phase), dtype=torch.long),
            }
        if self.direct_numeric_contact_fingers:
            contact = np.asarray(diagnostic_contact, dtype=bool)
            expected = self.direct_numeric_contact_fingers
            if contact.shape != (expected,):
                raise ValueError(
                    f"diagnostic_contact must have shape ({expected},)"
                )
            frame["diagnostic_contact"] = torch.from_numpy(contact.copy())
        self.history.append(frame)

    def robot_state_from_observation(
        self, observation: PulseObservation
    ) -> np.ndarray:
        """Build the exact MuJoCo recorder state contract at runtime.

        Episodes store controlled-arm joint positions, the 7-D TCP pose
        (wxyz quaternion), and task phase divided by 13.  Keeping this here
        prevents an online policy from silently using velocities or privileged
        pulse data that were absent from the training representation.
        """
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
        expected = int(self.model_config["state_dim"])
        if state.shape != (expected,):
            raise ValueError(
                "runtime MuJoCo state does not match checkpoint contract: "
                f"got {state.shape}, expected ({expected},)"
            )
        return state

    def _runtime_tactile(
        self,
        observation: PulseObservation,
        backend_info: dict | None,
    ) -> np.ndarray:
        """Match the online tactile view to the checkpoint dataset contract."""

        legacy = np.asarray(observation.tactile_kpa, dtype=np.float32)
        expected_fingers = self.direct_numeric_tactile_fingers
        if self.tactile_view != "active" or expected_fingers <= 2:
            return legacy
        # History priming happens before a live backend info frame exists.
        # At that point the robot is out of contact, so an all-zero active view
        # is the exact deployable observation rather than a fabricated split of
        # the legacy two-finger image.
        if backend_info is None:
            return np.zeros((expected_fingers, 10), dtype=np.float32)
        full = backend_info.get("tactile_full_56")
        diagnostic = tuple(
            str(name)
            for name in backend_info.get("diagnostic_finger_names", ())
        )
        if full is None or len(diagnostic) != expected_fingers:
            raise ValueError("active tactile runtime contract is unavailable")
        finger_names = tuple(str(name) for name in full["finger_names"])
        counts = np.asarray(full["channel_counts"], dtype=np.int64)
        offsets = np.asarray(full["channel_offsets"], dtype=np.int64)
        pressure = np.asarray(full["pressure_kpa"], dtype=np.float32)
        if counts.shape != offsets.shape or counts.size != len(finger_names):
            raise ValueError("invalid full-hand tactile channel metadata")
        index = {name: position for position, name in enumerate(finger_names)}
        rows = []
        for name in diagnostic:
            if name not in index:
                raise ValueError(f"diagnostic tactile finger is absent: {name}")
            position = index[name]
            count = int(counts[position])
            offset = int(offsets[position])
            if count != 10 or offset < 0 or offset + count > pressure.size:
                raise ValueError("active tactile runtime requires 10 channels per finger")
            rows.append(pressure[offset : offset + count])
        tactile = np.stack(rows).astype(np.float32, copy=False)
        if not np.isfinite(tactile).all():
            raise ValueError("active tactile runtime contains non-finite pressure")
        return tactile

    def append_observation(
        self,
        observation: PulseObservation,
        *,
        backend_info: dict | None = None,
    ) -> None:
        """Append one live MuJoCo observation without privileged fields."""
        if observation.rgb is None or observation.depth_m is None:
            raise ValueError("direct policy requires RGB-D observations")
        tactile = self._runtime_tactile(observation, backend_info)
        diagnostic_contact = None
        if self.direct_numeric_contact_fingers:
            if backend_info is None:
                diagnostic_contact = np.zeros(
                    self.direct_numeric_contact_fingers, dtype=bool
                )
            else:
                from pulse_mujoco.phase_gated_policy import (
                    observable_diagnostic_contact,
                )

                diagnostic_contact = observable_diagnostic_contact(
                    observation.contact, backend_info
                )
                if diagnostic_contact.size != self.direct_numeric_contact_fingers:
                    raise ValueError("runtime diagnostic contact count mismatch")
        self.append(
            rgb=observation.rgb,
            depth_m=observation.depth_m,
            tactile_kpa=tactile,
            # A connected tactile array is valid before contact too; its
            # calibrated pressure is simply near zero in that case.
            tactile_valid=bool(np.isfinite(tactile).all()),
            proximity_m=observation.proximity_m,
            proximity_valid=observation.proximity_valid,
            robot_state=self.robot_state_from_observation(observation),
            previous_action=observation.previous_action,
            task_phase=int(observation.task_phase),
            diagnostic_contact=diagnostic_contact,
        )

    @property
    def ready(self) -> bool:
        return len(self.history) == self.observation_horizon

    def _batch(self) -> dict[str, torch.Tensor]:
        if not self.ready:
            raise RuntimeError(
                f"need {self.observation_horizon} frames, "
                f"have {len(self.history)}"
            )
        return {
            key: torch.stack([frame[key] for frame in self.history])
            .unsqueeze(0)
            .to(self.device)
            for key in self.history[0]
        }

    @torch.no_grad()
    def predict_chunk(self) -> np.ndarray:
        normalized = self.model.predict_direct_action(self._batch())[0]
        return (
            normalized.detach().cpu().numpy().astype(np.float32)
            * self.action_scale[None]
        )

    def predict_shadow(
        self,
        *,
        frame: int,
        timestamp_s: float,
        task_phase: int,
        safety_observation: PulseObservation,
        safety_supervisor: SafetySupervisor,
        backend_info: dict | None = None,
    ) -> ShadowPrediction:
        chunk = self.predict_chunk()
        first = chunk[0]
        control_tick = chunk_action_to_control_tick(
            first, self.control_substeps
        )
        proposed = PulseAction(
            control_tick[:3],
            control_tick[3:6],
            target_pressure_kpa=float(control_tick[6]),
        )
        safe, info = safety_supervisor.filter_action(
            safety_observation, proposed, backend_info
        )
        previous = np.asarray(
            self.history[-1]["action"].cpu(), dtype=np.float32
        )
        normalized_first = first / self.action_scale
        temporal_delta = np.abs(normalized_first - previous)
        if int(task_phase) == int(TaskPhase.PRESSURE_REGULATION):
            # Pressure is an absolute setpoint. Its load rate is owned by the
            # pressure projection and SafetySupervisor, so an expected 0 to
            # setpoint transition must not be misclassified as pose-policy
            # uncertainty. Pose channels remain monitored normally.
            uncertainty = float(np.max(temporal_delta[:6]))
        else:
            uncertainty = float(np.max(temporal_delta))
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


class ShadowRecorder:
    """Align prediction at frame t with expert executed action at frame t+1."""

    def __init__(self, output: Path):
        self.output = Path(output)
        self.pending: ShadowPrediction | None = None
        self.records: list[dict] = []
        self.discarded_predictions: list[dict] = []

    def observe_expert_action(
        self, action, *, expert_label: str | None = None
    ) -> None:
        if self.pending is None:
            return
        expert = np.asarray(action, dtype=np.float32)
        if expert.shape != (7,):
            raise ValueError("expert action must have shape (7,)")
        predicted = self.pending.chunk_action[0]
        record = {
                "prediction_frame": self.pending.frame,
                "prediction_timestamp_s": self.pending.timestamp_s,
                "task_phase": self.pending.task_phase,
                "predicted_chunk_action": (
                    self.pending.chunk_action.tolist()
                ),
                "predicted_first_action": predicted.tolist(),
                "control_tick_action": (
                    self.pending.control_tick_action.tolist()
                ),
                "safety_preview_action": (
                    self.pending.safety_preview_action.tolist()
                ),
                "safety_modified": self.pending.safety_modified,
                "safety_reasons": list(self.pending.safety_reasons),
                "policy_uncertainty": self.pending.policy_uncertainty,
                "expert_executed_action": expert.tolist(),
                "absolute_error": np.abs(predicted - expert).tolist(),
            }
        if expert_label is not None:
            record["expert_label"] = str(expert_label)
        self.records.append(record)
        self.pending = None

    def set_prediction(self, prediction: ShadowPrediction) -> None:
        if self.pending is not None:
            raise RuntimeError(
                "pending shadow prediction was not aligned to expert action"
            )
        self.pending = prediction

    def discard_pending(self, reason: str) -> None:
        if self.pending is None:
            return
        self.discarded_predictions.append(
            {
                "prediction_frame": self.pending.frame,
                "task_phase": self.pending.task_phase,
                "reason": str(reason),
            }
        )
        self.pending = None

    def close(self, action_scale) -> dict:
        scale = np.asarray(action_scale, dtype=np.float32)
        errors = np.asarray(
            [record["absolute_error"] for record in self.records],
            dtype=np.float32,
        )
        normalized = errors / scale[None] if len(errors) else errors
        summary = {
            "schema_version": 1,
            "mode": "teacher_shadow_only",
            "policy_action_executed": False,
            "aligned_predictions": len(self.records),
            "unaligned_final_prediction": self.pending is not None,
            "discarded_predictions": list(self.discarded_predictions),
            "action_scale": scale.tolist(),
            "mae_by_channel": (
                errors.mean(axis=0).tolist()
                if len(errors)
                else [None] * 7
            ),
            "normalized_mae_by_channel": (
                normalized.mean(axis=0).tolist()
                if len(errors)
                else [None] * 7
            ),
            "active_6d_normalized_mae": (
                float(normalized[:, :6].mean())
                if len(errors)
                else None
            ),
            "pressure_mae_kpa": (
                float(errors[:, 6].mean()) if len(errors) else None
            ),
            "safety_modified_fraction": (
                float(
                    np.mean(
                        [record["safety_modified"] for record in self.records]
                    )
                )
                if self.records
                else None
            ),
            "uncertainty": (
                {
                    "mean": float(
                        np.mean(
                            [
                                record["policy_uncertainty"]
                                for record in self.records
                            ]
                        )
                    ),
                    "p95": float(
                        np.percentile(
                            [
                                record["policy_uncertainty"]
                                for record in self.records
                            ],
                            95,
                        )
                    ),
                }
                if self.records
                else None
            ),
            "records": self.records,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
