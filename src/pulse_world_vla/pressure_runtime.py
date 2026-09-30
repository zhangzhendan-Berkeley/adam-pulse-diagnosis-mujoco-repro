from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from pulse_mujoco.observation import PulseObservation
from pulse_mujoco.task import TaskPhase

from .pressure_control import build_pressure_features, load_pressure_checkpoint


class TemporalPressureRuntime:
    """Stateful residual pressure inference with explicit public deadband."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str | torch.device = "cpu",
    ):
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.device = requested
        self.model, self.payload = load_pressure_checkpoint(
            checkpoint, device=self.device
        )
        self.config = self.model.config
        self.history: deque[dict[str, Any]] = deque(
            maxlen=self.config.observation_horizon
        )

    def reset(self) -> None:
        self.history.clear()

    def observe(
        self,
        *,
        tactile_full56_kpa: np.ndarray,
        normal_force_n: np.ndarray,
        contact: np.ndarray,
        previous_command_kpa: float,
        target_pressure_kpa: float | None = None,
        target_force_n: float | None = None,
    ) -> None:
        item = {
            "tactile": np.asarray(tactile_full56_kpa, dtype=np.float32).copy(),
            "force": np.asarray(normal_force_n, dtype=np.float32).copy(),
            "contact": np.asarray(contact, dtype=np.float32).copy(),
            "command": float(previous_command_kpa),
            "target_pressure": float(
                self.config.target_pressure_kpa
                if target_pressure_kpa is None
                else target_pressure_kpa
            ),
            "target_force": float(
                self.config.target_force_n
                if target_force_n is None
                else target_force_n
            ),
        }
        if item["tactile"].shape != (self.config.tactile_channels,):
            raise ValueError("runtime pressure tactile must be full56")
        if item["force"].shape != (self.config.active_fingers,):
            raise ValueError("runtime pressure force must contain three fingers")
        if item["contact"].shape != (self.config.active_fingers,):
            raise ValueError("runtime pressure contact must contain three fingers")
        self.history.append(item)

    def ready(self) -> bool:
        return len(self.history) == self.config.observation_horizon

    @torch.inference_mode()
    def predict(self) -> dict[str, Any]:
        if not self.ready():
            raise RuntimeError("pressure runtime history is not ready")
        items = list(self.history)
        features = build_pressure_features(
            np.stack([item["tactile"] for item in items]),
            np.stack([item["force"] for item in items]),
            np.stack([item["contact"] for item in items]),
            np.asarray([item["command"] for item in items]),
            np.asarray([item["target_pressure"] for item in items]),
            np.asarray([item["target_force"] for item in items]),
            self.config,
        )
        output = self.model(
            torch.from_numpy(features)[None].to(self.device)
        )
        residual_chunk = output["residual_kpa"][0].cpu().numpy()
        force_chunk = output["future_force_n"][0].cpu().numpy()
        current = items[-1]
        raw_residual = float(residual_chunk[0])
        pressure_error = current["target_pressure"] - current["command"]
        reason = "model_residual"
        if not bool(np.all(current["contact"] > 0.5)):
            residual = -self.config.residual_limit_kpa
            reason = "contact_loss_unload"
        elif float(np.max(current["force"])) > self.config.soft_force_guard_n:
            residual = -self.config.residual_limit_kpa
            reason = "soft_force_guard_unload"
        elif abs(pressure_error) <= self.config.pressure_deadband_kpa:
            residual = 0.0
            reason = "public_target_deadband"
        else:
            residual = float(
                np.clip(
                    raw_residual,
                    -self.config.residual_limit_kpa,
                    self.config.residual_limit_kpa,
                )
            )
        command = float(
            np.clip(
                current["command"] + residual,
                0.0,
                self.config.maximum_command_kpa,
            )
        )
        return {
            "command_kpa": command,
            "residual_kpa": command - current["command"],
            "raw_residual_kpa": raw_residual,
            "residual_chunk_kpa": residual_chunk,
            "future_force_n": force_chunk,
            "pressure_error_kpa": pressure_error,
            "projection_reason": reason,
        }


class CompositeTemporalPressureRuntime:
    """Adapter exposing the existing composite specialist runtime contract."""

    observation_horizon = 4

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str | torch.device = "cpu",
    ):
        self.runtime = TemporalPressureRuntime(checkpoint, device=device)
        self.observation_horizon = self.runtime.config.observation_horizon
        self.last_route = "temporal_residual_pressure_gru"
        self._in_pressure_phase = False

    def reset(self) -> None:
        self.runtime.reset()
        self._in_pressure_phase = False

    @property
    def ready(self) -> bool:
        return self.runtime.ready()

    def append_observation(
        self,
        observation: PulseObservation,
        *,
        backend_info: dict | None = None,
    ) -> None:
        if int(observation.task_phase) != int(TaskPhase.PRESSURE_REGULATION):
            if self._in_pressure_phase:
                self.reset()
            return
        backend_info = backend_info or {}
        tactile = backend_info.get("tactile_full_56")
        if not isinstance(tactile, dict):
            raise ValueError("temporal pressure runtime requires full56 backend info")
        force = np.asarray(backend_info.get("normal_force_n"), dtype=np.float32)
        full_contact = np.asarray(tactile.get("contact"), dtype=bool)
        if force.shape != (3,) or full_contact.shape != (5,):
            raise ValueError("temporal pressure runtime requires three forces and five contacts")
        kwargs = {
            "tactile_full56_kpa": np.asarray(tactile["pressure_kpa"], dtype=np.float32),
            "normal_force_n": force,
            "contact": full_contact[1:4],
            "previous_command_kpa": float(observation.previous_action[6]),
            "target_pressure_kpa": self.runtime.config.target_pressure_kpa,
            "target_force_n": self.runtime.config.target_force_n,
        }
        if not self._in_pressure_phase:
            self.runtime.reset()
            for _ in range(self.observation_horizon - 1):
                self.runtime.observe(**kwargs)
            self._in_pressure_phase = True
        self.runtime.observe(**kwargs)

    def predict_shadow(self, **_kwargs):
        prediction = self.runtime.predict()
        action = np.zeros(7, dtype=np.float32)
        action[6] = float(prediction["command_kpa"])
        uncertainty = abs(float(prediction["residual_kpa"])) / float(
            self.runtime.config.maximum_command_kpa
        )
        return SimpleNamespace(
            control_tick_action=action,
            policy_uncertainty=uncertainty,
            pressure_prediction=prediction,
        )


def pressure_gate_metrics(
    command_kpa: np.ndarray,
    residual_kpa: np.ndarray,
    normal_force_n: np.ndarray,
    contact: np.ndarray,
    *,
    target_pressure_kpa: float,
    pressure_deadband_kpa: float,
    minimum_force_n: float,
    maximum_force_n: float,
    policy_rate_hz: float,
    stable_frames: int = 12,
) -> dict[str, Any]:
    command = np.asarray(command_kpa, dtype=np.float64)
    residual = np.asarray(residual_kpa, dtype=np.float64)
    force = np.asarray(normal_force_n, dtype=np.float64)
    contacts = np.asarray(contact, dtype=bool)
    count = command.size
    if (
        command.shape != (count,)
        or residual.shape != (count,)
        or force.shape != (count, 3)
        or contacts.shape != (count, 3)
        or count < stable_frames
    ):
        raise ValueError("invalid pressure gate trajectory")
    in_target = np.abs(command - target_pressure_kpa) <= pressure_deadband_kpa
    force_safe = np.all(
        (force >= minimum_force_n) & (force <= maximum_force_n), axis=1
    )
    valid = in_target & force_safe & np.all(contacts, axis=1)
    settled_index = None
    for start in range(count - stable_frames + 1):
        if bool(np.all(valid[start : start + stable_frames])):
            settled_index = start
            break
    tail_start = settled_index if settled_index is not None else max(0, count - stable_frames)
    command_delta = np.diff(command[tail_start:], prepend=command[tail_start])
    signs = np.sign(residual[np.abs(residual) > 1e-6])
    sign_flips = int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    return {
        "success": settled_index is not None,
        "settling_time_s": (
            None if settled_index is None else settled_index / policy_rate_hz
        ),
        "pressure_overshoot_kpa": float(
            max(0.0, np.max(command) - target_pressure_kpa)
        ),
        "pressure_undershoot_kpa": float(
            max(0.0, target_pressure_kpa - np.min(command))
        ),
        "steady_state_error_kpa": float(
            np.mean(np.abs(command[tail_start:] - target_pressure_kpa))
        ),
        "pressure_oscillation_rms_kpa": float(
            np.sqrt(np.mean(command_delta**2))
        ),
        "maximum_normal_force_n": force.max(axis=0).tolist(),
        "normal_force_overshoot_above_target_n": float(
            max(0.0, np.max(force) - 4.0)
        ),
        "normal_force_violation_n": float(
            max(0.0, np.max(force) - maximum_force_n)
        ),
        "action_sign_flip_count": sign_flips,
        "contact_loss_frames": int(np.count_nonzero(~np.all(contacts, axis=1))),
        "timeout": settled_index is None,
    }
