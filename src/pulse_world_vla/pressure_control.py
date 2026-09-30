from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


PRESSURE_CHECKPOINT_SCHEMA_VERSION = "adam_pulse_temporal_pressure_head_v1"


@dataclass(frozen=True)
class PressureControlConfig:
    observation_horizon: int = 4
    action_horizon: int = 4
    tactile_channels: int = 56
    active_fingers: int = 3
    hidden_dim: int = 64
    gru_layers: int = 1
    residual_limit_kpa: float = 4.0
    maximum_command_kpa: float = 127.5
    target_force_n: float = 4.0
    target_pressure_kpa: float = 40.0
    minimum_force_n: float = 1.25
    maximum_force_n: float = 10.0
    soft_force_guard_n: float = 6.8
    deadband_fraction_of_nearest_limit: float = 0.10
    tactile_scale_kpa: float = 320.0
    force_scale_n: float = 10.0
    command_scale_kpa: float = 127.5
    smoothness_weight: float = 0.02
    future_force_weight: float = 0.10

    def __post_init__(self) -> None:
        if self.observation_horizon < 2 or self.action_horizon < 1:
            raise ValueError("pressure control requires history >= 2 and horizon >= 1")
        if self.tactile_channels != 56 or self.active_fingers != 3:
            raise ValueError("the current pressure contract requires full56 and three fingers")
        if not self.minimum_force_n < self.target_force_n < self.maximum_force_n:
            raise ValueError("target force must be inside the task force limits")
        if not 0.0 < self.deadband_fraction_of_nearest_limit <= 1.0:
            raise ValueError("deadband fraction must be in (0, 1]")
        if min(
            self.residual_limit_kpa,
            self.maximum_command_kpa,
            self.target_pressure_kpa,
            self.tactile_scale_kpa,
            self.force_scale_n,
            self.command_scale_kpa,
        ) <= 0.0:
            raise ValueError("pressure scales and limits must be positive")

    @property
    def force_deadband_n(self) -> float:
        nearest_limit = min(
            self.target_force_n - self.minimum_force_n,
            self.maximum_force_n - self.target_force_n,
        )
        return nearest_limit * self.deadband_fraction_of_nearest_limit

    @property
    def pressure_deadband_kpa(self) -> float:
        # The public nominal conversion is 40 kPa -> 4 N per fingertip. Use
        # it only to express the task's force tolerance in command units.
        return (
            self.force_deadband_n
            * self.target_pressure_kpa
            / self.target_force_n
        )

    @property
    def feature_dim(self) -> int:
        # full tactile, tactile derivative, force, force derivative,
        # per-finger measured pressure/error/error derivative, contact,
        # previous command, public pressure target and public force target.
        return 2 * self.tactile_channels + 6 * self.active_fingers + 3


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def build_pressure_features(
    tactile_full56_kpa: np.ndarray,
    normal_force_n: np.ndarray,
    contact: np.ndarray,
    previous_command_kpa: np.ndarray,
    target_pressure_kpa: np.ndarray,
    target_force_n: np.ndarray,
    config: PressureControlConfig,
) -> np.ndarray:
    """Build explicit temporal feedback features from public observations."""

    history = config.observation_horizon
    tactile = _finite_array(
        tactile_full56_kpa,
        (history, config.tactile_channels),
        "tactile_full56_kpa",
    )
    force = _finite_array(
        normal_force_n,
        (history, config.active_fingers),
        "normal_force_n",
    )
    contacts = _finite_array(
        contact,
        (history, config.active_fingers),
        "contact",
    )
    command = _finite_array(
        previous_command_kpa, (history,), "previous_command_kpa"
    )
    target_pressure = _finite_array(
        target_pressure_kpa, (history,), "target_pressure_kpa"
    )
    target_force = _finite_array(target_force_n, (history,), "target_force_n")

    active = tactile[:, 16:46].reshape(history, config.active_fingers, 10)
    measured_pressure = active.mean(axis=-1)
    pressure_error = target_pressure[:, None] - measured_pressure
    tactile_delta = np.zeros_like(tactile)
    force_delta = np.zeros_like(force)
    error_delta = np.zeros_like(pressure_error)
    tactile_delta[1:] = tactile[1:] - tactile[:-1]
    force_delta[1:] = force[1:] - force[:-1]
    error_delta[1:] = pressure_error[1:] - pressure_error[:-1]

    features = np.concatenate(
        (
            tactile / config.tactile_scale_kpa,
            tactile_delta / config.tactile_scale_kpa,
            force / config.force_scale_n,
            force_delta / config.force_scale_n,
            measured_pressure / config.tactile_scale_kpa,
            pressure_error / config.tactile_scale_kpa,
            error_delta / config.tactile_scale_kpa,
            np.clip(contacts, 0.0, 1.0),
            (command / config.command_scale_kpa)[:, None],
            (target_pressure / config.command_scale_kpa)[:, None],
            (target_force / config.force_scale_n)[:, None],
        ),
        axis=-1,
    ).astype(np.float32)
    if features.shape != (history, config.feature_dim):
        raise AssertionError("pressure feature contract changed unexpectedly")
    return features


class ObservableResidualPressureTeacher:
    """Bounded feedback teacher using only policy-visible force and command."""

    def __init__(
        self,
        config: PressureControlConfig,
        *,
        proportional_gain: float = 0.8,
        derivative_gain: float = 0.1,
    ):
        self.config = config
        self.proportional_gain = float(proportional_gain)
        self.derivative_gain = float(derivative_gain)
        if self.proportional_gain <= 0.0 or self.derivative_gain < 0.0:
            raise ValueError("teacher gains must be non-negative with positive P gain")
        self._previous_error_n = 0.0

    def reset(self) -> None:
        self._previous_error_n = 0.0

    def propose(
        self,
        normal_force_n: np.ndarray,
        contact: np.ndarray,
        previous_command_kpa: float,
        *,
        target_force_n: float | None = None,
        target_pressure_kpa: float | None = None,
    ) -> tuple[float, dict[str, float | bool]]:
        force = _finite_array(
            normal_force_n, (self.config.active_fingers,), "normal_force_n"
        )
        contacts = np.asarray(contact, dtype=bool)
        if contacts.shape != (self.config.active_fingers,):
            raise ValueError("contact must contain the three active fingers")
        target = float(
            self.config.target_force_n if target_force_n is None else target_force_n
        )
        pressure_target = float(
            self.config.target_pressure_kpa
            if target_pressure_kpa is None
            else target_pressure_kpa
        )
        prior = float(
            np.clip(previous_command_kpa, 0.0, self.config.maximum_command_kpa)
        )
        observable_force = float(np.mean(force))
        force_error = target - observable_force
        command_error = pressure_target - prior
        deadband = self.config.pressure_deadband_kpa
        if not bool(np.all(contacts)):
            residual = -self.config.residual_limit_kpa
        elif float(np.max(force)) > self.config.soft_force_guard_n:
            residual = -self.config.residual_limit_kpa
        elif abs(command_error) <= deadband:
            residual = 0.0
        else:
            derivative = command_error - self._previous_error_n
            residual = (
                self.proportional_gain * command_error
                + self.derivative_gain * derivative
            )
            residual = float(
                np.clip(
                    residual,
                    -self.config.residual_limit_kpa,
                    self.config.residual_limit_kpa,
                )
            )
        command = float(
            np.clip(
                prior + residual,
                0.0,
                self.config.maximum_command_kpa,
            )
        )
        executed_residual = command - prior
        self._previous_error_n = command_error
        return command, {
            "residual_kpa": float(executed_residual),
            "observable_force_n": observable_force,
            "force_error_n": force_error,
            "command_error_kpa": command_error,
            "pressure_deadband_kpa": deadband,
            "in_deadband": bool(abs(command_error) <= deadband),
        }


class TemporalResidualPressureHead(nn.Module):
    def __init__(self, config: PressureControlConfig):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.feature_dim)
        self.gru = nn.GRU(
            config.feature_dim,
            config.hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
        )
        self.pressure_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_horizon),
        )
        self.future_force_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(
                config.hidden_dim,
                config.action_horizon * config.active_fingers,
            ),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[1:] != (
            self.config.observation_horizon,
            self.config.feature_dim,
        ):
            raise ValueError("pressure features do not match the temporal contract")
        _, hidden = self.gru(self.input_norm(features))
        encoded = hidden[-1]
        residual = torch.tanh(self.pressure_head(encoded))
        residual = residual * self.config.residual_limit_kpa
        future_force = self.future_force_head(encoded).reshape(
            -1,
            self.config.action_horizon,
            self.config.active_fingers,
        )
        return {"residual_kpa": residual, "future_force_n": future_force}

    def loss(
        self,
        features: torch.Tensor,
        target_residual_kpa: torch.Tensor,
        target_future_force_n: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self(features)
        action = nn.functional.smooth_l1_loss(
            output["residual_kpa"], target_residual_kpa
        )
        command_smoothness = output["residual_kpa"].diff(dim=1).abs().mean()
        future_force = nn.functional.smooth_l1_loss(
            output["future_force_n"], target_future_force_n
        )
        total = (
            action
            + self.config.smoothness_weight * command_smoothness
            + self.config.future_force_weight * future_force
        )
        return {
            "action": action,
            "smoothness": command_smoothness,
            "future_force": future_force,
            "total": total,
        }


class PressureTrajectoryDataset(Dataset):
    def __init__(
        self,
        path: Path | str,
        config: PressureControlConfig,
        *,
        split: str,
    ):
        self.path = Path(path)
        self.config = config
        with np.load(self.path) as payload:
            self.tactile = np.asarray(payload["tactile_full56_kpa"], dtype=np.float32)
            self.force = np.asarray(payload["normal_force_n"], dtype=np.float32)
            self.contact = np.asarray(payload["contact"], dtype=np.float32)
            self.command = np.asarray(payload["previous_command_kpa"], dtype=np.float32)
            self.target_pressure = np.asarray(payload["target_pressure_kpa"], dtype=np.float32)
            self.target_force = np.asarray(payload["target_force_n"], dtype=np.float32)
            self.residual = np.asarray(payload["teacher_residual_kpa"], dtype=np.float32)
            self.episode = np.asarray(payload["episode_index"], dtype=np.int64)
            self.split = np.asarray(payload["split"], dtype="U16")
        count = self.tactile.shape[0]
        expected = {
            "tactile": (count, config.tactile_channels),
            "force": (count, config.active_fingers),
            "contact": (count, config.active_fingers),
            "command": (count,),
            "target_pressure": (count,),
            "target_force": (count,),
            "residual": (count,),
            "episode": (count,),
            "split": (count,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"pressure dataset {name} has invalid shape")
        self.indices = []
        history = config.observation_horizon
        horizon = config.action_horizon
        for end in range(history - 1, count - horizon):
            indices = np.arange(end - history + 1, end + horizon + 1)
            if (
                self.split[end] == split
                and np.all(self.episode[indices] == self.episode[end])
                and np.all(self.split[indices] == split)
            ):
                self.indices.append(end)
        if not self.indices:
            raise ValueError(f"pressure dataset has no {split} windows")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        end = self.indices[item]
        history = self.config.observation_horizon
        horizon = self.config.action_horizon
        observed = slice(end - history + 1, end + 1)
        future = slice(end, end + horizon)
        features = build_pressure_features(
            self.tactile[observed],
            self.force[observed],
            self.contact[observed],
            self.command[observed],
            self.target_pressure[observed],
            self.target_force[observed],
            self.config,
        )
        return {
            "features": torch.from_numpy(features),
            "target_residual_kpa": torch.from_numpy(self.residual[future].copy()),
            "target_future_force_n": torch.from_numpy(self.force[future].copy()),
        }


def save_pressure_checkpoint(
    path: Path | str,
    model: TemporalResidualPressureHead,
    *,
    metrics: dict[str, Any],
    dataset: Path | str,
) -> None:
    torch.save(
        {
            "schema_version": PRESSURE_CHECKPOINT_SCHEMA_VERSION,
            "config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "dataset": str(Path(dataset).resolve()),
        },
        Path(path),
    )


def load_pressure_checkpoint(
    path: Path | str,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TemporalResidualPressureHead, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != PRESSURE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported temporal pressure checkpoint")
    model = TemporalResidualPressureHead(PressureControlConfig(**payload["config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload
