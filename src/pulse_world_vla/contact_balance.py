from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from .contact_descent import (
    observable_diagnostic_contact,
    observable_diagnostic_force,
)


def contact_balance_feature_vector(
    records: Sequence[dict],
    end_index: int,
    *,
    observation_horizon: int = 4,
    state_scale: float = float(np.pi),
) -> np.ndarray:
    """Build contact/force/robot-state history without expert or pulse truth."""

    start = int(end_index) - int(observation_horizon) + 1
    if start < 0 or end_index >= len(records):
        raise IndexError("contact-balance history crosses the episode boundary")
    if state_scale <= 0.0:
        raise ValueError("state_scale must be positive")
    frames = []
    for record in records[start : end_index + 1]:
        state = np.asarray(record.get("robot_state", ()), dtype=np.float32)
        if state.shape != (15,):
            raise ValueError("robot_state must have shape (15,)")
        frames.append(
            np.concatenate(
                (
                    observable_diagnostic_contact(record).astype(np.float32),
                    np.clip(observable_diagnostic_force(record) / 10.0, 0.0, 1.0),
                    state / float(state_scale),
                )
            )
        )
    return np.concatenate(frames).astype(np.float32, copy=False)


def contact_balance_features_from_observation(
    observation: dict[str, torch.Tensor],
) -> torch.Tensor:
    contact = observation.get("normal_force_valid")
    force = observation.get("normal_force")
    state = observation.get("robot_state")
    if contact is None or force is None or state is None:
        raise ValueError("runtime observation lacks contact-balance fields")
    if contact.ndim != 3 or contact.shape[-1] != 5:
        raise ValueError("normal_force_valid must have shape [B, H, 5]")
    if force.shape != contact.shape:
        raise ValueError("normal_force must match normal_force_valid")
    if state.shape[:2] != contact.shape[:2] or state.shape[-1] != 15:
        raise ValueError("robot_state must have shape [B, H, 15]")
    return torch.cat(
        (
            contact[..., 1:4].to(force.dtype),
            (force[..., 1:4] / 10.0).clamp(0.0, 1.0),
            state,
        ),
        dim=-1,
    ).flatten(1)


class ContactBalanceRotationHead(nn.Module):
    """Predict a unit posture-correction direction for partial contact."""

    def __init__(self, input_dim: int, *, hidden_dim: int = 32):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("contact-balance dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        with torch.no_grad():
            self.network[-1].bias.copy_(torch.tensor([0.0, 1.0, 0.0]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} contact-balance features, "
                f"got {features.shape[-1]}"
            )
        raw = self.network(features)
        return torch.nn.functional.normalize(raw, dim=-1, eps=1e-6)


def compose_contact_balance_rotation(
    baseline_rotation_rad: torch.Tensor,
    predicted_direction: torch.Tensor,
    contact_count: torch.Tensor,
    *,
    maximum_step_rad: float = 4e-5,
    active_contact_counts: tuple[int, ...] = (1, 2),
) -> torch.Tensor:
    if baseline_rotation_rad.shape != predicted_direction.shape:
        raise ValueError("rotation tensors must share shape")
    if baseline_rotation_rad.shape[-1] != 3:
        raise ValueError("rotation must have three channels")
    if contact_count.shape != baseline_rotation_rad.shape[:-1]:
        raise ValueError("contact_count shape does not match rotation")
    if maximum_step_rad <= 0.0:
        raise ValueError("maximum_step_rad must be positive")
    if not active_contact_counts or any(
        value not in (1, 2) for value in active_contact_counts
    ):
        raise ValueError("active_contact_counts must contain 1 and/or 2")
    partial = torch.zeros_like(contact_count, dtype=torch.bool)
    for value in active_contact_counts:
        partial |= contact_count == int(value)
    candidate = (
        torch.nn.functional.normalize(predicted_direction, dim=-1, eps=1e-6)
        * float(maximum_step_rad)
    )
    return torch.where(partial[..., None], candidate, baseline_rotation_rad)


def observable_force_balance_rotation(
    contact: np.ndarray,
    force_n: np.ndarray,
    finger_centers_world_m: np.ndarray,
    rotation_world_from_probe: np.ndarray,
    *,
    gain_rad_per_n: float = 3e-4,
    maximum_step_rad: float = 4e-5,
    minimum_single_contact_imbalance_n: float = 2.0,
) -> np.ndarray:
    """Compute a truth-free fingertip balance rotation from current forces.

    The axis comes from the known index-to-ring geometry. The signed step uses
    only the observable outer-finger force imbalance. This is a deterministic
    lower bound for evaluating learned recovery heads, not a pulse-location
    oracle.
    """

    contact = np.asarray(contact, dtype=bool)
    force_n = np.asarray(force_n, dtype=float)
    centers = np.asarray(finger_centers_world_m, dtype=float)
    rotation = np.asarray(rotation_world_from_probe, dtype=float)
    if contact.shape != (3,) or force_n.shape != (3,):
        raise ValueError("diagnostic contact and force must have shape (3,)")
    if centers.shape != (3, 3) or rotation.shape != (3, 3):
        raise ValueError("diagnostic centers and probe rotation must be 3-D")
    if not np.isfinite(force_n).all() or not np.isfinite(centers).all():
        raise ValueError("observable force-balance inputs must be finite")
    if not np.isfinite(rotation).all():
        raise ValueError("probe rotation must be finite")
    if gain_rad_per_n <= 0.0 or maximum_step_rad <= 0.0:
        raise ValueError("force-balance gain and step bound must be positive")
    if minimum_single_contact_imbalance_n < 0.0:
        raise ValueError("minimum single-contact imbalance must be non-negative")
    if contact.all():
        return np.zeros(3, dtype=np.float32)

    separation = centers[-1] - centers[0]
    separation[2] = 0.0
    axis_world = np.cross(separation, np.asarray([0.0, 0.0, 1.0]))
    axis_norm = float(np.linalg.norm(axis_world))
    if axis_norm <= 1e-12:
        raise ValueError("outer diagnostic finger centers are degenerate")
    axis_probe = rotation.T @ (axis_world / axis_norm)

    imbalance = float(force_n[-1] - force_n[0])
    if abs(imbalance) < 1e-6:
        imbalance = 1.0 if contact[-1] else -1.0
    if bool(contact[0]) != bool(contact[-1]):
        sign = 1.0 if contact[-1] else -1.0
        imbalance = sign * max(
            abs(imbalance), float(minimum_single_contact_imbalance_n)
        )
    step = float(
        np.clip(
            float(gain_rad_per_n) * imbalance,
            -float(maximum_step_rad),
            float(maximum_step_rad),
        )
    )
    return (axis_probe * step).astype(np.float32)
