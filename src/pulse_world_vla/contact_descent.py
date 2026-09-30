from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


DIAGNOSTIC_FINGER_INDICES = (1, 2, 3)
DEFAULT_ACTION_SCALE = np.asarray(
    [0.025, 0.042, 0.030, 0.112, 0.092, 0.086, 150.0],
    dtype=np.float32,
)


def observable_diagnostic_contact(record: dict) -> np.ndarray:
    """Return the policy-visible index/middle/ring contact mask."""

    full = record.get("full_hand_contact")
    if full is not None:
        contact = np.asarray(full, dtype=bool)
        if contact.shape != (5,):
            raise ValueError("full_hand_contact must contain five fingers")
        return contact[list(DIAGNOSTIC_FINGER_INDICES)]
    active = np.asarray(record.get("active_contact", ()), dtype=bool)
    if active.shape != (3,):
        raise ValueError("record does not provide a three-finger contact mask")
    return active


def observable_diagnostic_force(record: dict) -> np.ndarray:
    """Return index/middle/ring force without using privileged state."""

    full = record.get("full_hand_normal_force_n")
    if full is not None:
        force = np.asarray(full, dtype=np.float32)
        if force.shape != (5,):
            raise ValueError("full_hand_normal_force_n must contain five fingers")
        return force[list(DIAGNOSTIC_FINGER_INDICES)]
    active = np.asarray(record.get("active_normal_force_n", ()), dtype=np.float32)
    if active.shape != (3,):
        active = np.asarray(record.get("normal_force_n", ()), dtype=np.float32)
    if active.shape != (3,):
        raise ValueError("record does not provide three diagnostic forces")
    return active


def contact_mask_bucket(record: dict) -> str:
    return f"{int(observable_diagnostic_contact(record).sum())}/3"


def target_action(record: dict, *, source: str = "expert_proposed_action") -> np.ndarray:
    if source not in ("expert_label_action", "expert_proposed_action"):
        raise ValueError("unsupported contact-descent target action source")
    action = np.asarray(record.get(source, ()), dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"{source} must have shape (7,)")
    return action


def contact_descent_feature_vector(
    records: Sequence[dict],
    end_index: int,
    *,
    observation_horizon: int = 4,
    action_scale: np.ndarray = DEFAULT_ACTION_SCALE,
    force_scale_n: float = 10.0,
    include_previous_action: bool = True,
) -> np.ndarray:
    """Build a truth-free temporal feature vector ending at ``end_index``.

    Each frame contributes three contact bits and three bounded normal forces.
    It may additionally include the six deployment-visible previous TCP motion
    channels. The target expert action is deliberately excluded.
    """

    end_index = int(end_index)
    observation_horizon = int(observation_horizon)
    if observation_horizon < 1:
        raise ValueError("observation_horizon must be positive")
    start = end_index - observation_horizon + 1
    if start < 0 or end_index >= len(records):
        raise IndexError("contact-descent history crosses the episode boundary")
    scale = np.asarray(action_scale, dtype=np.float32)
    if scale.shape != (7,) or np.any(scale <= 0.0):
        raise ValueError("action_scale must contain seven positive values")
    if force_scale_n <= 0.0:
        raise ValueError("force_scale_n must be positive")
    frames = []
    for record in records[start : end_index + 1]:
        previous = np.asarray(record.get("previous_action", ()), dtype=np.float32)
        if previous.shape != (7,):
            raise ValueError("previous_action must have shape (7,)")
        parts = [
            observable_diagnostic_contact(record).astype(np.float32),
            np.clip(
                observable_diagnostic_force(record) / float(force_scale_n),
                0.0,
                1.0,
            ),
        ]
        if include_previous_action:
            parts.append(np.clip(previous[:6] / scale[:6], -1.0, 1.0))
        frames.append(np.concatenate(parts))
    return np.concatenate(frames).astype(np.float32, copy=False)


def contact_descent_target(
    record: dict,
    *,
    epsilon_m: float,
    target_action_source: str = "expert_proposed_action",
) -> tuple[bool, float]:
    if epsilon_m < 0.0:
        raise ValueError("epsilon_m must be non-negative")
    z_m = float(target_action(record, source=target_action_source)[2])
    descends = z_m < -float(epsilon_m)
    return descends, max(0.0, -z_m)


def contact_descent_features_from_observation(
    observation: dict[str, torch.Tensor],
    *,
    include_previous_action: bool,
    force_scale_n: float = 10.0,
) -> torch.Tensor:
    """Build the training-equivalent feature tensor from runtime history."""

    contact = observation.get("normal_force_valid")
    force = observation.get("normal_force")
    action = observation.get("action")
    if contact is None or force is None or action is None:
        raise ValueError("runtime observation lacks contact-descent fields")
    if contact.ndim != 3 or contact.shape[-1] != 5:
        raise ValueError("normal_force_valid must have shape [B, H, 5]")
    if force.shape != contact.shape:
        raise ValueError("normal_force must match normal_force_valid")
    if action.shape[:2] != contact.shape[:2] or action.shape[-1] != 7:
        raise ValueError("action history must have shape [B, H, 7]")
    parts = [
        contact[..., 1:4].to(force.dtype),
        (force[..., 1:4] / float(force_scale_n)).clamp(0.0, 1.0),
    ]
    if include_previous_action:
        parts.append(action[..., :6].clamp(-1.0, 1.0))
    return torch.cat(parts, dim=-1).flatten(1)


class ContactDescentHead(nn.Module):
    """Small recovery-only classifier plus positive descent magnitude head."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        maximum_descent_m: float = 1.8e-4,
    ):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("contact-descent dimensions must be positive")
        if maximum_descent_m <= 0.0:
            raise ValueError("maximum_descent_m must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.maximum_descent_m = float(maximum_descent_m)
        self.backbone = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.descent_logit = nn.Linear(self.hidden_dim, 1)
        self.descent_magnitude = nn.Linear(self.hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} contact-descent features, "
                f"got {features.shape[-1]}"
            )
        latent = self.backbone(features)
        logits = self.descent_logit(latent).squeeze(-1)
        magnitude_m = (
            torch.sigmoid(self.descent_magnitude(latent).squeeze(-1))
            * self.maximum_descent_m
        )
        return logits, magnitude_m


def compose_contact_descent_z(
    baseline_z_m: torch.Tensor,
    descent_logits: torch.Tensor,
    descent_magnitude_m: torch.Tensor,
    incomplete_contact: torch.Tensor,
    *,
    probability_threshold: float = 0.5,
) -> torch.Tensor:
    """Apply the optional head only when observable three-finger contact is incomplete."""

    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability_threshold must be in (0, 1)")
    if not (
        baseline_z_m.shape
        == descent_logits.shape
        == descent_magnitude_m.shape
        == incomplete_contact.shape
    ):
        raise ValueError("contact-descent composition tensors must share shape")
    decision = torch.sigmoid(descent_logits) >= probability_threshold
    descend = incomplete_contact.bool() & decision
    no_descent = incomplete_contact.bool() & ~decision
    result = torch.where(descend, -descent_magnitude_m.abs(), baseline_z_m)
    # A negative baseline would contradict the explicit no-descent decision.
    return torch.where(no_descent, result.clamp_min(0.0), result)
