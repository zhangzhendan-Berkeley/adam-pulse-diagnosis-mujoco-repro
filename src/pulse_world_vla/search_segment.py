from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .action_chunk import MultimodalHistoryEncoder


@dataclass(frozen=True)
class SearchSegmentPrediction:
    stop_probability: torch.Tensor
    direction_probability: torch.Tensor
    magnitude_normalized: torch.Tensor
    signed_displacement_normalized: torch.Tensor
    direction_sign: torch.Tensor

    @property
    def direction_confidence(self) -> torch.Tensor:
        """Confidence of the selected direction, in ``[0.5, 1.0]``."""

        return torch.maximum(
            self.direction_probability,
            1.0 - self.direction_probability,
        )


class SearchSegmentPolicy(nn.Module):
    """Predict a bounded next palpation segment, not a frame-action chunk."""

    def __init__(
        self,
        *,
        state_dim: int = 15,
        proximity_dim: int = 5,
        action_dim: int = 7,
        model_dim: int = 128,
        encoder_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        self.history = MultimodalHistoryEncoder(
            state_dim=state_dim,
            proximity_dim=proximity_dim,
            action_dim=action_dim,
            model_dim=model_dim,
            layers=encoder_layers,
            heads=heads,
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, 1)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 3),
        )

    def logits(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = self.history(observation)
        weights = self.attention(tokens).softmax(dim=1)
        return self.head((tokens * weights).sum(dim=1))

    def forward(self, observation: dict[str, torch.Tensor]) -> SearchSegmentPrediction:
        logits = self.logits(observation)
        stop = logits[:, 0].sigmoid()
        direction = logits[:, 1].sigmoid()
        magnitude = logits[:, 2].sigmoid()
        # Direction probability remains available for confidence calibration,
        # but it must not attenuate the independently predicted magnitude.
        # A 0.51 direction probability therefore means "+magnitude", not 2%
        # of that magnitude.  Resolve the exact 0.5 tie deterministically to
        # positive while exposing its low confidence to downstream gating.
        sign = torch.where(
            direction >= 0.5,
            torch.ones_like(direction),
            -torch.ones_like(direction),
        )
        return SearchSegmentPrediction(
            stop_probability=stop,
            direction_probability=direction,
            magnitude_normalized=magnitude,
            signed_displacement_normalized=sign * magnitude,
            direction_sign=sign,
        )

    def loss(
        self,
        observation: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        *,
        stop_positive_weight: float = 1.0,
        direction_weight: float = 1.0,
        magnitude_weight: float = 1.0,
        direction_min_magnitude_normalized: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        logits = self.logits(observation)
        stop_target = target["stop"].float()
        stop_loss = nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0],
            stop_target,
            pos_weight=logits.new_tensor(float(stop_positive_weight)),
        )
        moving = stop_target < 0.5
        magnitude_target = target["magnitude_normalized"].float()
        # Exact-zero displacement has no meaningful sign.  Callers can raise
        # the threshold (0.125 corresponds to 0.5 mm for a 4 mm radius) to
        # exclude sensor-scale near-zero targets from direction supervision.
        directional = (
            moving
            & (magnitude_target > 0.0)
            & (
                magnitude_target
                >= float(direction_min_magnitude_normalized)
            )
        )
        if directional.any():
            direction_loss = nn.functional.binary_cross_entropy_with_logits(
                logits[directional, 1],
                target["direction_positive"][directional].float(),
            )
        else:
            direction_loss = logits[:, 1].sum() * 0.0
        if moving.any():
            magnitude_loss = nn.functional.smooth_l1_loss(
                logits[moving, 2].sigmoid(),
                magnitude_target[moving],
            )
        else:
            magnitude_loss = logits[:, 2].sum() * 0.0
        total = (
            stop_loss
            + float(direction_weight) * direction_loss
            + float(magnitude_weight) * magnitude_loss
        )
        return {
            "loss": total,
            "stop": stop_loss,
            "direction": direction_loss,
            "magnitude": magnitude_loss,
        }
