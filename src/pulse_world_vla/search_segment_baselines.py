from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SegmentHeadOutput:
    stop_probability: torch.Tensor
    signed_displacement_normalized: torch.Tensor


def decode_segment_logits(logits: torch.Tensor) -> SegmentHeadOutput:
    """Decode the common three-logit segment contract.

    The third logit is an independent bounded magnitude. Direction is a
    discrete sign, so an uncertain direction never shrinks the magnitude.
    """

    stop = logits[:, 0].sigmoid()
    direction = logits[:, 1].sigmoid()
    magnitude = logits[:, 2].sigmoid()
    sign = torch.where(direction >= 0.5, 1.0, -1.0)
    return SegmentHeadOutput(stop, sign * magnitude)


class FeatureMLPSegmentPolicy(nn.Module):
    """Hand-crafted multimodal feature MLP baseline."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> SegmentHeadOutput:
        return decode_segment_logits(self.net(features))


class ACTSegmentPolicy(nn.Module):
    """Small ACT-style single-query decoder for one search segment.

    Unlike action-chunk BC, the query emits a bounded segment target directly.
    """

    def __init__(
        self,
        feature_dim: int,
        model_dim: int = 96,
        layers: int = 2,
        heads: int = 4,
        maximum_horizon: int = 4,
    ):
        super().__init__()
        self.maximum_horizon = int(maximum_horizon)
        self.frame = nn.Sequential(
            nn.Linear(feature_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.SiLU(),
        )
        self.position = nn.Parameter(torch.zeros(1, self.maximum_horizon, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.query = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.cross_attention = nn.MultiheadAttention(
            model_dim, heads, batch_first=True, dropout=0.0
        )
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 3),
        )

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] > self.maximum_horizon:
            raise ValueError("feature history exceeds ACT maximum horizon")
        memory = self.encoder(self.frame(features) + self.position[:, : features.shape[1]])
        query = self.query.expand(features.shape[0], -1, -1)
        attended, _ = self.cross_attention(query, memory, memory, need_weights=False)
        return self.head(attended[:, 0])

    def forward(self, features: torch.Tensor) -> SegmentHeadOutput:
        return decode_segment_logits(self.logits(features))


class FlowSegmentPolicy(nn.Module):
    """Tiny conditional flow-matching head for stop and signed displacement."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.time = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def velocity(
        self, features: torch.Tensor, noisy_target: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        condition = self.condition(features) + self.time(time[:, None])
        return self.head(torch.cat((condition, noisy_target), dim=-1))

    @torch.no_grad()
    def sample(
        self,
        features: torch.Tensor,
        *,
        steps: int = 8,
        generator: torch.Generator | None = None,
    ) -> SegmentHeadOutput:
        if steps < 1:
            raise ValueError("flow sampling steps must be positive")
        state = torch.randn(
            features.shape[0], 2, device=features.device, generator=generator
        )
        dt = 1.0 / float(steps)
        for index in range(steps):
            time = torch.full(
                (features.shape[0],), index / float(steps),
                device=features.device,
                dtype=features.dtype,
            )
            state = state + dt * self.velocity(features, state, time)
        return SegmentHeadOutput(state[:, 0].sigmoid(), state[:, 1])


def feature_frame_dim() -> int:
    # tactile mean/std/left-right gradient/delta and dynamic pulse centroid
    # (x/y), proximity + validity, state, previous action, tactile validity
    # and phase.
    return 6 + 5 + 5 + 15 + 7 + 1 + 1


def multimodal_feature_frames(observation: dict[str, torch.Tensor]) -> torch.Tensor:
    """Convert policy-visible history into compact per-frame features."""

    tactile = observation["tactile"][:, :, 0]
    batch, horizon, height, width = tactile.shape
    left = tactile[..., : width // 2].mean(dim=(-1, -2))
    right = tactile[..., width // 2 :].mean(dim=(-1, -2))
    tactile_mean = tactile.mean(dim=(-1, -2))
    tactile_std = tactile.flatten(-2).std(dim=-1, unbiased=False)
    tactile_gradient = right - left
    tactile_delta = torch.zeros_like(tactile_mean)
    delta_cx = torch.zeros_like(tactile_mean)
    delta_cy = torch.zeros_like(tactile_mean)
    if horizon > 1:
        delta_map = (tactile[:, 1:] - tactile[:, :-1]).abs()
        tactile_delta[:, 1:] = delta_map.mean(dim=(-1, -2))
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=tactile.device),
            torch.linspace(-1.0, 1.0, width, device=tactile.device),
            indexing="ij",
        )
        baseline = delta_map.median(dim=-1, keepdim=True).values.median(
            dim=-2, keepdim=True
        ).values
        weights = (delta_map - baseline).clamp_min(0.0)
        denominator = weights.sum(dim=(-1, -2)).clamp_min(1e-6)
        delta_cx[:, 1:] = (weights * xx).sum(dim=(-1, -2)) / denominator
        delta_cy[:, 1:] = (weights * yy).sum(dim=(-1, -2)) / denominator
    phase = observation["task_phase"].float().unsqueeze(-1)
    return torch.cat(
        (
            tactile_mean.unsqueeze(-1),
            tactile_std.unsqueeze(-1),
            tactile_gradient.unsqueeze(-1),
            tactile_delta.unsqueeze(-1),
            delta_cx.unsqueeze(-1),
            delta_cy.unsqueeze(-1),
            observation["proximity"],
            observation["proximity_valid"].float(),
            observation["robot_state"],
            observation["action"],
            observation["tactile_valid"].float().unsqueeze(-1),
            phase,
        ),
        dim=-1,
    )
