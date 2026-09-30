from __future__ import annotations

import torch
from torch import nn


class BCPolicy(nn.Module):
    """Small last-frame multimodal behavior-cloning baseline."""

    def __init__(
        self,
        state_dim=12,
        proximity_dim=5,
        action_dim=7,
        *,
        residual_action: bool = False,
        residual_scale: float = 0.5,
        contact_gated_pressure: bool = False,
    ):
        super().__init__()
        self.residual_action = bool(residual_action)
        self.residual_scale = float(residual_scale)
        self.contact_gated_pressure = bool(contact_gated_pressure)
        self.vision = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.tactile = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        feature_dim = (
            32 + 16 + proximity_dim + proximity_dim + state_dim
            + action_dim + 1
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        if self.residual_action:
            nn.init.zeros_(self.head[-2].weight)
            nn.init.zeros_(self.head[-2].bias)

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        vision = observation["vision"][:, -1]
        tactile = observation["tactile"][:, -1]
        tactile_valid = observation["tactile_valid"][:, -1, None].float()
        tactile_feature = self.tactile(tactile) * tactile_valid
        features = torch.cat(
            (
                self.vision(vision),
                tactile_feature,
                observation["proximity"][:, -1],
                observation["proximity_valid"][:, -1].float(),
                observation["robot_state"][:, -1],
                observation["action"][:, -1],
                tactile_valid,
            ),
            dim=-1,
        )
        prediction = self.head(features)
        if self.residual_action:
            previous = observation["action"][:, -1]
            prediction = torch.clamp(
                previous + self.residual_scale * prediction,
                min=-1.0,
                max=1.0,
            )
        if self.contact_gated_pressure and prediction.shape[-1] >= 7:
            prediction = prediction.clone()
            prediction[:, 6] *= tactile_valid[:, 0]
        return prediction


class TemporalTransformerBCPolicy(nn.Module):
    """Per-frame multimodal encoder followed by a temporal Transformer."""

    def __init__(
        self,
        state_dim=12,
        proximity_dim=5,
        action_dim=7,
        *,
        model_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        maximum_horizon: int = 16,
        residual_action: bool = True,
        residual_scale: float = 0.5,
        contact_gated_pressure: bool = False,
    ):
        super().__init__()
        self.residual_action = bool(residual_action)
        self.residual_scale = float(residual_scale)
        self.contact_gated_pressure = bool(contact_gated_pressure)
        self.maximum_horizon = int(maximum_horizon)
        self.vision = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.tactile = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        frame_feature_dim = (
            32
            + 16
            + proximity_dim
            + proximity_dim
            + state_dim
            + action_dim
            + 1
        )
        self.frame_projection = nn.Sequential(
            nn.Linear(frame_feature_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.SiLU(),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.maximum_horizon, model_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        if self.residual_action:
            nn.init.zeros_(self.head[-2].weight)
            nn.init.zeros_(self.head[-2].bias)

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        vision = observation["vision"]
        batch, horizon = vision.shape[:2]
        if horizon > self.maximum_horizon:
            raise ValueError(
                f"observation horizon {horizon} exceeds {self.maximum_horizon}"
            )
        tactile = observation["tactile"]
        tactile_valid = observation["tactile_valid"].float().unsqueeze(-1)
        vision_feature = self.vision(vision.flatten(0, 1)).unflatten(
            0, (batch, horizon)
        )
        tactile_feature = self.tactile(
            tactile.flatten(0, 1)
        ).unflatten(0, (batch, horizon))
        tactile_feature = tactile_feature * tactile_valid
        frame_features = torch.cat(
            (
                vision_feature,
                tactile_feature,
                observation["proximity"],
                observation["proximity_valid"].float(),
                observation["robot_state"],
                observation["action"],
                tactile_valid,
            ),
            dim=-1,
        )
        tokens = (
            self.frame_projection(frame_features)
            + self.position_embedding[:, :horizon]
        )
        encoded = self.temporal_encoder(tokens)
        prediction = self.head(encoded[:, -1])
        if self.residual_action:
            prediction = torch.clamp(
                observation["action"][:, -1]
                + self.residual_scale * prediction,
                min=-1.0,
                max=1.0,
            )
        if self.contact_gated_pressure and prediction.shape[-1] >= 7:
            prediction = prediction.clone()
            prediction[:, 6] *= tactile_valid[:, -1, 0]
        return prediction
