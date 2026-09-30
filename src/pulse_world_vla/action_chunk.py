from __future__ import annotations

import math

import torch
from torch import nn


FULL56_FINGER_SLICES = (
    slice(0, 16),
    slice(16, 26),
    slice(26, 36),
    slice(36, 46),
    slice(46, 56),
)


class FullHandTactileEncoder(nn.Module):
    """Topology-aware encoder for five fingers and 56 linear taxels."""

    def __init__(self, model_dim: int = 32):
        super().__init__()
        self.channel_position = nn.Parameter(torch.zeros(1, 16, model_dim))
        self.finger_embedding = nn.Parameter(torch.zeros(1, 5, model_dim))
        self.channel_projection = nn.Linear(3, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 32))

    def forward(
        self,
        values: torch.Tensor,
        delta: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape != delta.shape or values.shape != valid.shape:
            raise ValueError("full56 tactile tensors must share shape")
        if values.ndim != 3 or values.shape[-1] != 56:
            raise ValueError("full56 tactile input must be [batch,time,56]")
        batch, horizon = values.shape[:2]
        finger_tokens = []
        for finger_index, finger_slice in enumerate(FULL56_FINGER_SLICES):
            finger_values = values[..., finger_slice]
            finger_delta = delta[..., finger_slice]
            finger_valid = valid[..., finger_slice]
            count = finger_values.shape[-1]
            features = torch.stack(
                (finger_values, finger_delta, finger_valid.to(values.dtype)), dim=-1
            ).flatten(0, 1)
            tokens = (
                self.channel_projection(features)
                + self.channel_position[:, :count]
                + self.finger_embedding[:, finger_index : finger_index + 1]
            )
            flat_valid = finger_valid.flatten(0, 1)
            padding_mask = ~flat_valid
            all_invalid = ~flat_valid.any(dim=1)
            if all_invalid.any():
                padding_mask = padding_mask.clone()
                padding_mask[all_invalid, 0] = False
            encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
            mask = flat_valid.to(encoded.dtype)[..., None]
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            finger_tokens.append(pooled.unflatten(0, (batch, horizon)))
        output = self.output(torch.stack(finger_tokens, dim=2)).mean(dim=2)
        return output * valid.any(dim=-1).to(output.dtype)[..., None]


class MultimodalHistoryEncoder(nn.Module):
    """Encode a short synchronized multimodal history into temporal tokens."""

    def __init__(
        self,
        state_dim=12,
        proximity_dim=5,
        action_dim=7,
        model_dim=128,
        layers=2,
        heads=4,
        maximum_horizon=16,
        structured_tactile: bool = False,
        fusion_mode: str = "vanilla",
        fusion_architecture: str = "early_concat",
    ):
        super().__init__()
        self.maximum_horizon = int(maximum_horizon)
        self.structured_tactile = bool(structured_tactile)
        if fusion_mode not in {"vanilla", "tactile_gate", "contact_attention_bias"}:
            raise ValueError(f"unsupported fusion mode: {fusion_mode}")
        self.fusion_mode = str(fusion_mode)
        if fusion_architecture not in {"early_concat", "modality_tokens"}:
            raise ValueError(
                f"unsupported fusion architecture: {fusion_architecture}"
            )
        self.fusion_architecture = str(fusion_architecture)
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
        self.full_hand_tactile = (
            FullHandTactileEncoder() if self.structured_tactile else None
        )
        self.tactile_gate = nn.Sequential(nn.Linear(6, 32), nn.Sigmoid())
        if self.fusion_architecture == "early_concat":
            feature_dim = (
                32
                + 16
                + proximity_dim
                + proximity_dim
                + state_dim
                + action_dim
                + 1
                + (32 + 5 + 5 if self.structured_tactile else 0)
            )
            self.projection = nn.Sequential(
                nn.Linear(feature_dim, model_dim),
                nn.LayerNorm(model_dim),
                nn.SiLU(),
            )
        else:
            modality_widths = [32, 16, proximity_dim * 2, state_dim, action_dim]
            if self.structured_tactile:
                modality_widths.append(32 + 5 + 5)
            self.modality_projections = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(width, model_dim),
                    nn.LayerNorm(model_dim),
                    nn.SiLU(),
                )
                for width in modality_widths
            )
            self.modality_embedding = nn.Parameter(
                torch.zeros(1, 1, len(modality_widths), model_dim)
            )
            self.pooling_norm = nn.LayerNorm(model_dim)
        self.position = nn.Parameter(
            torch.zeros(1, self.maximum_horizon, model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        vision = observation["vision"]
        batch, horizon = vision.shape[:2]
        if horizon > self.maximum_horizon:
            raise ValueError(
                f"observation horizon {horizon} exceeds "
                f"{self.maximum_horizon}"
            )
        tactile_valid = observation["tactile_valid"].float().unsqueeze(-1)
        vision_feature = self.vision(vision.flatten(0, 1)).unflatten(
            0, (batch, horizon)
        )
        tactile_feature = self.tactile(
            observation["tactile"].flatten(0, 1)
        ).unflatten(0, (batch, horizon))
        tactile_feature = tactile_feature * tactile_valid
        structured_features = []
        if self.structured_tactile:
            required = (
                "tactile_full56",
                "tactile_full56_delta",
                "tactile_full56_valid",
                "normal_force",
                "normal_force_valid",
            )
            missing = [name for name in required if name not in observation]
            if missing:
                raise ValueError(f"structured tactile observation missing {missing}")
            full_feature = self.full_hand_tactile(
                observation["tactile_full56"],
                observation["tactile_full56_delta"],
                observation["tactile_full56_valid"],
            )
            contact_strength = torch.cat(
                (
                    observation["normal_force"].clamp(0, 10) / 10.0,
                    observation["tactile_full56_delta"].abs().mean(dim=-1, keepdim=True),
                ),
                dim=-1,
            )
            if self.fusion_mode != "vanilla":
                full_feature = full_feature * (1.0 + self.tactile_gate(contact_strength))
            structured_features = [
                full_feature,
                observation["normal_force"] / 10.0,
                observation["normal_force_valid"].to(vision.dtype),
            ]
        if self.fusion_architecture == "early_concat":
            features = torch.cat(
                (
                    vision_feature,
                    tactile_feature,
                    observation["proximity"],
                    observation["proximity_valid"].float(),
                    observation["robot_state"],
                    observation["action"],
                    tactile_valid,
                    *structured_features,
                ),
                dim=-1,
            )
            return self.temporal(
                self.projection(features) + self.position[:, :horizon]
            )

        proximity_valid = observation["proximity_valid"].bool()
        modality_features = [
            vision_feature,
            tactile_feature,
            torch.cat(
                (observation["proximity"], proximity_valid.to(vision.dtype)),
                dim=-1,
            ),
            observation["robot_state"],
            observation["action"],
        ]
        modality_valid = [
            torch.ones((batch, horizon), dtype=torch.bool, device=vision.device),
            observation["tactile_valid"].bool(),
            proximity_valid.any(dim=-1),
            torch.ones((batch, horizon), dtype=torch.bool, device=vision.device),
            torch.ones((batch, horizon), dtype=torch.bool, device=vision.device),
        ]
        if self.structured_tactile:
            modality_features.append(torch.cat(tuple(structured_features), dim=-1))
            modality_valid.append(observation["tactile_full56_valid"].any(dim=-1))

        tokens = torch.stack(
            [
                projection(feature)
                for projection, feature in zip(
                    self.modality_projections, modality_features, strict=True
                )
            ],
            dim=2,
        )
        valid = torch.stack(modality_valid, dim=2)
        tokens = (
            tokens
            + self.modality_embedding
            + self.position[:, :horizon, None, :]
        )
        encoded = self.temporal(
            tokens.flatten(1, 2),
            src_key_padding_mask=~valid.flatten(1, 2),
        ).unflatten(1, (horizon, len(modality_features)))
        weights = valid.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1)
        return self.pooling_norm(pooled)


class ACTStyleActionChunkPolicy(nn.Module):
    """Small ACT-style learned-query decoder for a future action chunk."""

    def __init__(
        self,
        state_dim=12,
        proximity_dim=5,
        action_dim=7,
        prediction_horizon=4,
        model_dim=128,
        encoder_layers=2,
        decoder_layers=2,
        heads=4,
        residual_scale=1.0,
        structured_tactile: bool = False,
        fusion_mode: str = "vanilla",
        fusion_architecture: str = "early_concat",
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.residual_scale = float(residual_scale)
        self.history = MultimodalHistoryEncoder(
            state_dim,
            proximity_dim,
            action_dim,
            model_dim,
            encoder_layers,
            heads,
            structured_tactile=structured_tactile,
            fusion_mode=fusion_mode,
            fusion_architecture=fusion_architecture,
        )
        self.action_queries = nn.Parameter(
            torch.zeros(1, self.prediction_horizon, model_dim)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, decoder_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, self.action_dim),
            nn.Tanh(),
        )
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        memory = self.history(observation)
        query = self.action_queries.expand(memory.shape[0], -1, -1)
        delta = self.head(self.decoder(query, memory))
        previous = observation["action"][:, -1, : self.action_dim]
        return torch.clamp(
            previous[:, None] + self.residual_scale * delta,
            min=-1.0,
            max=1.0,
        )


class DirectTransformerChunkPolicy(nn.Module):
    """Temporal encoder with a single direct action-chunk regression head."""

    def __init__(
        self,
        state_dim=12,
        proximity_dim=5,
        action_dim=7,
        prediction_horizon=4,
        model_dim=128,
        encoder_layers=2,
        heads=4,
        residual_scale=1.0,
        structured_tactile: bool = False,
        fusion_mode: str = "vanilla",
        fusion_architecture: str = "early_concat",
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.residual_scale = float(residual_scale)
        self.history = MultimodalHistoryEncoder(
            state_dim,
            proximity_dim,
            action_dim,
            model_dim,
            encoder_layers,
            heads,
            structured_tactile=structured_tactile,
            fusion_mode=fusion_mode,
            fusion_architecture=fusion_architecture,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, self.prediction_horizon * self.action_dim),
            nn.Tanh(),
        )
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.history(observation)[:, -1]
        delta = self.head(encoded).unflatten(
            -1, (self.prediction_horizon, self.action_dim)
        )
        previous = observation["action"][:, -1, : self.action_dim]
        return torch.clamp(
            previous[:, None] + self.residual_scale * delta,
            min=-1.0,
            max=1.0,
        )


class ActionChunkFlowPolicy(nn.Module):
    """Conditional rectified-flow model over a normalized action chunk."""

    def __init__(
        self,
        state_dim=12,
        proximity_dim=5,
        action_dim=7,
        prediction_horizon=4,
        model_dim=128,
        encoder_layers=2,
        flow_layers=4,
        heads=4,
        structured_tactile: bool = False,
        fusion_mode: str = "vanilla",
        fusion_architecture: str = "early_concat",
        prior_noise_scale: float = 1.0,
        warm_start: bool = False,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.model_dim = int(model_dim)
        self.prior_noise_scale = float(prior_noise_scale)
        self.warm_start = bool(warm_start)
        if not 0.0 <= self.prior_noise_scale <= 1.0:
            raise ValueError("prior_noise_scale must be in [0, 1]")
        self.history = MultimodalHistoryEncoder(
            state_dim,
            proximity_dim,
            action_dim,
            model_dim,
            encoder_layers,
            heads,
            structured_tactile=structured_tactile,
            fusion_mode=fusion_mode,
            fusion_architecture=fusion_architecture,
        )
        self.action_projection = nn.Linear(action_dim, model_dim)
        self.action_position = nn.Parameter(
            torch.zeros(1, self.prediction_horizon, model_dim)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.flow = nn.TransformerDecoder(layer, flow_layers)
        self.velocity = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, action_dim),
        )

    def _time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.model_dim // 2
        frequency = torch.exp(
            torch.arange(
                half, device=timestep.device, dtype=timestep.dtype
            )
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angle = timestep[:, None] * frequency[None] * 1000.0
        embedding = torch.cat((angle.sin(), angle.cos()), dim=-1)
        if embedding.shape[-1] < self.model_dim:
            embedding = nn.functional.pad(embedding, (0, 1))
        return self.time_mlp(embedding)

    def predict_velocity(
        self,
        observation: dict[str, torch.Tensor],
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.history(observation)
        tokens = (
            self.action_projection(noisy_action)
            + self.action_position[:, : noisy_action.shape[1]]
            + self._time_embedding(timestep)[:, None]
        )
        return self.velocity(self.flow(tokens, memory))

    def flow_loss(
        self,
        observation: dict[str, torch.Tensor],
        target: torch.Tensor,
        *,
        reduction="none",
    ) -> torch.Tensor:
        if self.warm_start:
            previous = observation["action"][:, -1:, : self.action_dim].expand_as(
                target
            )
            noise = (
                previous
                + self.prior_noise_scale * torch.randn_like(target)
            ).clamp(-1.0, 1.0)
        else:
            noise = torch.randn_like(target)
        timestep = torch.sigmoid(
            torch.randn(
                target.shape[0],
                device=target.device,
                dtype=target.dtype,
            )
        )
        mixed = (
            (1.0 - timestep[:, None, None]) * noise
            + timestep[:, None, None] * target
        )
        prediction = self.predict_velocity(observation, mixed, timestep)
        return nn.functional.smooth_l1_loss(
            prediction, target - noise, reduction=reduction
        )

    @torch.no_grad()
    def sample(
        self,
        observation: dict[str, torch.Tensor],
        *,
        steps=10,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        batch = observation["action"].shape[0]
        random_action = torch.randn(
            batch,
            self.prediction_horizon,
            self.action_dim,
            device=observation["action"].device,
            dtype=observation["action"].dtype,
            generator=generator,
        )
        if self.warm_start:
            previous = observation["action"][:, -1:, : self.action_dim].expand(
                batch, self.prediction_horizon, self.action_dim
            )
            action = (
                previous + self.prior_noise_scale * random_action
            ).clamp(-1.0, 1.0)
        else:
            action = random_action
        for index in range(steps):
            timestep = torch.full(
                (batch,),
                (index + 0.5) / steps,
                device=action.device,
                dtype=action.dtype,
            )
            action = action + self.predict_velocity(
                observation, action, timestep
            ) / steps
        return torch.clamp(action, -1.0, 1.0)
