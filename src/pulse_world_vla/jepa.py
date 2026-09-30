from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .action_chunk import FullHandTactileEncoder
from pulse_mujoco.task import TaskPhase


class MultimodalJEPAEncoder(nn.Module):
    """Lightweight modality-token encoder for multimodal future targets.

    Each sensor keeps a small modality-specific projection, while one shared
    Transformer models cross-modal and temporal dependencies.  The returned
    per-modality tokens make it possible to mask unavailable sensors in the
    JEPA target loss instead of treating missing data as zero-valued evidence.
    """

    MODALITIES = ("vision", "tactile", "full_hand_tactile", "proximity", "state")

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        proximity_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        maximum_horizon: int = 16,
    ):
        super().__init__()
        self.maximum_horizon = int(maximum_horizon)
        self.vision = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.tactile_image = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.full_hand = FullHandTactileEncoder()
        widths = (32, 16, 32 + 10, proximity_dim * 2, state_dim + action_dim)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, model_dim),
                nn.LayerNorm(model_dim),
                nn.SiLU(),
            )
            for width in widths
        )
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, 1, len(self.MODALITIES), model_dim)
        )
        self.position = nn.Parameter(torch.zeros(1, self.maximum_horizon, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.shared = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )

    def forward(
        self, sample: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        vision = sample["vision"]
        batch, horizon = vision.shape[:2]
        if horizon > self.maximum_horizon:
            raise ValueError("multimodal JEPA horizon exceeds configured maximum")
        vision_feature = self.vision(vision.flatten(0, 1)).unflatten(
            0, (batch, horizon)
        )
        tactile_feature = self.tactile_image(
            sample["tactile"].flatten(0, 1)
        ).unflatten(0, (batch, horizon))
        tactile_feature = tactile_feature * sample["tactile_valid"].float()[..., None]
        full_feature = self.full_hand(
            sample["tactile_full56"],
            sample["tactile_full56_delta"],
            sample["tactile_full56_valid"],
        )
        full_force = torch.cat(
            (
                sample["normal_force"].clamp(0, 10) / 10.0,
                sample["normal_force_valid"].to(full_feature.dtype),
            ),
            dim=-1,
        )
        features = (
            vision_feature,
            tactile_feature,
            torch.cat((full_feature, full_force), dim=-1),
            torch.cat(
                (sample["proximity"], sample["proximity_valid"].float()), dim=-1
            ),
            torch.cat((sample["robot_state"], sample["action"]), dim=-1),
        )
        valid = {
            "vision": torch.ones(
                (batch, horizon), dtype=torch.bool, device=vision.device
            ),
            "tactile": sample["tactile_valid"].bool(),
            "full_hand_tactile": sample["tactile_full56_valid"].any(dim=-1),
            "proximity": sample["proximity_valid"].any(dim=-1),
            "state": torch.ones(
                (batch, horizon), dtype=torch.bool, device=vision.device
            ),
        }
        tokens = torch.stack(
            [
                projection(feature)
                for projection, feature in zip(
                    self.projections, features, strict=True
                )
            ],
            dim=2,
        )
        token_valid = torch.stack(
            tuple(valid[name] for name in self.MODALITIES), dim=2
        )
        tokens = (
            tokens
            + self.modality_embedding
            + self.position[:, :horizon, None, :]
        )
        encoded = self.shared(
            tokens.flatten(1, 2),
            src_key_padding_mask=~token_valid.flatten(1, 2),
        ).unflatten(1, (horizon, len(self.MODALITIES)))
        return {
            name: encoded[:, :, index]
            for index, name in enumerate(self.MODALITIES)
        }, valid


class TactileStateEncoder(nn.Module):
    """Encode policy-visible tactile dynamics and robot state separately."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        maximum_horizon: int = 16,
    ):
        super().__init__()
        self.maximum_horizon = int(maximum_horizon)
        self.tactile = FullHandTactileEncoder()
        self.tactile_projection = nn.Sequential(
            nn.Linear(32 + 10, model_dim),
            nn.LayerNorm(model_dim),
            nn.SiLU(),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim + action_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.SiLU(),
        )
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
        self.tactile_temporal = nn.TransformerEncoder(
            deepcopy(layer), num_layers=layers, enable_nested_tensor=False
        )
        self.state_temporal = nn.TransformerEncoder(
            deepcopy(layer), num_layers=layers, enable_nested_tensor=False
        )
        self.joint = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.SiLU(),
        )

    def forward(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        horizon = sample["robot_state"].shape[1]
        if horizon > self.maximum_horizon:
            raise ValueError("JEPA horizon exceeds configured maximum")
        tactile = self.tactile(
            sample["tactile_full56"],
            sample["tactile_full56_delta"],
            sample["tactile_full56_valid"],
        )
        force = torch.cat(
            (
                sample["normal_force"].clamp(0, 10) / 10.0,
                sample["normal_force_valid"].to(tactile.dtype),
            ),
            dim=-1,
        )
        position = self.position[:, :horizon]
        tactile = self.tactile_temporal(
            self.tactile_projection(torch.cat((tactile, force), dim=-1))
            + position
        )
        state = self.state_temporal(
            self.state_projection(
                torch.cat((sample["robot_state"], sample["action"]), dim=-1)
            )
            + position
        )
        return {
            "tactile": tactile,
            "state": state,
            "joint": self.joint(torch.cat((tactile, state), dim=-1)),
        }


class JEPALatentWorldPolicy(nn.Module):
    """EMA-target latent world model with an auxiliary action-chunk head."""

    def __init__(
        self,
        state_dim=15,
        action_dim=7,
        prediction_horizon=4,
        model_dim=128,
        encoder_layers=2,
        predictor_layers=2,
        heads=4,
        ema_decay=0.99,
        action_loss_weight=4.0,
        latent_loss_weight=1.0,
        variance_loss_weight=0.1,
        residual_scale=1.0,
        encoder_architecture: str = "tactile_state",
        proximity_dim: int = 5,
        modality_loss_weights: dict[str, float] | None = None,
        future_steps: tuple[int, ...] | list[int] | None = None,
        loss_weighting: str = "fixed",
        adaptive_loss_ema_decay: float = 0.95,
        adaptive_loss_min_weight: float = 0.5,
        adaptive_loss_max_weight: float = 2.0,
        action_phase_loss_weights: dict[str, float] | None = None,
        action_channel_loss_weights: tuple[float, ...] | list[float] | None = None,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.ema_decay = float(ema_decay)
        self.action_loss_weight = float(action_loss_weight)
        self.latent_loss_weight = float(latent_loss_weight)
        self.variance_loss_weight = float(variance_loss_weight)
        self.residual_scale = float(residual_scale)
        if encoder_architecture not in {"tactile_state", "multimodal_shared"}:
            raise ValueError(
                f"unsupported JEPA encoder architecture: {encoder_architecture}"
            )
        self.encoder_architecture = str(encoder_architecture)
        self.modality_loss_weights = {
            name: 1.0 for name in MultimodalJEPAEncoder.MODALITIES
        }
        if modality_loss_weights:
            self.modality_loss_weights.update(
                {
                    str(name): float(value)
                    for name, value in modality_loss_weights.items()
                }
            )
        self.future_steps = tuple(
            range(1, self.prediction_horizon + 1)
            if future_steps is None
            else sorted({int(step) for step in future_steps})
        )
        if not self.future_steps or any(
            step < 1 or step > self.prediction_horizon
            for step in self.future_steps
        ):
            raise ValueError("JEPA future_steps must fall within prediction horizon")
        if loss_weighting not in {"fixed", "adaptive_modality"}:
            raise ValueError(f"unsupported JEPA loss weighting: {loss_weighting}")
        self.loss_weighting = str(loss_weighting)
        self.adaptive_loss_ema_decay = float(adaptive_loss_ema_decay)
        self.adaptive_loss_min_weight = float(adaptive_loss_min_weight)
        self.adaptive_loss_max_weight = float(adaptive_loss_max_weight)
        self.action_phase_loss_weights = {
            str(name): float(value)
            for name, value in (action_phase_loss_weights or {}).items()
        }
        channel_weights = torch.as_tensor(
            (
                [1.0] * self.action_dim
                if action_channel_loss_weights is None
                else action_channel_loss_weights
            ),
            dtype=torch.float32,
        )
        if (
            channel_weights.shape != (self.action_dim,)
            or not torch.isfinite(channel_weights).all()
            or torch.any(channel_weights <= 0.0)
        ):
            raise ValueError(
                "JEPA action channel loss weights must be finite, positive, "
                "and match action_dim"
            )
        self.register_buffer(
            "action_channel_loss_weights", channel_weights, persistent=False
        )
        if not (
            0.0 <= self.adaptive_loss_ema_decay < 1.0
            and 0.0 < self.adaptive_loss_min_weight
            <= self.adaptive_loss_max_weight
        ):
            raise ValueError("invalid adaptive JEPA loss-weighting parameters")
        if self.loss_weighting == "adaptive_modality":
            self.register_buffer(
                "modality_error_ema",
                torch.ones(len(MultimodalJEPAEncoder.MODALITIES)),
            )
        if self.encoder_architecture == "multimodal_shared":
            self.online_encoder = MultimodalJEPAEncoder(
                state_dim,
                action_dim,
                proximity_dim,
                model_dim,
                encoder_layers,
                heads,
            )
        else:
            self.online_encoder = TactileStateEncoder(
                state_dim, action_dim, model_dim, encoder_layers, heads
            )
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.future_queries = nn.Parameter(
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
        self.predictor = nn.TransformerDecoder(layer, predictor_layers)
        if self.encoder_architecture == "multimodal_shared":
            self.modality_heads = nn.ModuleDict(
                {
                    name: nn.Linear(model_dim, model_dim)
                    for name in MultimodalJEPAEncoder.MODALITIES
                }
            )
        else:
            self.tactile_head = nn.Linear(model_dim, model_dim)
            self.state_head = nn.Linear(model_dim, model_dim)
        self.action_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, action_dim),
            nn.Tanh(),
        )
        nn.init.zeros_(self.action_head[-2].weight)
        nn.init.zeros_(self.action_head[-2].bias)

    def predict(self, observation: dict[str, torch.Tensor]):
        encoded = self.online_encoder(observation)
        if self.encoder_architecture == "multimodal_shared":
            modality_tokens, modality_valid = encoded
            stacked = torch.stack(tuple(modality_tokens.values()), dim=2)
            weights = torch.stack(
                tuple(
                    modality_valid[name]
                    for name in MultimodalJEPAEncoder.MODALITIES
                ),
                dim=2,
            ).to(stacked.dtype)[..., None]
            context = (stacked * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1)
        else:
            context = encoded["joint"]
        query = self.future_queries.expand(context.shape[0], -1, -1)
        latent = self.predictor(query, context)
        previous = observation["action"][:, -1:, : self.action_dim]
        action = torch.clamp(
            previous + self.residual_scale * self.action_head(latent),
            -1.0,
            1.0,
        )
        if self.encoder_architecture == "multimodal_shared":
            predicted = {
                name: head(latent) for name, head in self.modality_heads.items()
            }
        else:
            predicted = {
                "tactile": self.tactile_head(latent),
                "state": self.state_head(latent),
            }
        return action, predicted

    def loss(self, batch: dict[str, dict[str, torch.Tensor]]):
        action, prediction = self.predict(batch["observation"])
        with torch.no_grad():
            target_encoded = self.target_encoder(batch["future"])
        action_error_by_channel = nn.functional.smooth_l1_loss(
            action, batch["future"]["action"], reduction="none"
        )
        action_error = (
            action_error_by_channel * self.action_channel_loss_weights
        ).sum(dim=-1) / self.action_channel_loss_weights.sum()
        action_weights = torch.ones_like(action_error)
        future_phase = batch["future"].get("task_phase")
        if future_phase is not None:
            for name, value in self.action_phase_loss_weights.items():
                try:
                    phase_id = int(TaskPhase[name.upper()])
                except KeyError as error:
                    raise ValueError(f"unknown action phase loss name: {name}") from error
                action_weights = torch.where(
                    future_phase == phase_id,
                    torch.full_like(action_weights, float(value)),
                    action_weights,
                )
        action_loss = (action_error * action_weights).mean()
        if self.encoder_architecture == "multimodal_shared":
            target, valid = target_encoded
            names = tuple(MultimodalJEPAEncoder.MODALITIES)
        else:
            target = target_encoded
            valid = {
                "tactile": torch.ones_like(
                    batch["future"]["tactile_valid"], dtype=torch.bool
                ),
                "state": torch.ones_like(
                    batch["future"]["tactile_valid"], dtype=torch.bool
                ),
            }
            names = ("tactile", "state")
        latent_losses = {}
        latent_observed = {}
        valid_predictions = []
        selected = torch.tensor(
            [step - 1 for step in self.future_steps],
            dtype=torch.long,
            device=action.device,
        )
        for name in names:
            predicted = prediction[name].index_select(1, selected)
            targeted = target[name].index_select(1, selected)
            selected_valid = valid[name].index_select(1, selected)
            mask = selected_valid.to(predicted.dtype)
            cosine = nn.functional.cosine_similarity(
                predicted, targeted.detach(), dim=-1
            )
            latent_losses[name] = (
                (1.0 - cosine) * mask
            ).sum() / mask.sum().clamp_min(1.0)
            latent_observed[name] = bool(selected_valid.any())
            valid_predictions.append(predicted[selected_valid])
        adaptive_weights = {name: 1.0 for name in names}
        if self.loss_weighting == "adaptive_modality":
            all_names = MultimodalJEPAEncoder.MODALITIES
            observed = torch.tensor(
                [
                    float(latent_losses[name].detach())
                    if name in latent_losses and latent_observed[name]
                    else float(self.modality_error_ema[index])
                    for index, name in enumerate(all_names)
                ],
                device=action.device,
            )
            if self.training:
                self.modality_error_ema.lerp_(
                    observed, 1.0 - self.adaptive_loss_ema_decay
                )
            normalized = self.modality_error_ema / self.modality_error_ema.mean().clamp_min(1e-6)
            normalized = normalized.clamp(
                self.adaptive_loss_min_weight,
                self.adaptive_loss_max_weight,
            )
            adaptive_weights = {
                name: normalized[index] for index, name in enumerate(all_names)
                if name in names
            }
        latent_loss = sum(
            self.modality_loss_weights.get(name, 1.0)
            * adaptive_weights[name]
            * latent_losses[name]
            for name in names
        )
        flattened = torch.cat(valid_predictions, dim=0)
        feature_std = flattened.float().std(dim=0, unbiased=False)
        variance_loss = nn.functional.relu(0.1 - feature_std).mean()
        total = (
            self.action_loss_weight * action_loss
            + self.latent_loss_weight * latent_loss
            + self.variance_loss_weight * variance_loss
        )
        return {
            "action": action_loss,
            **{f"latent_{name}": value for name, value in latent_losses.items()},
            **{
                f"weight_{name}": torch.as_tensor(
                    adaptive_weights[name], device=action.device
                ).detach()
                for name in names
            },
            "variance": variance_loss,
            "representation_std": feature_std.mean().detach(),
            "total": total,
        }

    @torch.no_grad()
    def update_target_encoder(self):
        decay = self.ema_decay
        for target, online in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters()
        ):
            target.lerp_(online, 1.0 - decay)
        for target, online in zip(
            self.target_encoder.buffers(), self.online_encoder.buffers()
        ):
            target.copy_(online)
