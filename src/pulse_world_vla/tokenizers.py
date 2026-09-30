from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .schema import FrameLayout, UnifiedLatentFrames


class ImageAutoencoder(nn.Module):
    """Small trainable image tokenizer with 8x spatial compression."""

    def __init__(self, input_channels: int, latent_channels: int, base_channels: int = 32):
        super().__init__()
        self.input_channels = int(input_channels)
        self.latent_channels = int(latent_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, latent_channels, 4, 2, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, base_channels * 2, 4, 2, 1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, input_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        batch, time = images.shape[:2]
        encoded = self.encoder(images.flatten(0, 1))
        return encoded.unflatten(0, (batch, time))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        batch, time = latents.shape[:2]
        decoded = self.decoder(latents.flatten(0, 1))
        return decoded.unflatten(0, (batch, time))

    def reconstruction_loss(self, images: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        reconstructed = self.decode(self.encode(images))
        error = (reconstructed - images).abs().mean(dim=(-3, -2, -1))
        if valid is None:
            return error.mean()
        weights = valid.to(error.dtype)
        return (error * weights).sum() / weights.sum().clamp_min(1)


@dataclass
class DecodedFrames:
    vision: torch.Tensor
    tactile: torch.Tensor
    proximity: torch.Tensor
    robot_state: torch.Tensor
    action: torch.Tensor
    full_hand_tactile_latent: torch.Tensor | None = None


FULL56_FINGER_SLICES = (
    slice(0, 16),
    slice(16, 26),
    slice(26, 36),
    slice(36, 46),
    slice(46, 56),
)


class FullHandTactileTokenizer(nn.Module):
    """Encode the documented five anatomical finger groups into five tokens.

    The 56 channels are deliberately not reshaped into an inferred 2-D grid.
    Each finger is encoded from pressure, temporal delta and validity, then
    augmented by the corresponding observable normal-force reading.
    """

    def __init__(self, token_width: int):
        super().__init__()
        self.channel_projection = nn.Linear(3, token_width)
        self.channel_position = nn.Parameter(torch.zeros(1, 16, token_width))
        self.finger_embedding = nn.Parameter(torch.zeros(1, 5, token_width))
        self.force_projection = nn.Linear(2, token_width)
        layer = nn.TransformerEncoderLayer(
            d_model=token_width,
            nhead=4,
            dim_feedforward=token_width * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(token_width)

    def forward(
        self,
        values: torch.Tensor,
        delta: torch.Tensor,
        valid: torch.Tensor,
        normal_force: torch.Tensor,
        normal_force_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.shape != delta.shape or values.shape != valid.shape:
            raise ValueError("full56 tactile tensors must share shape")
        if values.ndim != 3 or values.shape[-1] != 56:
            raise ValueError("full56 tactile input must be [batch,time,56]")
        if normal_force.shape != (*values.shape[:2], 5):
            raise ValueError("normal force must be [batch,time,5]")
        if normal_force_valid.shape != normal_force.shape:
            raise ValueError("normal force validity must match normal force")
        batch, horizon = values.shape[:2]
        finger_tokens = []
        finger_validity = []
        for finger_index, finger_slice in enumerate(FULL56_FINGER_SLICES):
            finger_values = values[..., finger_slice]
            finger_delta = delta[..., finger_slice]
            finger_valid = valid[..., finger_slice].bool()
            count = finger_values.shape[-1]
            features = torch.stack(
                (finger_values, finger_delta, finger_valid.to(values.dtype)), dim=-1
            ).flatten(0, 1)
            tokens = (
                self.channel_projection(features)
                + self.channel_position[:, :count]
                + self.finger_embedding[:, finger_index : finger_index + 1]
            )
            padding_mask = ~finger_valid.flatten(0, 1)
            all_invalid = padding_mask.all(dim=1)
            if torch.any(all_invalid):
                padding_mask = padding_mask.clone()
                padding_mask[all_invalid, 0] = False
            encoded = self.encoder(
                tokens, src_key_padding_mask=padding_mask
            )
            mask = finger_valid.flatten(0, 1).to(encoded.dtype)[..., None]
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            force = torch.stack(
                (
                    normal_force[..., finger_index].clamp(0, 10) / 10.0,
                    normal_force_valid[..., finger_index].to(values.dtype),
                ),
                dim=-1,
            ).flatten(0, 1)
            finger_tokens.append(
                self.output_norm(pooled + self.force_projection(force)).unflatten(
                    0, (batch, horizon)
                )
            )
            finger_validity.append(
                finger_valid.any(dim=-1)
                | normal_force_valid[..., finger_index].bool()
            )
        return (
            torch.stack(finger_tokens, dim=2),
            torch.stack(finger_validity, dim=2),
        )


class MultimodalFrameTokenizer(nn.Module):
    """Pack heterogeneous observations into one fixed-width latent frame."""

    def __init__(
        self,
        image_size: tuple[int, int],
        tactile_size: tuple[int, int],
        vision_channels: int = 8,
        tactile_channels: int = 4,
        proximity_dim: int = 5,
        state_dim: int = 12,
        action_dim: int = 3,
        token_width: int = 16,
        full_hand_tactile: bool = False,
    ):
        super().__init__()
        if any(size % 8 for size in (*image_size, *tactile_size)):
            raise ValueError("image_size and tactile_size must be divisible by 8")
        maximum_width = max(vision_channels, tactile_channels, proximity_dim, state_dim, action_dim)
        if token_width < maximum_width:
            raise ValueError(f"token_width={token_width} is smaller than modality width={maximum_width}")
        self.image_size = tuple(image_size)
        self.tactile_size = tuple(tactile_size)
        self.vision_shape = (vision_channels, image_size[0] // 8, image_size[1] // 8)
        self.tactile_shape = (tactile_channels, tactile_size[0] // 8, tactile_size[1] // 8)
        self.vision_autoencoder = ImageAutoencoder(4, vision_channels)
        self.tactile_autoencoder = ImageAutoencoder(1, tactile_channels, base_channels=16)
        self.full_hand_tactile = bool(full_hand_tactile)
        self.full_hand_tactile_tokenizer = (
            FullHandTactileTokenizer(token_width)
            if self.full_hand_tactile
            else None
        )
        self.layout = FrameLayout(
            vision_tokens=self.vision_shape[1] * self.vision_shape[2],
            tactile_tokens=self.tactile_shape[1] * self.tactile_shape[2],
            token_width=token_width,
            vision_channels=vision_channels,
            tactile_channels=tactile_channels,
            proximity_dim=proximity_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            full_hand_tactile_tokens=5 if self.full_hand_tactile else 0,
        )

    @staticmethod
    def _pack_spatial(latents: torch.Tensor, token_width: int) -> torch.Tensor:
        values = latents.permute(0, 1, 3, 4, 2).flatten(2, 3)
        return nn.functional.pad(values, (0, token_width - values.shape[-1]))

    @staticmethod
    def _pack_vector(values: torch.Tensor, token_width: int) -> torch.Tensor:
        return nn.functional.pad(values, (0, token_width - values.shape[-1])).unsqueeze(2)

    def encode(
        self,
        sample: dict[str, torch.Tensor],
        detach_autoencoder: bool = False,
        detach_structured_tactile: bool | None = None,
    ) -> UnifiedLatentFrames:
        vision = self.vision_autoencoder.encode(sample["vision"])
        tactile = self.tactile_autoencoder.encode(sample["tactile"])
        if detach_autoencoder:
            vision, tactile = vision.detach(), tactile.detach()
        full_hand_values = []
        full_hand_valid = []
        if self.full_hand_tactile_tokenizer is not None:
            required = (
                "tactile_full56",
                "tactile_full56_delta",
                "tactile_full56_valid",
                "normal_force",
                "normal_force_valid",
            )
            missing = [name for name in required if name not in sample]
            if missing:
                raise ValueError(f"full-hand tactile sample missing {missing}")
            full_hand_values, full_hand_valid = self.full_hand_tactile_tokenizer(
                sample["tactile_full56"],
                sample["tactile_full56_delta"],
                sample["tactile_full56_valid"],
                sample["normal_force"],
                sample["normal_force_valid"],
            )
            if (
                detach_autoencoder
                if detach_structured_tactile is None
                else detach_structured_tactile
            ):
                full_hand_values = full_hand_values.detach()
        values = torch.cat(
            (
                self._pack_spatial(vision, self.layout.token_width),
                self._pack_spatial(tactile, self.layout.token_width),
                *([full_hand_values] if self.full_hand_tactile else []),
                self._pack_vector(sample["proximity"], self.layout.token_width),
                self._pack_vector(sample["robot_state"], self.layout.token_width),
                self._pack_vector(sample["action"], self.layout.token_width),
            ),
            dim=2,
        )
        batch, time = values.shape[:2]
        ones = torch.ones(batch, time, self.layout.vision_tokens, device=values.device, dtype=values.dtype)
        tactile_valid = sample["tactile_valid"].to(values.dtype).unsqueeze(-1).expand(
            -1, -1, self.layout.tactile_tokens
        )
        proximity_valid = sample.get("proximity_valid")
        if proximity_valid is None:
            proximity_valid = torch.ones(batch, time, device=values.device, dtype=values.dtype)
        elif proximity_valid.ndim == 3:
            proximity_valid = proximity_valid.any(dim=-1)
        valid = torch.cat(
            (
                ones,
                tactile_valid,
                *([full_hand_valid.to(values.dtype)] if self.full_hand_tactile else []),
                proximity_valid.to(values.dtype).unsqueeze(-1),
                torch.ones(batch, time, 2, device=values.device, dtype=values.dtype),
            ),
            dim=2,
        )
        return UnifiedLatentFrames(
            values=values,
            valid_mask=valid,
            channel_mask=self.layout.channel_mask(values.device, values.dtype),
            layout=self.layout,
        )

    def decode(self, frames: torch.Tensor) -> DecodedFrames:
        parts = self.layout.slices()
        batch, time = frames.shape[:2]
        vision = frames[:, :, parts["vision"], : self.layout.vision_channels]
        vision = vision.reshape(batch, time, self.vision_shape[1], self.vision_shape[2], -1)
        vision = self.vision_autoencoder.decode(vision.permute(0, 1, 4, 2, 3))
        tactile = frames[:, :, parts["tactile"], : self.layout.tactile_channels]
        tactile = tactile.reshape(batch, time, self.tactile_shape[1], self.tactile_shape[2], -1)
        tactile = self.tactile_autoencoder.decode(tactile.permute(0, 1, 4, 2, 3))
        return DecodedFrames(
            vision=vision,
            tactile=tactile,
            proximity=frames[:, :, self.layout.proximity_index, : self.layout.proximity_dim],
            robot_state=frames[:, :, self.layout.state_index, : self.layout.state_dim],
            action=frames[:, :, self.layout.action_index, : self.layout.action_dim],
            full_hand_tactile_latent=(
                frames[:, :, parts["full_hand_tactile"]]
                if "full_hand_tactile" in parts
                else None
            ),
        )
