from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrameLayout:
    """Fixed token slots shared by observed and generated latent frames.

    Every slot has ``token_width`` scalars. Modalities with fewer channels are
    zero padded and excluded from the diffusion loss through ``channel_mask``.
    """

    vision_tokens: int
    tactile_tokens: int
    token_width: int
    vision_channels: int
    tactile_channels: int
    proximity_dim: int
    state_dim: int
    action_dim: int
    full_hand_tactile_tokens: int = 0

    @property
    def tactile_start(self) -> int:
        return self.vision_tokens

    @property
    def proximity_index(self) -> int:
        return (
            self.vision_tokens
            + self.tactile_tokens
            + self.full_hand_tactile_tokens
        )

    @property
    def full_hand_tactile_start(self) -> int:
        return self.vision_tokens + self.tactile_tokens

    @property
    def state_index(self) -> int:
        return self.proximity_index + 1

    @property
    def action_index(self) -> int:
        return self.state_index + 1

    @property
    def tokens_per_frame(self) -> int:
        return self.action_index + 1

    def slices(self) -> dict[str, slice]:
        result = {
            "vision": slice(0, self.vision_tokens),
            "tactile": slice(
                self.tactile_start, self.full_hand_tactile_start
            ),
            "proximity": slice(self.proximity_index, self.proximity_index + 1),
            "state": slice(self.state_index, self.state_index + 1),
            "action": slice(self.action_index, self.action_index + 1),
        }
        if self.full_hand_tactile_tokens:
            result = {
                "vision": result["vision"],
                "tactile": result["tactile"],
                "full_hand_tactile": slice(
                    self.full_hand_tactile_start, self.proximity_index
                ),
                "proximity": result["proximity"],
                "state": result["state"],
                "action": result["action"],
            }
        return result

    def modality_ids(self, device=None) -> torch.Tensor:
        values = (
            [0] * self.vision_tokens
            + [1] * self.tactile_tokens
            + [1] * self.full_hand_tactile_tokens
            + [2, 3, 4]
        )
        return torch.tensor(values, dtype=torch.long, device=device)

    def channel_mask(self, device=None, dtype=torch.float32) -> torch.Tensor:
        mask = torch.zeros(self.tokens_per_frame, self.token_width, device=device, dtype=dtype)
        widths = {
            "vision": self.vision_channels,
            "tactile": self.tactile_channels,
            "full_hand_tactile": self.token_width,
            "proximity": self.proximity_dim,
            "state": self.state_dim,
            "action": self.action_dim,
        }
        for name, slots in self.slices().items():
            mask[slots, : widths[name]] = 1
        return mask


@dataclass
class UnifiedLatentFrames:
    values: torch.Tensor
    valid_mask: torch.Tensor
    channel_mask: torch.Tensor
    layout: FrameLayout

    @property
    def loss_mask(self) -> torch.Tensor:
        return self.valid_mask[..., None] * self.channel_mask[None, None]

    def flattened(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, tokens, width = self.values.shape
        return (
            self.values.reshape(batch, time * tokens, width),
            self.valid_mask.reshape(batch, time * tokens),
        )
