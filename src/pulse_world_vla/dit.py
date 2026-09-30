from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(timestep: torch.Tensor, dim: int, maximum_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(maximum_period)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(1, half)
    )
    angles = timestep.float()[:, None] * frequencies[None]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if dim % 2:
        embedding = nn.functional.pad(embedding, (0, 1))
    return embedding.to(timestep.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Pre-norm transformer block with AdaLN-Zero diffusion conditioning."""

    def __init__(self, model_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = int(heads)
        self.norm1 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(condition).chunk(6, dim=-1)
        )
        normalized = modulate(self.norm1(x), shift_attention, scale_attention)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + gate_attention.unsqueeze(1) * attended
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        maximum_frames: int,
        maximum_tokens_per_frame: int,
        vocabulary_size: int = 4096,
        maximum_text_tokens: int = 32,
        dropout: float = 0.0,
        attention_gate_scale: float = 2.0,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.heads = int(heads)
        self.attention_gate_scale = float(attention_gate_scale)
        self.maximum_frames = int(maximum_frames)
        self.maximum_tokens_per_frame = int(maximum_tokens_per_frame)
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.valid_embedding = nn.Embedding(2, model_dim)
        self.modality_embedding = nn.Embedding(5, model_dim)
        self.frame_embedding = nn.Embedding(maximum_frames, model_dim)
        self.slot_embedding = nn.Embedding(maximum_tokens_per_frame, model_dim)
        self.known_embedding = nn.Embedding(2, model_dim)
        self.text_embedding = nn.Embedding(vocabulary_size, model_dim, padding_idx=0)
        self.text_position = nn.Embedding(maximum_text_tokens, model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(model_dim, heads, dropout) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 2))
        self.output = nn.Linear(model_dim, input_dim)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        modality_ids: torch.Tensor,
        frame_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        known: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        context_token_count: int,
        change_gates: dict[str, torch.Tensor] | None = None,
        prefix_cache: torch.Tensor | None = None,
        cache_depth: int = 0,
        return_prefix_cache: bool = False,
    ) -> torch.Tensor:
        batch, token_count = tokens.shape[:2]
        if frame_ids.max().item() >= self.maximum_frames:
            raise ValueError("Sequence exceeds configured maximum_frames")
        positions = (
            self.modality_embedding(modality_ids)
            + self.frame_embedding(frame_ids)
            + self.slot_embedding(slot_ids)
            + self.known_embedding(known.long())
        )
        x = (
            self.input_projection(tokens)
            + positions.unsqueeze(0)
            + self.valid_embedding(valid.long())
        )
        text_length = text_ids.shape[1]
        text_positions = torch.arange(text_length, device=x.device)
        text = self.text_embedding(text_ids) + self.text_position(text_positions)[None]
        x = torch.cat((text, x), dim=1)
        text_valid = text_ids.ne(0)
        padding = ~torch.cat((text_valid, valid.bool()), dim=1)
        key_padding = torch.zeros_like(padding, dtype=x.dtype).masked_fill(padding, -torch.inf)

        # Observed tokens cannot read noisy future tokens. Future queries may
        # attend to the full observation history and all jointly denoised slots.
        total = text_length + token_count
        history_end = text_length + context_token_count
        attention_mask = torch.zeros(total, total, device=x.device, dtype=x.dtype)
        attention_mask[:history_end, history_end:] = -torch.inf
        if change_gates:
            full_modality = torch.cat(
                (torch.full((text_length,), -1, device=x.device), modality_ids)
            )
            attention_mask = attention_mask.unsqueeze(0).expand(batch, -1, -1).clone()
            frame_queries = full_modality.ge(0)
            for name, modality_id in (("tactile", 1), ("proximity", 2)):
                gate = change_gates.get(name)
                if gate is None:
                    continue
                target_keys = full_modality.eq(modality_id)
                other_queries = frame_queries & full_modality.ne(modality_id)
                boost = other_queries[:, None] & target_keys[None, :]
                attention_mask[:, boost] += self.attention_gate_scale * gate[:, None]
            attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
        condition = self.time_mlp(timestep_embedding(timestep, self.model_dim))
        cached = None
        start_layer = 0
        if prefix_cache is not None:
            x = prefix_cache
            start_layer = int(cache_depth)
        for layer_index, block in enumerate(self.blocks[start_layer:], start=start_layer):
            x = block(x, condition, attention_mask, key_padding)
            if return_prefix_cache and layer_index + 1 == cache_depth:
                cached = x.detach()
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        output = self.output(x[:, text_length:])
        return (output, cached) if return_prefix_cache else output
