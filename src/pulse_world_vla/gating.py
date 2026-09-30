from __future__ import annotations

import torch


def robust_change_gate(
    signal: torch.Tensor,
    valid: torch.Tensor | None = None,
    minimum_gate: float = 0.0,
    maximum_gate: float = 1.0,
    temperature: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """MAD-normalized temporal difference gate requested for local sensors.

    Args:
        signal: ``[batch, time, ...]`` sensor values.
        valid: optional mask with the same shape or ``[batch, time]``.

    Per-channel absolute frame differences are averaged, robustly normalized by
    median absolute deviation, passed through sigmoid, and linearly rescaled.
    Samples without two consecutive valid readings return a zero gate.
    """

    if signal.ndim < 3 or signal.shape[1] < 2:
        return torch.zeros(signal.shape[0], device=signal.device, dtype=signal.dtype)
    difference = (signal[:, 1:] - signal[:, :-1]).abs()
    if valid is None:
        pair_valid = torch.ones_like(difference, dtype=torch.bool)
    else:
        valid = valid.bool()
        if valid.ndim == 2:
            valid = valid[..., None]
        pair_valid = valid[:, 1:] & valid[:, :-1]
        while pair_valid.ndim < difference.ndim:
            pair_valid = pair_valid.unsqueeze(-1)
        pair_valid = pair_valid.expand_as(difference)
    reduce_dims = tuple(range(2, difference.ndim))
    count = pair_valid.sum(dim=reduce_dims)
    temporal_change = (difference * pair_valid).sum(dim=reduce_dims) / count.clamp_min(1)
    temporal_valid = count.gt(0)

    # Per-trajectory temporal MAD is the primary statistic. A batch MAD
    # fallback avoids degeneracy for two-frame histories.
    masked = temporal_change.masked_fill(~temporal_valid, float("nan"))
    median = torch.nanmedian(masked, dim=1, keepdim=True).values
    mad = torch.nanmedian((masked - median).abs(), dim=1, keepdim=True).values
    degenerate = ~torch.isfinite(mad) | mad.lt(epsilon)
    if degenerate.any():
        batch_values = masked.flatten()
        batch_values = batch_values[torch.isfinite(batch_values)]
        if batch_values.numel() > 1:
            batch_median = batch_values.median()
            batch_mad = (batch_values - batch_median).abs().median().clamp_min(epsilon)
            median = torch.where(degenerate, batch_median, median)
            mad = torch.where(degenerate, batch_mad, mad)
        else:
            mad = mad.nan_to_num(0.0).clamp_min(epsilon)
    robust_z = (temporal_change - median) / (1.4826 * mad + epsilon)
    # Use the strongest recent change so a short contact transient is retained.
    robust_z = robust_z.masked_fill(~temporal_valid, -torch.inf).amax(dim=1)
    has_signal = temporal_valid.any(dim=1)
    # Rescale the upper half of sigmoid to [0, 1]: median/stationary change is
    # zero, while a positive robust outlier smoothly approaches one.
    strength = (2 * torch.sigmoid(robust_z / max(float(temperature), epsilon)) - 1).clamp(0, 1)
    gate = minimum_gate + (maximum_gate - minimum_gate) * strength
    return torch.where(has_signal, gate, torch.zeros_like(gate))


def multimodal_change_gates(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    proximity = robust_change_gate(
        observation["proximity"], observation.get("proximity_valid"), maximum_gate=1.0
    )
    tactile_valid = observation.get("tactile_valid")
    tactile = robust_change_gate(observation["tactile"], tactile_valid, maximum_gate=1.0)
    return {"proximity": proximity, "tactile": tactile}


def observable_contact_gate(
    observation: dict[str, torch.Tensor], epsilon: float = 1e-6
) -> torch.Tensor:
    """Contact strength derived only from policy-visible tactile and force."""

    full56 = observation.get("tactile_full56")
    full56_valid = observation.get("tactile_full56_valid")
    force = observation.get("normal_force")
    force_valid = observation.get("normal_force_valid")
    if full56 is None or full56_valid is None or force is None or force_valid is None:
        return robust_change_gate(
            observation["tactile"], observation.get("tactile_valid")
        )
    channel_valid = full56_valid[:, -1].bool()
    pressure = (
        full56[:, -1].abs() * channel_valid.to(full56.dtype)
    ).sum(dim=-1) / channel_valid.sum(dim=-1).clamp_min(1)
    force_mask = force_valid[:, -1].bool()
    force_strength = (
        force[:, -1].clamp(0, 10) * force_mask.to(force.dtype)
    ).sum(dim=-1) / force_mask.sum(dim=-1).clamp_min(1)
    force_strength = force_strength / 10.0
    delta = observation.get("tactile_full56_delta")
    if delta is None:
        delta_strength = torch.zeros_like(pressure)
    else:
        delta_strength = (
            delta[:, -1].abs() * channel_valid.to(delta.dtype)
        ).sum(dim=-1) / channel_valid.sum(dim=-1).clamp_min(1)
    strength = torch.maximum(pressure.clamp(0, 1), force_strength.clamp(0, 1))
    strength = torch.maximum(strength, delta_strength.clamp(0, 1))
    has_signal = channel_valid.any(dim=-1) | force_mask.any(dim=-1)
    return torch.where(has_signal, strength, torch.zeros_like(strength)).clamp(
        min=0.0, max=1.0
    )
