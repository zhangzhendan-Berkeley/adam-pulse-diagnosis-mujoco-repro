from __future__ import annotations

import torch

from pulse_mujoco.task import TaskPhase


CONTACT_PHASES = (
    TaskPhase.CONTACT_ESTABLISHMENT,
    TaskPhase.PRESSURE_REGULATION,
    TaskPhase.PULSE_SEARCH,
    TaskPhase.RECENTER,
    TaskPhase.STABLE_HOLD,
    TaskPhase.SIGNAL_CAPTURE,
)
APPROACH_PHASES = (
    TaskPhase.OBSERVE,
    TaskPhase.COARSE_APPROACH,
    TaskPhase.FINE_APPROACH,
    TaskPhase.PRE_CONTACT,
)


def _phase_mask(phase: torch.Tensor, values: tuple[TaskPhase, ...]) -> torch.Tensor:
    result = torch.zeros_like(phase, dtype=torch.bool)
    for value in values:
        result |= phase == int(value)
    return result


def apply_phase_aware_modality_dropout(
    observation: dict[str, torch.Tensor],
    config: dict,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Drop complete modalities per sample without mutating the input batch."""

    if "task_phase" not in observation:
        raise ValueError("phase-aware modality dropout requires task_phase")
    probabilities = dict(config.get("probabilities", {}))
    required = {"vision", "tactile", "proximity"}
    missing = required - probabilities.keys()
    if missing:
        raise ValueError(f"modality dropout probabilities missing {sorted(missing)}")
    for name, probability in probabilities.items():
        if name in required and not 0.0 <= float(probability) <= 1.0:
            raise ValueError(f"invalid {name} dropout probability")

    current_phase = observation["task_phase"][:, -1].long()
    contact = _phase_mask(current_phase, CONTACT_PHASES)
    approach = _phase_mask(current_phase, APPROACH_PHASES)
    batch = current_phase.shape[0]
    device = current_phase.device
    rates = {
        name: torch.full(
            (batch,), float(probabilities[name]), device=device, dtype=torch.float32
        )
        for name in required
    }
    rates["tactile"] = torch.where(
        contact,
        torch.full_like(
            rates["tactile"],
            float(config.get("contact_tactile_probability", probabilities["tactile"])),
        ),
        rates["tactile"],
    )
    rates["vision"] = torch.where(
        approach,
        torch.full_like(
            rates["vision"],
            float(config.get("approach_vision_probability", probabilities["vision"])),
        ),
        rates["vision"],
    )
    dropped = {name: torch.rand(batch, device=device) < rate for name, rate in rates.items()}
    return mask_observation_modalities(observation, dropped), dropped


def mask_observation_modalities(
    observation: dict[str, torch.Tensor],
    dropped: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply explicit per-sample modality masks without mutating observations."""

    required = {"vision", "tactile", "proximity"}
    if set(dropped) != required:
        raise ValueError(f"explicit modality masks must contain {sorted(required)}")
    augmented = dict(observation)

    def zero(name: str, mask: torch.Tensor) -> None:
        if name not in augmented:
            return
        value = augmented[name].clone()
        value[mask] = 0
        augmented[name] = value

    zero("vision", dropped["vision"])
    for name in (
        "tactile",
        "tactile_full56",
        "tactile_full56_delta",
        "normal_force",
    ):
        zero(name, dropped["tactile"])
    for name in (
        "tactile_valid",
        "tactile_full56_valid",
        "normal_force_valid",
    ):
        zero(name, dropped["tactile"])
    zero("proximity", dropped["proximity"])
    zero("proximity_valid", dropped["proximity"])
    return augmented
