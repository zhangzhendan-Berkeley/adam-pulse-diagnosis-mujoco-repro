from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .dataset import instruction_tokens


class PulseWorldVLAPolicy:
    """Receding-horizon inference adapter; safety remains in the existing governor."""

    def __init__(
        self, model, observation_horizon, action_scale_m=0.01, steps=10, guidance_scale=1.0,
        cache_refresh_steps=None, cache_depth=0,
    ):
        self.model = model.eval()
        self.history = deque(maxlen=int(observation_horizon))
        self.action_scale_m = float(action_scale_m)
        self.steps = int(steps)
        self.guidance_scale = float(guidance_scale)
        self.cache_refresh_steps = None if cache_refresh_steps is None else set(cache_refresh_steps)
        self.cache_depth = int(cache_depth)

    def reset(self):
        self.history.clear()

    def observe(self, frame: dict[str, torch.Tensor]):
        self.history.append(frame)

    @torch.no_grad()
    def predict(self, instruction: str):
        if len(self.history) != self.history.maxlen:
            raise RuntimeError(f"Need {self.history.maxlen} observations before prediction")
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        observation = {}
        for key in self.history[0]:
            values = torch.stack([frame[key] for frame in self.history]).unsqueeze(0).to(device)
            if torch.is_floating_point(values):
                values = values.to(dtype)
            observation[key] = values
        text = instruction_tokens(instruction).unsqueeze(0).to(device)
        decoded = self.model.sample(
            observation,
            text,
            self.steps,
            self.guidance_scale,
            cache_refresh_steps=self.cache_refresh_steps,
            cache_depth=self.cache_depth,
            deterministic_action=True,
        )
        action = decoded.action[0].float().cpu().numpy() * self.action_scale_m
        return {
            "action_delta_tcp_m": action,
            "future_vision": decoded.vision[0].float().cpu(),
            "future_tactile": decoded.tactile[0].float().cpu(),
            "future_proximity_normalized": decoded.proximity[0].float().cpu().numpy(),
            "future_robot_state_normalized": decoded.robot_state[0].float().cpu().numpy(),
        }
