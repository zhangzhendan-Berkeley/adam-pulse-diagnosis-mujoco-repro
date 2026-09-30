from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn

from .action_chunk import (
    ACTStyleActionChunkPolicy,
    ActionChunkFlowPolicy,
    DirectTransformerChunkPolicy,
)
from .model import PulseWorldVLA
from .jepa import JEPALatentWorldPolicy
from pulse_mujoco.task import TaskPhase


BENCHMARK_SCHEMA_VERSION = "adam_pulse_model_benchmark_v1"
CHECKPOINT_SCHEMA_VERSION = "adam_pulse_policy_checkpoint_v1"
MODEL_FAMILIES = (
    "direct_transformer",
    "act",
    "dit_world_vla",
    "flow_matching",
    "jepa_world_model",
    "hybrid_direct_generative",
)


@dataclass(frozen=True)
class PolicyPrediction:
    action_chunk: torch.Tensor
    future_latents: dict[str, torch.Tensor] | None = None
    uncertainty: torch.Tensor | None = None
    value: torch.Tensor | None = None


@dataclass(frozen=True)
class BenchmarkContract:
    model_family: str
    observation_horizon: int
    action_horizon: int
    action_dim: int
    state_dim: int
    tactile_view: str
    tactile_normalization: str
    proximity_optional: bool = True

    def __post_init__(self) -> None:
        if self.model_family not in MODEL_FAMILIES:
            raise ValueError(f"unsupported model family: {self.model_family}")
        if min(
            self.observation_horizon,
            self.action_horizon,
            self.action_dim,
            self.state_dim,
        ) < 1:
            raise ValueError("benchmark dimensions and horizons must be positive")


class PolicyModel(nn.Module, ABC):
    """Shared train/inference boundary for the model race.

    Models consume the same normalized observations and emit normalized action
    chunks. Task orchestration and deterministic safety stay outside this API.
    """

    def __init__(self, contract: BenchmarkContract):
        super().__init__()
        self.contract = contract
        self.family = contract.model_family

    def encode_observation(
        self, observation: dict[str, torch.Tensor]
    ) -> Any:
        return observation

    @abstractmethod
    def forward_train(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def predict_action_chunk(
        self, observation: dict[str, torch.Tensor], **kwargs
    ) -> PolicyPrediction:
        raise NotImplementedError

    def predict_future_latent(
        self, observation: dict[str, torch.Tensor], action_chunk: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        del observation, action_chunk
        return None

    def estimate_uncertainty(
        self, observation: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        del observation
        return None

    def estimate_value(
        self, observation: dict[str, torch.Tensor], action_chunk: torch.Tensor
    ) -> torch.Tensor | None:
        del observation, action_chunk
        return None

    def reset_episode(self) -> None:
        return None


class PhaseRoutedPolicy(PolicyModel):
    """Route complete action chunks using only the visible current task phase.

    This wrapper deliberately ignores event labels and future phases.  The
    phase used online is the final element of the observation history, so the
    offline evaluator exercises the same routing contract as deployment.
    """

    def __init__(
        self,
        default_policy: PolicyModel,
        specialist_policy: PolicyModel,
        *,
        specialist_phases: tuple[int | TaskPhase, ...] = (
            TaskPhase.PULSE_SEARCH,
            TaskPhase.RECENTER,
        ),
    ):
        if default_policy.contract != specialist_policy.contract:
            raise ValueError("phase-routed policies must share one benchmark contract")
        super().__init__(default_policy.contract)
        self.default_policy = default_policy
        self.specialist_policy = specialist_policy
        phases = tuple(sorted({int(phase) for phase in specialist_phases}))
        valid = {int(phase) for phase in TaskPhase}
        if not phases or any(phase not in valid for phase in phases):
            raise ValueError("specialist phases must be valid non-empty TaskPhase ids")
        self.specialist_phases = phases
        self.register_buffer(
            "_specialist_phase_ids",
            torch.tensor(phases, dtype=torch.long),
            persistent=False,
        )

    def forward_train(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        raise RuntimeError("PhaseRoutedPolicy is an inference-only composition")

    def route_mask(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        task_phase = observation.get("task_phase")
        if task_phase is None or task_phase.ndim != 2:
            raise ValueError("phase routing requires task_phase [batch, history]")
        if task_phase.shape[1] != self.contract.observation_horizon:
            raise ValueError("task_phase history does not match benchmark contract")
        current = task_phase[:, -1].long()
        ids = self._specialist_phase_ids.to(current.device)
        return (current[:, None] == ids[None, :]).any(dim=1)

    def predict_action_chunk(
        self, observation: dict[str, torch.Tensor], **kwargs
    ) -> PolicyPrediction:
        specialist_mask = self.route_mask(observation)
        default = self.default_policy.predict_action_chunk(observation, **kwargs)
        if not bool(specialist_mask.any()):
            return default
        specialist = self.specialist_policy.predict_action_chunk(
            observation, **kwargs
        )
        mask = specialist_mask[:, None, None]
        action = torch.where(mask, specialist.action_chunk, default.action_chunk)
        return PolicyPrediction(action_chunk=action)

    def reset_episode(self) -> None:
        self.default_policy.reset_episode()
        self.specialist_policy.reset_episode()


class ACTPolicyAdapter(PolicyModel):
    def __init__(self, contract: BenchmarkContract, **model_config):
        if contract.model_family not in {"direct_transformer", "act"}:
            raise ValueError("ACT adapter requires direct_transformer or act family")
        super().__init__(contract)
        self.model_config = dict(model_config)
        model_type = (
            DirectTransformerChunkPolicy
            if contract.model_family == "direct_transformer"
            else ACTStyleActionChunkPolicy
        )
        self.model = model_type(
            state_dim=contract.state_dim,
            action_dim=contract.action_dim,
            prediction_horizon=contract.action_horizon,
            **self.model_config,
        )

    def forward_train(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        prediction = self.model(batch["observation"])
        loss = nn.functional.smooth_l1_loss(
            prediction, batch["future"]["action"]
        )
        return {"action": loss, "total": loss}

    def predict_action_chunk(
        self, observation: dict[str, torch.Tensor], **kwargs
    ) -> PolicyPrediction:
        del kwargs
        return PolicyPrediction(action_chunk=self.model(observation))


class FlowPolicyAdapter(PolicyModel):
    def __init__(self, contract: BenchmarkContract, **model_config):
        if contract.model_family != "flow_matching":
            raise ValueError("Flow adapter requires flow_matching family")
        super().__init__(contract)
        self.model_config = dict(model_config)
        self.model = ActionChunkFlowPolicy(
            state_dim=contract.state_dim,
            action_dim=contract.action_dim,
            prediction_horizon=contract.action_horizon,
            **self.model_config,
        )

    def forward_train(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        loss = self.model.flow_loss(
            batch["observation"], batch["future"]["action"]
        ).mean()
        return {"flow": loss, "total": loss}

    def predict_action_chunk(
        self, observation: dict[str, torch.Tensor], **kwargs
    ) -> PolicyPrediction:
        return PolicyPrediction(
            action_chunk=self.model.sample(
                observation,
                steps=int(kwargs.get("steps", 4)),
                generator=kwargs.get("generator"),
            )
        )


class WorldVLAAdapter(PolicyModel):
    """Unified benchmark wrapper for the joint future-modality World-VLA."""

    def __init__(self, contract: BenchmarkContract, **model_config):
        if contract.model_family not in {
            "dit_world_vla",
            "hybrid_direct_generative",
        }:
            raise ValueError("World-VLA adapter requires a DiT world-model family")
        super().__init__(contract)
        config = dict(model_config)
        self.autoencoder_loss_weight = float(
            config.pop("autoencoder_loss_weight", 0.0)
        )
        self.default_sampling_steps = int(config.pop("sampling_steps", 4))
        self.default_use_direct_action = bool(
            config.pop("use_direct_action", True)
        )
        self.model_config = dict(model_config)
        self.model = PulseWorldVLA(
            observation_horizon=contract.observation_horizon,
            prediction_horizon=contract.action_horizon,
            state_dim=contract.state_dim,
            action_dim=contract.action_dim,
            **config,
        )

    def forward_train(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        losses = self.model.diffusion_loss(batch, detach_tokenizer=False)
        total = losses["diffusion"]
        if self.autoencoder_loss_weight > 0:
            reconstruction = self.model.autoencoder_loss(batch)
            losses.update(reconstruction)
            total = total + self.autoencoder_loss_weight * sum(
                reconstruction.values()
            )
        losses["total"] = total
        return losses

    def predict_action_chunk(
        self, observation: dict[str, torch.Tensor], **kwargs
    ) -> PolicyPrediction:
        batch = observation["action"].shape[0]
        text_ids = kwargs.get("text_ids")
        if text_ids is None:
            text_ids = torch.zeros(
                batch,
                24,
                dtype=torch.long,
                device=observation["action"].device,
            )
        use_direct = bool(
            kwargs.get("use_direct_action", self.default_use_direct_action)
        )
        if use_direct:
            return PolicyPrediction(
                action_chunk=self.model.predict_direct_action(observation)
            )
        decoded = self.model.sample(
            observation,
            text_ids,
            steps=int(kwargs.get("steps", self.default_sampling_steps)),
            deterministic_action=bool(kwargs.get("deterministic_action", True)),
            initial_action_chunk=(
                self.model.predict_direct_action(observation)
                if bool(kwargs.get("warm_start_direct", False))
                else None
            ),
            action_prior_noise_scale=float(
                kwargs.get("action_prior_noise_scale", 0.1)
            ),
        )
        return PolicyPrediction(
            action_chunk=decoded.action,
            future_latents={
                "vision": decoded.vision,
                "tactile": decoded.tactile,
                "proximity": decoded.proximity,
                "robot_state": decoded.robot_state,
                **(
                    {"full_hand_tactile": decoded.full_hand_tactile_latent}
                    if decoded.full_hand_tactile_latent is not None
                    else {}
                ),
            },
        )


class JEPAPolicyAdapter(PolicyModel):
    def __init__(self, contract: BenchmarkContract, **model_config):
        if contract.model_family != "jepa_world_model":
            raise ValueError("JEPA adapter requires jepa_world_model family")
        super().__init__(contract)
        self.model_config = dict(model_config)
        self.model = JEPALatentWorldPolicy(
            state_dim=contract.state_dim,
            action_dim=contract.action_dim,
            prediction_horizon=contract.action_horizon,
            **self.model_config,
        )

    def forward_train(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.model.loss(batch)

    def predict_action_chunk(
        self, observation: dict[str, torch.Tensor], **kwargs
    ) -> PolicyPrediction:
        del kwargs
        action, latent = self.model.predict(observation)
        return PolicyPrediction(action_chunk=action, future_latents=latent)

    def update_target_encoder(self) -> None:
        self.model.update_target_encoder()


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_policy_checkpoint(
    path: Path | str,
    model: PolicyModel,
    *,
    model_config: dict[str, Any],
    dataset: dict[str, Any],
    normalization: dict[str, Any],
    seed: int,
    metrics: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "contract": asdict(model.contract),
        "model_config": dict(model_config),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "epoch": epoch,
        "seed": int(seed),
        "dataset": dict(dataset),
        "normalization": dict(normalization),
        "metrics": {} if metrics is None else dict(metrics),
        "git_commit": git_commit,
        "parameters": model_parameter_count(model),
    }
    torch.save(payload, target)
    manifest = {
        key: value
        for key, value in payload.items()
        if key not in {"model", "optimizer"}
    }
    manifest["checkpoint"] = str(target.resolve())
    manifest["checkpoint_sha256"] = sha256_file(target)
    target.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_policy_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PolicyModel, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported unified policy checkpoint")
    contract = BenchmarkContract(**payload["contract"])
    config = dict(payload["model_config"])
    if contract.model_family in {"direct_transformer", "act"}:
        model: PolicyModel = ACTPolicyAdapter(contract, **config)
    elif contract.model_family == "flow_matching":
        model = FlowPolicyAdapter(contract, **config)
    elif contract.model_family in {"dit_world_vla", "hybrid_direct_generative"}:
        model = WorldVLAAdapter(contract, **config)
    elif contract.model_family == "jepa_world_model":
        model = JEPAPolicyAdapter(contract, **config)
    else:
        raise ValueError(
            f"checkpoint loader for {contract.model_family} is not implemented"
        )
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


@torch.no_grad()
def measure_policy_latency(
    model: PolicyModel,
    observation: dict[str, torch.Tensor],
    *,
    warmup: int = 10,
    iterations: int = 50,
    **predict_kwargs,
) -> dict[str, float]:
    if warmup < 0 or iterations < 1:
        raise ValueError("latency warmup/iterations are invalid")
    device = next(model.parameters()).device
    synchronize = device.type == "cuda"
    for _ in range(warmup):
        model.predict_action_chunk(observation, **predict_kwargs)
    if synchronize:
        torch.cuda.synchronize(device)
    values = []
    for _ in range(iterations):
        begin = time.perf_counter_ns()
        model.predict_action_chunk(observation, **predict_kwargs)
        if synchronize:
            torch.cuda.synchronize(device)
        values.append((time.perf_counter_ns() - begin) * 1e-6)
    ordered = torch.tensor(values, dtype=torch.float64)
    return {
        "p50_ms": float(torch.quantile(ordered, 0.50)),
        "p95_ms": float(torch.quantile(ordered, 0.95)),
        "mean_ms": float(ordered.mean()),
        "iterations": int(iterations),
    }
