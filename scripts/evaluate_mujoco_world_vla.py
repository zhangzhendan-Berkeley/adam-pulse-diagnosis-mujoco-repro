from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pulse_mujoco.event_labels import EVENT_TAG_NAMES
from pulse_mujoco.vla_dataset import MujocoWorldVLADataset, move_vla_batch
from pulse_world_vla.action_metrics import action_metric_groups
from pulse_world_vla.cuda import require_cuda
from pulse_world_vla.model import PulseWorldVLA


def _split_names(index_path: Path, split: str) -> list[str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    accepted = {item["episode"] for item in payload["accepted_episodes"]}
    names = [name for name in payload["splits"][split] if name in accepted]
    if not names:
        raise RuntimeError(f"dataset index has no accepted {split} episodes")
    return names


def _action_scale(data_config: dict) -> tuple[float, ...]:
    value = data_config.get("action_scale", data_config.get("action_scale_m"))
    if value is None or len(value) != 7:
        raise ValueError("data config must contain a seven-value action_scale")
    return tuple(float(item) for item in value)


def _latency(method, device: torch.device, batch_size: int, repeats: int) -> dict:
    with torch.no_grad():
        for _ in range(3):
            method()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            method()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "batch_size": int(batch_size),
        "batch_latency_ms_median": float(np.median(durations)),
        "batch_latency_ms_p95": float(np.percentile(durations, 95)),
        "per_sample_latency_ms_median": float(np.median(durations) / batch_size),
        "device": str(device),
        "repeats": int(repeats),
    }


def _world_errors(decoded, future: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    vision = torch.abs(decoded.vision - future["vision"]).flatten(2).mean(2)
    tactile_error = torch.abs(decoded.tactile - future["tactile"]).flatten(2).mean(2)
    tactile_valid = future["tactile_valid"].float()
    tactile = (tactile_error * tactile_valid).sum(1) / tactile_valid.sum(1).clamp_min(1)
    proximity_error = torch.abs(decoded.proximity - future["proximity"])
    proximity_valid = future["proximity_valid"].float()
    proximity = (
        (proximity_error * proximity_valid).flatten(1).sum(1)
        / proximity_valid.flatten(1).sum(1).clamp_min(1)
    )
    state = torch.abs(decoded.robot_state - future["robot_state"]).flatten(2).mean(2)
    return {
        "vision": vision.mean(1).cpu().numpy(),
        "tactile": tactile.cpu().numpy(),
        "proximity": proximity.cpu().numpy(),
        "robot_state": state.mean(1).cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MuJoCo World-VLA action and future-modality prediction."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--samples-per-episode", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--device", choices=("cuda",), default="cuda",
        help="CUDA is mandatory; formal evaluation never falls back to CPU.",
    )
    parser.add_argument("--cache-depth", type=int)
    parser.add_argument("--cache-refresh-steps", type=int, nargs="*")
    parser.add_argument("--latency-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.samples_per_episode < 1 or args.batch_size < 1:
        raise ValueError("samples-per-episode and batch-size must be positive")
    if args.steps is not None and args.steps < 1:
        raise ValueError("steps must be positive")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model_config = dict(config["model"])
    data_config = dict(config["data"])
    action_scale = _action_scale(data_config)
    device = require_cuda(purpose="MuJoCo World-VLA evaluation")
    names = _split_names(args.dataset_index, args.split)
    dataset = MujocoWorldVLADataset(
        root=args.data,
        episode_names=names,
        image_size=tuple(model_config["image_size"]),
        tactile_size=tuple(model_config["tactile_size"]),
        observation_horizon=int(model_config["observation_horizon"]),
        prediction_horizon=int(model_config["prediction_horizon"]),
        sensor_range_m=float(data_config["sensor_range_m"]),
        tactile_scale_kpa=float(data_config["tactile_scale_kpa"]),
        tactile_view=str(data_config.get("tactile_view", "diagnostic_2x10")),
        action_scale=action_scale,
        state_scale=float(data_config["state_scale"]),
        samples_per_episode=int(args.samples_per_episode),
        random_window=False,
        load_future_modalities=True,
        media_cache_frames=int(data_config.get("media_cache_frames", 0)),
    )
    if dataset.state_dim != int(model_config["state_dim"]):
        raise ValueError(
            f"dataset/model state dimension mismatch: {dataset.state_dim} != "
            f"{model_config['state_dim']}"
        )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )
    model = PulseWorldVLA(**model_config).to(device).eval()
    weights = payload.get("ema") or payload["model"]
    weight_source = "ema" if payload.get("ema") is not None else "model"
    model.load_state_dict(weights, strict=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    steps = int(args.steps or config.get("inference", {}).get("denoising_steps", 10))
    inference = config.get("inference", {})
    cache_depth = int(
        args.cache_depth
        if args.cache_depth is not None
        else inference.get("cache_depth", 0)
    )
    if cache_depth < 0 or cache_depth > len(model.transformer.blocks):
        raise ValueError("cache_depth must be within the transformer depth")
    refresh_steps = (
        set(args.cache_refresh_steps)
        if args.cache_refresh_steps is not None
        else set(int(item) for item in inference.get("cache_refresh_steps", []))
    )

    methods: dict[str, list[np.ndarray]] = {
        "hybrid_direct": [],
        "full_diffusion": [],
        "repeat_last": [],
    }
    targets: list[np.ndarray] = []
    previous: list[np.ndarray] = []
    task_phases: list[np.ndarray] = []
    tags = {name: [] for name in EVENT_TAG_NAMES}
    world_error_values = {name: [] for name in ("vision", "tactile", "proximity", "robot_state")}
    first_observation = None
    first_text = None
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = move_vla_batch(batch, device, torch.float32)
            torch.manual_seed(20_000 + batch_index)
            decoded = model.sample(
                batch["observation"],
                batch["text_ids"],
                steps=steps,
                cache_refresh_steps=None,
                cache_depth=0,
            )
            direct = model.predict_direct_action(batch["observation"])
            target = batch["future"]["action"]
            prior = batch["observation"]["action"][:, -1]
            methods["hybrid_direct"].append(direct.cpu().numpy())
            methods["full_diffusion"].append(decoded.action.cpu().numpy())
            methods["repeat_last"].append(prior[:, None].expand_as(target).cpu().numpy())
            targets.append(target.cpu().numpy())
            previous.append(prior.cpu().numpy())
            task_phases.append(
                batch["future"]["task_phase"][:, 0].cpu().numpy()
            )
            for name in EVENT_TAG_NAMES:
                tags[name].append(batch["event_tags"][name].cpu().numpy())
            for name, values in _world_errors(decoded, batch["future"]).items():
                world_error_values[name].append(values)
            if first_observation is None:
                first_observation = batch["observation"]
                first_text = batch["text_ids"]

    target_array = np.concatenate(targets)
    previous_array = np.concatenate(previous)
    task_phase_array = np.concatenate(task_phases)
    tag_arrays = {name: np.concatenate(values) for name, values in tags.items()}
    method_reports = {}
    for name, chunks in methods.items():
        prediction = np.concatenate(chunks)
        method_reports[name] = {
            "chunk_mae_normalized": float(np.mean(np.abs(prediction - target_array))),
            "chunk_endpoint_mae_normalized": float(
                np.mean(np.abs(prediction.sum(axis=1) - target_array.sum(axis=1)))
            ),
            "metrics_first_action": action_metric_groups(
                prediction[:, 0],
                target_array[:, 0],
                previous_array,
                tag_arrays,
                action_scale,
                task_phase=task_phase_array,
            ),
        }

    def _direct_infer():
        return model.predict_direct_action(first_observation)

    def _diffusion_infer():
        return model.sample(first_observation, first_text, steps=steps, cache_depth=0)

    method_reports["hybrid_direct"]["latency"] = _latency(
        _direct_infer,
        device,
        int(first_observation["vision"].shape[0]),
        int(args.latency_repeats),
    )
    method_reports["full_diffusion"]["latency"] = _latency(
        _diffusion_infer,
        device,
        int(first_observation["vision"].shape[0]),
        max(1, int(args.latency_repeats // 2)),
    )
    cache_report = None
    if cache_depth > 0 and refresh_steps:
        def _cached_infer():
            return model.sample(
                first_observation,
                first_text,
                steps=steps,
                cache_refresh_steps=refresh_steps,
                cache_depth=cache_depth,
            )

        cache_latency = _latency(
            _cached_infer,
            device,
            int(first_observation["vision"].shape[0]),
            max(1, int(args.latency_repeats // 2)),
        )
        torch.manual_seed(12345)
        full = model.sample(first_observation, first_text, steps=steps, cache_depth=0)
        torch.manual_seed(12345)
        cached = _cached_infer()
        cache_report = {
            "cache_depth": cache_depth,
            "refresh_steps_zero_based": sorted(refresh_steps),
            "latency": cache_latency,
            "full_to_cached_action_mae_normalized": float(
                torch.abs(full.action - cached.action).mean().cpu()
            ),
            "full_to_cached_vision_mae_normalized": float(
                torch.abs(full.vision - cached.vision).mean().cpu()
            ),
        }

    world_mae = {
        name: float(np.mean(np.concatenate(values)))
        for name, values in world_error_values.items()
    }
    report = {
        "schema_version": "mujoco_world_vla_evaluation_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_weight_source": weight_source,
        "dataset_index": str(args.dataset_index.resolve()),
        "split": args.split,
        "episodes": len(names),
        "windows": int(len(dataset)),
        "predicted_transitions": int(len(target_array) * target_array.shape[1]),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "denoising_steps": steps,
        "action_scale": list(action_scale),
        "methods": method_reports,
        "future_world_mae_normalized": world_mae,
        "cache": cache_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
