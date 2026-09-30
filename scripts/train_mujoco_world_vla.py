from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from pulse_mujoco.vla_dataset import MujocoWorldVLADataset, move_vla_batch
from pulse_world_vla.cuda import require_cuda
from pulse_world_vla.model import PulseWorldVLA
from pulse_world_vla.samplers import EventBalancedWindowSampler


ROOT = Path(__file__).resolve().parents[1]


def _splits(index_path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    accepted = {item["episode"] for item in payload["accepted_episodes"]}
    train = [name for name in payload["splits"]["train"] if name in accepted]
    validation = [
        name for name in payload["splits"]["validation"] if name in accepted
    ]
    if not train or not validation:
        raise RuntimeError("dataset index requires non-empty train and validation splits")
    return train, validation


@torch.no_grad()
def _update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source.detach(), 1.0 - decay)


def _checkpoint(
    model,
    ema,
    optimizer,
    config,
    epoch,
    history,
    train_names,
    validation_names,
    index_path,
    best,
    effective_sampling_probabilities,
    initialization,
    training_mode,
) -> dict:
    return {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "history": history,
        "best_validation_diffusion": float(best),
        "robot_model": "adam_pro",
        "hand_side": "right",
        "state_dim": 15,
        "action_dim": 7,
        "training_episodes": train_names,
        "validation_episodes": validation_names,
        "dataset_index": str(index_path.resolve()),
        "effective_sampling_probabilities": effective_sampling_probabilities,
        "initialization": initialization,
        "training_mode": training_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Adam Pro MuJoCo World-VLA.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/training/adam_pro_world_vla_stage2.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition-overlay", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="initialize model and EMA weights from a prior checkpoint, but start a new optimizer/history",
    )
    parser.add_argument(
        "--direct-head-only",
        action="store_true",
        help=(
            "freeze the tokenizer/world model and optimize only the low-latency "
            "direct action head"
        ),
    )
    parser.add_argument(
        "--direct-numeric-only",
        action="store_true",
        help="freeze all other weights and optimize only the direct numeric bypass",
    )
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    if args.resume and args.initialize_from is not None:
        parser.error("--resume and --initialize-from are mutually exclusive")
    if args.direct_head_only and args.direct_numeric_only:
        parser.error("direct-head-only and direct-numeric-only are mutually exclusive")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    sampling_config = config["sampling"]
    index_path = args.dataset_index or args.data / "splits/dataset_index.json"
    train_names, validation_names = _splits(index_path)
    dataset_kwargs = {
        "root": args.data,
        "image_size": tuple(model_config["image_size"]),
        "tactile_size": tuple(model_config["tactile_size"]),
        "observation_horizon": int(model_config["observation_horizon"]),
        "prediction_horizon": int(model_config["prediction_horizon"]),
        "sensor_range_m": float(data_config["sensor_range_m"]),
        "tactile_scale_kpa": float(data_config["tactile_scale_kpa"]),
        "tactile_view": str(data_config.get("tactile_view", "diagnostic_2x10")),
        "action_scale": tuple(data_config["action_scale"]),
        "state_scale": float(data_config["state_scale"]),
        "active_window_probability": float(
            data_config["active_window_probability"]
        ),
        "media_cache_frames": int(data_config["media_cache_frames"]),
        "transition_overlay": args.transition_overlay,
    }
    training = MujocoWorldVLADataset(
        **dataset_kwargs,
        episode_names=train_names,
        samples_per_episode=int(data_config["samples_per_episode"]),
        random_window=True,
    )
    validation = MujocoWorldVLADataset(
        **dataset_kwargs,
        episode_names=validation_names,
        samples_per_episode=int(data_config["validation_samples_per_episode"]),
        random_window=False,
    )
    expected_state_dim = int(model_config["state_dim"])
    if training.state_dim != expected_state_dim or validation.state_dim != expected_state_dim:
        raise ValueError(
            f"state dimension mismatch: train={training.state_dim}, "
            f"validation={validation.state_dim}, model={expected_state_dim}"
        )

    batch_size = args.batch_size or int(training_config["batch_size"])
    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(training_config["num_workers"])
    )
    sampler = EventBalancedWindowSampler(
        training,
        probabilities=sampling_config["probabilities"],
        num_samples=len(training),
        batch_size=batch_size,
        seed=args.seed,
    )
    device = require_cuda(purpose="MuJoCo World-VLA training")
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "prefetch_factor": 2 if workers > 0 else None,
    }
    training_loader = DataLoader(training, sampler=sampler, **loader_options)
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    model = PulseWorldVLA(**model_config).to(device=device, dtype=torch.float32)
    ema = copy.deepcopy(model).requires_grad_(False)
    if args.direct_numeric_only:
        if model.direct_numeric_head is None:
            raise ValueError(
                "--direct-numeric-only requires model.direct_numeric_bypass=true"
            )
        model.requires_grad_(False)
        model.direct_numeric_head.requires_grad_(True)
        optimized_parameters = list(model.direct_numeric_head.parameters())
        training_mode = "direct_numeric_only"
    elif args.direct_head_only:
        if model.direct_action_head is None:
            raise ValueError("--direct-head-only requires direct_action_head_weight > 0")
        model.requires_grad_(False)
        model.direct_action_head.requires_grad_(True)
        optimized_parameters = list(model.direct_action_head.parameters())
        training_mode = "direct_head_only"
    else:
        optimized_parameters = list(model.parameters())
        training_mode = "joint_world_vla"
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=float(
            args.learning_rate
            if args.learning_rate is not None
            else training_config["learning_rate"]
        ),
        weight_decay=float(training_config["weight_decay"]),
    )
    start_epoch = 0
    history: list[dict] = []
    best = float("inf")
    initialization = None
    last_path = args.output.with_suffix(".last.pt")
    if args.resume:
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        ema.load_state_dict(payload["ema"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        history = list(payload.get("history", []))
        best = float(payload.get("best_validation_diffusion", best))
        initialization = payload.get("initialization")
    elif args.initialize_from is not None:
        source = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        source_config = source.get("config", {}).get("model")
        source_weights = source.get("ema") or source["model"]
        missing, unexpected = model.load_state_dict(source_weights, strict=False)
        allowed_missing = {
            name
            for name in missing
            if name.startswith("direct_numeric_head.")
        }
        if unexpected or set(missing) != allowed_missing:
            raise RuntimeError(
                "initialization checkpoint is incompatible: "
                f"missing={missing}, unexpected={unexpected}"
            )
        ema.load_state_dict(model.state_dict(), strict=True)
        initialization = {
            "checkpoint": str(args.initialize_from.resolve()),
            "weight_source": "ema" if source.get("ema") is not None else "model",
            "source_epoch": source.get("epoch"),
            "source_best_validation_diffusion": source.get(
                "best_validation_diffusion"
            ),
            "source_model_config_differs": bool(
                source_config is not None and source_config != model_config
            ),
            "initialized_new_parameters": sorted(allowed_missing),
        }

    epochs = args.epochs or int(training_config["epochs"])
    accumulation = int(training_config["gradient_accumulation_steps"])
    use_bf16 = device.type == "cuda" and training_config["mixed_precision"] == "bf16"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        steps = 0
        warmup = epoch < int(training_config["autoencoder_warmup_epochs"])
        for step, batch in enumerate(training_loader):
            batch = move_vla_batch(batch, device, torch.float32)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                if args.direct_head_only or args.direct_numeric_only:
                    direct_objective = model.direct_action_loss(batch)
                    losses = {"direct_action_objective": direct_objective}
                    loss = direct_objective
                    reconstruction = {}
                else:
                    losses = model.diffusion_loss(
                        batch, detach_tokenizer=not warmup
                    )
                    loss = losses["diffusion"]
                    reconstruction = (
                        model.autoencoder_loss(batch) if warmup else {}
                    )
                if reconstruction:
                    loss = loss + float(
                        training_config["autoencoder_loss_weight"]
                    ) * sum(reconstruction.values())
            (loss / accumulation).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(training_loader):
                torch.nn.utils.clip_grad_norm_(
                    optimized_parameters,
                    float(training_config["gradient_clip_norm"]),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _update_ema(ema, model, float(training_config["ema_decay"]))
            for name, value in {"total": loss, **losses, **reconstruction}.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach().float().cpu())
            steps += 1

        ema.eval()
        validation_sums: dict[str, float] = {}
        validation_steps = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = move_vla_batch(batch, device, torch.float32)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    if args.direct_head_only or args.direct_numeric_only:
                        direct_objective = ema.direct_action_loss(batch)
                        losses = {"direct_action_objective": direct_objective}
                    else:
                        losses = ema.diffusion_loss(batch)
                for name, value in losses.items():
                    validation_sums[name] = validation_sums.get(name, 0.0) + float(
                        value.detach().float().cpu()
                    )
                validation_steps += 1
        metrics = {
            "epoch": epoch,
            "train": {name: value / steps for name, value in sums.items()},
            "validation": {
                name: value / validation_steps
                for name, value in validation_sums.items()
            },
        }
        history.append(metrics)
        selection_metric = (
            "direct_action_objective"
            if args.direct_head_only or args.direct_numeric_only
            else "diffusion"
        )
        validation_loss = metrics["validation"][selection_metric]
        payload = _checkpoint(
            model,
            ema,
            optimizer,
            config,
            epoch,
            history,
            train_names,
            validation_names,
            index_path,
            min(best, validation_loss),
            sampler.effective_probabilities,
            initialization,
            training_mode,
        )
        torch.save(payload, last_path)
        if validation_loss < best:
            best = validation_loss
            payload["best_validation_diffusion"] = best
            torch.save(payload, args.output)
        args.output.with_suffix(".metrics.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "training_mode": training_mode,
                    "selection_metric": selection_metric,
                    "train_selection_metric": metrics["train"][selection_metric],
                    "validation_selection_metric": validation_loss,
                    "best": best,
                    "device": str(device),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
