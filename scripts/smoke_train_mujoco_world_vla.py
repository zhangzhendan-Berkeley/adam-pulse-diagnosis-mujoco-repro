from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from pulse_mujoco.vla_dataset import MujocoWorldVLADataset, move_vla_batch
from pulse_world_vla.cuda import require_cuda
from pulse_world_vla.model import PulseWorldVLA


ROOT = Path(__file__).resolve().parents[1]
def _episode_splits(data: Path, index_path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    accepted = {item["episode"] for item in payload["accepted_episodes"]}
    train = [name for name in payload["splits"]["train"] if name in accepted]
    validation = [
        name for name in payload["splits"]["validation"] if name in accepted
    ]
    if not train or not validation:
        raise RuntimeError("dataset index needs non-empty train and validation splits")
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description="One-step MuJoCo-to-World-VLA training smoke.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/training/adam_pro_world_vla_smoke.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = config["model"]
    data_config = config["data"]
    train_names, validation_names = _episode_splits(args.data, args.dataset_index)
    dataset_kwargs = {
        "root": args.data,
        "image_size": tuple(model_config["image_size"]),
        "tactile_size": tuple(model_config["tactile_size"]),
        "observation_horizon": model_config["observation_horizon"],
        "prediction_horizon": model_config["prediction_horizon"],
        "sensor_range_m": data_config["sensor_range_m"],
        "tactile_scale_kpa": data_config["tactile_scale_kpa"],
        "tactile_view": data_config.get("tactile_view", "diagnostic_2x10"),
        "action_scale": tuple(data_config["action_scale_m"]),
        "state_scale": data_config["state_scale"],
        "samples_per_episode": 2,
    }
    training = MujocoWorldVLADataset(
        **dataset_kwargs, episode_names=train_names, random_window=True
    )
    validation = MujocoWorldVLADataset(
        **dataset_kwargs, episode_names=validation_names, random_window=False
    )
    expected_state_dim = int(model_config["state_dim"])
    if training.state_dim != expected_state_dim or validation.state_dim != expected_state_dim:
        raise ValueError(
            "dataset/model state dimension mismatch: "
            f"train={training.state_dim}, validation={validation.state_dim}, "
            f"model={expected_state_dim}"
        )
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = require_cuda(purpose="MuJoCo World-VLA training smoke")
    model = PulseWorldVLA(**model_config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    batch = move_vla_batch(next(iter(training_loader)), device, torch.float32)
    nonzero_action_ratio = float(
        (batch["future"]["action"].abs().sum(dim=-1) > 1e-8)
        .float()
        .mean()
        .cpu()
    )
    if nonzero_action_ratio <= 0.0:
        raise RuntimeError("training smoke selected no non-zero future actions")
    losses = model.diffusion_loss(batch, detach_tokenizer=False)
    loss = losses["diffusion"] + sum(model.autoencoder_loss(batch).values())
    if not torch.isfinite(loss):
        raise RuntimeError("training loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("model gradients are missing or non-finite")
    optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_batch = move_vla_batch(
            next(iter(validation_loader)), device, torch.float32
        )
        validation_losses = model.diffusion_loss(validation_batch)
    if not all(torch.isfinite(value) for value in validation_losses.values()):
        raise RuntimeError("validation loss is not finite")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "config": config,
        "adapter": "MujocoWorldVLADataset",
        "robot_model": "adam_pro",
        "hand_side": "right",
        "state_dim": training.state_dim,
        "dataset_index": str(args.dataset_index.resolve()),
        "training_episodes": train_names,
        "validation_episodes": validation_names,
    }
    torch.save(checkpoint, args.output)
    reloaded = PulseWorldVLA(**model_config)
    reloaded.load_state_dict(
        torch.load(args.output, map_location="cpu", weights_only=False)["model"],
        strict=True,
    )
    report = {
        "status": "passed",
        "device": str(device),
        "training_episodes": len(train_names),
        "validation_episodes": len(validation_names),
        "batch_size": int(batch["future"]["action"].shape[0]),
        "shapes": {
            "vision": list(batch["observation"]["vision"].shape),
            "tactile": list(batch["observation"]["tactile"].shape),
            "proximity": list(batch["observation"]["proximity"].shape),
            "robot_state": list(batch["observation"]["robot_state"].shape),
            "future_action": list(batch["future"]["action"].shape),
        },
        "training_loss": float(loss.detach().cpu()),
        "nonzero_future_action_ratio": nonzero_action_ratio,
        "validation_losses": {
            key: float(value.detach().cpu())
            for key, value in validation_losses.items()
        },
        "checkpoint": str(args.output.resolve()),
        "checkpoint_bytes": args.output.stat().st_size,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
