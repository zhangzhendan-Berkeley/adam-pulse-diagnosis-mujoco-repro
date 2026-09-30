from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .vla_dataset import MujocoWorldVLADataset


class SearchSegmentDataset(Dataset):
    """Load episode-level split search-point examples from a compact index."""

    def __init__(
        self,
        index_path: Path | str,
        split: str,
        *,
        image_size: tuple[int, int] = (64, 64),
        tactile_size: tuple[int, int] = (32, 32),
        observation_horizon: int = 4,
        sensor_range_m: float = 0.03,
        tactile_scale_kpa: float = 320.0,
        tactile_view: str = "diagnostic_2x10",
        action_scale: tuple[float, ...] = (0.025, 0.042, 0.030, 0.112, 0.092, 0.086, 150.0),
        state_scale: float = 3.141592653589793,
        media_cache_frames: int = 4096,
    ):
        self.index_path = Path(index_path)
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") not in {
            "adam_pulse_search_segment_v1",
            "adam_pulse_search_segment_v2",
            "adam_pulse_search_segment_v3_2d_centering",
        }:
            raise ValueError("unsupported search-segment index")
        if split not in payload["splits"]:
            raise KeyError(f"unknown split: {split}")
        self.maximum_displacement_m = float(payload["maximum_displacement_m"])
        self.target_contract = str(
            payload.get("target_contract", "signed_lateral_1d")
        )
        self.stop_gate = payload.get("stop_gate")
        selected = set(payload["splits"][split])
        self.samples = [item for item in payload["samples"] if item["sample_id"] in selected]
        self.observation_horizon = int(observation_horizon)
        indexed_horizon = int(payload.get("observation_horizon", observation_horizon))
        if indexed_horizon != self.observation_horizon:
            raise ValueError(
                "observation horizon does not match index: "
                f"{self.observation_horizon} != {indexed_horizon}"
            )
        if not self.samples:
            raise RuntimeError(f"search-segment split is empty: {split}")

        # Reuse the production normalization/media implementation without
        # constructing its episode window table.
        adapter = MujocoWorldVLADataset.__new__(MujocoWorldVLADataset)
        adapter.image_size = tuple(image_size)
        adapter.tactile_size = tuple(tactile_size)
        adapter.sensor_range_m = float(sensor_range_m)
        adapter.tactile_scale_kpa = float(tactile_scale_kpa)
        adapter.tactile_view = str(tactile_view)
        adapter.expose_full56 = False
        adapter.tactile_normalization = None
        adapter.action_scale = torch.as_tensor(action_scale).numpy()
        adapter.state_scale = float(state_scale)
        adapter.state_dim = int(payload["state_dim"])
        adapter.observation_horizon = self.observation_horizon
        adapter.load_future_modalities = True
        adapter.media_cache_frames = max(0, int(media_cache_frames))
        from collections import OrderedDict
        adapter._vision_cache = OrderedDict()
        adapter._tactile_cache = OrderedDict()
        self.adapter = adapter
        self._record_cache: dict[str, list[dict]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[int(index)]
        episode_dir = Path(sample["episode_dir"])
        cache_key = str(episode_dir)
        records = self._record_cache.get(cache_key)
        if records is None:
            records = [
                json.loads(line)
                for line in (episode_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self._record_cache[cache_key] = records
        end = int(sample["observation_end_frame"])
        start = end - self.observation_horizon + 1
        if start < 0:
            raise ValueError(f"insufficient history for {sample['sample_id']}")
        frames = self.adapter._frames(
            episode_dir,
            records[start : end + 1],
            action_source="previous_action",
        )
        # Contact state is an explicit input of the CH5 contact-establishment
        # specialist, not part of the frozen search-segment observation
        # contract. The shared media decoder exposes it for World-VLA users,
        # so remove it at this narrower policy boundary.
        for extended_field in (
            "diagnostic_contact",
            "tactile_full56",
            "tactile_full56_delta",
            "tactile_full56_valid",
            "normal_force",
            "normal_force_valid",
        ):
            frames.pop(extended_field, None)
        signed = float(sample["signed_displacement_m"])
        result = {
            "observation": frames,
            "target": {
                "stop": torch.tensor(float(sample["stop"])),
                "direction_positive": torch.tensor(float(signed > 0.0)),
                "magnitude_normalized": torch.tensor(
                    abs(signed) / self.maximum_displacement_m,
                    dtype=torch.float32,
                ),
                "signed_displacement_m": torch.tensor(signed, dtype=torch.float32),
            },
            "sample_id": sample["sample_id"],
            "episode": sample["episode"],
            "source": sample["source"],
            "point_index": int(sample["point_index"]),
            "training_weight": torch.tensor(
                float(sample.get("training_weight", 1.0)), dtype=torch.float32
            ),
        }
        if self.target_contract == "planned_delta_patient_uv_2d":
            delta_uv = torch.as_tensor(
                sample["planned_delta_patient_uv_m"], dtype=torch.float32
            )
            if tuple(delta_uv.shape) != (2,):
                raise ValueError(
                    f"invalid 2-D target for {sample['sample_id']}: {tuple(delta_uv.shape)}"
                )
            result["target"].update(
                {
                    "delta_uv_m": delta_uv,
                    "delta_uv_normalized": delta_uv / self.maximum_displacement_m,
                    "magnitude_normalized": torch.linalg.vector_norm(delta_uv)
                    / self.maximum_displacement_m,
                }
            )
        result["target_contract"] = self.target_contract
        quality = sample.get("observable_signal_quality")
        if quality is not None:
            # Keep the full-window safety gate beside, never inside, the
            # model observation.  Callers must explicitly opt into it.
            result["observable_signal_quality"] = {
                key: torch.tensor(float(value), dtype=torch.float32)
                for key, value in quality.items()
            }
        return result
