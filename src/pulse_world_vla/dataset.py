from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.media import load_depth
from src.datasets.event_labels import EVENT_TAG_NAMES, window_event_tags


def instruction_tokens(text: str, length: int = 24, vocabulary_size: int = 4096) -> torch.Tensor:
    """Dependency-free UTF-8 n-gram hashing; replaceable by a pretrained tokenizer."""
    encoded = text.encode("utf-8")
    tokens = []
    for index in range(max(0, len(encoded) - 2)):
        digest = hashlib.blake2s(encoded[index : index + 3], digest_size=2).digest()
        tokens.append(1 + int.from_bytes(digest, "little") % (vocabulary_size - 1))
        if len(tokens) == length:
            break
    return torch.tensor(tokens + [0] * (length - len(tokens)), dtype=torch.long)


class PulseWorldDataset(Dataset):
    def __init__(
        self,
        root,
        image_size=(128, 128),
        tactile_size=(32, 32),
        observation_horizon=2,
        prediction_horizon=4,
        sensor_range_m=0.03,
        action_scale_m=0.01,
        action_dim=3,
        state_scale=1.0,
        successful_only=True,
        random_window=True,
        episode_names=None,
        samples_per_episode=8,
        dynamic_window_probability=0.5,
        pressure_window_probability=0.0,
        dynamic_threshold_m=0.001,
        event_translation_threshold_m=5e-4,
        event_rotation_threshold_rad=2e-3,
        event_pressure_threshold_kpa=1e-3,
        evaluation_window_mode="uniform",
        tactile_scale_kpa=None,
        disable_tactile=False,
        allow_video_latch=False,
        load_future_modalities=True,
        observation_modalities="all",
        media_cache_root=None,
        require_media_cache=False,
    ):
        self.root = Path(root)
        self.image_size = tuple(image_size)
        self.tactile_size = tuple(tactile_size)
        self.observation_horizon = int(observation_horizon)
        self.prediction_horizon = int(prediction_horizon)
        self.sensor_range_m = float(sensor_range_m)
        self.action_dim = int(action_dim)
        action_scale = np.asarray(action_scale_m, dtype=np.float32)
        if action_scale.ndim == 0:
            action_scale = np.full(self.action_dim, float(action_scale))
        if action_scale.shape != (self.action_dim,) or np.any(action_scale <= 0):
            raise ValueError(
                f"action_scale_m must be positive scalar or shape "
                f"({self.action_dim},), got {action_scale.shape}"
            )
        self.action_scale = action_scale
        self.state_scale = float(state_scale)
        self.random_window = bool(random_window)
        self.samples_per_episode = max(1, int(samples_per_episode))
        self.dynamic_window_probability = float(dynamic_window_probability)
        self.pressure_window_probability = float(pressure_window_probability)
        if not 0.0 <= self.dynamic_window_probability <= 1.0:
            raise ValueError("dynamic_window_probability must be in [0, 1]")
        if not 0.0 <= self.pressure_window_probability <= 1.0:
            raise ValueError("pressure_window_probability must be in [0, 1]")
        self.dynamic_threshold_m = float(dynamic_threshold_m)
        self.event_translation_threshold_m = float(
            event_translation_threshold_m
        )
        self.event_rotation_threshold_rad = float(
            event_rotation_threshold_rad
        )
        self.event_pressure_threshold_kpa = float(
            event_pressure_threshold_kpa
        )
        if min(
            self.event_translation_threshold_m,
            self.event_rotation_threshold_rad,
            self.event_pressure_threshold_kpa,
        ) <= 0:
            raise ValueError("event thresholds must be positive")
        if evaluation_window_mode not in {
            "uniform",
            "dynamic",
            "proximity",
            "tactile",
            "pressure",
        }:
            raise ValueError(
                "evaluation_window_mode must be uniform, dynamic, proximity, "
                "tactile or pressure"
            )
        self.evaluation_window_mode = evaluation_window_mode
        self.tactile_scale_kpa = (
            None if tactile_scale_kpa is None else float(tactile_scale_kpa)
        )
        if self.tactile_scale_kpa is not None and self.tactile_scale_kpa <= 0:
            raise ValueError("tactile_scale_kpa must be positive")
        self.disable_tactile = bool(disable_tactile)
        self.allow_video_latch = bool(allow_video_latch)
        self.load_future_modalities = bool(load_future_modalities)
        if observation_modalities not in {"all", "last"}:
            raise ValueError("observation_modalities must be all or last")
        self.observation_modalities = observation_modalities
        self.media_cache_root = (
            None
            if media_cache_root is None
            else Path(media_cache_root)
        )
        self.require_media_cache = bool(require_media_cache)
        self._media_cache_arrays = {}
        if self.media_cache_root is not None:
            cache_manifest_path = (
                self.media_cache_root / "cache_manifest.json"
            )
            if not cache_manifest_path.exists():
                if self.require_media_cache:
                    raise RuntimeError(
                        f"media cache manifest not found: "
                        f"{cache_manifest_path}"
                    )
                self.media_cache_root = None
            else:
                cache_manifest = json.loads(
                    cache_manifest_path.read_text(encoding="utf-8")
                )
                if (
                    tuple(cache_manifest["image_size"])
                    != self.image_size
                    or tuple(cache_manifest["tactile_size"])
                    != self.tactile_size
                ):
                    raise ValueError(
                        "media cache resolution does not match dataset "
                        "configuration"
                    )
                if self.tactile_scale_kpa is not None and not np.isclose(
                    float(cache_manifest["tactile_scale_kpa"]),
                    self.tactile_scale_kpa,
                ):
                    raise ValueError(
                        "media cache tactile scale does not match dataset"
                    )
        episode_names = None if episode_names is None else set(episode_names)
        window = self.observation_horizon + self.prediction_horizon
        self.episodes = []
        self.all_window_candidates = []
        self.window_candidates_by_event = {
            name: [] for name in EVENT_TAG_NAMES
        }
        for metadata_path in sorted(self.root.glob("episode_*/episode.json")):
            if episode_names is not None and metadata_path.parent.name not in episode_names:
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                not self.allow_video_latch
                and metadata.get("control_mode")
                == "video_kinematic_contact_latch"
            ):
                continue
            if successful_only and not metadata.get("success", False):
                continue
            manifest = metadata_path.parent / "transitions.jsonl"
            records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            if len(records) >= window:
                episode_index = len(self.episodes)
                dynamic_starts = []
                proximity_starts = []
                tactile_starts = []
                pressure_starts = []
                event_starts = {name: [] for name in EVENT_TAG_NAMES}
                for start in range(len(records) - window + 1):
                    sequence = records[start : start + window]
                    future_sequence = sequence[self.observation_horizon :]
                    tags = window_event_tags(
                        records,
                        start,
                        self.observation_horizon,
                        self.prediction_horizon,
                        translation_threshold_m=(
                            self.event_translation_threshold_m
                        ),
                        rotation_threshold_rad=self.event_rotation_threshold_rad,
                        pressure_threshold_kpa=self.event_pressure_threshold_kpa,
                    )
                    self.all_window_candidates.append((episode_index, start))
                    for name, enabled in tags.items():
                        if enabled:
                            event_starts[name].append(start)
                            self.window_candidates_by_event[name].append(
                                (episode_index, start)
                            )
                    previous_pressure_action = np.asarray(
                        sequence[self.observation_horizon - 1].get(
                            "executed_action",
                            [
                                *sequence[self.observation_horizon - 1].get(
                                    "action_delta_tcp_m", [0, 0, 0]
                                ),
                                0,
                                0,
                                0,
                                0,
                            ],
                        ),
                        dtype=float,
                    )
                    first_future_pressure_action = np.asarray(
                        future_sequence[0].get(
                            "executed_action",
                            [
                                *future_sequence[0].get(
                                    "action_delta_tcp_m", [0, 0, 0]
                                ),
                                0,
                                0,
                                0,
                                0,
                            ],
                        ),
                        dtype=float,
                    )
                    if (
                        previous_pressure_action.size >= 7
                        and first_future_pressure_action.size >= 7
                        and abs(
                            float(first_future_pressure_action[6])
                            - float(previous_pressure_action[6])
                        )
                        > 1e-6
                    ):
                        # The first predicted action switches pressure intent.
                        # This is deliberately narrower than "dynamic": it
                        # isolates contact-on/contact-off supervision.
                        pressure_starts.append(start)
                    if any(
                        any(bool(value) for value in record.get("proximity_valid", []))
                        for record in future_sequence
                    ):
                        proximity_starts.append(start)
                    context_sequence = sequence[: self.observation_horizon]
                    if any(
                        record.get("tactile_valid", False)
                        and np.any(
                            np.abs(
                                np.asarray(
                                    record.get("tactile_raw_delta", [0.0]),
                                    dtype=float,
                                )
                            )
                            > 0.0
                        )
                        for record in context_sequence
                    ):
                        tactile_starts.append(start)
                    maximum_change = 0.0
                    for previous, current in zip(sequence, sequence[1:]):
                        for left, right in zip(
                            previous.get("proximity_distances_m", []),
                            current.get("proximity_distances_m", []),
                        ):
                            if left is not None and right is not None:
                                maximum_change = max(maximum_change, abs(float(right) - float(left)))
                        previous_action = np.asarray(
                            previous.get(
                                "executed_action",
                                [*previous.get("action_delta_tcp_m", [0, 0, 0]), 0, 0, 0, 0],
                            ),
                            dtype=float,
                        )
                        current_action = np.asarray(
                            current.get(
                                "executed_action",
                                [*current.get("action_delta_tcp_m", [0, 0, 0]), 0, 0, 0, 0],
                            ),
                            dtype=float,
                        )
                        if previous_action.size >= 3 and current_action.size >= 3:
                            maximum_change = max(
                                maximum_change,
                                float(np.linalg.norm(current_action[:3])),
                                float(
                                    np.linalg.norm(
                                        current_action[:3]
                                        - previous_action[:3]
                                    )
                                ),
                            )
                        if (
                            previous_action.shape == current_action.shape
                            and previous_action.size > 3
                            and np.any(
                                np.abs(current_action[3:] - previous_action[3:])
                                > 1e-8
                            )
                        ):
                            # Rotation and pressure have different units from
                            # translation, so treat any real change as dynamic.
                            maximum_change = max(
                                maximum_change, self.dynamic_threshold_m
                            )
                        if previous.get("task_phase") != current.get("task_phase"):
                            maximum_change = max(
                                maximum_change, self.dynamic_threshold_m
                            )
                    if maximum_change >= self.dynamic_threshold_m:
                        dynamic_starts.append(start)
                self.episodes.append(
                    (
                        metadata_path.parent,
                        metadata,
                        records,
                        {
                            "dynamic": dynamic_starts,
                            "proximity": proximity_starts,
                            "tactile": tactile_starts,
                            "pressure": pressure_starts,
                            **event_starts,
                        },
                    )
                )
        if not self.episodes:
            raise RuntimeError(f"No usable sequence episodes found in {self.root}")

    def __len__(self):
        return len(self.episodes) * self.samples_per_episode

    def _load_vision(self, episode_dir: Path, record: dict) -> torch.Tensor:
        cached = self._cached_episode_arrays(episode_dir)
        if cached is not None:
            frame = int(record["frame"])
            if 0 <= frame < len(cached["vision"]):
                return torch.from_numpy(
                    np.asarray(
                        cached["vision"][frame], dtype=np.float32
                    )
                )
            if self.require_media_cache:
                raise IndexError(
                    f"cached vision frame {frame} is out of range"
                )
        rgb_path = Path(record["rgb_path"])
        depth_path = Path(record["depth_path"])
        if not rgb_path.is_absolute():
            rgb_path = episode_dir / rgb_path
        if not depth_path.is_absolute():
            depth_path = episode_dir / depth_path
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        depth = load_depth(depth_path, record.get("depth_encoding"))
        rgb = cv2.resize(rgb, self.image_size[::-1], interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1
        depth = cv2.resize(depth, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        depth = np.nan_to_num(depth, nan=2.0, posinf=2.0, neginf=0.0)
        depth = np.clip(depth, 0.0, 2.0).astype(np.float32) - 1.0
        return torch.from_numpy(np.concatenate((rgb.transpose(2, 0, 1), depth[None]), axis=0)).float()

    def _load_tactile(self, episode_dir: Path, record: dict):
        if self.disable_tactile:
            return torch.zeros(1, *self.tactile_size), False
        cached = self._cached_episode_arrays(episode_dir)
        if cached is not None:
            frame = int(record["frame"])
            if 0 <= frame < len(cached["tactile"]):
                return (
                    torch.from_numpy(
                        np.asarray(
                            cached["tactile"][frame],
                            dtype=np.float32,
                        )
                    ),
                    bool(cached["tactile_valid"][frame]),
                )
            if self.require_media_cache:
                raise IndexError(
                    f"cached tactile frame {frame} is out of range"
                )
        path = record.get("tactile_path")
        if not path or not record.get("tactile_valid", False):
            return torch.zeros(1, *self.tactile_size), False
        path = Path(path)
        if not path.is_absolute():
            path = episode_dir / path
        tactile = np.load(path).astype(np.float32)
        if tactile.ndim == 3:
            tactile = tactile.mean(axis=-1)
        tactile = cv2.resize(tactile, self.tactile_size[::-1], interpolation=cv2.INTER_AREA)
        # A fixed training-split scale preserves pressure and pulse amplitude
        # across time. None retains the legacy per-frame normalization needed
        # to evaluate checkpoints trained before Schema-v2 tactile scaling.
        scale = self.tactile_scale_kpa
        if scale is None:
            scale = max(float(np.percentile(np.abs(tactile), 99)), 1e-6)
        return torch.from_numpy(np.clip(tactile / scale, -1, 1)[None]).float(), True

    def _cached_episode_arrays(self, episode_dir: Path):
        if self.media_cache_root is None:
            return None
        episode_name = episode_dir.name
        cached = self._media_cache_arrays.get(episode_name)
        if cached is not None:
            return cached
        cache_dir = self.media_cache_root / episode_name
        paths = {
            "vision": cache_dir / "vision.npy",
            "tactile": cache_dir / "tactile.npy",
            "tactile_valid": cache_dir / "tactile_valid.npy",
        }
        if not all(path.exists() for path in paths.values()):
            if self.require_media_cache:
                raise RuntimeError(
                    f"incomplete media cache for {episode_name}"
                )
            return None
        cached = {
            name: np.load(path, mmap_mode="r")
            for name, path in paths.items()
        }
        self._media_cache_arrays[episode_name] = cached
        return cached

    def _frames(self, episode_dir, records):
        (
            vision,
            tactile,
            tactile_valid,
            proximity,
            proximity_valid,
            state,
            action,
            task_phase,
        ) = ([] for _ in range(8))
        for frame_index, record in enumerate(records):
            observation_frame = frame_index < self.observation_horizon
            load_observation_frame = observation_frame and (
                self.observation_modalities == "all"
                or frame_index == self.observation_horizon - 1
            )
            load_future_frame = (
                not observation_frame and self.load_future_modalities
            )
            if load_observation_frame or load_future_frame:
                vision.append(self._load_vision(episode_dir, record))
                tactile_image, valid = self._load_tactile(episode_dir, record)
            else:
                # Action-only baselines do not consume future image targets.
                # A last-frame BC also does not consume earlier history. Keep
                # tensor shapes stable without decoding unused media.
                vision.append(torch.zeros(4, *self.image_size))
                tactile_image = torch.zeros(1, *self.tactile_size)
                valid = False
            tactile.append(tactile_image)
            tactile_valid.append(valid)
            raw_proximity = record.get("proximity_distances_m", [None] * 5)
            raw_valid = record.get("proximity_valid", [value is not None for value in raw_proximity])
            filled = [self.sensor_range_m if value is None else value for value in raw_proximity]
            proximity.append(np.clip(np.asarray(filled) / self.sensor_range_m, 0, 1) * 2 - 1)
            proximity_valid.append(raw_valid)
            state.append(np.asarray(record["robot_state"], dtype=np.float32) / self.state_scale)
            executed = record.get("executed_action")
            if executed is None:
                legacy_xyz = np.asarray(
                    record["action_delta_tcp_m"], dtype=np.float32
                )
                if self.action_dim == 3:
                    executed = legacy_xyz
                elif self.action_dim == 7:
                    executed = np.concatenate(
                        (legacy_xyz, np.zeros(4, dtype=np.float32))
                    )
                else:
                    raise ValueError(
                        "legacy action fallback supports only action_dim 3 or 7"
                    )
            executed = np.asarray(executed, dtype=np.float32)
            if executed.shape != (self.action_dim,):
                raise ValueError(
                    f"executed_action must have shape ({self.action_dim},), "
                    f"got {executed.shape}"
                )
            action.append(np.clip(executed / self.action_scale, -1, 1))
            task_phase.append(int(record.get("task_phase", 0)))
        return {
            "vision": torch.stack(vision),
            "tactile": torch.stack(tactile),
            "tactile_valid": torch.tensor(tactile_valid, dtype=torch.bool),
            "proximity": torch.tensor(np.asarray(proximity), dtype=torch.float32),
            "proximity_valid": torch.tensor(np.asarray(proximity_valid), dtype=torch.bool),
            "robot_state": torch.tensor(np.asarray(state), dtype=torch.float32),
            "action": torch.tensor(np.asarray(action), dtype=torch.float32),
            "task_phase": torch.tensor(
                np.asarray(task_phase), dtype=torch.long
            ),
        }

    def __getitem__(self, index):
        explicit_start = isinstance(index, tuple)
        if explicit_start:
            episode_index, start = (int(value) for value in index)
            if not 0 <= episode_index < len(self.episodes):
                raise IndexError("episode index is out of range")
            episode_dir, metadata, records, candidate_starts = self.episodes[
                episode_index
            ]
        else:
            episode_dir, metadata, records, candidate_starts = self.episodes[
                index % len(self.episodes)
            ]
        dynamic_starts = candidate_starts["dynamic"]
        pressure_starts = candidate_starts["pressure"]
        total = self.observation_horizon + self.prediction_horizon
        maximum_start = len(records) - total
        if explicit_start:
            if not 0 <= start <= maximum_start:
                raise IndexError("window start crosses episode boundary")
        elif self.random_window and maximum_start:
            draw = np.random.random()
            if pressure_starts and draw < self.pressure_window_probability:
                start = int(np.random.choice(pressure_starts))
            elif (
                dynamic_starts
                and draw
                < self.pressure_window_probability
                + (1.0 - self.pressure_window_probability)
                * self.dynamic_window_probability
            ):
                start = int(np.random.choice(dynamic_starts))
            else:
                start = np.random.randint(maximum_start + 1)
        elif maximum_start:
            replica = index // len(self.episodes)
            fraction = replica / max(1, self.samples_per_episode - 1)
            selected_starts = candidate_starts.get(self.evaluation_window_mode, [])
            if self.evaluation_window_mode != "uniform" and selected_starts:
                selected_index = int(round(fraction * (len(selected_starts) - 1)))
                start = int(selected_starts[selected_index])
            else:
                start = int(round(fraction * maximum_start))
        else:
            start = 0
        frames = self._frames(episode_dir, records[start : start + total])
        observation = {key: value[: self.observation_horizon] for key, value in frames.items()}
        future = {key: value[self.observation_horizon :] for key, value in frames.items()}
        return {
            "observation": observation,
            "future": future,
            "text_ids": instruction_tokens(metadata["instruction"]),
            "success": torch.tensor(float(metadata.get("success", False))),
            "event_tags": {
                name: torch.tensor(value, dtype=torch.bool)
                for name, value in window_event_tags(
                    records,
                    start,
                    self.observation_horizon,
                    self.prediction_horizon,
                    translation_threshold_m=self.event_translation_threshold_m,
                    rotation_threshold_rad=self.event_rotation_threshold_rad,
                    pressure_threshold_kpa=self.event_pressure_threshold_kpa,
                ).items()
            },
        }


def move_batch(batch, device, dtype):
    def move(value):
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        value = value.to(device)
        return value.to(dtype) if torch.is_floating_point(value) else value

    return move(batch)
