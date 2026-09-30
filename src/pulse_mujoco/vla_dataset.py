from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .episode_data import SUPPORTED_EPISODE_SCHEMA_VERSIONS, load_depth
from .event_labels import EVENT_TAG_NAMES, window_event_tags
from .m5_transition_overlay import (
    apply_transition_overlay,
    load_transition_overlay,
    sha256_file,
)
from .tactile_normalization import (
    FULL56_CHANNEL_COUNTS,
    FULL56_FINGER_NAMES,
    TactileNormalization,
    load_tactile_normalization,
)


def instruction_tokens(
    text: str, length: int = 24, vocabulary_size: int = 4096
) -> torch.Tensor:
    encoded = text.encode("utf-8")
    tokens: list[int] = []
    for index in range(max(0, len(encoded) - 2)):
        digest = hashlib.blake2s(encoded[index : index + 3], digest_size=2).digest()
        tokens.append(1 + int.from_bytes(digest, "little") % (vocabulary_size - 1))
        if len(tokens) == length:
            break
    return torch.tensor(tokens + [0] * (length - len(tokens)), dtype=torch.long)


class MujocoWorldVLADataset(Dataset):
    """Zero-conversion adapter from MuJoCo Episodes to World-VLA tensors."""

    def __init__(
        self,
        root: Path | str,
        *,
        episode_names: list[str] | tuple[str, ...] | None = None,
        image_size: tuple[int, int] = (64, 64),
        tactile_size: tuple[int, int] = (32, 32),
        observation_horizon: int = 4,
        prediction_horizon: int = 4,
        sensor_range_m: float = 0.03,
        tactile_scale_kpa: float = 320.0,
        tactile_view: str = "diagnostic_2x10",
        action_scale: tuple[float, ...] = (
            0.025,
            0.042,
            0.030,
            0.112,
            0.092,
            0.086,
            150.0,
        ),
        state_scale: float = np.pi,
        samples_per_episode: int = 8,
        random_window: bool = False,
        active_window_probability: float = 0.75,
        load_future_modalities: bool = True,
        event_translation_threshold_m: float = 5e-4,
        event_rotation_threshold_rad: float = 2e-3,
        event_pressure_threshold_kpa: float = 1e-3,
        media_cache_frames: int = 1024,
        transition_overlay: Path | str | None = None,
        expose_full56: bool = False,
        tactile_normalization: Path | str | TactileNormalization | None = None,
        allow_audited_recovery_failures: bool = False,
    ):
        self.root = Path(root)
        self.image_size = tuple(int(value) for value in image_size)
        self.tactile_size = tuple(int(value) for value in tactile_size)
        self.observation_horizon = int(observation_horizon)
        self.prediction_horizon = int(prediction_horizon)
        self.sensor_range_m = float(sensor_range_m)
        self.tactile_scale_kpa = float(tactile_scale_kpa)
        self.tactile_view = str(tactile_view)
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        self.state_scale = float(state_scale)
        self.samples_per_episode = max(1, int(samples_per_episode))
        self.random_window = bool(random_window)
        self.active_window_probability = float(active_window_probability)
        self.load_future_modalities = bool(load_future_modalities)
        self.expose_full56 = bool(expose_full56)
        self.allow_audited_recovery_failures = bool(
            allow_audited_recovery_failures
        )
        self.tactile_normalization = (
            tactile_normalization
            if isinstance(tactile_normalization, TactileNormalization)
            else (
                None
                if tactile_normalization is None
                else load_tactile_normalization(tactile_normalization)
            )
        )
        if self.tactile_normalization is not None:
            if self.tactile_normalization.source_split != "train":
                raise ValueError("tactile normalization must be fitted on train split")
            if (
                self.tactile_normalization.finger_names != FULL56_FINGER_NAMES
                or self.tactile_normalization.channel_counts
                != FULL56_CHANNEL_COUNTS
            ):
                raise ValueError("tactile normalization full56 topology mismatch")
        self.event_thresholds = {
            "translation_threshold_m": float(event_translation_threshold_m),
            "rotation_threshold_rad": float(event_rotation_threshold_rad),
            "pressure_threshold_kpa": float(event_pressure_threshold_kpa),
        }
        self.media_cache_frames = max(0, int(media_cache_frames))
        self._vision_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._tactile_cache: OrderedDict[str, tuple] = OrderedDict()
        self.transition_overlay_path = (
            None if transition_overlay is None else Path(transition_overlay)
        )
        self.transition_overlay = (
            None
            if self.transition_overlay_path is None
            else load_transition_overlay(self.transition_overlay_path)
        )
        if self.transition_overlay is not None:
            overlay_root = Path(self.transition_overlay["source_root"]).resolve()
            if overlay_root != self.root.resolve():
                raise ValueError(
                    "transition overlay source root does not match dataset root: "
                    f"{overlay_root} != {self.root.resolve()}"
                )
        if self.action_scale.shape != (7,) or np.any(self.action_scale <= 0):
            raise ValueError("action_scale must contain seven positive values")
        if self.sensor_range_m <= 0 or self.tactile_scale_kpa <= 0 or self.state_scale <= 0:
            raise ValueError("sensor, tactile and state scales must be positive")
        if self.tactile_view not in {"diagnostic_2x10", "active"}:
            raise ValueError(
                "tactile_view must be diagnostic_2x10 or active; full56 requires "
                "a topology-aware adapter and is not silently reshaped"
            )
        if not 0.0 <= self.active_window_probability <= 1.0:
            raise ValueError("active_window_probability must be in [0, 1]")
        if min(self.event_thresholds.values()) <= 0:
            raise ValueError("event thresholds must be positive")

        selected = None if episode_names is None else set(episode_names)
        required = self.observation_horizon + self.prediction_horizon
        self.episodes: list[tuple[Path, dict, list[dict], list[int]]] = []
        self.all_window_candidates: list[tuple[int, int]] = []
        self.window_candidates_by_event = {
            name: [] for name in EVENT_TAG_NAMES
        }
        for metadata_path in sorted(self.root.glob("episode_*/episode.json")):
            episode_dir = metadata_path.parent
            if selected is not None and episode_dir.name not in selected:
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") not in SUPPORTED_EPISODE_SCHEMA_VERSIONS:
                continue
            ordinary_success = bool(
                metadata.get("success", False)
                and metadata.get("signal_quality_accepted", False)
            )
            recovery = metadata.get("poststudent_recovery") or {}
            audited_recovery = bool(
                self.allow_audited_recovery_failures
                and recovery.get("schema_version")
                == "adam_pulse_search_poststudent_recovery_v1"
                and recovery.get("accepted") is True
                and recovery.get("train_only") is True
                and recovery.get("truth_in_observation") is False
                and int(recovery.get("student_segment_execution_count", -1)) == 1
                and int(recovery.get("poststudent_recovery_frame_count", 0)) > 0
                and int(recovery.get("safety_modifications", -1)) == 0
                and int(recovery.get("forbidden_contact_events", -1)) == 0
                and float(recovery.get("maximum_force_n", float("inf"))) < 10.0
            )
            if not ordinary_success and not audited_recovery:
                continue
            transition_path = episode_dir / "transitions.jsonl"
            records = [
                json.loads(line)
                for line in transition_path
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            if self.transition_overlay is not None:
                episode_overlay = self.transition_overlay["episodes"].get(
                    episode_dir.name
                )
                if episode_overlay is None:
                    if not bool(
                        self.transition_overlay.get("allow_unlisted_episodes", False)
                    ):
                        raise ValueError(
                            f"transition overlay has no entry for {episode_dir.name}"
                        )
                else:
                    if sha256_file(transition_path) != episode_overlay.get(
                        "source_transitions_sha256"
                    ):
                        raise ValueError(
                            "transition overlay source hash mismatch: "
                            f"{episode_dir.name}"
                        )
                    records = apply_transition_overlay(records, episode_overlay)
            if len(records) >= required:
                episode_index = len(self.episodes)
                active_starts = []
                for start in range(len(records) - required + 1):
                    future = records[
                        start + self.observation_horizon : start + required
                    ]
                    if any(
                        np.linalg.norm(
                            np.asarray(record["executed_action"], dtype=float)
                        )
                        > 1e-8
                        for record in future
                    ):
                        active_starts.append(start)
                    tags = window_event_tags(
                        records,
                        start,
                        self.observation_horizon,
                        self.prediction_horizon,
                        **self.event_thresholds,
                    )
                    self.all_window_candidates.append((episode_index, start))
                    for name, enabled in tags.items():
                        if enabled:
                            self.window_candidates_by_event[name].append(
                                (episode_index, start)
                            )
                self.episodes.append(
                    (episode_dir, metadata, records, active_starts)
                )
        if not self.episodes:
            raise RuntimeError(f"no usable MuJoCo Episode sequences in {self.root}")
        state_dims = {
            len(records[0]["robot_state"])
            for _, _, records, _ in self.episodes
        }
        if len(state_dims) != 1:
            raise ValueError(
                "episodes with different robot-state dimensions cannot be "
                f"mixed in one dataset: {sorted(state_dims)}"
            )
        self.state_dim = state_dims.pop()

    def __len__(self) -> int:
        return len(self.episodes) * self.samples_per_episode

    def _vision(self, episode_dir: Path, record: dict) -> torch.Tensor:
        cache_key = str(episode_dir / record["rgb_path"])
        cached = self._vision_cache.get(cache_key)
        if cached is not None:
            self._vision_cache.move_to_end(cache_key)
            return cached
        rgb_bgr = cv2.imread(str(episode_dir / record["rgb_path"]))
        if rgb_bgr is None:
            raise FileNotFoundError(episode_dir / record["rgb_path"])
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = load_depth(
            episode_dir / record["depth_path"], record.get("depth_encoding")
        )
        rgb = cv2.resize(rgb, self.image_size[::-1], interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        rgb = rgb.astype(np.float32) / 127.5 - 1.0
        depth = np.nan_to_num(depth, nan=2.0, posinf=2.0, neginf=0.0)
        depth = np.clip(depth, 0.0, 2.0).astype(np.float32) - 1.0
        channels = np.concatenate((rgb.transpose(2, 0, 1), depth[None]), axis=0)
        value = torch.from_numpy(channels)
        self._cache_put(self._vision_cache, cache_key, value)
        return value

    def _tactile(self, episode_dir: Path, record: dict) -> tuple:
        path = episode_dir / record["tactile_path"]
        cache_key = str(path)
        cached = self._tactile_cache.get(cache_key)
        if cached is not None:
            self._tactile_cache.move_to_end(cache_key)
            return cached
        full56 = None
        full56_valid = np.zeros(56, dtype=bool)
        with np.load(path) as payload:
            if self.tactile_view == "diagnostic_2x10":
                key = (
                    "tactile_diagnostic_2x10_kpa"
                    if "tactile_diagnostic_2x10_kpa" in payload
                    else "pressure_kpa"
                )
                pressure = np.asarray(payload[key], dtype=np.float32)
                if pressure.shape != (2, 10):
                    raise ValueError(
                        "diagnostic tactile pressure must have shape (2, 10), "
                        f"got {pressure.shape}"
                    )
            else:
                if "tactile_active_kpa" not in payload:
                    raise ValueError(
                        "episode does not provide the CH5 tactile_active_kpa view"
                    )
                pressure = np.asarray(payload["tactile_active_kpa"], dtype=np.float32)
                if pressure.ndim != 2 or pressure.shape[1] != 10:
                    raise ValueError(
                        "active tactile pressure must have shape (fingers, 10), "
                        f"got {pressure.shape}"
                    )
            if self.expose_full56:
                if "tactile_full_56_kpa" not in payload:
                    raise ValueError("episode does not provide tactile_full_56_kpa")
                full56 = np.asarray(
                    payload["tactile_full_56_kpa"], dtype=np.float32
                )
                if full56.shape != (56,):
                    raise ValueError(
                        f"full-hand tactile must have shape (56,), got {full56.shape}"
                    )
                full56_valid[:] = True
                if self.tactile_normalization is not None:
                    full56 = self.tactile_normalization.normalize(full56)
        image = cv2.resize(
            pressure, self.tactile_size[::-1], interpolation=cv2.INTER_LINEAR
        )
        image = np.clip(image / self.tactile_scale_kpa, -1.0, 1.0)
        value = (
            torch.from_numpy(image[None].astype(np.float32)),
            True,
            torch.from_numpy(
                np.zeros(56, dtype=np.float32)
                if full56 is None
                else full56.astype(np.float32)
            ),
            torch.from_numpy(full56_valid),
        )
        self._cache_put(self._tactile_cache, cache_key, value)
        return value

    def _cache_put(self, cache: OrderedDict, key: str, value) -> None:
        if self.media_cache_frames <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.media_cache_frames:
            cache.popitem(last=False)

    @staticmethod
    def _action_target(record: dict) -> np.ndarray:
        """Return the expert correction when a DAgger label is available."""
        return np.asarray(
            record.get("expert_label_action", record["executed_action"]),
            dtype=np.float32,
        )

    def _frames(
        self,
        episode_dir: Path,
        records: list[dict],
        *,
        action_source: str = "observation_then_expert_target",
    ) -> dict[str, torch.Tensor]:
        """Decode synchronized frames under an explicit action contract.

        The default split contract uses deployment-visible ``previous_action``
        for observation frames and expert correction targets only for future
        frames.  Keeping the choice explicit prevents a DAgger correction from
        leaking into the student observation that produced that correction.
        """
        if action_source not in {
            "observation_then_expert_target",
            "expert_target",
            "previous_action",
            "executed_action",
        }:
            raise ValueError(f"unsupported action source: {action_source}")
        result: dict[str, list] = {
            "vision": [],
            "tactile": [],
            "tactile_valid": [],
            "proximity": [],
            "proximity_valid": [],
            "robot_state": [],
            "action": [],
            "task_phase": [],
            "diagnostic_contact": [],
            "tactile_full56": [],
            "tactile_full56_valid": [],
            "normal_force": [],
            "normal_force_valid": [],
        }
        for frame_index, record in enumerate(records):
            load_media = (
                frame_index < self.observation_horizon or self.load_future_modalities
            )
            if load_media:
                vision = self._vision(episode_dir, record)
                (
                    tactile,
                    tactile_valid,
                    tactile_full56,
                    tactile_full56_valid,
                ) = self._tactile(episode_dir, record)
            else:
                vision = torch.zeros(4, *self.image_size)
                tactile = torch.zeros(1, *self.tactile_size)
                tactile_valid = False
                tactile_full56 = torch.zeros(56)
                tactile_full56_valid = torch.zeros(56, dtype=torch.bool)
            raw_proximity = np.asarray(record["proximity_m"], dtype=np.float32)
            proximity_valid = np.asarray(record["proximity_valid"], dtype=bool)
            filled = np.where(proximity_valid, raw_proximity, self.sensor_range_m)
            proximity = np.clip(filled / self.sensor_range_m, 0.0, 1.0) * 2.0 - 1.0
            state = np.asarray(record["robot_state"], dtype=np.float32)
            if action_source == "observation_then_expert_target":
                action = (
                    np.asarray(record["previous_action"], dtype=np.float32)
                    if frame_index < self.observation_horizon
                    else self._action_target(record)
                )
            elif action_source == "expert_target":
                action = self._action_target(record)
            else:
                action = np.asarray(record[action_source], dtype=np.float32)
            if state.shape != (self.state_dim,) or action.shape != (7,):
                raise ValueError(
                    "MuJoCo state/action contract mismatch: expected "
                    f"{self.state_dim}-D/7-D, got {state.shape}/{action.shape}"
                )
            result["vision"].append(vision)
            result["tactile"].append(tactile)
            result["tactile_valid"].append(tactile_valid)
            result["proximity"].append(proximity)
            result["proximity_valid"].append(proximity_valid)
            result["robot_state"].append(state / self.state_scale)
            result["action"].append(np.clip(action / self.action_scale, -1.0, 1.0))
            result["task_phase"].append(int(record["task_phase"]))
            result["diagnostic_contact"].append(
                np.asarray(
                    record.get("active_contact", record["contact"]), dtype=bool
                )
            )
            full_force_value = record.get("full_hand_normal_force_n")
            full_force = np.asarray(
                np.zeros(5)
                if full_force_value is None
                else full_force_value,
                dtype=np.float32,
            )
            if full_force.shape != (5,):
                raise ValueError(
                    f"full-hand normal force must have shape (5,), got {full_force.shape}"
                )
            result["tactile_full56"].append(tactile_full56)
            result["tactile_full56_valid"].append(tactile_full56_valid)
            result["normal_force"].append(full_force)
            full_contact_value = record.get("full_hand_contact")
            result["normal_force_valid"].append(
                np.asarray(
                    np.zeros(5)
                    if full_contact_value is None
                    else full_contact_value,
                    dtype=bool,
                )
            )
        tactile_full56 = torch.stack(result["tactile_full56"])
        tactile_delta = torch.zeros_like(tactile_full56)
        tactile_delta[1:] = tactile_full56[1:] - tactile_full56[:-1]
        return {
            "vision": torch.stack(result["vision"]),
            "tactile": torch.stack(result["tactile"]),
            "tactile_valid": torch.tensor(result["tactile_valid"], dtype=torch.bool),
            "proximity": torch.from_numpy(np.asarray(result["proximity"], dtype=np.float32)),
            "proximity_valid": torch.from_numpy(np.asarray(result["proximity_valid"], dtype=bool)),
            "robot_state": torch.from_numpy(np.asarray(result["robot_state"], dtype=np.float32)),
            "action": torch.from_numpy(np.asarray(result["action"], dtype=np.float32)),
            "task_phase": torch.tensor(result["task_phase"], dtype=torch.long),
            "diagnostic_contact": torch.from_numpy(
                np.asarray(result["diagnostic_contact"], dtype=bool)
            ),
            "tactile_full56": tactile_full56,
            "tactile_full56_delta": tactile_delta,
            "tactile_full56_valid": torch.stack(
                result["tactile_full56_valid"]
            ),
            "normal_force": torch.from_numpy(
                np.asarray(result["normal_force"], dtype=np.float32)
            ),
            "normal_force_valid": torch.from_numpy(
                np.asarray(result["normal_force_valid"], dtype=bool)
            ),
        }

    def __getitem__(self, index: int) -> dict:
        explicit_start = isinstance(index, tuple)
        if explicit_start:
            episode_index, start = (int(value) for value in index)
            if not 0 <= episode_index < len(self.episodes):
                raise IndexError("episode index is out of range")
            replica = 0
        else:
            episode_index = int(index) % len(self.episodes)
            replica = int(index) // len(self.episodes)
        episode_dir, metadata, records, active_starts = self.episodes[episode_index]
        total = self.observation_horizon + self.prediction_horizon
        maximum_start = len(records) - total
        if explicit_start:
            if not 0 <= start <= maximum_start:
                raise IndexError("window start crosses episode boundary")
        elif self.random_window and maximum_start:
            if (
                active_starts
                and np.random.random() < self.active_window_probability
            ):
                start = int(np.random.choice(active_starts))
            else:
                start = int(np.random.randint(maximum_start + 1))
        elif maximum_start:
            fraction = replica / max(1, self.samples_per_episode - 1)
            start = int(round(fraction * maximum_start))
        else:
            start = 0
        frames = self._frames(episode_dir, records[start : start + total])
        return {
            "observation": {
                key: value[: self.observation_horizon] for key, value in frames.items()
            },
            "future": {
                key: value[self.observation_horizon :] for key, value in frames.items()
            },
            "text_ids": instruction_tokens(metadata["instruction"]),
            "success": torch.tensor(float(metadata.get("success", False))),
            "episode": episode_dir.name,
            "start_frame": start,
            "event_tags": {
                name: torch.tensor(value, dtype=torch.bool)
                for name, value in window_event_tags(
                    records,
                    start,
                    self.observation_horizon,
                    self.prediction_horizon,
                    **self.event_thresholds,
                ).items()
            },
        }


def move_vla_batch(batch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    def move(value):
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            return value.to(dtype) if torch.is_floating_point(value) else value
        return value

    return move(batch)
