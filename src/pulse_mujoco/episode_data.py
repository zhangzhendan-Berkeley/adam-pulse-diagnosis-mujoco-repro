from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .actions import PulseAction
from .observation import PulseObservation
from .task import TaskPhase


EPISODE_SCHEMA_VERSION = "adam_pulse_episode_v4_ch5"
SUPPORTED_EPISODE_SCHEMA_VERSIONS = {
    "adam_pulse_episode_v2",
    "adam_pulse_episode_v3",
    EPISODE_SCHEMA_VERSION,
}
DEPTH_ENCODING = "uint16_png_0p1mm"
TACTILE_SCHEMA_VERSION = "adam_pulse_tactile_v2_full56"
TACTILE_POLICY_FINGERS = ("index", "middle")
FULL_HAND_FINGERS = ("thumb", "index", "middle", "ring", "little")
FULL_HAND_CHANNEL_COUNTS = (16, 10, 10, 10, 10)
POLICY_OBSERVATION_FIELDS = (
    "vision",
    "tactile",
    "tactile_valid",
    "proximity",
    "proximity_valid",
    "robot_state",
    "action",
    "task_phase",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_depth_uint16(path: Path, depth_m: np.ndarray) -> dict:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D, got {depth.shape}")
    scale = 1e-4
    valid = np.isfinite(depth) & (depth >= 0.0) & (depth <= 6.5535)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    encoded[valid] = np.clip(
        np.rint(depth[valid] / scale), 1, 65535
    ).astype(np.uint16)
    if not cv2.imwrite(str(path), encoded, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise OSError(f"failed to save depth image: {path}")
    return {
        "format": DEPTH_ENCODING,
        "scale_m_per_unit": scale,
        "invalid_value": 0,
    }


def load_depth(path: Path, encoding: dict | None = None) -> np.ndarray:
    details = encoding or {}
    if details.get("format", DEPTH_ENCODING) != DEPTH_ENCODING:
        raise ValueError(f"unsupported depth encoding: {details}")
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None or encoded.dtype != np.uint16 or encoded.ndim != 2:
        raise ValueError(f"invalid uint16 depth image: {path}")
    depth = encoded.astype(np.float32) * float(
        details.get("scale_m_per_unit", 1e-4)
    )
    depth[encoded == int(details.get("invalid_value", 0))] = np.inf
    return depth


class MujocoEpisodeRecorder:
    """Synchronized compressed recorder with an explicit state contract."""

    def __init__(
        self,
        root: Path | str,
        episode_id: str,
        instruction: str,
        *,
        capture_interval: int = 1,
        jpeg_quality: int = 88,
        record_size: tuple[int, int] | None = None,
    ):
        self.episode_dir = Path(root) / f"episode_{episode_id}"
        self.frame_dir = self.episode_dir / "frames"
        if self.episode_dir.exists() and any(self.episode_dir.iterdir()):
            raise FileExistsError(f"episode already exists: {self.episode_dir}")
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.instruction = str(instruction)
        self.capture_interval = max(1, int(capture_interval))
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.record_size = (
            None
            if record_size is None
            else (int(record_size[0]), int(record_size[1]))
        )
        if self.record_size is not None and min(self.record_size) <= 0:
            raise ValueError("record_size must contain positive width and height")
        self.callback_steps = 0
        self.frames = 0
        self.records: list[dict] = []
        self.arm_joint_dim: int | None = None
        self.duplicate_timestamp_callbacks = 0
        self.tactile_contract: dict | None = None
        self._last_rgb: np.ndarray | None = None
        self._last_depth: np.ndarray | None = None
        self._last_rgb_path: Path | None = None
        self._last_depth_path: Path | None = None
        self._last_depth_encoding: dict | None = None
        self.reused_rgbd_frames = 0

    @staticmethod
    def _robot_state(observation: dict) -> np.ndarray:
        # The controlled arm dimension is model-specific: Lite=4, Pro=7.
        return np.concatenate(
            (
                np.asarray(observation["joint_position_rad"], dtype=np.float32),
                np.asarray(observation["tcp_pose"], dtype=np.float32),
                np.asarray([observation["task_phase"] / 13.0], dtype=np.float32),
            )
        )

    @staticmethod
    def _full_hand_arrays(frame, field: str) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
        names = tuple(str(name) for name in frame.finger_names)
        counts = tuple(int(value) for value in frame.channel_counts)
        if names != FULL_HAND_FINGERS or counts != FULL_HAND_CHANNEL_COUNTS:
            raise ValueError(
                "full-hand tactile order/count mismatch: "
                f"{names}/{counts} != {FULL_HAND_FINGERS}/{FULL_HAND_CHANNEL_COUNTS}"
            )
        values = np.asarray(getattr(frame, field))
        if values.shape != (sum(FULL_HAND_CHANNEL_COUNTS),):
            raise ValueError(f"full-hand {field} must have shape (56,), got {values.shape}")
        return values, names, counts

    def _tactile_payload(self, backend, observation: dict) -> tuple[dict, dict]:
        pressure = np.asarray(observation["tactile_kpa"], dtype=np.float32)
        raw = np.asarray(observation["tactile_raw"], dtype=np.uint16)
        raw_delta = np.asarray(observation["tactile_raw_delta"], dtype=np.float32)
        if pressure.shape != (2, 10) or raw.shape != (2, 10) or raw_delta.shape != (2, 10):
            raise ValueError("policy-compatible tactile arrays must have shape (2, 10)")
        dynamic = (
            np.zeros((2, 10), dtype=np.float32)
            if backend.last_tactile is None
            else np.asarray(
                backend.last_tactile.dynamic_arrays_kpa, dtype=np.float32
            )
        )
        if dynamic.shape != (2, 10):
            raise ValueError("policy-compatible tactile dynamic array must have shape (2, 10)")

        payload = {
            # Frozen aliases used by all pre-CH5 loaders.
            "pressure_kpa": pressure,
            "dynamic_kpa": dynamic,
            "raw": raw,
            "raw_delta": raw_delta,
            # Explicit CH5 names. These are analysis/supervision data unless a
            # later ablation deliberately selects another policy view.
            "tactile_diagnostic_2x10_kpa": pressure,
            "tactile_diagnostic_2x10_dynamic_kpa": dynamic,
            "tactile_diagnostic_2x10_raw": raw,
            "tactile_diagnostic_2x10_raw_delta": raw_delta,
        }
        active_names = tuple(
            str(name)
            for name in getattr(backend, "diagnostic_finger_names", TACTILE_POLICY_FINGERS)
        )
        frame = getattr(backend, "last_tactile_full", None)
        contract = {
            "schema_version": TACTILE_SCHEMA_VERSION,
            "policy_view": "tactile_diagnostic_2x10_kpa",
            "policy_finger_names": list(TACTILE_POLICY_FINGERS),
            "policy_shape": [2, 10],
            "active_view": None,
            "active_finger_names": list(active_names),
            "active_shape": None,
            "full_hand_view": None,
            "full_hand_finger_names": list(FULL_HAND_FINGERS),
            "full_hand_channel_counts": list(FULL_HAND_CHANNEL_COUNTS),
            "full_hand_channel_offsets": [0, 16, 26, 36, 46],
            "full_hand_available": bool(frame is not None),
        }
        record_fields = {
            "active_contact": None,
            "active_normal_force_n": None,
            "full_hand_contact": None,
            "full_hand_normal_force_n": None,
        }
        if frame is not None:
            full_pressure, names, counts = self._full_hand_arrays(frame, "pressure_kpa")
            full_dynamic, _, _ = self._full_hand_arrays(frame, "dynamic_pressure_kpa")
            full_raw, _, _ = self._full_hand_arrays(frame, "raw_values")
            full_raw_delta, _, _ = self._full_hand_arrays(frame, "raw_delta")
            finger_slices = {name: frame.finger_slice(name) for name in names}
            if any(name not in finger_slices for name in active_names):
                raise ValueError(f"unknown active tactile fingers: {active_names}")
            if any(counts[names.index(name)] != 10 for name in active_names):
                raise ValueError("CH5 active diagnostic fingers must each have 10 channels")
            active_pressure = np.stack(
                [full_pressure[finger_slices[name]] for name in active_names]
            ).astype(np.float32)
            active_dynamic = np.stack(
                [full_dynamic[finger_slices[name]] for name in active_names]
            ).astype(np.float32)
            active_raw = np.stack(
                [full_raw[finger_slices[name]] for name in active_names]
            ).astype(np.uint16)
            active_raw_delta = np.stack(
                [full_raw_delta[finger_slices[name]] for name in active_names]
            ).astype(np.float32)
            payload.update(
                {
                    "tactile_active_kpa": active_pressure,
                    "tactile_active_dynamic_kpa": active_dynamic,
                    "tactile_active_raw": active_raw,
                    "tactile_active_raw_delta": active_raw_delta,
                    "tactile_full_56_kpa": full_pressure.astype(np.float32),
                    "tactile_full_56_dynamic_kpa": full_dynamic.astype(np.float32),
                    "tactile_full_56_raw": full_raw.astype(np.uint16),
                    "tactile_full_56_raw_delta": full_raw_delta.astype(np.float32),
                }
            )
            full_contact = np.asarray(frame.contact, dtype=bool)
            full_force = np.asarray(frame.total_normal_force_n, dtype=np.float32)
            if full_contact.shape != (5,) or full_force.shape != (5,):
                raise ValueError("full-hand contact/force arrays must have shape (5,)")
            active_indices = [names.index(name) for name in active_names]
            record_fields = {
                "active_contact": full_contact[active_indices].tolist(),
                "active_normal_force_n": full_force[active_indices].tolist(),
                "full_hand_contact": full_contact.tolist(),
                "full_hand_normal_force_n": full_force.tolist(),
            }
            contract.update(
                {
                    "active_view": "tactile_active_kpa",
                    "active_shape": [len(active_names), 10],
                    "full_hand_view": "tactile_full_56_kpa",
                }
            )
        if self.tactile_contract is None:
            self.tactile_contract = contract
        elif contract != self.tactile_contract:
            raise ValueError("tactile contract changed within an episode")
        return payload, record_fields

    def __call__(self, backend, observation: dict, info: dict, label: str) -> None:
        self.callback_steps += 1
        if self.callback_steps % self.capture_interval:
            return
        timestamp_s = float(observation["timestamp_s"])
        # The expert emits a final success/failure sync without advancing the
        # MuJoCo clock.  At capture_interval=1 that callback is observable and
        # would create a zero-duration transition, which is invalid for both
        # replay and temporal training.  Keep the first physical sample and
        # record the skipped callback in metadata instead of fabricating time.
        if self.records and timestamp_s <= float(self.records[-1]["timestamp_s"]):
            self.duplicate_timestamp_callbacks += 1
            return
        rgb = observation.get("rgb")
        depth = observation.get("depth_m")
        if rgb is None or depth is None:
            raise RuntimeError("episode recording requires include_rgbd=True")
        arm_joint_dim = int(
            np.asarray(observation["joint_position_rad"]).size
        )
        if self.arm_joint_dim is None:
            self.arm_joint_dim = arm_joint_dim
        elif arm_joint_dim != self.arm_joint_dim:
            raise ValueError(
                "arm joint dimension changed within an episode: "
                f"{self.arm_joint_dim} -> {arm_joint_dim}"
            )
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(f"invalid RGB shape: {rgb.shape}")
        rgb = rgb[:, :, :3]
        if self.record_size is not None:
            width, height = self.record_size
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(
                np.asarray(depth, dtype=np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        stem = f"{self.frames:06d}"
        tactile_path = self.frame_dir / f"tactile_{stem}.npz"
        reuse_rgbd = bool(
            self._last_rgb is not None
            and self._last_depth is not None
            and np.array_equal(rgb, self._last_rgb)
            and np.array_equal(depth, self._last_depth, equal_nan=True)
        )
        if reuse_rgbd:
            assert self._last_rgb_path is not None
            assert self._last_depth_path is not None
            assert self._last_depth_encoding is not None
            rgb_path = self._last_rgb_path
            depth_path = self._last_depth_path
            depth_encoding = dict(self._last_depth_encoding)
            self.reused_rgbd_frames += 1
        else:
            rgb_path = self.frame_dir / f"rgb_{stem}.jpg"
            depth_path = self.frame_dir / f"depth_{stem}.png"
            if not cv2.imwrite(
                str(rgb_path),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            ):
                raise OSError(f"failed to save RGB image: {rgb_path}")
            depth_encoding = save_depth_uint16(depth_path, depth)
            self._last_rgb = rgb.copy()
            self._last_depth = depth.copy()
            self._last_rgb_path = rgb_path
            self._last_depth_path = depth_path
            self._last_depth_encoding = dict(depth_encoding)
        tactile_payload, tactile_record_fields = self._tactile_payload(
            backend, observation
        )
        np.savez_compressed(tactile_path, **tactile_payload)
        executed = np.asarray(
            info.get("executed_action", observation["previous_action"]),
            dtype=np.float32,
        )
        proposed = np.asarray(info.get("proposed_action", executed), dtype=np.float32)
        safe = np.asarray(info.get("safe_action", executed), dtype=np.float32)
        execution = np.asarray(
            info.get("execution_action", safe), dtype=np.float32
        )
        expert_label = np.asarray(
            info.get("expert_label_action", executed), dtype=np.float32
        )
        safety = info.get("safety") or {}
        execution_safety = info.get("execution_safety") or {}
        record = {
            "frame": self.frames,
            "sim_step": self.callback_steps,
            "timestamp_s": timestamp_s,
            "label": str(label),
            "task_phase": int(observation["task_phase"]),
            "task_phase_name": TaskPhase(int(observation["task_phase"])).name.lower(),
            "rgb_path": str(rgb_path.relative_to(self.episode_dir)),
            "rgb_encoding": {"format": "jpeg", "quality": self.jpeg_quality},
            "depth_path": str(depth_path.relative_to(self.episode_dir)),
            "depth_encoding": depth_encoding,
            "tactile_path": str(tactile_path.relative_to(self.episode_dir)),
            "joint_position_rad": _jsonable(observation["joint_position_rad"]),
            "joint_velocity_rad_s": _jsonable(observation["joint_velocity_rad_s"]),
            "tcp_pose": _jsonable(observation["tcp_pose"]),
            "tcp_velocity": _jsonable(observation["tcp_velocity"]),
            "robot_state": self._robot_state(observation).tolist(),
            "proximity_m": _jsonable(observation["proximity_m"]),
            "proximity_valid": _jsonable(observation["proximity_valid"]),
            "contact": _jsonable(observation["contact"]),
            "normal_force_n": _jsonable(info.get("normal_force_n", [0.0, 0.0])),
            **tactile_record_fields,
            "previous_action": _jsonable(observation["previous_action"]),
            "expert_proposed_action": proposed.tolist(),
            "safe_action": safe.tolist(),
            "execution_action": execution.tolist(),
            "executed_action": executed.tolist(),
            # DAgger target for the state actually visited.  In pure expert
            # episodes it equals executed_action; in phase-gated episodes it
            # remains the safety-filtered expert correction even when a
            # learned action was physically executed.
            "expert_label_action": expert_label.tolist(),
            "action_modified": bool(
                safety.get("modified", False)
                or execution_safety.get("modified", False)
            ),
            "action_modification_reasons": sorted(
                set(safety.get("reasons", []))
                | set(execution_safety.get("reasons", []))
            ),
            "expert_intervened": bool(
                (info.get("expert_recovery") or {}).get("active", False)
                or (info.get("policy_control") or {}).get(
                    "expert_intervened", False
                )
            ),
            "recovery_mode": (
                (info.get("expert_recovery") or {}).get("mode")
            ),
            "risk_score": float(
                1.0
                if (info.get("expert_recovery") or {}).get("active", False)
                else 0.0
            ),
            "policy_control": _jsonable(info.get("policy_control")),
            "perturbation_state": _jsonable(
                info.get(
                    "perturbation_state",
                    {
                        "active": False,
                        "events": [],
                        "recoverable": True,
                    },
                )
            ),
            "collision_safe": bool(info.get("collision_safe", True)),
            # Backend-only recovery diagnostics.  These are intentionally
            # outside the policy observation contract and are persisted for
            # post-run contact-loss audits.
            "pressure_recovery_audit": _jsonable(
                {
                    key: info[key]
                    for key in (
                        "diagnostic_contacts_before",
                        "diagnostic_contacts_after",
                        "diagnostic_contact_mask_changed",
                        "pressure_latched_before",
                        "pressure_latched_after",
                        "pressure_recovery_hold_active_before",
                        "pressure_recovery_hold_active_after",
                        "pressure_feedforward_torque_norm_nm_before",
                        "pressure_feedforward_torque_norm_nm_after",
                        "pressure_recovery_target_delta_rad",
                        "pressure_contact_lost",
                        "pressure_recovery_joint_hold_active",
                        "pressure_hold_target_retained_on_ik_rejection",
                        "action_accepted",
                        "ik_solver_converged",
                        "ik_position_error_m",
                    )
                    if key in info
                }
            ),
        }
        self.records.append(record)
        self.frames += 1

    def close(self, result: dict, *, extra: dict | None = None) -> Path:
        if self.arm_joint_dim is None:
            raise RuntimeError("cannot close an episode without recorded frames")
        manifest = self.episode_dir / "transitions.jsonl"
        manifest.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self.records),
            encoding="utf-8",
        )
        metadata = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "simulator": "mujoco",
            "instruction": self.instruction,
            "frames": self.frames,
            "duplicate_timestamp_callbacks": self.duplicate_timestamp_callbacks,
            "reused_rgbd_frames": self.reused_rgbd_frames,
            "success": bool(result.get("success", False)),
            "failure": result.get("failure"),
            "search_strategy": result.get("search_strategy"),
            "simulation_time_s": result.get("simulation_time_s"),
            "unique_search_positions": result.get("unique_search_positions"),
            "maximum_force_n_per_finger": _jsonable(
                result.get("maximum_force_n_per_finger")
            ),
            "forbidden_contact_events": int(
                result.get("forbidden_contact_events", 0)
            ),
            "selected_signal_quality": _jsonable(
                result.get("selected_signal_quality")
            ),
            "signal_quality_schema_version": result.get(
                "signal_quality_schema_version"
            ),
            "signal_quality_accepted": bool(
                result.get("signal_quality_accepted", False)
            ),
            "control_mode": "expert_dynamic_control",
            "expert_version": (
                "mujoco_signal_quality_v1"
                if result.get("search_strategy") == "signal_quality_early_stop"
                else "mujoco_directional_v1"
            ),
            "state_contract": {
                "name": "controlled_arm_qpos_tcp_pose_phase",
                "arm_joint_dim": self.arm_joint_dim,
                "tcp_pose_dim": 7,
                "phase_dim": 1,
                "robot_state_dim": self.arm_joint_dim + 8,
            },
            "quaternion_convention": "wxyz",
            "media_storage": {
                "rgb": {
                    "format": "jpeg",
                    "quality": self.jpeg_quality,
                    "size": (
                        None
                        if self.record_size is None
                        else list(self.record_size)
                    ),
                },
                "depth": {"format": DEPTH_ENCODING, "scale_m_per_unit": 1e-4},
                "tactile": {"format": "npz_compressed"},
            },
            "tactile_contract": self.tactile_contract,
            "observation_contract": {
                "schema_version": "adam_pulse_policy_observation_v2",
                "fields": list(POLICY_OBSERVATION_FIELDS),
                "tactile_view": "tactile_diagnostic_2x10_kpa",
                "truth_in_observation": False,
            },
            "action_contract": {
                "schema_version": "adam_pulse_action_history_v2",
                "previous_action": "last physically executed action visible at current frame",
                "executed_action": "action physically executed to reach current frame",
                "expert_label_action": "supervision only; never policy history",
                "planned_and_executed_landing_separate": True,
            },
        }
        if extra:
            collisions = sorted(set(extra) & set(metadata))
            if collisions:
                raise ValueError(
                    "episode extras cannot override recorder contracts: "
                    + ", ".join(collisions)
                )
            metadata.update(_jsonable(extra))
        (self.episode_dir / "episode.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.episode_dir


class EpisodeReplayBackend:
    """Replay recorded observations through the live environment contract."""

    def __init__(self, episode_dir: Path | str):
        self.episode_dir = Path(episode_dir)
        self.metadata = json.loads(
            (self.episode_dir / "episode.json").read_text(encoding="utf-8")
        )
        self.records = [
            json.loads(line)
            for line in (self.episode_dir / "transitions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError(f"empty episode: {self.episode_dir}")
        self.index = 0

    def _observation(self, index: int) -> PulseObservation:
        record = self.records[index]
        rgb_bgr = cv2.imread(str(self.episode_dir / record["rgb_path"]))
        if rgb_bgr is None:
            raise FileNotFoundError(record["rgb_path"])
        tactile = np.load(self.episode_dir / record["tactile_path"])
        return PulseObservation(
            timestamp_s=float(record["timestamp_s"]),
            rgb=cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
            depth_m=load_depth(
                self.episode_dir / record["depth_path"],
                record["depth_encoding"],
            ),
            joint_position_rad=np.asarray(record["joint_position_rad"]),
            joint_velocity_rad_s=np.asarray(record["joint_velocity_rad_s"]),
            tcp_pose=np.asarray(record["tcp_pose"]),
            tcp_velocity=np.asarray(record["tcp_velocity"]),
            proximity_m=np.asarray(record["proximity_m"]),
            proximity_valid=np.asarray(record["proximity_valid"]),
            tactile_kpa=tactile["pressure_kpa"],
            tactile_raw=tactile["raw"],
            tactile_raw_delta=tactile["raw_delta"],
            contact=np.asarray(record["contact"]),
            task_phase=int(record["task_phase"]),
            previous_action=np.asarray(record["previous_action"]),
        )

    def reset(self, seed: int | None = None) -> tuple[PulseObservation, dict]:
        self.index = 0
        return self._observation(0), {
            "seed": seed,
            "replay": True,
            "collision_safe": True,
        }

    def apply_action(self, action: PulseAction):
        next_index = min(self.index + 1, len(self.records) - 1)
        expert = np.asarray(
            self.records[next_index]["executed_action"], dtype=np.float32
        )
        error = float(np.linalg.norm(action.as_array() - expert))
        self.index = next_index
        terminated = self.index == len(self.records) - 1
        return self._observation(self.index), -error, terminated, False, {
            "replay": True,
            "collision_safe": True,
            "executed_action": expert,
            "expert_action": expert,
            "action_error": error,
            "environment_step": self.index,
        }

    def get_privileged_state(self) -> dict:
        return {"episode_success": bool(self.metadata["success"]), "frame": self.index}

    def save_state(self) -> dict:
        return {"schema_version": 1, "index": self.index}

    def load_state(self, snapshot: dict) -> None:
        if snapshot.get("schema_version") != 1:
            raise ValueError("unsupported replay snapshot schema")
        index = int(snapshot["index"])
        if not 0 <= index < len(self.records):
            raise ValueError("replay index is out of range")
        self.index = index

    def emergency_retreat(self) -> PulseObservation:
        return self._observation(self.index)

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class EpisodeValidation:
    valid: bool
    errors: tuple[str, ...]
    frames: int


def validate_episode(episode_dir: Path | str, *, check_media: bool = True) -> EpisodeValidation:
    root = Path(episode_dir)
    errors: list[str] = []
    try:
        metadata = json.loads((root / "episode.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (root / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        return EpisodeValidation(False, (str(error),), 0)
    schema_version = metadata.get("schema_version")
    if schema_version not in SUPPORTED_EPISODE_SCHEMA_VERSIONS:
        errors.append("unsupported episode schema")
    state_contract = metadata.get("state_contract")
    if isinstance(state_contract, dict):
        arm_joint_dim = int(state_contract.get("arm_joint_dim", -1))
        robot_state_dim = int(state_contract.get("robot_state_dim", -1))
    elif state_contract == "qpos4_tcp_pose7_phase1":
        arm_joint_dim, robot_state_dim = 4, 12
    elif records:
        arm_joint_dim = len(records[0].get("joint_position_rad", []))
        robot_state_dim = arm_joint_dim + 8
    else:
        arm_joint_dim, robot_state_dim = -1, -1
    if arm_joint_dim <= 0 or robot_state_dim != arm_joint_dim + 8:
        errors.append("invalid state contract")
    if metadata.get("frames") != len(records) or not records:
        errors.append("metadata frame count mismatch or empty episode")
    tactile_contract = metadata.get("tactile_contract")
    if schema_version == EPISODE_SCHEMA_VERSION:
        if not isinstance(tactile_contract, dict):
            errors.append("missing tactile contract")
            tactile_contract = {}
        elif tactile_contract.get("schema_version") != TACTILE_SCHEMA_VERSION:
            errors.append("unsupported tactile contract")
        observation_contract = metadata.get("observation_contract")
        if not isinstance(observation_contract, dict):
            errors.append("missing observation contract")
        else:
            fields = tuple(observation_contract.get("fields", ()))
            if fields != POLICY_OBSERVATION_FIELDS:
                errors.append("policy observation allowlist mismatch")
            if observation_contract.get("truth_in_observation") is not False:
                errors.append("truth_in_observation must be false")
        action_contract = metadata.get("action_contract")
        if not isinstance(action_contract, dict):
            errors.append("missing action contract")
    else:
        tactile_contract = {}
    active_shape = tuple(tactile_contract.get("active_shape") or ())
    full_hand_available = bool(tactile_contract.get("full_hand_available", False))
    previous_time = -np.inf
    for index, record in enumerate(records):
        where = f"frame {index}"
        if record.get("frame") != index:
            errors.append(f"{where}: non-contiguous frame index")
        timestamp = float(record.get("timestamp_s", np.nan))
        if not np.isfinite(timestamp) or timestamp <= previous_time:
            errors.append(f"{where}: timestamp is not strictly increasing")
        previous_time = timestamp
        expected_shapes = {
            "joint_position_rad": (arm_joint_dim,),
            "joint_velocity_rad_s": (arm_joint_dim,),
            "tcp_pose": (7,), "tcp_velocity": (6,),
            "robot_state": (robot_state_dim,),
            "proximity_m": (5,), "proximity_valid": (5,), "contact": (2,),
            "expert_proposed_action": (7,), "safe_action": (7,),
            "executed_action": (7,),
        }
        if schema_version == EPISODE_SCHEMA_VERSION:
            expected_shapes["execution_action"] = (7,)
            expected_shapes["expert_label_action"] = (7,)
            expected_shapes["previous_action"] = (7,)
            if active_shape:
                expected_shapes["active_contact"] = (active_shape[0],)
                expected_shapes["active_normal_force_n"] = (active_shape[0],)
            if full_hand_available:
                expected_shapes["full_hand_contact"] = (5,)
                expected_shapes["full_hand_normal_force_n"] = (5,)
        for key, shape in expected_shapes.items():
            if np.asarray(record.get(key)).shape != shape:
                errors.append(f"{where}: {key} must have shape {shape}")
        if not record.get("collision_safe", False):
            errors.append(f"{where}: collision_safe is false")
        if check_media:
            try:
                rgb = cv2.imread(str(root / record["rgb_path"]))
                if rgb is None:
                    raise ValueError("unreadable RGB")
                load_depth(root / record["depth_path"], record["depth_encoding"])
                with np.load(root / record["tactile_path"]) as tactile:
                    if tactile["pressure_kpa"].shape != (2, 10):
                        raise ValueError("invalid tactile shape")
                    if schema_version == EPISODE_SCHEMA_VERSION:
                        required = {
                            "tactile_diagnostic_2x10_kpa": (2, 10),
                            "tactile_diagnostic_2x10_dynamic_kpa": (2, 10),
                            "tactile_diagnostic_2x10_raw": (2, 10),
                            "tactile_diagnostic_2x10_raw_delta": (2, 10),
                        }
                        if active_shape:
                            required.update(
                                {
                                    "tactile_active_kpa": active_shape,
                                    "tactile_active_dynamic_kpa": active_shape,
                                    "tactile_active_raw": active_shape,
                                    "tactile_active_raw_delta": active_shape,
                                }
                            )
                        if full_hand_available:
                            required.update(
                                {
                                    "tactile_full_56_kpa": (56,),
                                    "tactile_full_56_dynamic_kpa": (56,),
                                    "tactile_full_56_raw": (56,),
                                    "tactile_full_56_raw_delta": (56,),
                                }
                            )
                        for key, shape in required.items():
                            if key not in tactile or tactile[key].shape != shape:
                                raise ValueError(
                                    f"{key} must have shape {shape}"
                                )
            except (OSError, KeyError, ValueError) as error:
                errors.append(f"{where}: invalid media: {error}")
    return EpisodeValidation(not errors, tuple(errors), len(records))
