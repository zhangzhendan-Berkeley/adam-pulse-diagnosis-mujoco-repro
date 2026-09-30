from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.datasets.media import (
    DEPTH_FLOAT32_NPY,
    DEPTH_UINT16_PNG,
    save_depth,
)
from src.envs.pulse_diagnosis_env import parse_task_phase


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _quaternion_delta_rotation_vector(previous_xyzw, current_xyzw):
    previous = np.asarray(previous_xyzw, dtype=float)
    current = np.asarray(current_xyzw, dtype=float)
    previous /= max(np.linalg.norm(previous), 1e-12)
    current /= max(np.linalg.norm(current), 1e-12)
    px, py, pz, pw = previous
    cx, cy, cz, cw = current
    # current * inverse(previous), both in xyzw convention.
    x = -cw * px + cx * pw - cy * pz + cz * py
    y = -cw * py + cx * pz + cy * pw - cz * px
    z = -cw * pz - cx * py + cy * px + cz * pw
    w = cw * pw + cx * px + cy * py + cz * pz
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector = np.asarray([x, y, z], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        return (2.0 * vector).astype(np.float32)
    angle = 2.0 * np.arctan2(norm, np.clip(w, -1.0, 1.0))
    return (angle * vector / norm).astype(np.float32)


class SynchronizedEpisodeRecorder:
    """Record camera, proximity and robot signals on one simulation clock."""

    def __init__(
        self,
        root,
        episode_id,
        instruction,
        capture_interval=4,
        *,
        depth_storage=DEPTH_FLOAT32_NPY,
        depth_scale_m_per_unit=1e-4,
        rgb_storage="png",
        rgb_jpeg_quality=95,
    ):
        self.root = Path(root)
        self.episode_dir = self.root / f"episode_{episode_id}"
        self.frame_dir = self.episode_dir / "frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.instruction = str(instruction)
        self.capture_interval = max(1, int(capture_interval))
        if depth_storage not in (DEPTH_FLOAT32_NPY, DEPTH_UINT16_PNG):
            raise ValueError(f"unsupported depth storage: {depth_storage}")
        if rgb_storage not in ("png", "jpeg"):
            raise ValueError(f"unsupported RGB storage: {rgb_storage}")
        if not 1 <= int(rgb_jpeg_quality) <= 100:
            raise ValueError("RGB JPEG quality must be in [1, 100]")
        self.depth_storage = str(depth_storage)
        self.depth_scale_m_per_unit = float(depth_scale_m_per_unit)
        self.rgb_storage = str(rgb_storage)
        self.rgb_jpeg_quality = int(rgb_jpeg_quality)
        self.step = 0
        self.frames = 0
        self.previous_tcp = None
        self.previous_tcp_pose = None
        self.previous_timestamp_s = None
        self.previous_executed_action = np.zeros(7, dtype=np.float32)
        self.pending_pressure_target_kpa = 0.0
        self.manifest = self.episode_dir / "transitions.jsonl"
        self.events_manifest = self.episode_dir / "events.jsonl"
        self.recorded_event_ids = set()
        self.recorded_event_markers = set()
        self.active_recovery_event = None
        self.recovery_event_count = 0
        self.expert_intervention_frames = 0
        self.pressure_label_frames = 0

    def _write_event(self, event_record):
        with self.events_manifest.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    _jsonable(event_record), ensure_ascii=False
                )
                + "\n"
            )

    def capture(
        self,
        rgb,
        depth,
        proximity_distances_m,
        robot_state,
        tcp_position_m,
        tactile_image=None,
        tactile_valid=False,
        phase="unknown",
        timestamp_s=None,
        joint_position=None,
        joint_velocity=None,
        tcp_pose=None,
        tcp_velocity=None,
        tactile_raw=None,
        tactile_raw_delta=None,
        contact=None,
        expert_proposed_action=None,
        safe_action=None,
        executed_action=None,
        target_pressure_kpa=None,
        next_target_pressure_kpa=None,
        action_clipping=None,
        perturbation_state=None,
        environment_perturbation=None,
        expert_intervened=False,
        recovery_mode=None,
        risk_score=0.0,
    ):
        self.step += 1
        if self.step % self.capture_interval:
            return False
        rgb = np.asarray(rgb)
        depth = np.asarray(depth, dtype=np.float32)
        if rgb.size == 0 or depth.size == 0:
            return False
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if np.max(rgb) <= 1.5:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        tcp = np.asarray(tcp_position_m, dtype=np.float32)
        action = np.zeros(3, dtype=np.float32) if self.previous_tcp is None else tcp - self.previous_tcp
        if tcp_velocity is None:
            dt = (
                None
                if timestamp_s is None
                or self.previous_timestamp_s is None
                else float(timestamp_s) - self.previous_timestamp_s
            )
            tcp_linear_velocity = (
                np.zeros(3, dtype=np.float32)
                if self.previous_tcp is None or dt is None or dt <= 0.0
                else action / dt
            )
            tcp_velocity = np.concatenate(
                (tcp_linear_velocity, np.zeros(3, dtype=np.float32))
            )
        tcp_pose = (
            np.concatenate((tcp, np.asarray([0.0, 0.0, 0.0, 1.0])))
            if tcp_pose is None
            else np.asarray(tcp_pose, dtype=np.float32)
        )
        rotation_delta = (
            np.zeros(3, dtype=np.float32)
            if self.previous_tcp_pose is None
            else _quaternion_delta_rotation_vector(
                self.previous_tcp_pose[3:], tcp_pose[3:]
            )
        )
        pressure_label_provided = (
            target_pressure_kpa is not None
            or next_target_pressure_kpa is not None
        )
        pressure_target = (
            self.pending_pressure_target_kpa
            if next_target_pressure_kpa is not None
            else 0.0
            if target_pressure_kpa is None
            else float(target_pressure_kpa)
        )
        next_pressure_target = (
            None
            if next_target_pressure_kpa is None
            else float(next_target_pressure_kpa)
        )
        if not np.isfinite(pressure_target) or pressure_target < 0.0:
            raise ValueError("target_pressure_kpa must be finite and non-negative")
        if next_pressure_target is not None and (
            not np.isfinite(next_pressure_target)
            or next_pressure_target < 0.0
        ):
            raise ValueError(
                "next_target_pressure_kpa must be finite and non-negative"
            )
        inferred_executed = np.concatenate(
            (action, rotation_delta, np.asarray([pressure_target], dtype=np.float32))
        )
        executed = np.asarray(
            inferred_executed
            if executed_action is None
            else executed_action,
            dtype=np.float32,
        )
        if target_pressure_kpa is not None and not np.isclose(
            float(executed[6]), pressure_target, atol=1e-6
        ):
            raise ValueError(
                "executed_action pressure must match target_pressure_kpa"
            )
        proposed = np.asarray(
            executed if expert_proposed_action is None
            else expert_proposed_action,
            dtype=np.float32,
        )
        safe = np.asarray(
            executed if safe_action is None else safe_action,
            dtype=np.float32,
        )
        for name, value in (
            ("tcp_pose", tcp_pose),
            ("tcp_velocity", tcp_velocity),
            ("expert_proposed_action", proposed),
            ("safe_action", safe),
            ("executed_action", executed),
        ):
            expected = 7 if "action" in name else (7 if name == "tcp_pose" else 6)
            if np.asarray(value).shape != (expected,):
                raise ValueError(
                    f"{name} must have shape ({expected},), got "
                    f"{np.asarray(value).shape}"
                )
        self.previous_tcp = tcp.copy()
        self.previous_tcp_pose = np.asarray(tcp_pose, dtype=np.float32).copy()
        self.previous_timestamp_s = (
            None if timestamp_s is None else float(timestamp_s)
        )
        stem = f"{self.frames:06d}"
        rgb_suffix = ".png" if self.rgb_storage == "png" else ".jpg"
        depth_suffix = (
            ".npy" if self.depth_storage == DEPTH_FLOAT32_NPY else ".png"
        )
        rgb_path = self.frame_dir / f"rgb_{stem}{rgb_suffix}"
        depth_path = self.frame_dir / f"depth_{stem}{depth_suffix}"
        import cv2

        rgb_parameters = (
            []
            if self.rgb_storage == "png"
            else [cv2.IMWRITE_JPEG_QUALITY, self.rgb_jpeg_quality]
        )
        if not cv2.imwrite(
            str(rgb_path),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            rgb_parameters,
        ):
            raise OSError(f"failed to write RGB image: {rgb_path}")
        depth_encoding = save_depth(
            depth_path,
            depth,
            encoding=self.depth_storage,
            scale_m_per_unit=self.depth_scale_m_per_unit,
        )
        tactile_path = None
        if tactile_image is not None:
            tactile_path = self.frame_dir / f"tactile_{stem}.npy"
            np.save(tactile_path, np.asarray(tactile_image, dtype=np.float32))
        proximity = [None if value is None else float(value) for value in proximity_distances_m]
        canonical_phase = parse_task_phase(phase)
        record = {
            "frame": self.frames,
            "sim_step": self.step,
            "timestamp_s": (
                None if timestamp_s is None else float(timestamp_s)
            ),
            "phase": str(phase),
            "task_phase": int(canonical_phase),
            "task_phase_name": canonical_phase.name.lower(),
            "rgb_path": str(rgb_path.relative_to(self.episode_dir)),
            "rgb_encoding": {
                "format": self.rgb_storage,
                "jpeg_quality": (
                    self.rgb_jpeg_quality
                    if self.rgb_storage == "jpeg"
                    else None
                ),
            },
            "depth_path": str(depth_path.relative_to(self.episode_dir)),
            "depth_encoding": depth_encoding,
            "tactile_path": None if tactile_path is None else str(tactile_path.relative_to(self.episode_dir)),
            "tactile_valid": bool(tactile_valid),
            "proximity_distances_m": proximity,
            "proximity_valid": [value is not None for value in proximity_distances_m],
            "robot_state": np.asarray(robot_state, dtype=np.float32).tolist(),
            "action_delta_tcp_m": action.tolist(),
            "tcp_position_m": tcp.tolist(),
            "joint_position": (
                None
                if joint_position is None
                else np.asarray(joint_position, dtype=np.float32).tolist()
            ),
            "joint_velocity": (
                None
                if joint_velocity is None
                else np.asarray(joint_velocity, dtype=np.float32).tolist()
            ),
            "tcp_pose": np.asarray(tcp_pose, dtype=np.float32).tolist(),
            "tcp_velocity": np.asarray(
                tcp_velocity, dtype=np.float32
            ).tolist(),
            "tactile_raw": (
                None
                if tactile_raw is None
                else np.asarray(tactile_raw, dtype=np.uint16).tolist()
            ),
            "tactile_raw_delta": (
                None
                if tactile_raw_delta is None
                else np.asarray(tactile_raw_delta, dtype=np.float32).tolist()
            ),
            "contact": (
                [bool(tactile_valid), bool(tactile_valid)]
                if contact is None
                else np.asarray(contact, dtype=bool).tolist()
            ),
            "previous_action": self.previous_executed_action.tolist(),
            "expert_proposed_action": proposed.tolist(),
            "safe_action": safe.tolist(),
            "executed_action": executed.tolist(),
            "target_pressure_kpa": pressure_target,
            "next_target_pressure_kpa": next_pressure_target,
            "pressure_label_available": pressure_label_provided,
            "pressure_label_semantics": (
                "unavailable"
                if not pressure_label_provided
                else "latched_expert_nominal_pad_pressure_intent"
                if next_target_pressure_kpa is not None
                else "expert_nominal_pad_pressure_intent"
            ),
            "action_source": (
                (
                    "inferred_from_capture_delta"
                    if not pressure_label_provided
                    else "inferred_tcp_delta_with_latched_pressure_intent"
                    if next_target_pressure_kpa is not None
                    else "inferred_tcp_delta_with_expert_pressure_intent"
                )
                if executed_action is None
                else "environment_executed"
            ),
            "action_clipping": action_clipping or {},
            "perturbation_state": perturbation_state or {},
            "environment_perturbation": _jsonable(
                environment_perturbation
            ),
            "expert_intervened": bool(expert_intervened),
            "recovery_mode": recovery_mode,
            "risk_score": float(risk_score),
        }
        perturbation = perturbation_state or {}
        physical = environment_perturbation or {}
        physical_events = list(physical.get("events", []))
        for event in list(perturbation.get("events", [])) + physical_events:
            event_id = str(event.get("event_id", ""))
            if not event_id:
                continue
            is_physical = event in physical_events
            stages = ["observed"]
            if is_physical:
                progress = float(
                    event.get("interpolation_progress", 0.0)
                )
                stages = []
                if event_id not in self.recorded_event_ids:
                    stages.append("started")
                if 0.0 < progress < 1.0:
                    stages.append("interpolating")
                if bool(event.get("completed", False)):
                    stages.append("completed")
            for lifecycle_stage in stages:
                marker = (event_id, lifecycle_stage)
                if marker in self.recorded_event_markers:
                    continue
                event_record = dict(event)
                event_record["lifecycle_stage"] = lifecycle_stage
                if is_physical:
                    event_record["physical_environment_event"] = True
                    event_record["applied_state"] = {
                        key: _jsonable(value)
                        for key, value in physical.items()
                        if key != "events"
                    }
                event_record["recorded_frame"] = self.frames
                event_record["recorded_sim_step"] = self.step
                self._write_event(event_record)
                self.recorded_event_markers.add(marker)
            self.recorded_event_ids.add(event_id)

        current_recovery_mode = (
            None
            if not expert_intervened or recovery_mode is None
            else str(recovery_mode)
        )
        if (
            self.active_recovery_event is not None
            and current_recovery_mode
            != self.active_recovery_event["mode"]
        ):
            self._write_event(
                {
                    **self.active_recovery_event,
                    "lifecycle_stage": "completed",
                    "recorded_frame": self.frames,
                    "recorded_sim_step": self.step,
                }
            )
            self.active_recovery_event = None
        if (
            current_recovery_mode is not None
            and self.active_recovery_event is None
        ):
            self.recovery_event_count += 1
            self.active_recovery_event = {
                "event_id": (
                    f"recovery_{self.recovery_event_count:04d}_"
                    f"{current_recovery_mode}"
                ),
                "kind": "expert_recovery",
                "mode": current_recovery_mode,
                "recoverable": True,
            }
            self._write_event(
                {
                    **self.active_recovery_event,
                    "lifecycle_stage": "started",
                    "recorded_frame": self.frames,
                    "recorded_sim_step": self.step,
                }
            )
        with self.manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.frames += 1
        self.expert_intervention_frames += int(bool(expert_intervened))
        self.pressure_label_frames += int(pressure_label_provided)
        if next_pressure_target is not None:
            self.pending_pressure_target_kpa = next_pressure_target
        self.previous_executed_action = executed.copy()
        return True

    def close(self, success, result_path=None, extra=None):
        if self.active_recovery_event is not None:
            self._write_event(
                {
                    **self.active_recovery_event,
                    "lifecycle_stage": "completed",
                    "recorded_frame": self.frames,
                    "recorded_sim_step": self.step,
                    "completion_reason": "episode_closed",
                }
            )
            self.active_recovery_event = None
        metadata = {
            "schema_version": 2,
            "instruction": self.instruction,
            "frames": self.frames,
            "success": bool(success),
            "result_path": None if result_path is None else str(result_path),
            "tactile_available": False,
            "perturbation_event_count": len(self.recorded_event_ids),
            "recovery_event_count": self.recovery_event_count,
            "expert_intervention_frames": self.expert_intervention_frames,
            "pressure_label_available": bool(
                self.frames > 0 and self.pressure_label_frames == self.frames
            ),
            "pressure_label_frames": self.pressure_label_frames,
            "media_storage": {
                "rgb": {
                    "format": self.rgb_storage,
                    "jpeg_quality": (
                        self.rgb_jpeg_quality
                        if self.rgb_storage == "jpeg"
                        else None
                    ),
                },
                "depth": {
                    "format": self.depth_storage,
                    "scale_m_per_unit": (
                        self.depth_scale_m_per_unit
                        if self.depth_storage == DEPTH_UINT16_PNG
                        else None
                    ),
                },
            },
        }
        if extra:
            metadata.update(extra)
        (self.episode_dir / "episode.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
