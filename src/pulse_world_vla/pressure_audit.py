from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pulse_mujoco.m5_transition_overlay import apply_transition_overlay
from pulse_mujoco.task import TaskPhase


PRESSURE_AUDIT_SCHEMA_VERSION = "adam_pulse_pressure_dataset_audit_v1"
ACTIVE_FINGERS = ("index", "middle", "ring")


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
        "standard_deviation": float(array.std()),
        "unique_rounded_1e6": int(np.unique(np.round(array, 6)).size),
    }


def _sign_flip_count(values: Iterable[float], *, deadband: float = 1e-6) -> int:
    delta = np.asarray(list(values), dtype=np.float64)
    signs = np.sign(delta[np.abs(delta) > float(deadband)])
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def _lagged_correlations(
    command: np.ndarray, response: np.ndarray, *, maximum_lag: int = 8
) -> list[dict[str, Any]]:
    command = np.asarray(command, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    result = []
    for lag in range(maximum_lag + 1):
        x = command[: command.size - lag or None]
        y = response[lag:]
        if x.size < 3 or np.std(x) <= 1e-9 or np.std(y) <= 1e-9:
            correlation = None
        else:
            correlation = float(np.corrcoef(x, y)[0, 1])
        result.append({"lag_frames": lag, "correlation": correlation})
    return result


def _target_action(record: dict[str, Any]) -> np.ndarray:
    value = np.asarray(
        record.get("expert_label_action", record["executed_action"]),
        dtype=np.float64,
    )
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("pressure audit requires finite 7-D target actions")
    return value


def _action(record: dict[str, Any], name: str) -> np.ndarray:
    value = np.asarray(record[name], dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError(f"pressure audit requires finite 7-D {name}")
    return value


def _active_force(record: dict[str, Any]) -> np.ndarray:
    value = record.get("active_normal_force_n", record.get("normal_force_n"))
    force = np.asarray(value, dtype=np.float64)
    if force.shape != (3,) or not np.isfinite(force).all():
        raise ValueError("pressure audit requires three active normal forces")
    return force


def _active_tactile(record: dict[str, Any], episode_dir: Path) -> np.ndarray:
    path = Path(record["tactile_path"])
    if not path.is_absolute():
        path = episode_dir / path
    with np.load(path) as payload:
        if "tactile_active_kpa" in payload:
            value = np.asarray(payload["tactile_active_kpa"], dtype=np.float64)
        elif "tactile_full_56_kpa" in payload:
            full = np.asarray(payload["tactile_full_56_kpa"], dtype=np.float64)
            value = np.stack((full[16:26], full[26:36], full[36:46]))
        else:
            raise ValueError(f"active tactile pressure is absent: {path}")
    if value.shape != (3, 10) or not np.isfinite(value).all():
        raise ValueError(f"invalid active tactile pressure: {path}")
    return value


def _contiguous_segments(records: list[dict[str, Any]], phase: int) -> list[list[int]]:
    selected = [index for index, record in enumerate(records) if int(record["task_phase"]) == phase]
    segments: list[list[int]] = []
    for index in selected:
        if not segments or index != segments[-1][-1] + 1:
            segments.append([index])
        else:
            segments[-1].append(index)
    return segments


def _window_counts(
    records: list[dict[str, Any]], *, observation_horizon: int, action_horizon: int
) -> dict[str, int]:
    total = observation_horizon + action_horizon
    counts = {phase.name.lower(): 0 for phase in TaskPhase}
    counts["any_pressure_related"] = 0
    pressure_related = {
        int(TaskPhase.CONTACT_ESTABLISHMENT),
        int(TaskPhase.PRESSURE_REGULATION),
        int(TaskPhase.STABLE_HOLD),
        int(TaskPhase.SIGNAL_CAPTURE),
    }
    for start in range(max(0, len(records) - total + 1)):
        future = {
            int(item["task_phase"])
            for item in records[start + observation_horizon : start + total]
        }
        for phase in TaskPhase:
            counts[phase.name.lower()] += int(int(phase) in future)
        counts["any_pressure_related"] += int(bool(future & pressure_related))
    return {name: value for name, value in counts.items() if value}


def _episode_metrics(
    records: list[dict[str, Any]],
    episode_dir: Path,
    *,
    target_force_n: float,
    minimum_force_n: float,
    maximum_force_n: float,
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    frames: list[dict[str, Any]] = []
    phase_durations: dict[str, list[float]] = {}
    for phase in (
        TaskPhase.CONTACT_ESTABLISHMENT,
        TaskPhase.PRESSURE_REGULATION,
        TaskPhase.STABLE_HOLD,
        TaskPhase.SIGNAL_CAPTURE,
    ):
        for segment in _contiguous_segments(records, int(phase)):
            first, last = segment[0], segment[-1]
            duration = max(
                0.0,
                float(records[last]["timestamp_s"])
                - float(records[first]["timestamp_s"]),
            )
            phase_durations.setdefault(phase.name.lower(), []).append(duration)
            for local_index, record_index in enumerate(segment):
                record = records[record_index]
                force = _active_force(record)
                tactile = _active_tactile(record, episode_dir)
                target = _target_action(record)
                executed = _action(record, "executed_action")
                previous = _action(record, "previous_action")
                prior = frames[-1] if local_index and frames else None
                frames.append(
                    {
                        "phase": int(phase),
                        "timestamp_s": float(record["timestamp_s"]),
                        "target_command_kpa": float(target[6]),
                        "executed_command_kpa": float(executed[6]),
                        "previous_command_kpa": float(previous[6]),
                        "force": force,
                        "force_mean_n": float(force.mean()),
                        "force_minimum_n": float(force.min()),
                        "force_maximum_n": float(force.max()),
                        "tactile_mean_kpa": float(tactile.mean()),
                        "tactile_peak_kpa": float(tactile.max()),
                        "tactile_finger_mean_kpa": tactile.mean(axis=1),
                        "delta_command_kpa": (
                            0.0
                            if prior is None
                            else float(executed[6] - prior["executed_command_kpa"])
                        ),
                        "delta_force_mean_n": (
                            0.0 if prior is None else float(force.mean() - prior["force_mean_n"])
                        ),
                        "delta_tactile_mean_kpa": (
                            0.0
                            if prior is None
                            else float(tactile.mean() - prior["tactile_mean_kpa"])
                        ),
                        "undershoot": bool(force.mean() < 0.9 * target_force_n),
                        "overshoot": bool(force.mean() > 1.1 * target_force_n),
                        "in_task_force_window": bool(
                            np.all((force >= minimum_force_n) & (force <= maximum_force_n))
                        ),
                        "steady_hold": bool(
                            abs(executed[6] - previous[6]) <= 0.1
                            and np.all((force >= minimum_force_n) & (force <= maximum_force_n))
                        ),
                    }
                )
    return phase_durations, frames


def _aggregate_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        return {"frame_count": 0}
    command = np.asarray([item["executed_command_kpa"] for item in frames])
    force = np.asarray([item["force_mean_n"] for item in frames])
    delta_command = np.asarray([item["delta_command_kpa"] for item in frames])
    return {
        "frame_count": len(frames),
        "target_pressure_command_kpa": _distribution(
            item["target_command_kpa"] for item in frames
        ),
        "executed_pressure_command_kpa": _distribution(command),
        "previous_pressure_command_kpa": _distribution(
            item["previous_command_kpa"] for item in frames
        ),
        "active_tactile_mean_kpa": _distribution(
            item["tactile_mean_kpa"] for item in frames
        ),
        "active_tactile_peak_kpa": _distribution(
            item["tactile_peak_kpa"] for item in frames
        ),
        "active_tactile_per_finger_mean_kpa": {
            name: _distribution(item["tactile_finger_mean_kpa"][index] for item in frames)
            for index, name in enumerate(ACTIVE_FINGERS)
        },
        "active_normal_force_mean_n": _distribution(force),
        "active_normal_force_minimum_n": _distribution(
            item["force_minimum_n"] for item in frames
        ),
        "active_normal_force_maximum_n": _distribution(
            item["force_maximum_n"] for item in frames
        ),
        "delta_pressure_command_kpa": _distribution(delta_command),
        "delta_tactile_mean_kpa": _distribution(
            item["delta_tactile_mean_kpa"] for item in frames
        ),
        "delta_force_mean_n": _distribution(
            item["delta_force_mean_n"] for item in frames
        ),
        "command_delta_rms_kpa": float(np.sqrt(np.mean(delta_command**2))),
        "force_delta_rms_n": float(
            np.sqrt(np.mean(np.asarray([item["delta_force_mean_n"] for item in frames]) ** 2))
        ),
        "command_action_sign_flip_count": _sign_flip_count(delta_command),
        "coverage": {
            "undershoot_frames": sum(item["undershoot"] for item in frames),
            "overshoot_frames": sum(item["overshoot"] for item in frames),
            "in_task_force_window_frames": sum(
                item["in_task_force_window"] for item in frames
            ),
            "steady_hold_frames": sum(item["steady_hold"] for item in frames),
            "pressure_transition_frames": int(np.count_nonzero(np.abs(delta_command) > 0.1)),
        },
        "command_to_force_lag_correlation": _lagged_correlations(command, force),
    }


def audit_pressure_dataset(
    dataset_root: Path | str,
    dataset_index: Path | str,
    transition_overlay: Path | str,
    *,
    observation_horizon: int = 4,
    action_horizon: int = 4,
    target_force_n: float = 4.0,
    minimum_force_n: float = 1.25,
    maximum_force_n: float = 10.0,
    failed_closed_loop_episode: Path | str | None = None,
    successful_closed_loop_episode: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root)
    index = json.loads(Path(dataset_index).read_text(encoding="utf-8"))
    overlay = json.loads(Path(transition_overlay).read_text(encoding="utf-8"))
    accepted = {item["episode"] for item in index["accepted_episodes"]}
    split_reports: dict[str, Any] = {}
    all_dataset_frames: list[dict[str, Any]] = []
    all_durations: dict[str, list[float]] = {}
    for split, raw_names in index["splits"].items():
        names = [name for name in raw_names if name in accepted]
        frames: list[dict[str, Any]] = []
        windows: dict[str, int] = {}
        durations: dict[str, list[float]] = {}
        for name in names:
            episode_dir = root / name
            records = [
                json.loads(line)
                for line in (episode_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if name in overlay["episodes"]:
                records = apply_transition_overlay(records, overlay["episodes"][name])
            for key, value in _window_counts(
                records,
                observation_horizon=observation_horizon,
                action_horizon=action_horizon,
            ).items():
                windows[key] = windows.get(key, 0) + value
            episode_durations, episode_frames = _episode_metrics(
                records,
                episode_dir,
                target_force_n=target_force_n,
                minimum_force_n=minimum_force_n,
                maximum_force_n=maximum_force_n,
            )
            frames.extend(episode_frames)
            for key, value in episode_durations.items():
                durations.setdefault(key, []).extend(value)
        all_dataset_frames.extend(frames)
        for key, value in durations.items():
            all_durations.setdefault(key, []).extend(value)
        split_reports[split] = {
            "episode_count": len(names),
            "window_counts_h4": windows,
            "phase_duration_s": {key: _distribution(value) for key, value in durations.items()},
            "pressure_frames": _aggregate_frames(frames),
        }

    closed_loop: dict[str, Any] = {}
    for name, raw_path in (
        ("failed_unified", failed_closed_loop_episode),
        ("successful_specialist", successful_closed_loop_episode),
    ):
        if raw_path is None:
            continue
        episode_dir = Path(raw_path)
        records = [
            json.loads(line)
            for line in (episode_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        durations, frames = _episode_metrics(
            records,
            episode_dir,
            target_force_n=target_force_n,
            minimum_force_n=minimum_force_n,
            maximum_force_n=maximum_force_n,
        )
        metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
        closed_loop[name] = {
            "path": str(episode_dir.resolve()),
            "success": bool(metadata.get("success", False)),
            "failure": metadata.get("failure"),
            "simulation_time_s": metadata.get("simulation_time_s"),
            "phase_duration_s": {
                phase: _distribution(values) for phase, values in durations.items()
            },
            "pressure_frames": _aggregate_frames(frames),
        }

    aggregate = _aggregate_frames(all_dataset_frames)
    target_unique = aggregate.get("target_pressure_command_kpa", {}).get(
        "unique_rounded_1e6", 0
    )
    coverage = aggregate.get("coverage", {})
    recommendations = []
    if target_unique <= 1:
        recommendations.append(
            "The pressure target is constant; absolute-action imitation cannot teach feedback correction by itself."
        )
    if not coverage.get("pressure_transition_frames", 0):
        recommendations.append(
            "No command transitions are supervised; collect or synthesize bounded residual-control transitions before training a dynamic head."
        )
    if not coverage.get("overshoot_frames", 0):
        recommendations.append(
            "Overshoot recovery is absent and must be added to the pressure mini-gate or targeted dataset."
        )
    return {
        "schema_version": PRESSURE_AUDIT_SCHEMA_VERSION,
        "contract": {
            "observation_horizon": observation_horizon,
            "action_horizon": action_horizon,
            "active_fingers": list(ACTIVE_FINGERS),
            "target_force_n_per_finger": target_force_n,
            "task_force_window_n_per_finger": [minimum_force_n, maximum_force_n],
            "pressure_target_is_public_task_state": True,
            "privileged_pulse_truth_used": False,
        },
        "dataset": {
            "root": str(root.resolve()),
            "index": str(Path(dataset_index).resolve()),
            "transition_overlay": str(Path(transition_overlay).resolve()),
            "splits": split_reports,
            "aggregate_pressure_frames": aggregate,
            "phase_duration_s": {
                key: _distribution(value) for key, value in all_durations.items()
            },
        },
        "closed_loop_comparison": closed_loop,
        "audit_decision": {
            "dynamic_residual_head_ready_from_existing_labels": bool(
                coverage.get("pressure_transition_frames", 0)
                and coverage.get("undershoot_frames", 0)
                and coverage.get("overshoot_frames", 0)
            ),
            "recommendations": recommendations,
        },
    }
