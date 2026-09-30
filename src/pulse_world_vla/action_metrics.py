from __future__ import annotations

import numpy as np

from pulse_mujoco.task import TaskPhase


def _safe_mean(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def action_metric_groups(
    prediction,
    target,
    previous,
    event_tags,
    action_scale,
    *,
    pressure_on_threshold_kpa: float = 30.0,
    task_phase=None,
) -> dict:
    """Compute comparable physical and normalized metrics by event family."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    scale = np.asarray(action_scale, dtype=np.float64).reshape(1, -1)
    if (
        prediction.shape != target.shape
        or previous.shape != target.shape
        or target.ndim != 2
    ):
        raise ValueError("action arrays must share shape [samples, action_dim]")
    if scale.shape[1] != target.shape[1]:
        raise ValueError("action scale does not match action dimension")
    tags = {
        name: np.asarray(value, dtype=bool)
        for name, value in event_tags.items()
    }
    count = len(target)
    if any(value.shape != (count,) for value in tags.values()):
        raise ValueError("event tag arrays must have shape [samples]")

    physical_prediction = prediction * scale
    physical_target = target * scale
    normalized_error = prediction - target
    physical_error = physical_prediction - physical_target
    active_6d = tags["active_translation"] | tags["active_rotation"]
    masks = {
        "overall": np.ones(count, dtype=bool),
        "active_6d": active_6d,
        "translation": tags["active_translation"],
        "rotation": tags["active_rotation"],
        "contact_transition": tags["contact_transition"],
        "pressure_transition": tags["pressure_transition"],
        "search": tags["search_move"],
        "recenter": tags["recenter"],
        "recovery": tags["recovery"],
        "steady_hold": tags["steady_hold"],
    }
    if task_phase is not None:
        phase_values = np.asarray(task_phase)
        if phase_values.shape != (count,):
            raise ValueError("task_phase must have shape [samples]")
        if not np.issubdtype(phase_values.dtype, np.integer):
            if not np.all(np.isfinite(phase_values)) or not np.all(
                phase_values == np.floor(phase_values)
            ):
                raise ValueError("task_phase must contain integer phase ids")
        phase_values = phase_values.astype(np.int64)
        valid_ids = {int(phase) for phase in TaskPhase}
        if any(int(value) not in valid_ids for value in phase_values):
            raise ValueError("task_phase contains an unknown phase id")
        masks.update(
            {
                f"phase_{phase.name.lower()}": phase_values == int(phase)
                for phase in TaskPhase
            }
        )
    result = {}
    for name, mask in masks.items():
        selected = np.flatnonzero(mask)
        if not len(selected):
            result[name] = {"samples": 0}
            continue
        norm_error = normalized_error[selected]
        phys_error = physical_error[selected]
        pred6 = physical_prediction[selected, :6]
        target6 = physical_target[selected, :6]
        dot = np.sum(pred6 * target6, axis=1)
        pred_norm = np.linalg.norm(pred6, axis=1)
        target_norm = np.linalg.norm(target6, axis=1)
        directional = target_norm > 1e-9
        cosine = np.divide(
            dot,
            np.maximum(pred_norm * target_norm, 1e-12),
        )
        pressure_target = physical_target[selected, 6]
        pressure_prediction = physical_prediction[selected, 6]
        pressure_target_on = (
            pressure_target > pressure_on_threshold_kpa
        )
        pressure_prediction_on = (
            pressure_prediction > pressure_on_threshold_kpa
        )
        channel_direction_accuracy = []
        channel_direction_samples = []
        for channel in range(target.shape[1]):
            channel_active = np.abs(
                physical_target[selected, channel]
            ) > 1e-9
            channel_direction_samples.append(
                int(np.count_nonzero(channel_active))
            )
            channel_direction_accuracy.append(
                _safe_mean(
                    (
                        np.sign(physical_prediction[selected, channel][channel_active])
                        == np.sign(physical_target[selected, channel][channel_active])
                    ).astype(np.float64)
                )
            )
        result[name] = {
            "samples": int(len(selected)),
            "mae_normalized": float(np.mean(np.abs(norm_error))),
            "mse_normalized": float(np.mean(norm_error**2)),
            "active_6d_mae_normalized": float(
                np.mean(np.abs(norm_error[:, :6]))
            ),
            "translation_mae_mm": float(
                np.mean(np.abs(phys_error[:, :3])) * 1000.0
            ),
            "translation_endpoint_error_mm": float(
                np.mean(np.linalg.norm(phys_error[:, :3], axis=1))
                * 1000.0
            ),
            "rotation_mae_rad": float(
                np.mean(np.abs(phys_error[:, 3:6]))
            ),
            "rotation_endpoint_error_rad": float(
                np.mean(np.linalg.norm(phys_error[:, 3:6], axis=1))
            ),
            "pressure_mae_kpa": float(
                np.mean(np.abs(phys_error[:, 6]))
            ),
            "pressure_on_accuracy": float(
                np.mean(pressure_prediction_on == pressure_target_on)
            ),
            "active_6d_cosine": _safe_mean(cosine[directional]),
            "action_direction_accuracy": _safe_mean(
                (dot[directional] > 0.0).astype(np.float64)
            ),
            "channel_mae_normalized": np.mean(
                np.abs(norm_error), axis=0
            ).tolist(),
            "channel_mae_physical": np.mean(
                np.abs(phys_error), axis=0
            ).tolist(),
            "channel_direction_samples": channel_direction_samples,
            "channel_direction_accuracy": channel_direction_accuracy,
        }
    return result
