from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_DIRECTION_MINIMUM_M = 0.5e-3


def _float_vector(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [samples]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _binary_target(name: str, values: Any) -> np.ndarray:
    array = _float_vector(name, values)
    if np.any((array != 0.0) & (array != 1.0)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return array.astype(bool)


def _recall(correct: np.ndarray, selected: np.ndarray) -> float | None:
    return None if not np.any(selected) else float(np.mean(correct[selected]))


def _balanced_accuracy(*recalls: float | None) -> float | None:
    if any(value is None for value in recalls):
        return None
    return float(np.mean(np.asarray(recalls, dtype=np.float64)))


def search_segment_metrics(
    *,
    stop_target: Any,
    stop_prediction: Any,
    signed_target_m: Any,
    raw_signed_prediction_m: Any,
    maximum_displacement_m: float,
    stop_threshold: float = 0.5,
    direction_minimum_m: float = DEFAULT_DIRECTION_MINIMUM_M,
    radius_tolerance_m: float = 1e-9,
) -> dict[str, Any]:
    """Compute the common next-search-segment metric contract.

    ``raw_signed_prediction_m`` must be the value before any safety clipping.
    Displacement errors are measured on ground-truth continue examples.  The
    direction metrics additionally exclude near-zero targets, whose sign is
    not a meaningful supervision signal.  Radius violations are deliberately
    measured over every raw model output, including outputs paired with a
    predicted or target stop, so an invalid bounded head cannot be hidden by a
    downstream stop gate.
    """

    stop_true = _binary_target("stop_target", stop_target)
    stop_score = _float_vector("stop_prediction", stop_prediction)
    signed_true = _float_vector("signed_target_m", signed_target_m)
    signed_raw = _float_vector(
        "raw_signed_prediction_m", raw_signed_prediction_m
    )
    sample_count = len(stop_true)
    for name, value in (
        ("stop_prediction", stop_score),
        ("signed_target_m", signed_true),
        ("raw_signed_prediction_m", signed_raw),
    ):
        if len(value) != sample_count:
            raise ValueError(
                f"{name} has {len(value)} samples; expected {sample_count}"
            )

    maximum_displacement_m = float(maximum_displacement_m)
    stop_threshold = float(stop_threshold)
    direction_minimum_m = float(direction_minimum_m)
    radius_tolerance_m = float(radius_tolerance_m)
    if not np.isfinite(maximum_displacement_m) or maximum_displacement_m <= 0.0:
        raise ValueError("maximum_displacement_m must be finite and positive")
    if not np.isfinite(stop_threshold) or not 0.0 <= stop_threshold <= 1.0:
        raise ValueError("stop_threshold must be within [0, 1]")
    if np.any((stop_score < 0.0) | (stop_score > 1.0)):
        raise ValueError("stop_prediction must contain probabilities in [0, 1]")
    if not np.isfinite(direction_minimum_m) or direction_minimum_m < 0.0:
        raise ValueError("direction_minimum_m must be finite and non-negative")
    if not np.isfinite(radius_tolerance_m) or radius_tolerance_m < 0.0:
        raise ValueError("radius_tolerance_m must be finite and non-negative")

    stop_pred = stop_score >= stop_threshold
    continue_true = ~stop_true
    continue_pred = ~stop_pred
    stop_correct = stop_pred == stop_true
    continue_recall = _recall(continue_pred, continue_true)
    stop_recall = _recall(stop_pred, stop_true)

    # A mathematically exact zero has no direction even when callers choose a
    # zero threshold.  The default 0.5 mm threshold also keeps sensor-scale
    # jitter out of the primary direction score.
    nonzero_target = np.abs(signed_true) > 0.0
    directional = (
        continue_true
        & nonzero_target
        & (np.abs(signed_true) >= direction_minimum_m)
    )
    near_zero = continue_true & ~directional
    target_positive = directional & (signed_true > 0.0)
    target_negative = directional & (signed_true < 0.0)
    predicted_positive = signed_raw > 0.0
    predicted_negative = signed_raw < 0.0
    predicted_zero = signed_raw == 0.0
    positive_recall = _recall(predicted_positive, target_positive)
    negative_recall = _recall(predicted_negative, target_negative)
    direction_correct = np.sign(signed_raw) == np.sign(signed_true)
    direction_accuracy = _recall(direction_correct, directional)

    if np.any(continue_true):
        signed_error = np.abs(signed_raw[continue_true] - signed_true[continue_true])
        magnitude_error = np.abs(
            np.abs(signed_raw[continue_true])
            - np.abs(signed_true[continue_true])
        )
        signed_mae_mm = float(np.mean(signed_error) * 1e3)
        magnitude_mae_mm = float(np.mean(magnitude_error) * 1e3)
    else:
        signed_mae_mm = None
        magnitude_mae_mm = None

    raw_radius_violation = (
        np.abs(signed_raw)
        > maximum_displacement_m + radius_tolerance_m
    )
    raw_radius_violation_rate = (
        None
        if sample_count == 0
        else float(np.mean(raw_radius_violation))
    )
    continue_radius_violation_rate = (
        None
        if not np.any(continue_true)
        else float(np.mean(raw_radius_violation[continue_true]))
    )

    stop_confusion = {
        "target_continue": {
            "predicted_continue": int(np.sum(continue_true & continue_pred)),
            "predicted_stop": int(np.sum(continue_true & stop_pred)),
        },
        "target_stop": {
            "predicted_continue": int(np.sum(stop_true & continue_pred)),
            "predicted_stop": int(np.sum(stop_true & stop_pred)),
        },
    }
    direction_confusion = {
        "target_negative": {
            "predicted_negative": int(np.sum(target_negative & predicted_negative)),
            "predicted_zero": int(np.sum(target_negative & predicted_zero)),
            "predicted_positive": int(np.sum(target_negative & predicted_positive)),
        },
        "target_positive": {
            "predicted_negative": int(np.sum(target_positive & predicted_negative)),
            "predicted_zero": int(np.sum(target_positive & predicted_zero)),
            "predicted_positive": int(np.sum(target_positive & predicted_positive)),
        },
    }

    return {
        "samples": int(sample_count),
        "continue_samples": int(np.sum(continue_true)),
        "stop_samples": int(np.sum(stop_true)),
        "stop_accuracy": (
            None if sample_count == 0 else float(np.mean(stop_correct))
        ),
        "stop_balanced_accuracy": _balanced_accuracy(
            continue_recall, stop_recall
        ),
        "continue_recall": continue_recall,
        "stop_recall": stop_recall,
        "stop_confusion_matrix": stop_confusion,
        "direction_minimum_mm": direction_minimum_m * 1e3,
        "direction_samples": int(np.sum(directional)),
        "positive_direction_samples": int(np.sum(target_positive)),
        "negative_direction_samples": int(np.sum(target_negative)),
        "near_zero_direction_samples": int(np.sum(near_zero)),
        "direction_accuracy": direction_accuracy,
        "direction_balanced_accuracy": _balanced_accuracy(
            positive_recall, negative_recall
        ),
        "positive_direction_recall": positive_recall,
        "negative_direction_recall": negative_recall,
        "direction_confusion_matrix": direction_confusion,
        "signed_displacement_mae_mm": signed_mae_mm,
        "magnitude_mae_mm": magnitude_mae_mm,
        "raw_radius_violation_count": int(np.sum(raw_radius_violation)),
        "raw_radius_violation_rate": raw_radius_violation_rate,
        "continue_raw_radius_violation_rate": continue_radius_violation_rate,
        # Kept while downstream reports migrate from the v1 name.  It has the
        # same raw/pre-clip semantics as raw_radius_violation_rate.
        "radius_violation_rate": raw_radius_violation_rate,
    }

