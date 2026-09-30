from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import median_filter


PULSE_SIGNAL_QUALITY_SCHEMA_VERSION = "pulse_signal_quality_v1"
PULSE_DIAGNOSIS_QUALITY_GATE_SCHEMA_VERSION = "pulse_diagnosis_quality_gate_v1"

# Sensor-derived fields that are sufficient to replay the deterministic stop
# gate without retaining privileged simulator state.  These values describe
# only the completed sampling window at the current contact point.
PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS = (
    "sample_count",
    "duration_s",
    "dual_contact_ratio",
    "any_finger_peak_to_peak_kpa",
    "aggregate_peak_to_peak_kpa",
    "dominant_frequency_hz",
    "band_energy_ratio",
    "force_safe_ratio",
)

# These three fields are the original aggregate spatial contract.  Keep the
# explicit base tuple so old aggregate-only configurations stay compatible.
PULSE_TACTILE_COVERAGE_BASE_OBSERVABLE_FIELDS = (
    "central_energy_ratio",
    "weighted_centroid_offset_mm",
    "per_finger_energy_fraction",
)

# Complete set persisted by current index builders.  The strict vectors are
# required only when their corresponding threshold is enabled.
PULSE_TACTILE_COVERAGE_OBSERVABLE_FIELDS = (
    *PULSE_TACTILE_COVERAGE_BASE_OBSERVABLE_FIELDS,
    # These arrays are still entirely tactile-derived.  Persisting them makes
    # a configured per-finger centering gate replayable from an offline index
    # instead of silently degrading it to the legacy aggregate-only gate.
    "per_finger_central_energy_ratio",
    "per_finger_weighted_centroid_offset_mm",
    "per_finger_center_to_edge_contrast_ratio",
    "per_finger_peak_longitudinal_offset_mm",
    "per_finger_center_core_mean_kpa",
    "per_finger_minimum_bidirectional_decay_ratio",
)


@dataclass(frozen=True)
class PulseSignalQualityThresholds:
    minimum_samples: int = 60
    minimum_dual_contact_ratio: float = 0.95
    minimum_any_finger_peak_to_peak_kpa: float = 0.45
    minimum_aggregate_peak_to_peak_kpa: float = 0.30
    minimum_coherent_any_finger_peak_to_peak_kpa: float = 0.45
    minimum_coherent_aggregate_peak_to_peak_kpa: float = 0.30
    minimum_coherent_band_energy_ratio: float = 0.25
    minimum_frequency_hz: float = 0.8
    maximum_frequency_hz: float = 1.5
    minimum_band_energy_ratio: float = 0.25
    minimum_force_n: float = 2.0
    maximum_force_n: float = 8.0
    minimum_force_safe_ratio: float = 0.95

    @classmethod
    def from_mapping(cls, values: dict | None) -> "PulseSignalQualityThresholds":
        if not values:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{name: values[name] for name in fields if name in values})

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PulseTactileCoverageThresholds:
    minimum_central_energy_ratio: float = 0.75
    maximum_weighted_centroid_offset_mm: float = 8.0
    minimum_per_finger_energy_fraction: float = 0.0
    minimum_per_finger_central_energy_ratio: float = 0.0
    maximum_per_finger_weighted_centroid_offset_mm: float = float("inf")
    minimum_per_finger_center_to_edge_contrast_ratio: float = 0.0
    maximum_per_finger_peak_longitudinal_offset_mm: float = float("inf")
    minimum_per_finger_center_core_mean_kpa: float = 0.0
    minimum_per_finger_bidirectional_decay_ratio: float = 0.0

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "PulseTactileCoverageThresholds":
        if not values:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{name: values[name] for name in fields if name in values})

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PulseSignalQuality:
    schema_version: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    sample_count: int
    duration_s: float
    dual_contact_ratio: float
    per_finger_peak_to_peak_kpa: tuple[float, ...]
    any_finger_peak_to_peak_kpa: float
    aggregate_peak_to_peak_kpa: float
    coherent_low_amplitude_fallback_used: bool
    dominant_frequency_hz: float
    band_energy_ratio: float
    force_safe_ratio: float
    per_finger_force_p05_n: tuple[float, ...]
    per_finger_force_p95_n: tuple[float, ...]

    def as_dict(self) -> dict:
        result = asdict(self)
        result["rejection_reasons"] = list(self.rejection_reasons)
        result["per_finger_peak_to_peak_kpa"] = list(
            self.per_finger_peak_to_peak_kpa
        )
        result["per_finger_force_p05_n"] = list(self.per_finger_force_p05_n)
        result["per_finger_force_p95_n"] = list(self.per_finger_force_p95_n)
        return result


def should_extend_pulse_quality_confirmation(
    quality: PulseSignalQuality,
    *,
    thresholds: PulseSignalQualityThresholds,
    band_energy_margin: float,
) -> bool:
    """Request a longer window only for a safe, plausible pulse.

    This is deliberately narrower than the acceptance gate.  It never changes
    a threshold or accepts a sample itself; it only authorizes collecting more
    public tactile/contact/force samples when contact and force checks already
    pass.  Besides the low-amplitude coherent fallback, a strong pulse whose
    short-window spectrum has not settled may receive one longer confirmation
    window before the search controller chooses a direction.
    """

    margin = float(band_energy_margin)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("band_energy_margin must be finite and non-negative")
    if quality.accepted:
        return False
    amplitude_reasons = {
        "any_finger_peak_to_peak",
        "aggregate_peak_to_peak",
    }
    spectral_reasons = {"dominant_frequency", "band_energy_ratio"}
    reasons = set(quality.rejection_reasons)
    if not reasons or not reasons.issubset(
        amplitude_reasons | spectral_reasons
    ):
        return False
    if reasons & spectral_reasons:
        return bool(
            quality.any_finger_peak_to_peak_kpa
            >= thresholds.minimum_any_finger_peak_to_peak_kpa
            and quality.aggregate_peak_to_peak_kpa
            >= thresholds.minimum_aggregate_peak_to_peak_kpa
        )
    return bool(
        quality.any_finger_peak_to_peak_kpa
        >= thresholds.minimum_coherent_any_finger_peak_to_peak_kpa
        and quality.aggregate_peak_to_peak_kpa
        >= thresholds.minimum_coherent_aggregate_peak_to_peak_kpa
        and quality.band_energy_ratio
        >= thresholds.minimum_coherent_band_energy_ratio - margin
    )


def pulse_spatial_direction_is_reliable(
    quality: PulseSignalQuality | Mapping[str, Any],
) -> bool:
    """Return whether a window may safely guide tactile spatial motion."""

    if isinstance(quality, PulseSignalQuality):
        reasons = set(quality.rejection_reasons)
    else:
        reasons = set(quality.get("rejection_reasons", ()))
    invalidating_reasons = {
        "insufficient_samples",
        "dual_contact_ratio",
        "dominant_frequency",
        "band_energy_ratio",
        "force_safe_ratio",
    }
    return not bool(reasons & invalidating_reasons)


def despike_tactile_window(
    tactile_kpa: np.ndarray,
    *,
    window_samples: int = 5,
    threshold_mad: float = 6.0,
    minimum_delta_kpa: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace isolated temporal taxel spikes by neighboring interpolation.

    The mask requires both a large local residual and a channel-wise robust
    residual z-score.  Smooth physiological pulse extrema therefore remain
    untouched, while one or two corrupted sensor frames are removed before
    band-pass filtering can spread them across the whole quality window.
    """

    values = np.asarray(tactile_kpa, dtype=float)
    size = int(window_samples)
    z_threshold = float(threshold_mad)
    minimum_delta = float(minimum_delta_kpa)
    if values.ndim != 3:
        raise ValueError("tactile_kpa must have shape [samples, fingers, channels]")
    if len(values) < size:
        raise ValueError("tactile window is shorter than the despike window")
    if size < 3 or size % 2 == 0:
        raise ValueError("window_samples must be an odd integer >= 3")
    if not np.isfinite(values).all():
        raise ValueError("tactile_kpa must be finite")
    if not np.isfinite(z_threshold) or z_threshold <= 0.0:
        raise ValueError("threshold_mad must be finite and positive")
    if not np.isfinite(minimum_delta) or minimum_delta < 0.0:
        raise ValueError("minimum_delta_kpa must be finite and non-negative")

    local_median = median_filter(
        values,
        size=(size, 1, 1),
        mode="nearest",
    )
    residual = values - local_median
    residual_median = np.median(residual, axis=0, keepdims=True)
    residual_mad = np.median(
        np.abs(residual - residual_median), axis=0, keepdims=True
    )
    robust_scale = 1.4826 * residual_mad
    threshold = np.maximum(z_threshold * robust_scale, minimum_delta)
    mask = np.abs(residual - residual_median) > threshold
    filtered = values.copy()
    sample_indices = np.arange(len(values), dtype=float)
    for finger in range(values.shape[1]):
        for channel in range(values.shape[2]):
            channel_mask = mask[:, finger, channel]
            if not channel_mask.any():
                continue
            valid = ~channel_mask
            if valid.sum() < 2:
                filtered[:, finger, channel] = local_median[
                    :, finger, channel
                ]
                continue
            filtered[channel_mask, finger, channel] = np.interp(
                sample_indices[channel_mask],
                sample_indices[valid],
                values[valid, finger, channel],
            )
    return filtered, mask


def _robust_peak_to_peak(values: np.ndarray, axis=0) -> np.ndarray:
    return np.percentile(values, 95.0, axis=axis) - np.percentile(
        values, 5.0, axis=axis
    )


def _quality_decision(
    *,
    sample_count: int,
    sample_rate_hz: float,
    dual_contact_ratio: float,
    any_finger_peak_to_peak_kpa: float,
    aggregate_peak_to_peak_kpa: float,
    dominant_frequency_hz: float,
    band_energy_ratio: float,
    force_safe_ratio: float,
    thresholds: PulseSignalQualityThresholds,
) -> tuple[tuple[str, ...], bool]:
    """Apply the common signal-quality decision to observable statistics."""

    standard_amplitude = bool(
        any_finger_peak_to_peak_kpa
        >= thresholds.minimum_any_finger_peak_to_peak_kpa
        and aggregate_peak_to_peak_kpa
        >= thresholds.minimum_aggregate_peak_to_peak_kpa
    )
    coherent_amplitude = bool(
        any_finger_peak_to_peak_kpa
        >= thresholds.minimum_coherent_any_finger_peak_to_peak_kpa
        and aggregate_peak_to_peak_kpa
        >= thresholds.minimum_coherent_aggregate_peak_to_peak_kpa
        and band_energy_ratio
        >= thresholds.minimum_coherent_band_energy_ratio
    )
    coherent_fallback_used = bool(coherent_amplitude and not standard_amplitude)

    reasons = []
    if sample_count < thresholds.minimum_samples:
        reasons.append("insufficient_samples")
    if dual_contact_ratio < thresholds.minimum_dual_contact_ratio:
        reasons.append("dual_contact_ratio")
    if (
        any_finger_peak_to_peak_kpa
        < thresholds.minimum_any_finger_peak_to_peak_kpa
        and not coherent_amplitude
    ):
        reasons.append("any_finger_peak_to_peak")
    if (
        aggregate_peak_to_peak_kpa
        < thresholds.minimum_aggregate_peak_to_peak_kpa
        and not coherent_amplitude
    ):
        reasons.append("aggregate_peak_to_peak")

    # A finite FFT can place a component exactly at the closest bin just
    # outside a continuous physiological boundary (for example 0.78125 Hz
    # for a 0.8 Hz lower bound at 20 Hz / 512 bins).  Allow half one FFT bin
    # on each side; this accounts for quantization without relaxing the
    # configured physiological band by an arbitrary fixed amount.
    n_fft = max(512, 1 << int(np.ceil(np.log2(sample_count))))
    frequency_bin_hz = float(sample_rate_hz / n_fft)
    frequency_tolerance_hz = 0.5 * frequency_bin_hz
    if not (
        thresholds.minimum_frequency_hz - frequency_tolerance_hz
        <= dominant_frequency_hz
        <= thresholds.maximum_frequency_hz + frequency_tolerance_hz
    ):
        reasons.append("dominant_frequency")
    if band_energy_ratio < thresholds.minimum_band_energy_ratio:
        reasons.append("band_energy_ratio")
    if force_safe_ratio < thresholds.minimum_force_safe_ratio:
        reasons.append("force_safe_ratio")
    return tuple(reasons), coherent_fallback_used


def pulse_signal_quality_gate_from_metrics(
    metrics: Mapping[str, Any],
    *,
    thresholds: PulseSignalQualityThresholds | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Replay the deterministic stop gate from a completed sensor window.

    The input is deliberately restricted to current-window tactile/contact/
    force statistics.  Pulse-center truth, expert guidance and future state
    are neither required nor accepted by this interface.
    """

    thresholds = thresholds or PulseSignalQualityThresholds()
    missing = [
        field for field in PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS if field not in metrics
    ]
    if missing:
        raise KeyError(f"missing observable pulse quality fields: {missing}")
    values = {
        field: float(metrics[field]) for field in PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS
    }
    sample_count = int(values["sample_count"])
    duration_s = values["duration_s"]
    if sample_count < 4:
        raise ValueError("observable pulse quality requires at least four samples")
    if duration_s <= 0.0:
        raise ValueError("observable pulse quality duration_s must be positive")
    if not np.isfinite(np.asarray(list(values.values()), dtype=float)).all():
        raise ValueError("observable pulse quality metrics must be finite")
    sample_rate_hz = sample_count / duration_s
    reasons, _ = _quality_decision(
        sample_count=sample_count,
        sample_rate_hz=sample_rate_hz,
        dual_contact_ratio=values["dual_contact_ratio"],
        any_finger_peak_to_peak_kpa=values["any_finger_peak_to_peak_kpa"],
        aggregate_peak_to_peak_kpa=values["aggregate_peak_to_peak_kpa"],
        dominant_frequency_hz=values["dominant_frequency_hz"],
        band_energy_ratio=values["band_energy_ratio"],
        force_safe_ratio=values["force_safe_ratio"],
        thresholds=thresholds,
    )
    return not reasons, reasons


def pulse_tactile_coverage_gate_from_metrics(
    metrics: Mapping[str, Any],
    *,
    thresholds: PulseTactileCoverageThresholds | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Replay the sensor-derived spatial coverage part of the stop gate."""

    thresholds = thresholds or PulseTactileCoverageThresholds()
    missing = [
        field
        for field in PULSE_TACTILE_COVERAGE_BASE_OBSERVABLE_FIELDS
        if field not in metrics
    ]
    if missing:
        raise KeyError(f"missing observable tactile coverage fields: {missing}")
    central_ratio = float(metrics["central_energy_ratio"])
    centroid_offset = float(metrics["weighted_centroid_offset_mm"])
    per_finger = np.asarray(metrics["per_finger_energy_fraction"], dtype=float)
    if per_finger.ndim != 1 or per_finger.size < 1:
        raise ValueError("per_finger_energy_fraction must be a non-empty vector")
    values = np.concatenate(
        (np.asarray([central_ratio, centroid_offset], dtype=float), per_finger)
    )
    if not np.isfinite(values).all():
        raise ValueError("observable tactile coverage metrics must be finite")
    reasons = []
    if central_ratio < thresholds.minimum_central_energy_ratio:
        reasons.append("tactile_central_energy")
    if centroid_offset > thresholds.maximum_weighted_centroid_offset_mm:
        reasons.append("tactile_centroid_offset")
    if np.any(per_finger < thresholds.minimum_per_finger_energy_fraction):
        reasons.append("tactile_per_finger_energy")
    optional_vector_gates = (
        (
            "per_finger_central_energy_ratio",
            thresholds.minimum_per_finger_central_energy_ratio,
            np.less,
            "tactile_per_finger_central_energy",
            thresholds.minimum_per_finger_central_energy_ratio > 0.0,
        ),
        (
            "per_finger_weighted_centroid_offset_mm",
            thresholds.maximum_per_finger_weighted_centroid_offset_mm,
            np.greater,
            "tactile_per_finger_centroid_offset",
            np.isfinite(
                thresholds.maximum_per_finger_weighted_centroid_offset_mm
            ),
        ),
        (
            "per_finger_center_to_edge_contrast_ratio",
            thresholds.minimum_per_finger_center_to_edge_contrast_ratio,
            np.less,
            "tactile_per_finger_center_edge_contrast",
            thresholds.minimum_per_finger_center_to_edge_contrast_ratio > 0.0,
        ),
        (
            "per_finger_peak_longitudinal_offset_mm",
            thresholds.maximum_per_finger_peak_longitudinal_offset_mm,
            np.greater,
            "tactile_per_finger_peak_location",
            np.isfinite(
                thresholds.maximum_per_finger_peak_longitudinal_offset_mm
            ),
        ),
        (
            "per_finger_center_core_mean_kpa",
            thresholds.minimum_per_finger_center_core_mean_kpa,
            np.less,
            "tactile_per_finger_center_amplitude",
            thresholds.minimum_per_finger_center_core_mean_kpa > 0.0,
        ),
        (
            "per_finger_minimum_bidirectional_decay_ratio",
            thresholds.minimum_per_finger_bidirectional_decay_ratio,
            np.less,
            "tactile_per_finger_bidirectional_decay",
            thresholds.minimum_per_finger_bidirectional_decay_ratio > 0.0,
        ),
    )
    for field, threshold, comparison, reason, enabled in optional_vector_gates:
        if not enabled:
            continue
        if field not in metrics:
            raise KeyError(f"missing observable tactile coverage field: {field}")
        vector = np.asarray(metrics[field], dtype=float)
        if vector.shape != per_finger.shape or not np.isfinite(vector).all():
            raise ValueError(f"{field} must be a finite per-finger vector")
        if np.any(comparison(vector, threshold)):
            reasons.append(reason)
    return not reasons, tuple(reasons)


def pulse_diagnosis_quality_gate_from_metrics(
    signal_metrics: Mapping[str, Any],
    *,
    signal_thresholds: PulseSignalQualityThresholds | None = None,
    spatial_metrics: Mapping[str, Any] | None = None,
    spatial_thresholds: PulseTactileCoverageThresholds | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Replay the complete deployable stop gate for one palpation window.

    ``spatial_thresholds=None`` preserves the historical signal-only gate.
    When spatial thresholds are configured, coverage is mandatory and may be
    supplied separately (offline index) or as ``tactile_coverage`` in the live
    expert summary.
    """

    signal_accepted, signal_reasons = pulse_signal_quality_gate_from_metrics(
        signal_metrics, thresholds=signal_thresholds
    )
    if spatial_thresholds is None:
        return signal_accepted, signal_reasons
    coverage = spatial_metrics
    if coverage is None:
        nested = signal_metrics.get("tactile_coverage")
        coverage = nested if isinstance(nested, Mapping) else None
    if coverage is None:
        return False, tuple(signal_reasons) + ("tactile_coverage_unavailable",)
    spatial_accepted, spatial_reasons = pulse_tactile_coverage_gate_from_metrics(
        coverage, thresholds=spatial_thresholds
    )
    return signal_accepted and spatial_accepted, tuple(signal_reasons) + tuple(
        spatial_reasons
    )


def evaluate_pulse_signal_quality(
    dynamic_tactile_kpa: np.ndarray,
    contact: np.ndarray,
    force_n: np.ndarray,
    *,
    sample_rate_hz: float,
    thresholds: PulseSignalQualityThresholds | None = None,
) -> PulseSignalQuality:
    thresholds = thresholds or PulseSignalQualityThresholds()
    dynamic = np.asarray(dynamic_tactile_kpa, dtype=float)
    contacts = np.asarray(contact, dtype=bool)
    forces = np.asarray(force_n, dtype=float)
    if dynamic.ndim != 3:
        raise ValueError("dynamic_tactile_kpa must have shape [samples, fingers, channels]")
    if contacts.shape != dynamic.shape[:2]:
        raise ValueError("contact must have shape [samples, fingers]")
    if forces.shape != dynamic.shape[:2]:
        raise ValueError("force_n must have shape [samples, fingers]")
    if dynamic.shape[0] < 4:
        raise ValueError("at least four tactile samples are required")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if not np.isfinite(dynamic).all() or not np.isfinite(forces).all():
        raise ValueError("quality inputs must be finite")

    finger_signal = dynamic.mean(axis=2)
    finger_amplitude = _robust_peak_to_peak(finger_signal, axis=0)
    aggregate_signal = finger_signal.mean(axis=1)
    aggregate_amplitude = float(_robust_peak_to_peak(aggregate_signal))
    sample_time = np.arange(len(aggregate_signal), dtype=float) / sample_rate_hz
    detrended = aggregate_signal - np.polyval(
        np.polyfit(sample_time, aggregate_signal, 1), sample_time
    )
    windowed = detrended * np.hanning(len(detrended))
    n_fft = max(512, 1 << int(np.ceil(np.log2(len(windowed)))))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    power = np.square(np.abs(np.fft.rfft(windowed, n=n_fft)))
    positive = frequencies > 0.0
    positive_power = power[positive]
    if positive_power.size and float(positive_power.sum()) > 1e-12:
        positive_frequencies = frequencies[positive]
        dominant_frequency_hz = float(
            positive_frequencies[int(np.argmax(positive_power))]
        )
        physiological = (
            (positive_frequencies >= thresholds.minimum_frequency_hz)
            & (positive_frequencies <= thresholds.maximum_frequency_hz)
        )
        band_energy_ratio = float(
            positive_power[physiological].sum() / positive_power.sum()
        )
    else:
        dominant_frequency_hz = 0.0
        band_energy_ratio = 0.0

    dual_contact_ratio = float(contacts.all(axis=1).mean())
    force_safe = (
        (forces >= thresholds.minimum_force_n)
        & (forces <= thresholds.maximum_force_n)
    ).all(axis=1)
    force_safe_ratio = float(force_safe.mean())
    force_p05 = np.percentile(forces, 5.0, axis=0)
    force_p95 = np.percentile(forces, 95.0, axis=0)

    reasons, coherent_fallback_used = _quality_decision(
        sample_count=len(dynamic),
        sample_rate_hz=sample_rate_hz,
        dual_contact_ratio=dual_contact_ratio,
        any_finger_peak_to_peak_kpa=float(finger_amplitude.max()),
        aggregate_peak_to_peak_kpa=aggregate_amplitude,
        dominant_frequency_hz=dominant_frequency_hz,
        band_energy_ratio=band_energy_ratio,
        force_safe_ratio=force_safe_ratio,
        thresholds=thresholds,
    )

    return PulseSignalQuality(
        schema_version=PULSE_SIGNAL_QUALITY_SCHEMA_VERSION,
        accepted=not reasons,
        rejection_reasons=reasons,
        sample_count=len(dynamic),
        duration_s=float(len(dynamic) / sample_rate_hz),
        dual_contact_ratio=dual_contact_ratio,
        per_finger_peak_to_peak_kpa=tuple(float(value) for value in finger_amplitude),
        any_finger_peak_to_peak_kpa=float(finger_amplitude.max()),
        aggregate_peak_to_peak_kpa=aggregate_amplitude,
        coherent_low_amplitude_fallback_used=coherent_fallback_used,
        dominant_frequency_hz=dominant_frequency_hz,
        band_energy_ratio=band_energy_ratio,
        force_safe_ratio=force_safe_ratio,
        per_finger_force_p05_n=tuple(float(value) for value in force_p05),
        per_finger_force_p95_n=tuple(float(value) for value in force_p95),
    )
