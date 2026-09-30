import numpy as np

from pulse_mujoco import (
    PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS,
    PULSE_SIGNAL_QUALITY_SCHEMA_VERSION,
    PulseSignalQualityThresholds,
    PulseTactileCoverageThresholds,
    despike_tactile_window,
    evaluate_pulse_signal_quality,
    pulse_spatial_direction_is_reliable,
    pulse_diagnosis_quality_gate_from_metrics,
    pulse_signal_quality_gate_from_metrics,
    should_extend_pulse_quality_confirmation,
)


def _window(*, amplitude=0.35, frequency_hz=1.2, samples=90):
    time_s = np.arange(samples, dtype=float) / 20.0
    wave = np.sin(2.0 * np.pi * frequency_hz * time_s)
    dynamic = np.empty((samples, 2, 10), dtype=float)
    dynamic[:, 0, :] = (0.50 + amplitude * wave)[:, None]
    dynamic[:, 1, :] = (0.35 + 0.75 * amplitude * wave)[:, None]
    contact = np.ones((samples, 2), dtype=bool)
    force = np.full((samples, 2), 6.0, dtype=float)
    return dynamic, contact, force


def test_quality_evaluator_accepts_clear_stable_pulse():
    quality = evaluate_pulse_signal_quality(*_window(), sample_rate_hz=20.0)
    assert quality.schema_version == PULSE_SIGNAL_QUALITY_SCHEMA_VERSION
    assert quality.accepted
    assert not quality.rejection_reasons
    assert 0.8 <= quality.dominant_frequency_hz <= 1.5
    assert quality.dual_contact_ratio == 1.0
    assert quality.force_safe_ratio == 1.0


def test_observable_metric_gate_exactly_replays_raw_window_decision():
    thresholds = PulseSignalQualityThresholds()
    for amplitude in (0.02, 0.13, 0.35):
        quality = evaluate_pulse_signal_quality(
            *_window(amplitude=amplitude),
            sample_rate_hz=20.0,
            thresholds=thresholds,
        )
        metrics = {
            field: getattr(quality, field)
            for field in PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS
        }
        accepted, reasons = pulse_signal_quality_gate_from_metrics(
            metrics, thresholds=thresholds
        )
        assert accepted == quality.accepted
        assert reasons == quality.rejection_reasons


def test_observable_metric_gate_rejects_missing_and_nonfinite_fields():
    quality = evaluate_pulse_signal_quality(*_window(), sample_rate_hz=20.0)
    metrics = {
        field: getattr(quality, field)
        for field in PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS
    }
    incomplete = dict(metrics)
    incomplete.pop("band_energy_ratio")
    try:
        pulse_signal_quality_gate_from_metrics(incomplete)
    except KeyError as error:
        assert "band_energy_ratio" in str(error)
    else:
        raise AssertionError("missing signal-quality field was accepted")

    nonfinite = dict(metrics, force_safe_ratio=float("nan"))
    try:
        pulse_signal_quality_gate_from_metrics(nonfinite)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite signal-quality metrics were accepted")


def test_complete_quality_gate_requires_configured_spatial_coverage():
    quality = evaluate_pulse_signal_quality(*_window(), sample_rate_hz=20.0)
    signal = {
        field: getattr(quality, field)
        for field in PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS
    }
    spatial_thresholds = PulseTactileCoverageThresholds(
        minimum_central_energy_ratio=0.75,
        maximum_weighted_centroid_offset_mm=8.0,
        minimum_per_finger_energy_fraction=0.08,
    )
    accepted, reasons = pulse_diagnosis_quality_gate_from_metrics(
        signal,
        spatial_metrics={
            "central_energy_ratio": 0.90,
            "weighted_centroid_offset_mm": 1.0,
            "per_finger_energy_fraction": [0.20, 0.55, 0.25],
        },
        spatial_thresholds=spatial_thresholds,
    )
    assert accepted
    assert not reasons

    rejected, reasons = pulse_diagnosis_quality_gate_from_metrics(
        signal,
        spatial_metrics={
            "central_energy_ratio": 0.90,
            "weighted_centroid_offset_mm": 1.0,
            "per_finger_energy_fraction": [0.079, 0.38, 0.541],
        },
        spatial_thresholds=spatial_thresholds,
    )
    assert not rejected
    assert reasons == ("tactile_per_finger_energy",)


def test_complete_quality_gate_replays_strict_per_finger_centering():
    quality = evaluate_pulse_signal_quality(*_window(), sample_rate_hz=20.0)
    signal = {
        field: getattr(quality, field)
        for field in PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS
    }
    thresholds = PulseTactileCoverageThresholds(
        minimum_per_finger_central_energy_ratio=0.55,
        maximum_per_finger_weighted_centroid_offset_mm=4.0,
        minimum_per_finger_center_to_edge_contrast_ratio=1.10,
        maximum_per_finger_peak_longitudinal_offset_mm=2.5,
    )
    spatial = {
        "central_energy_ratio": 0.90,
        "weighted_centroid_offset_mm": 1.0,
        "per_finger_energy_fraction": [0.2, 0.5, 0.3],
        "per_finger_central_energy_ratio": [0.7, 0.8, 0.7],
        "per_finger_weighted_centroid_offset_mm": [2.0, 1.0, 3.0],
        "per_finger_center_to_edge_contrast_ratio": [1.4, 1.8, 1.3],
        "per_finger_peak_longitudinal_offset_mm": [2.5, 2.5, 2.5],
    }
    accepted, reasons = pulse_diagnosis_quality_gate_from_metrics(
        signal,
        spatial_metrics=spatial,
        spatial_thresholds=thresholds,
    )
    assert accepted
    assert not reasons

    edge_biased = dict(spatial)
    edge_biased["per_finger_center_to_edge_contrast_ratio"] = [1.4, 0.8, 1.3]
    edge_biased["per_finger_peak_longitudinal_offset_mm"] = [2.5, 7.5, 2.5]
    accepted, reasons = pulse_diagnosis_quality_gate_from_metrics(
        signal,
        spatial_metrics=edge_biased,
        spatial_thresholds=thresholds,
    )
    assert not accepted
    assert "tactile_per_finger_center_edge_contrast" in reasons
    assert "tactile_per_finger_peak_location" in reasons


def test_complete_quality_gate_replays_bidirectional_decay_fields():
    quality = evaluate_pulse_signal_quality(*_window(), sample_rate_hz=20.0)
    signal = {
        field: getattr(quality, field)
        for field in PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS
    }
    thresholds = PulseTactileCoverageThresholds(
        minimum_per_finger_center_core_mean_kpa=0.15,
        minimum_per_finger_bidirectional_decay_ratio=1.25,
    )
    spatial = {
        "central_energy_ratio": 0.90,
        "weighted_centroid_offset_mm": 1.0,
        "per_finger_energy_fraction": [0.2, 0.5, 0.3],
        "per_finger_center_core_mean_kpa": [0.2, 0.4, 0.25],
        "per_finger_minimum_bidirectional_decay_ratio": [1.5, 2.0, 1.3],
    }
    accepted, reasons = pulse_diagnosis_quality_gate_from_metrics(
        signal,
        spatial_metrics=spatial,
        spatial_thresholds=thresholds,
    )
    assert accepted, reasons

    spatial["per_finger_minimum_bidirectional_decay_ratio"] = [1.5, 1.1, 1.3]
    accepted, reasons = pulse_diagnosis_quality_gate_from_metrics(
        signal,
        spatial_metrics=spatial,
        spatial_thresholds=thresholds,
    )
    assert not accepted
    assert "tactile_per_finger_bidirectional_decay" in reasons


def test_complete_quality_gate_preserves_signal_only_compatibility():
    quality = evaluate_pulse_signal_quality(*_window(), sample_rate_hz=20.0)
    signal = {
        field: getattr(quality, field)
        for field in PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS
    }
    accepted, reasons = pulse_diagnosis_quality_gate_from_metrics(signal)
    assert accepted == quality.accepted
    assert reasons == quality.rejection_reasons


def test_quality_evaluator_allows_half_fft_bin_at_frequency_boundary():
    quality = evaluate_pulse_signal_quality(
        *_window(frequency_hz=0.8), sample_rate_hz=20.0
    )
    assert quality.dominant_frequency_hz == 0.78125
    assert quality.accepted, quality.rejection_reasons


def test_quality_evaluator_rejects_weak_first_point():
    quality = evaluate_pulse_signal_quality(
        *_window(amplitude=0.02), sample_rate_hz=20.0
    )
    assert not quality.accepted
    assert "any_finger_peak_to_peak" in quality.rejection_reasons
    assert "aggregate_peak_to_peak" in quality.rejection_reasons


def test_quality_evaluator_accepts_low_amplitude_only_when_coherent():
    dynamic, contact, force = _window(amplitude=0.13)
    thresholds = PulseSignalQualityThresholds(
        minimum_coherent_any_finger_peak_to_peak_kpa=0.20,
        minimum_coherent_aggregate_peak_to_peak_kpa=0.15,
        minimum_coherent_band_energy_ratio=0.80,
    )
    quality = evaluate_pulse_signal_quality(
        dynamic, contact, force, sample_rate_hz=20.0, thresholds=thresholds
    )
    assert quality.accepted, quality.rejection_reasons
    assert quality.coherent_low_amplitude_fallback_used

    rng = np.random.default_rng(7)
    incoherent = rng.normal(0.0, 0.20, size=dynamic.shape)
    rejected = evaluate_pulse_signal_quality(
        incoherent, contact, force, sample_rate_hz=20.0, thresholds=thresholds
    )
    assert not rejected.accepted
    assert not rejected.coherent_low_amplitude_fallback_used


def test_quality_evaluator_rejects_contact_force_and_frequency_failures():
    dynamic, contact, force = _window(frequency_hz=2.5)
    contact[:20, 1] = False
    force[:20, 0] = 9.0
    quality = evaluate_pulse_signal_quality(
        dynamic,
        contact,
        force,
        sample_rate_hz=20.0,
        thresholds=PulseSignalQualityThresholds(),
    )
    assert not quality.accepted
    assert "dual_contact_ratio" in quality.rejection_reasons
    assert "dominant_frequency" in quality.rejection_reasons
    assert "force_safe_ratio" in quality.rejection_reasons


def test_confirmation_extension_only_admits_safe_near_coherent_window():
    thresholds = PulseSignalQualityThresholds(
        minimum_any_finger_peak_to_peak_kpa=0.30,
        minimum_aggregate_peak_to_peak_kpa=0.18,
        minimum_coherent_any_finger_peak_to_peak_kpa=0.18,
        minimum_coherent_aggregate_peak_to_peak_kpa=0.10,
        minimum_coherent_band_energy_ratio=0.80,
    )
    quality = evaluate_pulse_signal_quality(
        *_window(amplitude=0.13, frequency_hz=1.2),
        sample_rate_hz=20.0,
        thresholds=thresholds,
    )
    # The clean synthetic signal already passes, so it never wastes time on
    # an extension.
    assert not should_extend_pulse_quality_confirmation(
        quality, thresholds=thresholds, band_energy_margin=0.05
    )

    from dataclasses import replace

    near = replace(
        quality,
        accepted=False,
        rejection_reasons=(
            "any_finger_peak_to_peak",
            "aggregate_peak_to_peak",
        ),
        band_energy_ratio=0.78,
    )
    assert should_extend_pulse_quality_confirmation(
        near, thresholds=thresholds, band_energy_margin=0.05
    )
    unsafe = replace(
        near,
        rejection_reasons=("force_safe_ratio",),
    )
    assert not should_extend_pulse_quality_confirmation(
        unsafe, thresholds=thresholds, band_energy_margin=0.05
    )
    far = replace(near, band_energy_ratio=0.70)
    assert not should_extend_pulse_quality_confirmation(
        far, thresholds=thresholds, band_energy_margin=0.05
    )


def test_tactile_despike_removes_two_frame_transient_without_flattening_pulse():
    clean, _, _ = _window(amplitude=0.13, samples=90)
    corrupted = clean.copy()
    corrupted[41:43] += 0.27

    filtered, mask = despike_tactile_window(
        corrupted,
        window_samples=5,
        threshold_mad=6.0,
        minimum_delta_kpa=0.08,
    )

    assert mask[41:43].all()
    assert np.max(np.abs(filtered[41:43] - clean[41:43])) < 0.04
    unchanged = ~mask
    assert np.array_equal(filtered[unchanged], corrupted[unchanged])


def test_tactile_despike_rejects_invalid_configuration():
    dynamic, _, _ = _window(samples=10)
    for kwargs in (
        {"window_samples": 4},
        {"window_samples": 11},
        {"threshold_mad": 0.0},
        {"minimum_delta_kpa": -0.1},
    ):
        try:
            despike_tactile_window(dynamic, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid despike settings accepted: {kwargs}")


def test_spatial_direction_reliability_ignores_amplitude_but_not_time_safety():
    assert pulse_spatial_direction_is_reliable(
        {
            "rejection_reasons": [
                "any_finger_peak_to_peak",
                "aggregate_peak_to_peak",
            ]
        }
    )
    assert not pulse_spatial_direction_is_reliable(
        {"rejection_reasons": ["dominant_frequency"]}
    )
    assert not pulse_spatial_direction_is_reliable(
        {
            "rejection_reasons": [
                "any_finger_peak_to_peak",
                "force_safe_ratio",
            ]
        }
    )


def test_quality_confirmation_extends_strong_spectrally_unsettled_window():
    thresholds = PulseSignalQualityThresholds()
    quality = evaluate_pulse_signal_quality(
        *_window(amplitude=0.35, frequency_hz=0.70),
        sample_rate_hz=20.0,
        thresholds=thresholds,
    )

    assert "dominant_frequency" in quality.rejection_reasons
    assert should_extend_pulse_quality_confirmation(
        quality,
        thresholds=thresholds,
        band_energy_margin=0.05,
    )


def test_quality_confirmation_rejects_spectral_noise_without_pulse_amplitude():
    thresholds = PulseSignalQualityThresholds()
    quality = evaluate_pulse_signal_quality(
        *_window(amplitude=0.01, frequency_hz=0.70),
        sample_rate_hz=20.0,
        thresholds=thresholds,
    )

    assert "dominant_frequency" in quality.rejection_reasons
    assert not should_extend_pulse_quality_confirmation(
        quality,
        thresholds=thresholds,
        band_energy_margin=0.05,
    )
