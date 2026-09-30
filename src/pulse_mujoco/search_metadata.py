from __future__ import annotations

from typing import Any, Mapping


def compact_search_episode_metadata(result: Mapping[str, Any]) -> dict:
    """Preserve search decisions without copying full tactile time series."""
    return {
        "recovery_events": result.get("recovery_events", []),
        # These labels and policy records are authoritative.  Never rebuild
        # them from the selected position or privileged post-hoc metrics.
        "search_segment_plans": result.get("search_segment_plans", []),
        "search_segment_policy_records": result.get(
            "search_segment_policy_records", []
        ),
        "search_sample_summaries": [
            {
                "point_index": sample.get("point_index"),
                "position_m": sample.get("position_m"),
                "pulse_score_kpa": sample.get("pulse_score_kpa"),
                "pulse_direction_xy": sample.get("pulse_direction_xy"),
                "pulse_centered_taxel_offset_xy_m": sample.get(
                    "pulse_centered_taxel_offset_xy_m"
                ),
                "pulse_within_pad_offset_xy_m": sample.get(
                    "pulse_within_pad_offset_xy_m"
                ),
                "pulse_interfinger_energy_offset_xy_m": sample.get(
                    "pulse_interfinger_energy_offset_xy_m"
                ),
                "pulse_weighted_taxel_center_m": sample.get(
                    "pulse_weighted_taxel_center_m"
                ),
                "per_channel_pulse_score_kpa": sample.get(
                    "per_channel_pulse_score_kpa"
                ),
                "pulse_direction_magnitude_mm": sample.get(
                    "pulse_direction_magnitude_mm"
                ),
                "tactile_despike_replacements": sample.get(
                    "tactile_despike_replacements", 0
                ),
                "quality_confirmation_extended": sample.get(
                    "quality_confirmation_extended", False
                ),
                "quality_confirmation_extra_steps": sample.get(
                    "quality_confirmation_extra_steps", 0
                ),
                "preliminary_signal_quality": sample.get(
                    "preliminary_signal_quality"
                ),
                "signal_quality": sample.get("signal_quality"),
            }
            for sample in result.get("search_samples", [])
        ],
    }


def search_policy_record_audit(
    *, proposal_count: int, result: Mapping[str, Any]
) -> dict:
    """Make missing persisted model proposals a fail-visible condition."""
    records = result.get("search_segment_policy_records", [])
    record_count = len(records)
    return {
        "proposal_count": int(proposal_count),
        "persisted_record_count": int(record_count),
        "proposal_records_complete": int(proposal_count) == record_count,
    }
