from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _episode_basename(record: Mapping[str, Any]) -> str:
    raw = record.get("episode")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("episode record has no valid episode path")
    return Path(raw).name


def _episodes(summary: Mapping[str, Any], *, expected: int, label: str) -> list[dict]:
    raw = summary.get("episodes")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{label} summary episodes is not an array")
    episodes = list(raw)
    if len(episodes) != expected:
        raise ValueError(
            f"{label} summary must contain exactly {expected} episode(s)"
        )
    if summary.get("episode_count") != expected:
        raise ValueError(f"{label} summary episode_count does not match")
    if any(not isinstance(item, Mapping) for item in episodes):
        raise ValueError(f"{label} summary contains an invalid episode record")
    return [dict(item) for item in episodes]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sum_integer(records: Sequence[Mapping[str, Any]], key: str) -> int:
    total = 0
    for record in records:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"episode has invalid nonnegative integer: {key}")
        total += value
    return total


def repair_collection_summary(
    base_summary: Mapping[str, Any],
    replacement_summary: Mapping[str, Any],
    *,
    expected_episode: str,
    base_summary_path: Path,
    replacement_summary_path: Path,
) -> dict[str, Any]:
    """Replace one failed record while preserving auditable source evidence.

    This is intentionally strict and never mutates either source summary.  It
    is for deterministic reruns of one failed scenario, not for cherry-picking
    an arbitrary successful episode from a larger collection.
    """

    base_records = _episodes(base_summary, expected=30, label="base")
    replacement_records = _episodes(
        replacement_summary, expected=1, label="replacement"
    )
    if not expected_episode or Path(expected_episode).name != expected_episode:
        raise ValueError("expected_episode must be one basename")

    matching_indices = [
        index
        for index, record in enumerate(base_records)
        if _episode_basename(record) == expected_episode
    ]
    if len(matching_indices) != 1:
        raise ValueError("base summary must contain exactly one matching episode")
    base_index = matching_indices[0]
    original = base_records[base_index]
    replacement = replacement_records[0]

    if _episode_basename(replacement) != expected_episode:
        raise ValueError("replacement episode basename does not match")
    if original.get("accepted") is not False:
        raise ValueError("base record selected for replacement is not rejected")
    if replacement.get("accepted") is not True:
        raise ValueError("replacement record is not accepted")
    if replacement.get("success") is not True:
        raise ValueError("replacement record did not complete successfully")
    if replacement.get("signal_quality_accepted") is not True:
        raise ValueError("replacement record did not pass signal quality")
    if replacement_summary.get("accepted_count") != 1:
        raise ValueError("replacement summary accepted_count must be exactly one")
    if original.get("scenario") != replacement.get("scenario"):
        raise ValueError("replacement scenario does not exactly match base scenario")

    result = copy.deepcopy(dict(base_summary))
    repaired_records = copy.deepcopy(base_records)
    repaired_records[base_index] = copy.deepcopy(replacement)
    result["episodes"] = repaired_records
    result["episode_count"] = len(repaired_records)
    result["accepted_count"] = sum(
        record.get("accepted") is True for record in repaired_records
    )
    result["policy_action_episode_count"] = sum(
        record.get("policy_action_executed") is True for record in repaired_records
    )
    result["policy_action_frame_count"] = _sum_integer(
        repaired_records, "policy_action_frames"
    )
    result["expert_action_queries"] = _sum_integer(
        repaired_records, "expert_action_queries"
    )
    result["privileged_state_reads"] = _sum_integer(
        repaired_records, "privileged_state_reads"
    )
    result["posthoc_privileged_state_reads"] = _sum_integer(
        repaired_records, "posthoc_privileged_state_reads"
    )
    result["expert_direction_queries"] = _sum_integer(
        repaired_records, "expert_direction_queries"
    )
    result["forbidden_contact_events"] = _sum_integer(
        repaired_records, "forbidden_contact_events"
    )
    result["segment_policy_proposals"] = sum(
        int((record.get("segment_policy") or {}).get("proposals", 0))
        for record in repaired_records
    )
    result["repair_audit"] = {
        "schema_version": "adam_pulse_collection_repair_audit_v1",
        "method": "single_matched_scenario_replacement",
        "expected_base_episode_count": 30,
        "episode": expected_episode,
        "record_index": base_index,
        "scenario_exact_match": True,
        "original_record_accepted": False,
        "replacement_record_accepted": True,
        "base_summary": {
            "path": str(base_summary_path.resolve()),
            "sha256": sha256_file(base_summary_path),
        },
        "replacement_summary": {
            "path": str(replacement_summary_path.resolve()),
            "sha256": sha256_file(replacement_summary_path),
        },
        "original_episode_path": str(original["episode"]),
        "replacement_episode_path": str(replacement["episode"]),
        "original_failure": original.get("failure"),
    }
    return result
