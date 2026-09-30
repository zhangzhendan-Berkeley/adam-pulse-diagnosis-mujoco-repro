from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def sealed_scenario_contract_failures(
    manifest: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    summary_roots: Sequence[Path],
) -> list[str]:
    """Validate that both blind sources exactly replay the sealed scenarios."""

    failures: list[str] = []
    raw_scenarios = manifest.get("scenarios")
    manifest_scenarios = (
        list(raw_scenarios)
        if isinstance(raw_scenarios, Sequence)
        and not isinstance(raw_scenarios, (str, bytes))
        else []
    )
    expected_scenarios: dict[str, Mapping[str, Any]] = {}
    for scenario in manifest_scenarios:
        if not isinstance(scenario, Mapping) or "scenario_id" not in scenario:
            failures.append("manifest_missing_scenario_metadata")
            continue
        scenario_id = str(scenario["scenario_id"])
        if scenario_id in expected_scenarios:
            failures.append("manifest_duplicate_scenario_id")
        expected_scenarios[scenario_id] = scenario

    expected_per_source = manifest.get("scenario_count_per_source")
    if isinstance(expected_per_source, bool) or not isinstance(
        expected_per_source, int
    ):
        failures.append("manifest_invalid_scenario_count")
        expected_per_source = -1
    if len(manifest_scenarios) != expected_per_source:
        failures.append("manifest_scenario_count_mismatch")

    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        sources = {}
        failures.append("manifest_missing_source_roots")
    expected_roots = (
        sources.get("observable_expert"),
        sources.get("privileged_expert"),
    )
    if len(summaries) != 2 or len(summary_roots) != 2:
        failures.append("blind_source_count_mismatch")
        return sorted(set(failures))
    for expected, actual in zip(expected_roots, summary_roots, strict=True):
        if not isinstance(expected, str) or not expected:
            failures.append("manifest_missing_source_roots")
        elif Path(expected).resolve() != Path(actual).resolve():
            failures.append("collection_source_root_mismatch")

    for source_index, summary in enumerate(summaries):
        if int(summary.get("episode_count", -1)) != expected_per_source:
            failures.append(f"source_{source_index}_episode_count_mismatch")
        raw_episodes = summary.get("episodes")
        episodes = (
            list(raw_episodes)
            if isinstance(raw_episodes, Sequence)
            and not isinstance(raw_episodes, (str, bytes))
            else []
        )
        observed_by_id: dict[str, Mapping[str, Any]] = {}
        for episode in episodes:
            if not isinstance(episode, Mapping):
                failures.append(f"source_{source_index}_invalid_episode_metadata")
                continue
            scenario = episode.get("scenario")
            if not isinstance(scenario, Mapping) or "scenario_id" not in scenario:
                failures.append(f"source_{source_index}_missing_scenario_metadata")
                continue
            scenario_id = str(scenario["scenario_id"])
            if scenario_id in observed_by_id:
                failures.append(f"source_{source_index}_duplicate_scenario_id")
            observed_by_id[scenario_id] = scenario
        if set(observed_by_id) != set(expected_scenarios):
            failures.append(f"source_{source_index}_scenario_id_set_mismatch")
        for scenario_id, scenario in observed_by_id.items():
            if expected_scenarios.get(scenario_id) != scenario:
                failures.append(f"source_{source_index}_scenario_parameter_mismatch")

    return sorted(set(failures))

