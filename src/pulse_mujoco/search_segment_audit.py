"""Fail-fast leakage and observation-contract audit for search segments.

The search-segment index contains several samples per episode.  Checking only
sample identifiers is therefore insufficient: aliases of an episode, symlinks
to the same directory, and repeated scenario templates must also remain in one
split.  This module derives all four identities independently and raises on the
first violated invariant.

Direction statistics intentionally cover continue samples only.  A stop target
has no direction and would otherwise inflate the near-zero count.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .pulse_signal_quality import (
    PulseSignalQualityThresholds,
    PulseTactileCoverageThresholds,
    pulse_diagnosis_quality_gate_from_metrics,
)


AUDIT_SCHEMA_VERSION = "adam_pulse_search_segment_audit_v1"
DEFAULT_NEAR_ZERO_THRESHOLD_M = 0.25e-3

# These are the tensors returned under SearchSegmentDataset[...]["observation"].
# Keep this as an allowlist: adding a recorder field must never implicitly make
# it policy-visible.
SEARCH_SEGMENT_OBSERVATION_ALLOWLIST = frozenset(
    {
        "vision",
        "tactile",
        "tactile_valid",
        "proximity",
        "proximity_valid",
        "robot_state",
        "action",
        "task_phase",
    }
)

# Exact names are supplemented by conservative token checks in
# _is_privileged_observation_key().  These labels may be valid supervision or
# offline audit data, but never model observations.
PRIVILEGED_OBSERVATION_KEYS = frozenset(
    {
        "pulse_center_m",
        "expert_guidance_direction_xy",
        "pulse_weighted_taxel_center_m",
        "expert_label_action",
        "privileged_state",
        "privileged_observation",
        "ground_truth",
        "future_contact",
        "future_success",
        "future_signal_quality",
    }
)

_OBSERVATION_DECLARATION_KEYS = (
    "observation_fields",
    "observation_field_allowlist",
    "observation_allowlist",
    "observation_schema",
    "observation_contract",
    "observation",
)


class SearchSegmentAuditError(ValueError):
    """A machine-identifiable, fail-fast search-segment audit failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {message}")


def _fail(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    raise SearchSegmentAuditError(code, message, details=details)


def _nonempty_string(value: Any, field: str, *, sample: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        suffix = "" if sample is None else f" for sample {sample!r}"
        _fail("invalid_field", f"{field} must be a non-empty string{suffix}")
    return value.strip()


def _as_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("invalid_index", f"{field} must be a list")
    return value


def _stable_group_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _read_episode_identity(episode_dir: Path) -> dict[str, Any]:
    """Read only metadata needed for grouping, without retaining large shadows."""

    metadata_path = episode_dir / "episode.json"
    if not metadata_path.is_file():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(
            "invalid_episode_metadata",
            f"cannot read {metadata_path}: {exc}",
            details={"episode_dir": str(episode_dir)},
        )
    if not isinstance(metadata, Mapping):
        _fail(
            "invalid_episode_metadata",
            f"{metadata_path} must contain a JSON object",
            details={"episode_dir": str(episode_dir)},
        )
    scenario = metadata.get("scenario")
    return {
        "scenario_profile": metadata.get("scenario_profile"),
        "scenario": dict(scenario) if isinstance(scenario, Mapping) else scenario,
        "scenario_group": metadata.get("scenario_group"),
        "split_group": metadata.get("split_group"),
    }


def _nested_scenario(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _scenario_identities(
    sample: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Return a report label and every independently auditable group identity.

    A declared group is checked in addition to a group rebuilt from source
    metadata.  Thus a builder cannot hide template reuse by assigning a unique
    declared group to every alias.
    """

    sample_scenario = _nested_scenario(sample.get("scenario"))
    metadata_scenario = _nested_scenario(metadata.get("scenario"))
    declared = _first_present(
        sample.get("scenario_group"), metadata.get("scenario_group")
    )
    declared_value = _stable_group_value(declared)

    profile = _stable_group_value(
        _first_present(
            sample.get("scenario_profile"),
            sample_scenario.get("scenario_profile"),
            metadata.get("scenario_profile"),
            metadata_scenario.get("scenario_profile"),
        )
    )
    scenario_id = _stable_group_value(
        _first_present(
            sample.get("scenario_id"),
            sample_scenario.get("scenario_id"),
            metadata_scenario.get("scenario_id"),
        )
    )

    identities: list[tuple[str, str]] = []
    rebuilt: str | None = None
    if profile is not None and scenario_id is not None:
        rebuilt = f"profile={profile}|scenario_id={scenario_id}"
        identities.append(("profile_scenario_id", rebuilt))
    else:
        # Legacy fallback from the frozen plan: profile + lateral-offset bucket
        # + command-drop pattern.  Prefer an explicit bucket/pattern, otherwise
        # use the exact recorded values so repeated templates still collide.
        offset = _first_present(
            sample.get("lateral_offset_bucket"),
            sample.get("offset_bucket"),
            sample_scenario.get("lateral_offset_bucket"),
            sample_scenario.get("lateral_offset_mm"),
            metadata_scenario.get("lateral_offset_bucket"),
            metadata_scenario.get("lateral_offset_mm"),
        )
        command_drop = _first_present(
            sample.get("command_drop_pattern"),
            sample_scenario.get("command_drop_pattern"),
            metadata_scenario.get("command_drop_pattern"),
        )
        if command_drop is None:
            drop_start = _first_present(
                sample_scenario.get("command_drop_start_step"),
                metadata_scenario.get("command_drop_start_step"),
            )
            drop_steps = _first_present(
                sample_scenario.get("command_drop_steps"),
                metadata_scenario.get("command_drop_steps"),
            )
            if drop_start is not None or drop_steps is not None:
                command_drop = {"start_step": drop_start, "steps": drop_steps}
        if profile is not None and (offset is not None or command_drop is not None):
            rebuilt = (
                f"profile={profile}|offset={_stable_group_value(offset)}|"
                f"command_drop={_stable_group_value(command_drop)}"
            )
            identities.append(("legacy_template", rebuilt))

    if declared_value is not None:
        identities.append(("declared", declared_value))
    if not identities:
        split_group = _stable_group_value(metadata.get("split_group"))
        if split_group is not None:
            identities.append(("metadata_split_group", split_group))
            rebuilt = split_group
    if not identities:
        _fail(
            "missing_scenario_group",
            "cannot derive scenario group; record scenario_profile/scenario_id "
            "or an explicit scenario_group",
            details={"sample_id": sample.get("sample_id")},
        )

    # De-duplicate while retaining deterministic preference order.
    identities = list(dict.fromkeys(identities))
    report_label = declared_value or rebuilt or identities[0][1]
    return report_label, tuple(identities)


def _resolve_episode_path(
    sample: Mapping[str, Any], *, index_directory: Path
) -> Path:
    raw = _first_present(
        sample.get("episode_dir"),
        sample.get("episode_resolved_path"),
        sample.get("resolved_episode_dir"),
        sample.get("resolved_path"),
    )
    sample_id = str(sample.get("sample_id", "<unknown>"))
    raw_string = _nonempty_string(raw, "episode_dir", sample=sample_id)
    path = Path(raw_string).expanduser()
    if not path.is_absolute():
        path = index_directory / path
    resolved = path.resolve(strict=False)

    declared_resolved = _first_present(
        sample.get("episode_resolved_path"),
        sample.get("resolved_episode_dir"),
        sample.get("resolved_path"),
    )
    if declared_resolved is not None and sample.get("episode_dir") is not None:
        declared_path = Path(
            _nonempty_string(
                declared_resolved, "episode_resolved_path", sample=sample_id
            )
        ).expanduser()
        if not declared_path.is_absolute():
            declared_path = index_directory / declared_path
        declared_path = declared_path.resolve(strict=False)
        if declared_path != resolved:
            _fail(
                "resolved_path_mismatch",
                f"sample {sample_id!r} declares {declared_path} but episode_dir "
                f"resolves to {resolved}",
                details={
                    "sample_id": sample_id,
                    "episode_dir": str(resolved),
                    "declared_resolved_path": str(declared_path),
                },
            )
    return resolved


def _identity_cross_split(
    seen: dict[str, tuple[str, str]],
    *,
    identity: str,
    split: str,
    sample_id: str,
    code: str,
    label: str,
) -> None:
    previous = seen.get(identity)
    if previous is not None and previous[0] != split:
        _fail(
            code,
            f"{label} {identity!r} occurs in both {previous[0]!r} and {split!r}",
            details={
                label: identity,
                "first_split": previous[0],
                "second_split": split,
                "first_sample_id": previous[1],
                "second_sample_id": sample_id,
            },
        )
    seen.setdefault(identity, (split, sample_id))


def _normalise_field_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _is_privileged_observation_key(value: str) -> bool:
    key = _normalise_field_name(value).split(".")[-1]
    if key in PRIVILEGED_OBSERVATION_KEYS:
        return True
    return (
        "privileged" in key
        or "ground_truth" in key
        or "pulse_center" in key
        or "expert_guidance" in key
        or "expert_label" in key
        or key.startswith("future_contact")
        or key.startswith("future_success")
        or key.startswith("future_signal_quality")
    )


def _declaration_top_fields(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        for wrapper in ("fields", "keys", "allowlist"):
            if wrapper in value:
                return _declaration_top_fields(value[wrapper])
        return {str(key) for key in value}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        result: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                _fail(
                    "invalid_observation_contract",
                    "observation field lists may contain only strings",
                )
            result.add(item)
        return result
    _fail(
        "invalid_observation_contract",
        "observation field declaration must be a list, mapping, or string",
    )


def _privileged_names_in_declaration(value: Any, path: str = "") -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_string = str(key)
            key_path = f"{path}.{key_string}" if path else key_string
            if _is_privileged_observation_key(key_string):
                result.add(key_path)
            result.update(_privileged_names_in_declaration(nested, key_path))
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            item_path = f"{path}[{index}]"
            if isinstance(nested, str) and _is_privileged_observation_key(nested):
                result.add(nested)
            else:
                result.update(_privileged_names_in_declaration(nested, item_path))
    return result


def _observation_contract(
    payload: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    allowlist: frozenset[str],
) -> tuple[set[str], str]:
    declarations: list[tuple[str, Any]] = []
    for key in _OBSERVATION_DECLARATION_KEYS:
        if key in payload:
            declarations.append((f"index.{key}", payload[key]))
    for sample in samples:
        sample_id = str(sample.get("sample_id", "<unknown>"))
        for key in _OBSERVATION_DECLARATION_KEYS:
            if key in sample:
                declarations.append((f"sample[{sample_id}].{key}", sample[key]))

    # v1 did not persist a declaration.  Its deployed encoded-tensor contract
    # is fixed and can still be audited by key, while the provenance flag makes
    # the legacy inference explicit in the report.
    if not declarations:
        return set(allowlist), "legacy_runtime_contract_inferred"

    fields: set[str] = set()
    privileged: set[str] = set()
    for location, declaration in declarations:
        fields.update(_declaration_top_fields(declaration))
        privileged.update(
            f"{location}:{value}"
            for value in _privileged_names_in_declaration(declaration)
        )
    normalised = {_normalise_field_name(field) for field in fields}
    privileged.update(field for field in normalised if _is_privileged_observation_key(field))
    if privileged:
        _fail(
            "privileged_observation_key",
            "observation contract contains privileged keys: "
            + ", ".join(sorted(privileged)),
            details={"keys": sorted(privileged)},
        )
    unexpected = sorted(normalised - allowlist)
    if unexpected:
        _fail(
            "observation_field_not_allowlisted",
            "observation contract contains fields outside the allowlist: "
            + ", ".join(unexpected),
            details={
                "unexpected": unexpected,
                "allowlist": sorted(allowlist),
            },
        )
    if not normalised:
        _fail("invalid_observation_contract", "observation field list is empty")
    return normalised, "declared"


def _signed_displacement(sample: Mapping[str, Any]) -> float:
    value = _first_present(
        sample.get("signed_displacement_m"),
        sample.get("planned_signed_lateral_m"),
    )
    if value is None and isinstance(sample.get("target"), Mapping):
        value = _first_present(
            sample["target"].get("signed_displacement_m"),
            sample["target"].get("planned_signed_lateral_m"),
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(
            "invalid_target",
            f"sample {sample.get('sample_id')!r} has no numeric signed displacement",
        )
    result = float(value)
    if not math.isfinite(result):
        _fail(
            "invalid_target",
            f"sample {sample.get('sample_id')!r} has non-finite displacement",
        )
    return result


def _delta_uv(sample: Mapping[str, Any]) -> tuple[float, float]:
    value = sample.get("planned_delta_patient_uv_m")
    if value is None and isinstance(sample.get("target"), Mapping):
        value = sample["target"].get("planned_delta_patient_uv_m")
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all():
        _fail(
            "invalid_target",
            f"sample {sample.get('sample_id')!r} has no finite 2-D UV target",
        )
    return float(array[0]), float(array[1])


def _empty_statistics() -> dict[str, Any]:
    return {
        "samples": 0,
        "episodes": 0,
        "stop": 0,
        "continue": 0,
        "positive": 0,
        "negative": 0,
        "near_zero": 0,
        "by_uv_direction": Counter(),
        "by_source": Counter(),
        "by_scenario": Counter(),
        "_episodes": set(),
    }


def _finalise_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    statistics = dict(statistics)
    statistics["episodes"] = len(statistics.pop("_episodes"))
    statistics["by_source"] = dict(sorted(statistics["by_source"].items()))
    statistics["by_scenario"] = dict(sorted(statistics["by_scenario"].items()))
    statistics["by_uv_direction"] = dict(
        sorted(statistics["by_uv_direction"].items())
    )
    return statistics


def _update_statistics(
    statistics: dict[str, Any],
    *,
    sample: Mapping[str, Any],
    episode: str,
    source: str,
    scenario: str,
    stop: bool,
    displacement: float,
    delta_uv: tuple[float, float] | None,
    near_zero_threshold_m: float,
) -> None:
    statistics["samples"] += 1
    statistics["_episodes"].add(episode)
    statistics["by_source"][source] += 1
    statistics["by_scenario"][scenario] += 1
    if stop:
        statistics["stop"] += 1
        return
    statistics["continue"] += 1
    if delta_uv is not None:
        u, v = delta_uv
        u_name = "+u" if u >= near_zero_threshold_m else "-u" if u <= -near_zero_threshold_m else "u0"
        v_name = "+v" if v >= near_zero_threshold_m else "-v" if v <= -near_zero_threshold_m else "v0"
        statistics["by_uv_direction"][f"{u_name}/{v_name}"] += 1
    if abs(displacement) < near_zero_threshold_m:
        statistics["near_zero"] += 1
    elif displacement > 0.0:
        statistics["positive"] += 1
    else:
        statistics["negative"] += 1


def audit_search_segment_index(
    index_path: Path | str,
    *,
    observation_allowlist: Iterable[str] = SEARCH_SEGMENT_OBSERVATION_ALLOWLIST,
    near_zero_threshold_m: float = DEFAULT_NEAR_ZERO_THRESHOLD_M,
) -> dict[str, Any]:
    """Read and audit a search-segment index, raising on the first violation."""

    path = Path(index_path).expanduser().resolve(strict=False)
    if not path.is_file():
        _fail("missing_index", f"search-segment index does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail("invalid_index", f"cannot read {path}: {exc}")
    if not isinstance(payload, Mapping):
        _fail("invalid_index", "search-segment index must contain a JSON object")

    raw_samples = payload.get("samples")
    samples_sequence = _as_sequence(raw_samples, "samples")
    samples: list[Mapping[str, Any]] = []
    samples_by_id: dict[str, Mapping[str, Any]] = {}
    for position, raw_sample in enumerate(samples_sequence):
        if not isinstance(raw_sample, Mapping):
            _fail("invalid_index", f"samples[{position}] must be an object")
        sample_id = _nonempty_string(raw_sample.get("sample_id"), "sample_id")
        if sample_id in samples_by_id:
            _fail(
                "duplicate_sample_id",
                f"sample_id {sample_id!r} occurs more than once in samples",
                details={"sample_id": sample_id},
            )
        samples.append(raw_sample)
        samples_by_id[sample_id] = raw_sample
    if not samples:
        _fail("invalid_index", "samples must not be empty")

    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, Mapping) or not raw_splits:
        _fail("invalid_index", "splits must be a non-empty object")
    split_by_sample_id: dict[str, str] = {}
    split_order: list[str] = []
    for raw_split, raw_ids in raw_splits.items():
        split = _nonempty_string(raw_split, "split name")
        split_order.append(split)
        local_ids: set[str] = set()
        for raw_sample_id in _as_sequence(raw_ids, f"splits.{split}"):
            sample_id = _nonempty_string(raw_sample_id, f"splits.{split} sample_id")
            if sample_id in local_ids:
                _fail(
                    "duplicate_split_sample_id",
                    f"sample_id {sample_id!r} is repeated within split {split!r}",
                    details={"sample_id": sample_id, "split": split},
                )
            local_ids.add(sample_id)
            previous_split = split_by_sample_id.get(sample_id)
            if previous_split is not None and previous_split != split:
                _fail(
                    "sample_id_cross_split",
                    f"sample_id {sample_id!r} occurs in both {previous_split!r} "
                    f"and {split!r}",
                    details={
                        "sample_id": sample_id,
                        "first_split": previous_split,
                        "second_split": split,
                    },
                )
            split_by_sample_id[sample_id] = split

    unknown_ids = sorted(set(split_by_sample_id) - set(samples_by_id))
    if unknown_ids:
        _fail(
            "unknown_split_sample_id",
            "splits reference unknown sample ids: " + ", ".join(unknown_ids),
            details={"sample_ids": unknown_ids},
        )
    unassigned_ids = sorted(set(samples_by_id) - set(split_by_sample_id))
    if unassigned_ids:
        _fail(
            "unassigned_sample_id",
            "samples are absent from all splits: " + ", ".join(unassigned_ids),
            details={"sample_ids": unassigned_ids},
        )

    try:
        threshold = float(near_zero_threshold_m)
    except (TypeError, ValueError):
        _fail("invalid_threshold", "near_zero_threshold_m must be numeric")
    if not math.isfinite(threshold) or threshold <= 0.0:
        _fail("invalid_threshold", "near_zero_threshold_m must be finite and positive")
    allowlist = frozenset(
        _normalise_field_name(_nonempty_string(field, "observation allowlist field"))
        for field in observation_allowlist
    )
    if not allowlist:
        _fail("invalid_observation_contract", "observation allowlist is empty")

    episode_seen: dict[str, tuple[str, str]] = {}
    path_seen: dict[str, tuple[str, str]] = {}
    scenario_seen: dict[str, tuple[str, str]] = {}
    episode_to_path: dict[str, str] = {}
    episode_to_scenario: dict[str, str] = {}
    metadata_cache: dict[Path, dict[str, Any]] = {}
    scenario_by_sample_id: dict[str, str] = {}
    source_by_sample_id: dict[str, str] = {}
    target_by_sample_id: dict[
        str, tuple[bool, float, tuple[float, float] | None]
    ] = {}
    target_contract = str(payload.get("target_contract", "signed_lateral_1d"))
    if target_contract not in {"signed_lateral_1d", "planned_delta_patient_uv_2d"}:
        _fail("invalid_target_contract", f"unsupported target contract: {target_contract}")
    maximum_displacement_m = float(payload.get("maximum_displacement_m", math.inf))
    stop_gate = payload.get("stop_gate")
    replay_stop_gate = bool(
        isinstance(stop_gate, Mapping) and isinstance(stop_gate.get("thresholds"), Mapping)
    )
    signal_thresholds = (
        PulseSignalQualityThresholds.from_mapping(stop_gate.get("thresholds"))
        if replay_stop_gate
        else None
    )
    spatial_gate = stop_gate.get("spatial_gate", {}) if replay_stop_gate else {}
    spatial_thresholds = (
        PulseTactileCoverageThresholds.from_mapping(spatial_gate.get("thresholds"))
        if isinstance(spatial_gate, Mapping) and spatial_gate.get("enabled", False)
        else None
    )
    stop_gate_samples_checked = 0

    # Identity leakage checks precede the observation declaration check.  This
    # lets the legacy v1 index expose its real scenario-template leakage even
    # though v1 did not persist an explicit observation_fields declaration.
    for sample in samples:
        sample_id = str(sample["sample_id"])
        split = split_by_sample_id[sample_id]
        declared_split = sample.get("split")
        if declared_split is not None and declared_split != split:
            _fail(
                "sample_split_mismatch",
                f"sample {sample_id!r} declares split {declared_split!r} but is "
                f"listed under {split!r}",
                details={
                    "sample_id": sample_id,
                    "declared_split": declared_split,
                    "index_split": split,
                },
            )
        episode = _nonempty_string(sample.get("episode"), "episode", sample=sample_id)
        resolved_path = _resolve_episode_path(sample, index_directory=path.parent)
        resolved_string = str(resolved_path)

        _identity_cross_split(
            episode_seen,
            identity=episode,
            split=split,
            sample_id=sample_id,
            code="episode_cross_split",
            label="episode",
        )
        previous_path = episode_to_path.get(episode)
        if previous_path is not None and previous_path != resolved_string:
            _fail(
                "episode_path_mismatch",
                f"episode {episode!r} maps to both {previous_path!r} and "
                f"{resolved_string!r}",
                details={
                    "episode": episode,
                    "first_resolved_path": previous_path,
                    "second_resolved_path": resolved_string,
                },
            )
        episode_to_path.setdefault(episode, resolved_string)

        _identity_cross_split(
            path_seen,
            identity=resolved_string,
            split=split,
            sample_id=sample_id,
            code="resolved_path_cross_split",
            label="resolved_path",
        )

        metadata = metadata_cache.get(resolved_path)
        if metadata is None:
            metadata = _read_episode_identity(resolved_path)
            metadata_cache[resolved_path] = metadata
        scenario, scenario_identities = _scenario_identities(sample, metadata)
        previous_scenario = episode_to_scenario.get(episode)
        if previous_scenario is not None and previous_scenario != scenario:
            _fail(
                "episode_scenario_mismatch",
                f"episode {episode!r} maps to inconsistent scenario groups",
                details={
                    "episode": episode,
                    "first_scenario": previous_scenario,
                    "second_scenario": scenario,
                },
            )
        episode_to_scenario.setdefault(episode, scenario)
        for kind, identity in scenario_identities:
            _identity_cross_split(
                scenario_seen,
                identity=f"{kind}:{identity}",
                split=split,
                sample_id=sample_id,
                code="scenario_group_cross_split",
                label="scenario_group",
            )

        source = _nonempty_string(sample.get("source"), "source", sample=sample_id)
        stop = sample.get("stop")
        if not isinstance(stop, bool):
            _fail(
                "invalid_target",
                f"sample {sample_id!r} stop must be a boolean",
            )
        if replay_stop_gate:
            signal_metrics = sample.get("observable_signal_quality")
            if not isinstance(signal_metrics, Mapping):
                _fail(
                    "missing_stop_gate_metrics",
                    f"sample {sample_id!r} lacks observable signal metrics",
                )
            spatial_metrics = sample.get("observable_spatial_quality")
            try:
                replayed_stop, replay_reasons = (
                    pulse_diagnosis_quality_gate_from_metrics(
                        signal_metrics,
                        signal_thresholds=signal_thresholds,
                        spatial_metrics=(
                            spatial_metrics
                            if isinstance(spatial_metrics, Mapping)
                            else None
                        ),
                        spatial_thresholds=spatial_thresholds,
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                _fail(
                    "invalid_stop_gate_metrics",
                    f"sample {sample_id!r} stop gate cannot be replayed: {error}",
                )
            if replayed_stop != stop:
                _fail(
                    "stop_gate_label_mismatch",
                    f"sample {sample_id!r} stop={stop} but hard gate returns "
                    f"{replayed_stop}: {list(replay_reasons)}",
                )
            stop_gate_samples_checked += 1
        displacement = _signed_displacement(sample)
        delta_uv = (
            _delta_uv(sample)
            if target_contract == "planned_delta_patient_uv_2d"
            else None
        )
        if delta_uv is not None:
            if not math.isclose(displacement, delta_uv[0], abs_tol=1e-9):
                _fail(
                    "invalid_target",
                    f"sample {sample_id!r} signed displacement disagrees with UV u",
                )
            norm = math.hypot(*delta_uv)
            if norm > maximum_displacement_m + 1e-9:
                _fail(
                    "target_out_of_bounds",
                    f"sample {sample_id!r} 2-D displacement exceeds maximum step",
                )
            if stop and norm > 1e-9:
                _fail(
                    "invalid_stop_target",
                    f"sample {sample_id!r} stop target must have zero UV displacement",
                )
        scenario_by_sample_id[sample_id] = scenario
        source_by_sample_id[sample_id] = source
        target_by_sample_id[sample_id] = (stop, displacement, delta_uv)

    observation_fields, observation_contract_source = _observation_contract(
        payload, samples, allowlist=allowlist
    )

    overall = _empty_statistics()
    by_split = {split: _empty_statistics() for split in split_order}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        split = split_by_sample_id[sample_id]
        stop, displacement, delta_uv = target_by_sample_id[sample_id]
        arguments = {
            "sample": sample,
            "episode": str(sample["episode"]),
            "source": source_by_sample_id[sample_id],
            "scenario": scenario_by_sample_id[sample_id],
            "stop": stop,
            "displacement": displacement,
            "delta_uv": delta_uv,
            "near_zero_threshold_m": threshold,
        }
        _update_statistics(overall, **arguments)
        _update_statistics(by_split[split], **arguments)

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "index": str(path),
        "index_schema_version": payload.get("schema_version"),
        "passed": True,
        "near_zero_threshold_m": threshold,
        "direction_population": "continue_samples_only",
        "target_contract": target_contract,
        "stop_gate_replay": {
            "enabled": replay_stop_gate,
            "samples_checked": stop_gate_samples_checked,
            "all_labels_match": (
                stop_gate_samples_checked == len(samples)
                if replay_stop_gate
                else None
            ),
        },
        "observation_contract": {
            "source": observation_contract_source,
            "fields": sorted(observation_fields),
            "allowlist": sorted(allowlist),
            "privileged_keys_found": [],
        },
        "split_invariants": {
            "sample_id_disjoint": True,
            "episode_disjoint": True,
            "resolved_path_disjoint": True,
            "scenario_group_disjoint": True,
        },
        "statistics": {
            "overall": _finalise_statistics(overall),
            "by_split": {
                split: _finalise_statistics(by_split[split]) for split in split_order
            },
        },
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast audit of a search-segment dataset index."
    )
    parser.add_argument("index", type=Path)
    parser.add_argument(
        "--near-zero-mm",
        type=float,
        default=DEFAULT_NEAR_ZERO_THRESHOLD_M * 1e3,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_search_segment_index(
        args.index, near_zero_threshold_m=args.near_zero_mm * 1e-3
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    _main()
