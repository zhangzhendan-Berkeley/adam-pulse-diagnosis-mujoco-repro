from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from pulse_mujoco.config_loader import load_project_config
from pulse_mujoco.dataset_index import episode_split_counts
from pulse_mujoco.pulse_signal_quality import (
    PULSE_DIAGNOSIS_QUALITY_GATE_SCHEMA_VERSION,
    PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS,
    PULSE_SIGNAL_QUALITY_SCHEMA_VERSION,
    PULSE_TACTILE_COVERAGE_BASE_OBSERVABLE_FIELDS,
    PULSE_TACTILE_COVERAGE_OBSERVABLE_FIELDS,
    PulseSignalQualityThresholds,
    PulseTactileCoverageThresholds,
)
from pulse_mujoco.search_dagger import (
    initialized_policy_retention_episode_audit,
    initialized_policy_retention_sample_audit,
    onpolicy_branch_reviewed_sample_audit,
    onpolicy_branch_sample_audit,
    paired_dagger_first_state_episode_audit,
    paired_dagger_first_state_sample_audit,
    paired_dagger_hard_sample_audit,
)


SCHEMA_VERSION_1D = "adam_pulse_search_segment_v2"
SCHEMA_VERSION_2D = "adam_pulse_search_segment_v3_2d_centering"
PLAN_SCHEMA_VERSION_2D = "pulse_search_segment_plan_v3_2d_centering"
INITIALIZED_POLICY_RETENTION_HORIZON = 32
INITIALIZED_POLICY_RETENTION_STRIDE = 2
OBSERVATION_FIELDS = (
    "vision",
    "tactile",
    "tactile_valid",
    "proximity",
    "proximity_valid",
    "robot_state",
    "action",
    "task_phase",
)
FORBIDDEN_POLICY_FIELDS = (
    "pulse_center_m",
    "expert_guidance_direction_xy",
    "pulse_weighted_taxel_center_m",
    "expert_label_action",
)


def initialized_policy_checkpoint_contract(
    checkpoint: Path, *, observation_horizon: int
) -> dict:
    """Audit the exact initialized checkpoint used by retention episodes."""

    path = checkpoint.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing initialized policy checkpoint: {path}")
    report_path = path.with_suffix(".json")
    if not report_path.is_file():
        raise FileNotFoundError(
            f"missing initialized policy checkpoint report: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model_horizon = int((report.get("model_config") or {}).get("maximum_horizon", -1))
    runtime = dict(report.get("runtime_config") or {})
    runtime_horizon = int(runtime.get("observation_horizon", -1))
    runtime_stride = int(runtime.get("observation_stride_policy_steps", -1))
    expected_horizon = int(INITIALIZED_POLICY_RETENTION_HORIZON)
    if int(observation_horizon) != expected_horizon:
        raise ValueError(
            "initialized-policy retention requires observation horizon "
            f"{expected_horizon}, got {observation_horizon}"
        )
    if model_horizon != expected_horizon or runtime_horizon != expected_horizon:
        raise ValueError(
            "initialized checkpoint horizon contract differs: "
            f"model={model_horizon}, runtime={runtime_horizon}, "
            f"expected={expected_horizon}"
        )
    if runtime_stride != INITIALIZED_POLICY_RETENTION_STRIDE:
        raise ValueError(
            "initialized checkpoint observation stride differs: "
            f"{runtime_stride} != {INITIALIZED_POLICY_RETENTION_STRIDE}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": "adam_pulse_initialized_policy_retention_v2",
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "checkpoint_report": str(report_path.resolve()),
        "observation_horizon": expected_horizon,
        "observation_stride_policy_steps": INITIALIZED_POLICY_RETENTION_STRIDE,
        "label_source": "initialized_policy_retention",
        "target_semantics": "raw_initialized_policy_model_delta_uv_m",
        "deployment_audit": (
            "radial_clip(raw_model_delta_uv_m, maximum_step_m) "
            "== deployed_delta_uv_m == executed_delta_uv_m"
        ),
        "stop_samples_in_action_training": False,
        "truth_in_observation": False,
    }


def accepted_onpolicy_collection_episodes(data_root: Path) -> set[str]:
    """Return accepted branch directory names under the recorded contract.

    Collection v1 required both the full parent and branch to succeed.  The
    v2 collector records the parent only as stable-snapshot provenance, so an
    independently safe, audited student branch is accepted on its own.
    """

    summary_path = (
        data_root.parent / "collection_summary.json"
        if data_root.name == "branches"
        else data_root / "collection_summary.json"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"missing on-policy collection summary: {summary_path}"
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version not in {
        "adam_pulse_search_onpolicy_branch_collection_v1",
        "adam_pulse_search_onpolicy_branch_collection_v2",
    }:
        raise ValueError("unsupported on-policy collection summary schema")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != int(
        payload.get("episode_count", -1)
    ):
        raise ValueError("on-policy collection summary record count differs")
    accepted: set[str] = set()
    for record in records:
        episode = str(record.get("episode", ""))
        parent = record.get("parent") or {}
        branch = record.get("branch") or {}
        top_accepted = record.get("accepted") is True
        if schema_version.endswith("_v1"):
            contract_accepted = bool(
                parent.get("accepted") is True
                and branch.get("accepted") is True
            )
        else:
            contract_accepted = branch.get("accepted") is True
        if top_accepted != contract_accepted:
            raise ValueError(
                f"on-policy collection acceptance mismatch: {episode}"
            )
        if top_accepted:
            accepted.add(f"episode_{episode}")
    return accepted


def contiguous_blocks(indices: list[int]) -> list[list[int]]:
    blocks: list[list[int]] = []
    for value in indices:
        if not blocks or value != blocks[-1][-1] + 1:
            blocks.append([value])
        else:
            blocks[-1].append(value)
    return blocks


def scenario_group(metadata: dict, episode_dir: Path) -> str:
    profile = str(metadata.get("scenario_profile", "unspecified"))
    scenario = metadata.get("scenario")
    if profile in {
        "ch5_multifinger_train_v1",
        "ch5_multifinger_axial_balance_v1",
        "ch5_multifinger_positive_axial_v1",
        "ch5_multifinger_approach_axial_v1",
        "ch5_search_recovery_gate_v1",
    } and isinstance(scenario, dict):
        # Pair ids were introduced after the first formal collection started.
        # Hashing the exact nuisance template (while excluding only the two
        # intended offset axes and the display id) keeps both old and new
        # paired episodes in one split without conflating different draws.
        nuisance = {
            key: value
            for key, value in scenario.items()
            if key not in {
                "scenario_id",
                "lateral_offset_mm",
                "axial_offset_mm",
                "approach_axial_offset_mm",
            }
        }
        encoded = json.dumps(nuisance, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        return f"profile:{profile}|nuisance:{digest}"
    if isinstance(scenario, dict) and scenario.get("scenario_id") is not None:
        # Deliberately omit the collection root: repeated scenario templates
        # from observable/privileged runs must remain in the same split.
        return f"profile:{profile}|scenario:{scenario['scenario_id']}"
    if metadata.get("split_group") is not None:
        return f"upstream:{metadata['split_group']}"
    return f"episode:{episode_dir.resolve()}"


def assign_group_splits(
    episode_groups: dict[str, str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for episode, group in episode_groups.items():
        groups[group].append(episode)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    targets = episode_split_counts(
        len(episode_groups), train_fraction, validation_fraction
    )
    split_names = ("train", "validation", "test")
    split_episodes = {name: [] for name in split_names}
    for group in names:
        deficits = [
            targets[index] - len(split_episodes[name])
            for index, name in enumerate(split_names)
        ]
        destination = split_names[int(np.argmax(deficits))]
        split_episodes[destination].extend(sorted(groups[group]))
    return {
        episode: split
        for split, episodes in split_episodes.items()
        for episode in episodes
    }


def assign_episode_splits(
    episode_groups: dict[str, str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    sealed_all_test: bool = False,
) -> dict[str, str]:
    if sealed_all_test:
        return {episode: "test" for episode in episode_groups}
    return assign_group_splits(
        episode_groups,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )


def assign_ch5_stratified_sample_splits(
    samples: list[dict], *, seed: int
) -> None:
    """Keep nuisance pairs together while covering each geometry family."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["scenario_group"])].append(sample)
    by_family: dict[str, list[str]] = defaultdict(list)
    for group, group_samples in grouped.items():
        families = set()
        for sample in group_samples:
            lateral = abs(float(sample.get("lateral_offset_mm") or 0.0))
            axial = abs(float(sample.get("axial_offset_mm") or 0.0))
            axial = max(
                axial,
                abs(float(sample.get("approach_axial_offset_mm") or 0.0)),
            )
            if lateral < 1e-9 and axial < 1e-9:
                families.add("nominal")
            elif lateral >= 1e-9 and axial >= 1e-9:
                families.add("corner")
            elif axial >= 1e-9:
                families.add("axial")
            else:
                families.add("lateral")
        if len(families) != 1:
            raise ValueError(
                f"CH5 nuisance group mixes geometry families: {group}: {families}"
            )
        by_family[families.pop()].append(group)
    rng = np.random.default_rng(int(seed))
    assignment: dict[str, str] = {}
    for family in sorted(by_family):
        groups = sorted(by_family[family])
        rng.shuffle(groups)
        if len(groups) < 3:
            raise ValueError(
                f"CH5 stratified split needs at least three {family} groups, "
                f"got {len(groups)}"
            )
        assignment[groups[0]] = "validation"
        assignment[groups[1]] = "test"
        for group in groups[2:]:
            assignment[group] = "train"
    for sample in samples:
        sample["split"] = assignment[str(sample["scenario_group"])]


def _read_records(episode_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (episode_dir / "transitions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def _observable_signal_quality(summary: dict, *, sample_id: str) -> dict:
    """Extract only deployable current-window signal statistics."""

    quality = summary.get("signal_quality")
    if not isinstance(quality, dict):
        raise ValueError(f"missing signal quality for {sample_id}")
    if quality.get("schema_version") != PULSE_SIGNAL_QUALITY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported signal quality schema for {sample_id}: "
            f"{quality.get('schema_version')}"
        )
    missing = [
        field
        for field in PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS
        if field not in quality
    ]
    if missing:
        raise ValueError(
            f"missing observable signal quality fields for {sample_id}: {missing}"
        )
    result = {
        field: (
            int(quality[field])
            if field == "sample_count"
            else float(quality[field])
        )
        for field in PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS
    }
    if not np.isfinite(np.asarray(list(result.values()), dtype=float)).all():
        raise ValueError(f"non-finite observable signal quality for {sample_id}")
    return result


def _observable_spatial_quality(
    summary: dict, *, sample_id: str, required: bool = False
) -> dict | None:
    """Keep sensor-derived centering evidence outside the policy observation."""

    coverage = (summary.get("signal_quality") or {}).get("tactile_coverage")
    if not isinstance(coverage, dict):
        if required:
            raise ValueError(f"missing tactile coverage for {sample_id}")
        return None
    scalar_fields = (
        "central_energy_ratio",
        "weighted_centroid_offset_mm",
        "minimum_central_energy_ratio",
        "maximum_weighted_centroid_offset_mm",
        "minimum_per_finger_energy_fraction",
    )
    result = {field: float(coverage[field]) for field in scalar_fields}
    for field in (
        "minimum_per_finger_center_core_mean_kpa",
        "minimum_per_finger_bidirectional_decay_ratio",
    ):
        if field in coverage:
            result[field] = float(coverage[field])
    result["weighted_centroid_local_mm"] = [
        float(value) for value in coverage["weighted_centroid_local_mm"]
    ]
    result["per_finger_energy_fraction"] = [
        float(value) for value in coverage["per_finger_energy_fraction"]
    ]
    # These are measured tactile-window summaries, not pulse truth.  Keep
    # every field used by the optional strict per-finger coverage gates so a
    # rebuilt index can replay the deployment stop contract exactly.
    strict_fields = tuple(
        field
        for field in PULSE_TACTILE_COVERAGE_OBSERVABLE_FIELDS
        if field not in PULSE_TACTILE_COVERAGE_BASE_OBSERVABLE_FIELDS
    )
    for field in strict_fields:
        if field in coverage:
            result[field] = [float(value) for value in coverage[field]]
    values = np.asarray(
        list(result[field] for field in scalar_fields)
        + result["weighted_centroid_local_mm"]
        + result["per_finger_energy_fraction"]
        + [
            value
            for field in strict_fields
            for value in result.get(field, ())
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite tactile coverage for {sample_id}")
    return result


def _legacy_plan(
    records: list[dict],
    start: int,
    end: int,
    lateral: np.ndarray,
) -> dict | None:
    commands = [
        np.asarray(
            record.get("expert_label_action", record["executed_action"]),
            dtype=float,
        )[:2]
        for record in records[start:end]
        if record.get("label") in {"search_lift", "search_translate"}
    ]
    if not commands:
        return None
    delta = np.sum(commands, axis=0)
    return {
        "schema_version": "pulse_search_segment_plan_reconstructed_v1",
        "planned_delta_xy_m": delta.tolist(),
        "planned_signed_lateral_m": float(np.dot(delta, lateral)),
        "label_source": "reconstructed_legacy",
        "recovery_mode": None,
    }


def _validate_explicit_plan(
    plan: dict,
    *,
    expected_point: int,
    lateral: np.ndarray,
    maximum_m: float,
    axis: np.ndarray | None = None,
) -> dict:
    schema_version = plan.get("schema_version")
    if schema_version not in {"pulse_search_segment_plan_v2", PLAN_SCHEMA_VERSION_2D}:
        raise ValueError(f"unsupported explicit plan schema: {plan.get('schema_version')}")
    if int(plan.get("from_point_index", -1)) != expected_point:
        raise ValueError(
            f"segment plan point mismatch: {plan.get('from_point_index')} != {expected_point}"
        )
    sample = np.asarray(plan["sample_position_xy_m"], dtype=float)
    target = np.asarray(plan["planned_target_xy_m"], dtype=float)
    delta = np.asarray(plan["planned_delta_xy_m"], dtype=float)
    if sample.shape != (2,) or target.shape != (2,) or delta.shape != (2,):
        raise ValueError("segment plan XY fields must all have shape (2,)")
    np.testing.assert_allclose(delta, target - sample, atol=1e-9, rtol=0.0)
    signed = float(np.dot(delta, lateral))
    if not np.isclose(
        signed,
        float(plan["planned_signed_lateral_m"]),
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError("planned signed lateral displacement is inconsistent")
    axial = 0.0
    delta_uv = np.asarray([signed, 0.0], dtype=float)
    displacement = abs(signed)
    if schema_version == PLAN_SCHEMA_VERSION_2D:
        if axis is None:
            raise ValueError("2-D segment validation requires patient axis")
        plan_lateral = np.asarray(plan["patient_lateral_xy"], dtype=float)
        plan_axis = np.asarray(plan["patient_axis_xy"], dtype=float)
        if plan_lateral.shape != (2,) or plan_axis.shape != (2,):
            raise ValueError("plan patient axes must have shape (2,)")
        np.testing.assert_allclose(plan_lateral, lateral, atol=1e-9, rtol=0.0)
        np.testing.assert_allclose(plan_axis, axis, atol=1e-9, rtol=0.0)
        if not np.isclose(np.dot(plan_lateral, plan_axis), 0.0, atol=1e-9):
            raise ValueError("plan patient axes are not orthogonal")
        axial = float(np.dot(delta, axis))
        delta_uv = np.asarray(plan["planned_delta_patient_uv_m"], dtype=float)
        if delta_uv.shape != (2,):
            raise ValueError("planned_delta_patient_uv_m must have shape (2,)")
        np.testing.assert_allclose(
            delta_uv, [signed, axial], atol=1e-9, rtol=0.0
        )
        if not np.isclose(
            float(plan["planned_signed_axial_m"]), axial, atol=1e-9, rtol=0.0
        ):
            raise ValueError("planned signed axial displacement is inconsistent")
        displacement = float(np.linalg.norm(delta_uv))
        if not np.isclose(
            float(plan["planned_delta_norm_m"]), displacement, atol=1e-9, rtol=0.0
        ):
            raise ValueError("planned 2-D displacement norm is inconsistent")
    if displacement > maximum_m + 1e-9:
        limit_name = (
            "step" if schema_version == PLAN_SCHEMA_VERSION_2D else "radius"
        )
        raise ValueError(
            f"explicit planned segment exceeds deployment {limit_name}: "
            f"{displacement * 1000.0:.3f} mm"
        )
    return {
        "signed_lateral_m": signed,
        "signed_axial_m": axial,
        "delta_uv_m": delta_uv.tolist(),
        "norm_m": displacement,
        "schema_version": schema_version,
    }


def _is_terminal_stop_only_episode(
    blocks: list[list[int]], summaries: list[dict]
) -> bool:
    """Return whether an episode contains one accepted sample and no move.

    An episode that satisfies the signal-quality gate at its first sampled
    contact has no outgoing search segment by construction.  It is still a
    valid supervised stop sample and must not be rejected merely because its
    explicit segment-plan list is empty.
    """

    return bool(
        len(blocks) == 1
        and len(summaries) == 1
        and summaries[0].get("signal_quality", {}).get("accepted", False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build explicit next-search-segment supervision."
    )
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        required=True,
        help="accepted episode root; repeat for a zero-copy multi-source view",
    )
    parser.add_argument(
        "--alignment", type=Path, default=Path("configs/alignment_pro.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--maximum-displacement-mm", type=float, default=4.0)
    parser.add_argument("--split-seed", type=int, default=20260821)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--sealed-all-test",
        action="store_true",
        help=(
            "assign every scenario group to test for a pre-declared blind "
            "holdout; never use this index for training or calibration"
        ),
    )
    parser.add_argument(
        "--ch5-stratified-split",
        action="store_true",
        help=(
            "reserve nuisance-paired nominal/lateral/axial/corner groups in "
            "validation and test; requires at least three groups per family"
        ),
    )
    parser.add_argument(
        "--include-reconstructed-legacy",
        action="store_true",
        help="include approximate lift+translate labels from episodes predating plan-v2",
    )
    parser.add_argument(
        "--paired-dagger-hard-only",
        action="store_true",
        help=(
            "keep only truth-free student hard states whose explicit teacher "
            "counterfactual is privileged and exactly matches the dataset target"
        ),
    )
    parser.add_argument(
        "--paired-dagger-first-state-only",
        action="store_true",
        help=(
            "keep the truth-free first moving trusted-teacher state from each paired "
            "episode, including student-correct states for retention replay"
        ),
    )
    parser.add_argument(
        "--initialized-policy-retention-only",
        action="store_true",
        help=(
            "build train-only replay from every moving output of an accepted "
            "expert-free initialized-policy episode"
        ),
    )
    parser.add_argument(
        "--initialized-policy-checkpoint",
        type=Path,
        help=(
            "exact checkpoint recorded by initialized-policy retention episodes; "
            "required with --initialized-policy-retention-only"
        ),
    )
    parser.add_argument(
        "--onpolicy-branch-hard-only",
        action="store_true",
        help=(
            "accept only explicitly safe, train-only student-visited branch "
            "artifacts with privileged counterfactual labels"
        ),
    )
    parser.add_argument(
        "--onpolicy-branch-reviewed-only",
        action="store_true",
        help=(
            "accept every exactly labeled, truth-free reviewed proposal up to "
            "the safe on-policy branch takeover boundary"
        ),
    )
    parser.add_argument(
        "--onpolicy-branch-reviewed-first-state-only",
        action="store_true",
        help=(
            "accept only point zero from the audited reviewed on-policy "
            "branch prefix"
        ),
    )
    parser.add_argument(
        "--require-onpolicy-collection-acceptance",
        action="store_true",
        help=(
            "also require the source collection summary to accept the branch "
            "under its versioned acceptance contract; valid only with "
            "--onpolicy-branch-hard-only"
        ),
    )
    args = parser.parse_args()
    if args.history < 1:
        parser.error("--history must be positive")
    if sum(
        int(value)
        for value in (
            args.paired_dagger_hard_only,
            args.initialized_policy_retention_only,
            args.onpolicy_branch_hard_only,
            args.onpolicy_branch_reviewed_only,
            args.onpolicy_branch_reviewed_first_state_only,
        )
    ) > 1 or (args.paired_dagger_hard_only and args.paired_dagger_first_state_only):
        parser.error(
            "retention, paired DAgger, and on-policy filters are mutually exclusive"
        )
    if args.initialized_policy_retention_only != bool(
        args.initialized_policy_checkpoint
    ):
        parser.error(
            "--initialized-policy-retention-only and "
            "--initialized-policy-checkpoint must be supplied together"
        )
    if (
        args.require_onpolicy_collection_acceptance
        and not (
            args.onpolicy_branch_hard_only
            or args.onpolicy_branch_reviewed_only
            or args.onpolicy_branch_reviewed_first_state_only
        )
    ):
        parser.error(
            "--require-onpolicy-collection-acceptance requires an "
            "on-policy branch filter"
        )

    config = load_project_config(args.alignment)
    forearm = config["scene"]["patient_forearm"]
    axis = np.asarray(forearm["to_m"], dtype=float)[:2] - np.asarray(
        forearm["from_m"], dtype=float
    )[:2]
    axis /= np.linalg.norm(axis)
    lateral = np.asarray([-axis[1], axis[0]], dtype=float)
    maximum = float(args.maximum_displacement_mm) * 1e-3
    two_dimensional = bool(
        config.get("task", {}).get("pulse_search_two_dimensional_enabled", False)
    )
    output_schema_version = (
        SCHEMA_VERSION_2D if two_dimensional else SCHEMA_VERSION_1D
    )
    retention_contract = (
        initialized_policy_checkpoint_contract(
            args.initialized_policy_checkpoint,
            observation_horizon=args.history,
        )
        if args.initialized_policy_retention_only
        else None
    )
    if args.initialized_policy_retention_only and not two_dimensional:
        parser.error("initialized-policy retention requires the 2-D target contract")
    target_contract = (
        "planned_delta_patient_uv_2d"
        if two_dimensional
        else "signed_lateral_1d"
    )

    episodes: list[tuple[str, Path, dict, list[dict], list[list[int]]]] = []
    episode_groups: dict[str, str] = {}
    rejected_episodes: list[dict] = []
    retention_episode_audits: dict[str, dict] = {}
    # Preserve the caller-provided directory for collection-summary lookup.
    # A branch-only training view may intentionally symlink `branches` while
    # providing a newer audited summary alongside that view.  Episode reads
    # can resolve to the physical artifact directory, but the acceptance
    # contract belongs to the logical collection root supplied by the caller.
    collection_roots = [path.absolute() for path in args.data]
    data_roots = [path.resolve() for path in args.data]
    multiple_roots = len(data_roots) > 1
    for root_index, data_root in enumerate(data_roots):
        collection_accepted = (
            accepted_onpolicy_collection_episodes(collection_roots[root_index])
            if args.require_onpolicy_collection_acceptance
            else None
        )
        for link in sorted(data_root.glob("episode_*")):
            if not link.is_dir():
                continue
            episode_dir = link.resolve()
            episode_name = (
                f"source_{root_index:02d}_{link.name}"
                if multiple_roots
                else link.name
            )
            if collection_accepted is not None and link.name not in collection_accepted:
                rejected_episodes.append(
                    {
                        "episode": episode_name,
                            "reason": "onpolicy_branch_or_collection_rejected",
                    }
                )
                continue
            metadata_path = episode_dir / "episode.json"
            transitions_path = episode_dir / "transitions.jsonl"
            if not metadata_path.is_file() or not transitions_path.is_file():
                rejected_episodes.append(
                    {
                        "episode": episode_name,
                        "reason": "incomplete_episode_artifacts",
                        "episode_json_present": metadata_path.is_file(),
                        "transitions_present": transitions_path.is_file(),
                    }
                )
                continue
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            branch_contract = metadata.get("onpolicy_branch") or {}
            accepted_branch = bool(
                (
                    args.onpolicy_branch_hard_only
                    or args.onpolicy_branch_reviewed_only
                    or args.onpolicy_branch_reviewed_first_state_only
                )
                and branch_contract.get("schema_version")
                == "adam_pulse_search_onpolicy_branch_v1"
                and branch_contract.get("accepted") is True
            )
            paired_first_state_contract = metadata.get("paired_dagger") or {}
            accepted_first_state = bool(
                paired_first_state_contract.get("collection_mode")
                == "first_state_teacher_label"
                and paired_first_state_contract.get("accepted") is True
            )
            if not accepted_branch and not accepted_first_state and (
                not metadata.get("success", False)
                or not metadata.get("signal_quality_accepted", False)
            ):
                rejected_episodes.append(
                    {"episode": episode_name, "reason": "episode_not_accepted"}
                )
                continue
            records = _read_records(episode_dir)
            blocks = contiguous_blocks(
                [
                    index
                    for index, record in enumerate(records)
                    if record.get("label") == "search_sample"
                ]
            )
            summaries = metadata.get("search_sample_summaries") or []
            if not summaries or len(blocks) != len(summaries):
                rejected_episodes.append(
                    {
                        "episode": episode_name,
                        "reason": "sample_summary_block_mismatch",
                        "summaries": len(summaries),
                        "blocks": len(blocks),
                    }
                )
                continue
            if accepted_first_state:
                collection_audit = paired_dagger_first_state_episode_audit(
                    metadata
                )
                if not collection_audit["accepted"]:
                    rejected_episodes.append(
                        {
                            "episode": episode_name,
                            "reason": "paired_dagger_first_state_episode_filter",
                            "details": collection_audit["reasons"],
                        }
                    )
                    continue
            explicit = metadata.get("search_segment_plans") or []
            incompatible_plans = [
                plan.get("schema_version")
                for plan in explicit
                if (
                    two_dimensional
                    and plan.get("schema_version") != PLAN_SCHEMA_VERSION_2D
                )
                or (
                    not two_dimensional
                    and plan.get("schema_version") == PLAN_SCHEMA_VERSION_2D
                )
            ]
            if incompatible_plans:
                rejected_episodes.append(
                    {
                        "episode": episode_name,
                        "reason": "segment_plan_target_contract_mismatch",
                        "plan_schema_versions": incompatible_plans,
                    }
                )
                continue
            if args.initialized_policy_retention_only:
                actual_checkpoint = str(
                    Path(str(metadata.get("segment_checkpoint", ""))).resolve()
                )
                retention_metadata = dict(metadata)
                retention_metadata["segment_checkpoint"] = actual_checkpoint
                retention_audit = initialized_policy_retention_episode_audit(
                    retention_metadata,
                    expected_checkpoint=retention_contract["checkpoint"],
                    expected_horizon=retention_contract["observation_horizon"],
                    expected_stride=retention_contract[
                        "observation_stride_policy_steps"
                    ],
                    transition_count=len(records),
                    search_block_count=len(blocks),
                )
                retention_episode_audits[episode_name] = retention_audit
                if not retention_audit["accepted"]:
                    rejected_episodes.append(
                        {
                            "episode": episode_name,
                            "reason": "initialized_policy_retention_episode_filter",
                            "details": retention_audit["reasons"],
                        }
                    )
                    continue
            elif (
                not explicit
                and not args.include_reconstructed_legacy
                and not _is_terminal_stop_only_episode(blocks, summaries)
            ):
                rejected_episodes.append(
                    {
                        "episode": episode_name,
                        "reason": "missing_explicit_segment_plans",
                    }
                )
                continue
            episodes.append(
                (episode_name, episode_dir, metadata, records, blocks)
            )
            episode_groups[episode_name] = scenario_group(
                metadata, episode_dir
            )

    if not episodes:
        raise RuntimeError("no accepted episodes with usable search supervision")
    episode_split = assign_episode_splits(
        episode_groups,
        seed=args.split_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        sealed_all_test=args.sealed_all_test,
    )
    if (
        args.onpolicy_branch_hard_only
        or args.onpolicy_branch_reviewed_only
        or args.onpolicy_branch_reviewed_first_state_only
        or args.initialized_policy_retention_only
    ):
        episode_split = {episode: "train" for episode in episode_split}

    samples: list[dict] = []
    skipped: list[dict] = []
    for episode, episode_dir, metadata, records, blocks in episodes:
        summaries = metadata["search_sample_summaries"]
        explicit_by_point = {
            int(plan["from_point_index"]): plan
            for plan in (metadata.get("search_segment_plans") or [])
        }
        policy_by_point = {
            int(record["point_index"]): record
            for record in (metadata.get("search_segment_policy_records") or [])
        }
        for point, (summary, block) in enumerate(zip(summaries, blocks)):
            end = block[-1]
            if end + 1 < args.history:
                skipped.append(
                    {"episode": episode, "point": point, "reason": "insufficient_history"}
                )
                continue
            is_final = point == len(blocks) - 1
            stop = is_final and bool(
                summary.get("signal_quality", {}).get("accepted", False)
            )
            plan = explicit_by_point.get(point)
            retention_sample_audit = None
            planned_delta_xy_m = None
            executed_landing_delta_xy_m = None
            recovery_mode = None
            if args.initialized_policy_retention_only:
                if stop:
                    skipped.append(
                        {
                            "episode": episode,
                            "point": point,
                            "reason": "initialized_policy_retention_stop_excluded",
                        }
                    )
                    continue
                retention_sample_audit = (
                    initialized_policy_retention_sample_audit(
                        policy_by_point.get(point),
                        point_index=point,
                        maximum_displacement_m=maximum,
                    )
                )
                if not retention_sample_audit["accepted"]:
                    skipped.append(
                        {
                            "episode": episode,
                            "point": point,
                            "reason": "initialized_policy_retention_sample_filter",
                            "details": retention_sample_audit["reasons"],
                        }
                    )
                    continue
                delta_uv = list(retention_sample_audit["target_delta_uv_m"])
                signed = float(delta_uv[0])
                axial = float(delta_uv[1])
                delta_norm = float(np.linalg.norm(delta_uv))
                source = "initialized_policy_retention"
                plan_schema = None
                planned_delta_xy_m = (
                    signed * lateral + axial * axis
                ).tolist()
                executed_landing_delta_xy_m = policy_by_point[point].get(
                    "executed_landing_delta_xy_m"
                )
            else:
                if not stop and plan is None and point + 1 < len(blocks):
                    plan = _legacy_plan(
                        records, end + 1, blocks[point + 1][0], lateral
                    )
                if stop:
                    signed = 0.0
                    axial = 0.0
                    delta_uv = [0.0, 0.0]
                    delta_norm = 0.0
                    # A terminal zero-motion target is replayed from the
                    # deployable signal/spatial-quality gate.  Do not attribute
                    # it to an episode-level privileged-assist flag: the flag only
                    # says that oracle fallback was available, not that truth was
                    # read for this sample.
                    source = "deterministic_stop_gate"
                    plan_schema = None
                elif plan is None:
                    skipped.append(
                        {
                            "episode": episode,
                            "point": point,
                            "reason": "missing_segment_plan",
                        }
                    )
                    continue
                elif plan.get("schema_version") in {
                    "pulse_search_segment_plan_v2",
                    PLAN_SCHEMA_VERSION_2D,
                }:
                    validated_plan = _validate_explicit_plan(
                        plan,
                        expected_point=point,
                        lateral=lateral,
                        maximum_m=maximum,
                        axis=axis,
                    )
                    signed = float(validated_plan["signed_lateral_m"])
                    axial = float(validated_plan["signed_axial_m"])
                    delta_uv = list(validated_plan["delta_uv_m"])
                    delta_norm = float(validated_plan["norm_m"])
                    source = str(plan.get("label_source", "observable_expert"))
                    plan_schema = plan["schema_version"]
                    planned_delta_xy_m = plan.get("planned_delta_xy_m")
                    executed_landing_delta_xy_m = plan.get(
                        "executed_landing_delta_xy_m"
                    )
                    recovery_mode = plan.get("recovery_mode")
                else:
                    signed = float(plan["planned_signed_lateral_m"])
                    axial = 0.0
                    delta_uv = [signed, 0.0]
                    delta_norm = abs(signed)
                    if abs(signed) > maximum + 1e-9:
                        skipped.append(
                            {
                                "episode": episode,
                                "point": point,
                                "reason": "legacy_segment_out_of_bounds",
                                "value_m": signed,
                            }
                        )
                        continue
                    source = "reconstructed_legacy"
                    plan_schema = plan.get("schema_version")
                    planned_delta_xy_m = plan.get("planned_delta_xy_m")
                    executed_landing_delta_xy_m = plan.get(
                        "executed_landing_delta_xy_m"
                    )
                    recovery_mode = plan.get("recovery_mode")

            paired_hard_audit = None
            paired_first_state_audit = None
            onpolicy_branch_audit = None
            onpolicy_branch_reviewed_audit = None
            paired_contract = metadata.get("paired_dagger") or {}
            has_paired_contract = paired_contract.get("schema_version") == (
                "adam_pulse_search_gate_paired_dagger_v1"
            )
            if (
                (args.paired_dagger_hard_only or args.paired_dagger_first_state_only)
                and has_paired_contract
            ):
                if stop:
                    skipped.append(
                        {
                            "episode": episode,
                            "point": point,
                            "reason": (
                                "paired_dagger_stop_not_hard"
                                if args.paired_dagger_hard_only
                                else "paired_dagger_stop_not_first_state"
                            ),
                        }
                    )
                    continue
                if args.paired_dagger_hard_only:
                    paired_hard_audit = paired_dagger_hard_sample_audit(
                        policy_by_point.get(point), plan, delta_uv
                    )
                    audit = paired_hard_audit
                    reason = "paired_dagger_hard_filter"
                else:
                    paired_first_state_audit = (
                        paired_dagger_first_state_sample_audit(
                            policy_by_point.get(point),
                            plan,
                            delta_uv,
                            point_index=point,
                        )
                    )
                    audit = paired_first_state_audit
                    reason = "paired_dagger_first_state_filter"
                if not audit["accepted"]:
                    skipped.append(
                        {
                            "episode": episode,
                            "point": point,
                            "reason": reason,
                            "details": audit["reasons"],
                        }
                    )
                    continue
            elif args.onpolicy_branch_hard_only:
                onpolicy_branch_audit = onpolicy_branch_sample_audit(
                    policy_by_point.get(point),
                    plan,
                    delta_uv,
                    point_index=point,
                    branch_contract=metadata.get("onpolicy_branch"),
                )
                if not onpolicy_branch_audit["accepted"]:
                    skipped.append(
                        {
                            "episode": episode,
                            "point": point,
                            "reason": "onpolicy_branch_hard_filter",
                            "details": onpolicy_branch_audit["reasons"],
                        }
                    )
                    continue
            elif (
                args.onpolicy_branch_reviewed_only
                or args.onpolicy_branch_reviewed_first_state_only
            ):
                onpolicy_branch_reviewed_audit = (
                    onpolicy_branch_reviewed_sample_audit(
                        policy_by_point.get(point),
                        plan,
                        delta_uv,
                        point_index=point,
                        branch_contract=metadata.get("onpolicy_branch"),
                    )
                )
                if not onpolicy_branch_reviewed_audit["accepted"]:
                    skipped.append(
                        {
                            "episode": episode,
                            "point": point,
                            "reason": "onpolicy_branch_reviewed_filter",
                            "details": onpolicy_branch_reviewed_audit["reasons"],
                        }
                    )
                    continue
            elif args.paired_dagger_hard_only or args.paired_dagger_first_state_only:
                skipped.append(
                    {
                        "episode": episode,
                        "point": point,
                        "reason": "missing_paired_dagger_contract",
                    }
                )
                continue
                if args.onpolicy_branch_reviewed_first_state_only and point != 0:
                    skipped.append(
                        {
                            "episode": episode,
                            "point": point,
                            "reason": "onpolicy_branch_reviewed_not_first_state",
                        }
                    )
                    continue

            scenario = metadata.get("scenario") or {}
            sample_id = f"{episode}_point_{point:02d}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "episode": episode,
                    "episode_dir": str(episode_dir),
                    "episode_resolved_path": str(episode_dir.resolve()),
                    "scenario_group": episode_groups[episode],
                    "scenario_profile": metadata.get("scenario_profile"),
                    "scenario_id": scenario.get("scenario_id"),
                    "lateral_offset_mm": scenario.get("lateral_offset_mm"),
                    "axial_offset_mm": scenario.get("axial_offset_mm", 0.0),
                    "approach_axial_offset_mm": scenario.get(
                        "approach_axial_offset_mm", 0.0
                    ),
                    "source": source,
                    "split": episode_split[episode],
                    "point_index": point,
                    "observation_end_frame": end,
                    "stop": bool(stop),
                    "signed_displacement_m": signed,
                    "planned_signed_lateral_m": signed,
                    "planned_signed_axial_m": axial,
                    "planned_delta_patient_uv_m": delta_uv,
                    "planned_delta_norm_m": delta_norm,
                    "plan_schema_version": plan_schema,
                    "planned_delta_xy_m": planned_delta_xy_m,
                    "executed_landing_delta_xy_m": executed_landing_delta_xy_m,
                    "recovery_mode": recovery_mode,
                    "signal_quality_accepted": bool(
                        summary.get("signal_quality", {}).get("accepted", False)
                    ),
                    # This is an offline/deployment gate input, not part of
                    # the model observation.  It contains only statistics
                    # computed from the completed tactile/contact/force
                    # window at the current search point.
                    "observable_signal_quality": _observable_signal_quality(
                        summary, sample_id=sample_id
                    ),
                    "observable_spatial_quality": _observable_spatial_quality(
                        summary,
                        sample_id=sample_id,
                        required=two_dimensional,
                    ),
                    "sensor_centroid_offset_xy_m": summary.get(
                        "pulse_centered_taxel_offset_xy_m"
                    ),
                    "paired_dagger_hard_audit": paired_hard_audit,
                    "paired_dagger_first_state_audit": paired_first_state_audit,
                    "onpolicy_branch_audit": onpolicy_branch_audit,
                    "onpolicy_branch_reviewed_audit": (
                        onpolicy_branch_reviewed_audit
                    ),
                    "initialized_policy_retention_audit": (
                        retention_sample_audit
                    ),
                }
            )

    if not samples:
        raise RuntimeError("no valid search-segment samples were constructed")
    if args.ch5_stratified_split:
        if args.sealed_all_test:
            parser.error(
                "--ch5-stratified-split and --sealed-all-test are mutually exclusive"
            )
        profiles = {str(item.get("scenario_profile")) for item in samples}
        supported_ch5_profiles = {
            "ch5_multifinger_train_v1",
            "ch5_multifinger_axial_balance_v1",
            "ch5_multifinger_positive_axial_v1",
            "ch5_multifinger_approach_axial_v1",
        }
        if not profiles or not profiles.issubset(supported_ch5_profiles):
            raise ValueError(
                "CH5 stratified split requires only approved CH5 paired profiles"
            )
        assign_ch5_stratified_sample_splits(samples, seed=args.split_seed)
    if (
        args.paired_dagger_hard_only
        or args.paired_dagger_first_state_only
        or args.onpolicy_branch_hard_only
        or args.onpolicy_branch_reviewed_only
        or args.onpolicy_branch_reviewed_first_state_only
        or args.initialized_policy_retention_only
    ):
        # These are explicitly training-only paired replay views. Evaluation
        # remains on the frozen base validation/test IDs after incremental
        # merge, so do not split the small paired direction set.
        for sample in samples:
            sample["split"] = "train"
    sample_ids = [item["sample_id"] for item in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("duplicate search-segment sample IDs")

    by_split = {
        split: [item["sample_id"] for item in samples if item["split"] == split]
        for split in ("train", "validation", "test")
    }
    source_counts = Counter(item["source"] for item in samples)
    def direction_bucket(item: dict) -> str:
        if item["stop"]:
            return "stop"
        u, v = (float(value) for value in item["planned_delta_patient_uv_m"])
        threshold = 0.00025
        u_name = "+u" if u >= threshold else "-u" if u <= -threshold else "u0"
        v_name = "+v" if v >= threshold else "-v" if v <= -threshold else "v0"
        return f"{u_name}/{v_name}" if two_dimensional else u_name

    direction_counts = Counter(direction_bucket(item) for item in samples)
    quality_thresholds = PulseSignalQualityThresholds.from_mapping(
        config.get("task", {}).get("pulse_signal_quality")
    )
    coverage_config = config.get("task", {}).get("pulse_tactile_coverage", {})
    coverage_enabled = bool(coverage_config.get("enabled", False))
    coverage_thresholds = PulseTactileCoverageThresholds.from_mapping(
        coverage_config
    )
    payload = {
        "schema_version": output_schema_version,
        "source_dataset": (
            str(data_roots[0]) if len(data_roots) == 1 else None
        ),
        "source_datasets": [str(path) for path in data_roots],
        "alignment": str(args.alignment.resolve()),
        "state_dim": 15,
        "observation_horizon": int(args.history),
        "observation_action_semantics": "previous_action_policy_visible",
        "target_contract": target_contract,
        "observation_fields": list(OBSERVATION_FIELDS),
        "forbidden_policy_fields": list(FORBIDDEN_POLICY_FIELDS),
        "patient_lateral_xy": lateral.tolist(),
        "maximum_displacement_m": maximum,
        "near_zero_direction_threshold_m": 0.0005,
        "patient_axis_xy": axis.tolist(),
        "label_semantics": (
            "raw initialized-policy model delta in patient lateral/axial coordinates; "
            "deployment clipping and execution equality audited separately"
            if args.initialized_policy_retention_only
            else "explicit expert planned delta in patient lateral/axial coordinates"
            if two_dimensional
            else "explicit expert planned target projected onto patient lateral axis"
        ),
        "paired_dagger_hard_only": bool(args.paired_dagger_hard_only),
        "paired_dagger_first_state_only": bool(
            args.paired_dagger_first_state_only
        ),
        "initialized_policy_retention_only": bool(
            args.initialized_policy_retention_only
        ),
        "initialized_policy_retention": retention_contract,
        "initialized_policy_retention_episode_audits": retention_episode_audits,
        "onpolicy_branch_hard_only": bool(args.onpolicy_branch_hard_only),
        "onpolicy_branch_reviewed_only": bool(
            args.onpolicy_branch_reviewed_only
        ),
        "onpolicy_branch_reviewed_first_state_only": bool(
            args.onpolicy_branch_reviewed_first_state_only
        ),
        "onpolicy_collection_acceptance_required": bool(
            args.require_onpolicy_collection_acceptance
        ),
        "truth_in_observation": False,
        "stop_gate": {
            "schema_version": PULSE_DIAGNOSIS_QUALITY_GATE_SCHEMA_VERSION,
            "type": "deterministic_current_window_signal_and_spatial_quality",
            "input_scope": "completed_current_search_point_window",
            "policy_observation": False,
            "observable_fields": list(PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS),
            "thresholds": quality_thresholds.as_dict(),
            "spatial_gate": {
                "enabled": coverage_enabled,
                "observable_fields": list(
                    PULSE_TACTILE_COVERAGE_OBSERVABLE_FIELDS
                ),
                "thresholds": coverage_thresholds.as_dict(),
            },
        },
        "split_seed": None if args.sealed_all_test else int(args.split_seed),
        "split_unit": (
            "initialized_policy_retention_train_only"
            if args.initialized_policy_retention_only
            else "onpolicy_branch_train_only"
            if (
                args.onpolicy_branch_hard_only
                or args.onpolicy_branch_reviewed_only
                or args.onpolicy_branch_reviewed_first_state_only
            )
            else "scenario_group_sealed_all_test"
            if args.sealed_all_test
            else "scenario_group"
        ),
        "sealed_blind_holdout": bool(args.sealed_all_test),
        "samples": samples,
        "splits": by_split,
        "statistics": {
            "samples": len(samples),
            "continue": sum(not item["stop"] for item in samples),
            "stop": sum(item["stop"] for item in samples),
            "skipped": len(skipped),
            "rejected_episodes": len(rejected_episodes),
            "maximum_command_displacement_mm": max(
                float(item["planned_delta_norm_m"]) for item in samples
            )
            * 1e3,
            "by_source": dict(sorted(source_counts.items())),
            "by_direction": dict(sorted(direction_counts.items())),
            "by_split": {name: len(values) for name, values in by_split.items()},
        },
        "rejected_episodes": rejected_episodes,
        "skipped": skipped,
    }
    canonical = json.dumps(payload["samples"], ensure_ascii=False, sort_keys=True)
    payload["samples_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["statistics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
