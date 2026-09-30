from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .episode_data import validate_episode


DATASET_INDEX_SCHEMA_VERSION = "adam_pulse_dataset_index_v1"


def episode_split_counts(
    count: int, train_fraction: float, validation_fraction: float
) -> tuple[int, int, int]:
    if count <= 0:
        return 0, 0, 0
    fractions = np.asarray(
        [train_fraction, validation_fraction, 1.0 - train_fraction - validation_fraction],
        dtype=np.float64,
    )
    if np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("invalid split fractions")
    raw = fractions * count
    values = np.floor(raw).astype(int)
    for index in sorted(
        range(3), key=lambda item: (-(raw[item] - values[item]), item)
    )[: count - int(values.sum())]:
        values[index] += 1
    if count >= 3:
        for required in (1, 2):
            if values[required] == 0:
                donor = int(np.argmax(values))
                if values[donor] > 1:
                    values[donor] -= 1
                    values[required] += 1
    return tuple(int(value) for value in values)


def _read_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _statistics(root: Path, names: list[str]) -> dict:
    phase_counts: Counter[str] = Counter()
    actions, proximity, tactile_raw = [], [], []
    frames = 0
    for name in names:
        records = _read_records(root / name / "transitions.jsonl")
        frames += len(records)
        for record in records:
            phase_counts[str(record["task_phase_name"])] += 1
            actions.append(np.asarray(record["executed_action"], dtype=np.float64))
            proximity.append(np.asarray(record["proximity_m"], dtype=np.float64))
            tactile = np.load(root / name / record["tactile_path"])
            tactile_raw.append(tactile["raw"].reshape(-1).astype(np.float64))

    def describe(values: list[np.ndarray]) -> dict:
        if not values:
            return {"count": 0}
        array = np.asarray(values)
        return {
            "count": len(array),
            "mean": array.mean(axis=0).tolist(),
            "std": array.std(axis=0).tolist(),
            "minimum": array.min(axis=0).tolist(),
            "maximum": array.max(axis=0).tolist(),
            "percentile_01": np.percentile(array, 1, axis=0).tolist(),
            "percentile_99": np.percentile(array, 99, axis=0).tolist(),
        }

    return {
        "episodes": len(names),
        "frames": frames,
        "phase_counts": dict(sorted(phase_counts.items())),
        "executed_action": describe(actions),
        "proximity_m": describe(proximity),
        "tactile_raw": describe(tactile_raw),
    }


def build_dataset_index(
    root: Path | str,
    *,
    split_seed: int = 20260731,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    check_media: bool = True,
) -> dict:
    """Validate and split complete episodes without seed/profile leakage."""
    root = Path(root).resolve()
    accepted: list[dict] = []
    rejected: list[dict] = []
    for episode_dir in sorted(root.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        validation = validate_episode(episode_dir, check_media=check_media)
        metadata = json.loads(
            (episode_dir / "episode.json").read_text(encoding="utf-8")
        )
        reasons = list(validation.errors)
        if not metadata.get("success", False):
            reasons.append("episode success is false")
        if not metadata.get("signal_quality_accepted", False):
            reasons.append("signal quality was not accepted")
        if reasons:
            rejected.append({"episode": episode_dir.name, "reasons": sorted(set(reasons))})
            continue
        explicit_split_group = metadata.get("split_group")
        scenario = metadata.get("scenario")
        scenario_group = (
            scenario.get("scenario_id")
            if isinstance(scenario, dict)
            else None
        )
        split_group = str(
            explicit_split_group
            or scenario_group
            or (
                f"seed:{metadata.get('random_seed') or episode_dir.name}|"
                f"patient:{metadata.get('patient_profile', 'default')}|"
                f"pulse:{metadata.get('pulse_profile', 'default')}"
            )
        )
        accepted.append(
            {
                "episode": episode_dir.name,
                "frames": validation.frames,
                "random_seed": metadata.get("random_seed"),
                "split_group": split_group,
                "search_strategy": metadata.get("search_strategy"),
            }
        )

    groups: dict[str, list[str]] = defaultdict(list)
    for episode in accepted:
        groups[episode["split_group"]].append(episode["episode"])
    group_names = sorted(groups)
    np.random.default_rng(split_seed).shuffle(group_names)
    targets = episode_split_counts(
        len(accepted), train_fraction, validation_fraction
    )
    split_names = ("train", "validation", "test")
    splits = {name: [] for name in split_names}
    for group in group_names:
        # Assign a whole leakage group to the split furthest below its target.
        deficits = [targets[i] - len(splits[name]) for i, name in enumerate(split_names)]
        destination = split_names[int(np.argmax(deficits))]
        splits[destination].extend(sorted(groups[group]))
    all_names = [item["episode"] for item in accepted]
    index = {
        "schema_version": DATASET_INDEX_SCHEMA_VERSION,
        "root": str(root),
        "split_seed": int(split_seed),
        "split_unit": "seed_patient_pulse_group",
        "accepted_episodes": accepted,
        "rejected_episodes": rejected,
        "splits": splits,
        "statistics": _statistics(root, all_names),
        "training_statistics": _statistics(root, splits["train"]),
    }
    return index


def write_dataset_index(root: Path | str, index: dict) -> Path:
    path = Path(root) / "splits" / "dataset_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
