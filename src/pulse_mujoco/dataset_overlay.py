from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable

from .dataset_index import _statistics, build_dataset_index, write_dataset_index


TRAIN_ONLY_OVERLAY_SCHEMA_VERSION = "adam_pulse_train_only_dataset_overlay_v1"


def supplement_episode_alias(source_root: Path, episode_name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", source_root.name).strip("_")
    suffix = re.sub(r"[^a-zA-Z0-9_]+", "_", episode_name).strip("_")
    if suffix.startswith("episode_"):
        suffix = suffix[len("episode_") :]
    if not token or not suffix:
        raise ValueError("supplement source and episode names must be non-empty")
    return f"episode_supplement_{token}_{suffix}"


def train_only_splits(
    base_splits: dict[str, list[str]], supplemental_names: Iterable[str]
) -> dict[str, list[str]]:
    required = {"train", "validation", "test"}
    if set(base_splits) != required:
        raise ValueError("base dataset must define train/validation/test splits")
    supplement = list(supplemental_names)
    existing = {name for values in base_splits.values() for name in values}
    overlap = sorted(existing.intersection(supplement))
    if overlap:
        raise ValueError(f"supplement aliases overlap base splits: {overlap}")
    return {
        "train": list(base_splits["train"]) + supplement,
        "validation": list(base_splits["validation"]),
        "test": list(base_splits["test"]),
    }


def _link_episode(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source, target_is_directory=True)


def build_train_only_dataset_overlay(
    *,
    base_root: Path,
    base_index_path: Path,
    supplemental_roots: Iterable[Path],
    output_root: Path,
    check_media: bool = True,
) -> dict:
    """Build a reversible symlink view with supplements confined to train."""
    base_root = base_root.resolve()
    base_index_path = base_index_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    base_index = json.loads(base_index_path.read_text(encoding="utf-8"))
    if Path(base_index["root"]).resolve() != base_root:
        raise ValueError("base index root does not match --base-data")

    output_root.mkdir(parents=True)
    base_names = [item["episode"] for item in base_index["accepted_episodes"]]
    for name in base_names:
        _link_episode(base_root / name, output_root / name)

    supplemental_names: list[str] = []
    supplemental_provenance: list[dict] = []
    for source_root_value in supplemental_roots:
        source_root = source_root_value.resolve()
        summary_path = source_root / "collection_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        accepted = [
            item["episode"]
            for item in summary.get("episodes", [])
            if bool(item.get("accepted", False))
        ]
        if not accepted:
            raise ValueError(f"supplement has no accepted episodes: {source_root}")
        aliases = []
        for name in accepted:
            alias = supplement_episode_alias(source_root, name)
            _link_episode(source_root / name, output_root / alias)
            supplemental_names.append(alias)
            aliases.append({"alias": alias, "source_episode": name})
        supplemental_provenance.append(
            {
                "root": str(source_root),
                "collection_summary": str(summary_path),
                "episodes": aliases,
            }
        )

    index = build_dataset_index(
        output_root,
        split_seed=int(base_index["split_seed"]),
        check_media=check_media,
    )
    expected = set(base_names + supplemental_names)
    observed = {item["episode"] for item in index["accepted_episodes"]}
    if observed != expected or index["rejected_episodes"]:
        raise RuntimeError(
            "overlay validation did not preserve all accepted episodes: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}, "
            f"rejected={index['rejected_episodes']}"
        )
    index["splits"] = train_only_splits(
        base_index["splits"], supplemental_names
    )
    index["training_statistics"] = _statistics(
        output_root, index["splits"]["train"]
    )
    index["overlay"] = {
        "schema_version": TRAIN_ONLY_OVERLAY_SCHEMA_VERSION,
        "base_root": str(base_root),
        "base_index": str(base_index_path),
        "base_split_counts": {
            name: len(values) for name, values in base_index["splits"].items()
        },
        "supplemental_episode_count": len(supplemental_names),
        "supplemental_episodes_train_only": list(supplemental_names),
        "supplemental_sources": supplemental_provenance,
        "validation_ids_preserved": (
            index["splits"]["validation"] == base_index["splits"]["validation"]
        ),
        "test_ids_preserved": (
            index["splits"]["test"] == base_index["splits"]["test"]
        ),
    }
    write_dataset_index(output_root, index)
    return index
