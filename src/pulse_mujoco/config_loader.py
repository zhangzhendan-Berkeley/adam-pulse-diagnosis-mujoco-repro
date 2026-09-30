from __future__ import annotations

"""Load versioned YAML overlays without duplicating frozen base configs."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_project_config(path: Path | str) -> dict[str, Any]:
    """Load a full config or a small ``extends`` overlay.

    Relative base paths are resolved beside the overlay. A short ancestry
    guard rejects accidental recursive includes.
    """

    current = Path(path).resolve()
    ancestry: set[Path] = set()

    def load(item: Path) -> dict[str, Any]:
        resolved = item.resolve()
        if resolved in ancestry:
            raise ValueError(f"recursive config inheritance: {resolved}")
        ancestry.add(resolved)
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"project config must be a mapping: {resolved}")
        base_name = payload.pop("extends", None)
        result = payload
        if base_name is not None:
            base_path = Path(str(base_name))
            if not base_path.is_absolute():
                base_path = resolved.parent / base_path
            result = _deep_merge(load(base_path), payload)
        ancestry.remove(resolved)
        return result

    return load(current)
