from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


TACTILE_NORMALIZATION_SCHEMA_VERSION = "adam_pulse_tactile_normalization_v1"
FULL56_FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
FULL56_CHANNEL_COUNTS = (16, 10, 10, 10, 10)
FULL56_OFFSETS = (0, 16, 26, 36, 46)


@dataclass(frozen=True)
class TactileNormalization:
    mode: str
    center: np.ndarray
    scale: np.ndarray
    finger_names: tuple[str, ...]
    channel_counts: tuple[int, ...]
    source_split: str = "train"
    epsilon: float = 1e-6
    scale_fallback: str = "none"
    fallback_channels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"standard", "median_mad"}:
            raise ValueError(f"unsupported tactile normalization: {self.mode}")
        center = np.asarray(self.center, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        channels = sum(int(value) for value in self.channel_counts)
        if center.shape != (channels,) or scale.shape != (channels,):
            raise ValueError("tactile normalization vector shape mismatch")
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("tactile normalization must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("tactile normalization scale must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    def normalize(self, values: np.ndarray, *, clip: float = 8.0) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != self.center.shape[0]:
            raise ValueError("tactile input channel count does not match statistics")
        result = (array - self.center) / self.scale
        return np.clip(result, -float(clip), float(clip)).astype(np.float32)

    def as_dict(self) -> dict:
        return {
            "schema_version": TACTILE_NORMALIZATION_SCHEMA_VERSION,
            "mode": self.mode,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "finger_names": list(self.finger_names),
            "channel_counts": list(self.channel_counts),
            "source_split": self.source_split,
            "epsilon": float(self.epsilon),
            "scale_fallback": self.scale_fallback,
            "fallback_channels": list(self.fallback_channels),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TactileNormalization":
        if payload.get("schema_version") != TACTILE_NORMALIZATION_SCHEMA_VERSION:
            raise ValueError("unsupported tactile normalization artifact")
        return cls(
            mode=str(payload["mode"]),
            center=np.asarray(payload["center"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            finger_names=tuple(payload["finger_names"]),
            channel_counts=tuple(int(x) for x in payload["channel_counts"]),
            source_split=str(payload.get("source_split", "train")),
            epsilon=float(payload.get("epsilon", 1e-6)),
            scale_fallback=str(payload.get("scale_fallback", "none")),
            fallback_channels=tuple(
                int(x) for x in payload.get("fallback_channels", ())
            ),
        )


def fit_tactile_normalization(
    samples: Iterable[np.ndarray],
    *,
    mode: str,
    finger_names: tuple[str, ...] = FULL56_FINGER_NAMES,
    channel_counts: tuple[int, ...] = FULL56_CHANNEL_COUNTS,
    epsilon: float = 1e-6,
) -> TactileNormalization:
    values = [np.asarray(sample, dtype=np.float32).reshape(-1) for sample in samples]
    if not values:
        raise ValueError("cannot fit tactile normalization without training samples")
    matrix = np.stack(values)
    channels = sum(channel_counts)
    if matrix.shape[1] != channels:
        raise ValueError(f"expected {channels} tactile channels, got {matrix.shape[1]}")
    if not np.isfinite(matrix).all():
        raise ValueError("tactile training samples contain non-finite values")
    if mode == "standard":
        center = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        fallback_channels: tuple[int, ...] = ()
        scale_fallback = "none"
    elif mode == "median_mad":
        center = np.median(matrix, axis=0)
        scale = 1.4826 * np.median(np.abs(matrix - center[None]), axis=0)
        standard_scale = matrix.std(axis=0)
        fallback = scale < float(epsilon)
        scale[fallback] = standard_scale[fallback]
        fallback_channels = tuple(np.flatnonzero(fallback).tolist())
        scale_fallback = "training_standard_deviation"
    else:
        raise ValueError(f"unsupported tactile normalization: {mode}")
    scale = np.maximum(scale, float(epsilon))
    return TactileNormalization(
        mode=mode,
        center=center,
        scale=scale,
        finger_names=finger_names,
        channel_counts=channel_counts,
        epsilon=epsilon,
        scale_fallback=scale_fallback,
        fallback_channels=fallback_channels,
    )


def save_tactile_normalization(
    path: Path | str, artifact: TactileNormalization
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact.as_dict(), indent=2) + "\n", encoding="utf-8")


def load_tactile_normalization(path: Path | str) -> TactileNormalization:
    return TactileNormalization.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
