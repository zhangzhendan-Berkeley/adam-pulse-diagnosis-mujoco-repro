from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
from torch.utils.data import Sampler


DEFAULT_EVENT_PROBABILITIES = {
    "uniform": 0.20,
    "fine_approach": 0.00,
    "active_translation": 0.15,
    "active_rotation": 0.10,
    "contact_transition": 0.15,
    "pressure_transition": 0.15,
    "search_move": 0.10,
    "recovery": 0.10,
    "steady_hold": 0.05,
}


class EventBalancedWindowSampler(Sampler[tuple[int, int]]):
    """Yield explicit episode/window indices with event-balanced batches."""

    def __init__(
        self,
        dataset,
        *,
        probabilities: Mapping[str, float] | None = None,
        num_samples: int | None = None,
        batch_size: int = 1,
        seed: int = 0,
        balance_directions: bool = False,
    ):
        self.dataset = dataset
        self.probabilities = dict(
            DEFAULT_EVENT_PROBABILITIES
            if probabilities is None
            else probabilities
        )
        unknown = set(self.probabilities) - {
            "uniform",
            *dataset.window_candidates_by_event,
        }
        if unknown:
            raise ValueError(f"unknown sampling events: {sorted(unknown)}")
        if any(value < 0 for value in self.probabilities.values()):
            raise ValueError("sampling probabilities must be non-negative")
        total = float(sum(self.probabilities.values()))
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"sampling probabilities must sum to 1, got {total}")
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        self.batch_size = int(batch_size)
        if self.num_samples < 1 or self.batch_size < 1:
            raise ValueError("num_samples and batch_size must be positive")
        self.seed = int(seed)
        self.balance_directions = bool(balance_directions)
        self.epoch = 0
        self._all = tuple(dataset.all_window_candidates)
        self._pools = {
            name: tuple(values)
            for name, values in dataset.window_candidates_by_event.items()
        }
        self._direction_pools = {
            name: self._group_by_direction(values)
            for name, values in self._pools.items()
        }
        if not self._all:
            raise ValueError("dataset contains no valid windows")
        # A valid curriculum may not contain every event (for example an
        # early-stop collection has no search/recovery windows). Do not turn
        # that missing mass into uniform samples: renormalize among pools that
        # actually exist, while keeping uniform available for coverage.
        available = {
            name
            for name in self.probabilities
            if name == "uniform" or self._pools.get(name)
        }
        available_mass = sum(
            value for name, value in self.probabilities.items() if name in available
        )
        if available_mass <= 0:
            raise ValueError("sampler has no available event probability mass")
        self.effective_probabilities = {
            name: (value / available_mass if name in available else 0.0)
            for name, value in self.probabilities.items()
        }
        self.dynamic_events = tuple(
            name
            for name in (
                "fine_approach",
                "active_translation",
                "active_rotation",
                "contact_transition",
                "pressure_transition",
                "search_move",
                "recenter",
                "recovery",
            )
            if self._pools.get(name)
        )
        if not self.dynamic_events:
            raise ValueError("dataset contains no dynamic event windows")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def _choose_from_event(self, rng: np.random.Generator, event: str):
        pool = self._all if event == "uniform" else self._pools.get(event, ())
        if self.balance_directions and event != "uniform":
            direction_pools = self._direction_pools.get(event, {})
            if direction_pools:
                buckets = tuple(direction_pools)
                bucket = buckets[int(rng.integers(len(buckets)))]
                pool = direction_pools[bucket]
        if not pool:
            pool = self._all
        return pool[int(rng.integers(len(pool)))]

    def _group_by_direction(self, candidates):
        """Group windows by dominant normalized action axis and sign."""

        grouped = {}
        horizon = int(self.dataset.observation_horizon)
        prediction = int(self.dataset.prediction_horizon)
        scale = np.asarray(self.dataset.action_scale, dtype=np.float64)
        for episode_index, start in candidates:
            records = self.dataset.episodes[episode_index][2]
            actions = np.asarray(
                [
                    self.dataset._action_target(record)
                    for record in records[
                        start + horizon : start + horizon + prediction
                    ]
                ],
                dtype=np.float64,
            )
            # Direction balancing concerns TCP translation/rotation only.
            # Pressure is sampled by its own transition event and must not
            # become the dominant "direction" merely because it changes.
            normalized = actions[:, :6] / scale[:6]
            flat_index = int(np.argmax(np.abs(normalized)))
            time_index, axis = np.unravel_index(flat_index, normalized.shape)
            value = float(normalized[time_index, axis])
            bucket = (
                "stationary"
                if abs(value) <= 1e-8
                else f"axis_{axis}_{'positive' if value > 0 else 'negative'}"
            )
            grouped.setdefault(bucket, []).append((episode_index, start))
        return {key: tuple(value) for key, value in grouped.items()}

    def __iter__(self) -> Iterator[tuple[int, int]]:
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        names = tuple(self.effective_probabilities)
        probability = np.asarray(
            [self.effective_probabilities[name] for name in names], dtype=float
        )
        produced = 0
        while produced < self.num_samples:
            current_batch = min(self.batch_size, self.num_samples - produced)
            selections = []
            dynamic_event = self.dynamic_events[
                int(rng.integers(len(self.dynamic_events)))
            ]
            selections.append(self._choose_from_event(rng, dynamic_event))
            for _ in range(1, current_batch):
                event = names[int(rng.choice(len(names), p=probability))]
                selections.append(self._choose_from_event(rng, event))
            rng.shuffle(selections)
            yield from selections
            produced += current_batch
