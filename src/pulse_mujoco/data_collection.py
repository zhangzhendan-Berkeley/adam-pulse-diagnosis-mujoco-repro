from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .episode_data import EpisodeValidation


COLLECTION_MANIFEST_SCHEMA_VERSION = "adam_pulse_collection_manifest_v1"


@dataclass(frozen=True)
class CollectionScenario:
    scenario_id: str
    lateral_offset_mm: float
    tactile_noise_std_kpa: float
    heart_rate_bpm: float
    target_force_n: float
    pulse_amplitude_scale: float
    axial_offset_mm: float = 0.0
    approach_axial_offset_mm: float = 0.0
    proximity_bias_m: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    proximity_dropout_channel: int = -1
    tactile_gain: float = 1.0
    tactile_spike_kpa: float = 0.0
    action_scale: float = 1.0
    command_drop_start_step: int = -1
    command_drop_steps: int = 0
    ood_class: str = "unspecified"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stage3_collection_scenario(
    index: int, seed: int, profile: str = "random_recovery"
) -> CollectionScenario:
    """Build deterministic Stage-3 scenarios with paired nuisance control."""

    if profile == "contactfirst_collection_v1":
        # Replay the full-episode collection distribution without changing
        # its target forces, nuisance variables, or observation schema.
        return build_collection_scenarios(int(index) + 1, int(seed))[int(index)]

    if profile == "ch5_multifinger_balanced":
        # Low-nuisance contract pilot for the new embodiment. Directions are
        # expressed in the patient tangent basis, not hard-coded world axes.
        # Noise and execution faults are introduced only after recorder,
        # schema and 2-D label round trips pass.
        offsets_uv_mm = (
            (0.0, 0.0),
            (12.0, 0.0),
            (-12.0, 0.0),
            (0.0, 15.0),
            (0.0, -18.0),
            (10.0, 10.0),
            (-10.0, -10.0),
            (10.0, -10.0),
            (-10.0, 10.0),
        )
        lateral_mm, axial_mm = offsets_uv_mm[int(index) % len(offsets_uv_mm)]
        return CollectionScenario(
            scenario_id=f"ch5_multifinger_balanced_{index:03d}",
            lateral_offset_mm=lateral_mm,
            axial_offset_mm=axial_mm,
            tactile_noise_std_kpa=0.006,
            heart_rate_bpm=72.0,
            target_force_n=4.0,
            pulse_amplitude_scale=1.0,
            proximity_bias_m=(0.0, 0.0, 0.0, 0.0, 0.0),
            proximity_dropout_channel=-1,
            tactile_gain=1.0,
            tactile_spike_kpa=0.0,
            action_scale=1.0,
            command_drop_start_step=-1,
            command_drop_steps=0,
            ood_class="ch5_contract_pilot",
        )
    if profile == "ch5_multifinger_train_v1":
        # Adjacent episodes are paired: every nuisance is identical and only
        # the patient-tangent pulse offset changes.  This supplies explicit
        # +u/-u and +v/-v evidence without letting noise become a direction
        # shortcut.  The 12-case grid repeats with new deterministic nuisance
        # draws when a collection extends beyond one cycle.
        offsets_uv_mm = (
            (0.0, 0.0),
            (0.0, 0.0),
            (14.0, 0.0),
            (-14.0, 0.0),
            (0.0, 15.0),
            (0.0, -18.0),
            (10.0, 10.0),
            (-10.0, -10.0),
            (10.0, -10.0),
            (-10.0, 10.0),
            (9.0, 0.0),
            (-9.0, 0.0),
        )
        lateral_mm, axial_mm = offsets_uv_mm[int(index) % len(offsets_uv_mm)]
        pair_index = int(index) // 2
        rng = np.random.default_rng(int(seed) + pair_index)
        return CollectionScenario(
            # A scenario id names the nuisance template, not the episode.
            # Sharing it across the pair keeps near-identical nuisance draws
            # in one dataset split while episode/sample ids remain unique.
            scenario_id=f"ch5_multifinger_train_v1_pair_{pair_index:03d}",
            lateral_offset_mm=lateral_mm,
            axial_offset_mm=axial_mm,
            tactile_noise_std_kpa=float(rng.uniform(0.006, 0.014)),
            heart_rate_bpm=float(rng.uniform(58.0, 84.0)),
            target_force_n=4.0,
            pulse_amplitude_scale=float(rng.uniform(0.72, 1.15)),
            proximity_bias_m=tuple(rng.normal(0.0, 0.0006, 5)),
            proximity_dropout_channel=(pair_index % 5 if pair_index % 5 == 4 else -1),
            tactile_gain=float(rng.uniform(0.90, 1.10)),
            tactile_spike_kpa=(0.14 if pair_index % 6 == 5 else 0.0),
            action_scale=float(rng.uniform(0.95, 1.05)),
            command_drop_start_step=-1,
            command_drop_steps=0,
            ood_class="ch5_multifinger_train",
        )
    if profile == "ch5_multifinger_axial_balance_v1":
        # The ordinary CH5 grid deliberately accepts a good +15 mm first
        # contact, so it produces almost no positive-v correction labels.
        # This narrow supplement stays inside the validated wrist workcell
        # while moving to the observed +18 mm correction boundary.  Adjacent
        # episodes retain matched nuisance draws and opposite axial evidence.
        offsets_uv_mm = (
            (0.0, 18.0),
            (0.0, -18.0),
            (0.0, 18.0),
            (0.0, -18.0),
            (0.0, 18.0),
            (0.0, -18.0),
            (0.0, 18.0),
            (0.0, -18.0),
            (0.0, 18.0),
            (0.0, -18.0),
            (0.0, 18.0),
            (0.0, -18.0),
        )
        lateral_mm, axial_mm = offsets_uv_mm[int(index) % len(offsets_uv_mm)]
        pair_index = int(index) // 2
        rng = np.random.default_rng(int(seed) + pair_index)
        return CollectionScenario(
            scenario_id=f"ch5_multifinger_axial_balance_v1_pair_{pair_index:03d}",
            lateral_offset_mm=lateral_mm,
            axial_offset_mm=axial_mm,
            tactile_noise_std_kpa=float(rng.uniform(0.006, 0.008)),
            heart_rate_bpm=float(rng.uniform(60.0, 82.0)),
            target_force_n=4.0,
            pulse_amplitude_scale=float(rng.uniform(0.90, 1.10)),
            proximity_bias_m=(0.0, 0.0, 0.0, 0.0, 0.0),
            proximity_dropout_channel=-1,
            tactile_gain=float(rng.uniform(0.98, 1.02)),
            tactile_spike_kpa=0.0,
            # This profile exists to balance spatial labels, not to mix in a
            # control-tracking confound. Execution-scale robustness remains
            # covered by the ordinary CH5 train/OOD profiles.
            action_scale=1.0,
            command_drop_start_step=-1,
            command_drop_steps=0,
            ood_class="ch5_multifinger_axial_balance",
        )
    if profile == "ch5_multifinger_positive_axial_v1":
        # +18 mm is outside the repeatable 10 mm local search envelope: some
        # nuisance draws oscillate at the radius boundary. +16 mm remains a
        # deliberately weak first contact but leaves enough reachable margin
        # for safe positive-v labels. Each episode is its own nuisance group
        # so train/validation/test can all contain independent +v evidence.
        rng = np.random.default_rng(int(seed) + int(index))
        return CollectionScenario(
            scenario_id=f"ch5_multifinger_positive_axial_v1_{int(index):03d}",
            lateral_offset_mm=0.0,
            axial_offset_mm=16.0,
            tactile_noise_std_kpa=float(rng.uniform(0.006, 0.008)),
            heart_rate_bpm=float(rng.uniform(62.0, 80.0)),
            target_force_n=4.0,
            pulse_amplitude_scale=float(rng.uniform(0.92, 1.08)),
            proximity_bias_m=(0.0, 0.0, 0.0, 0.0, 0.0),
            proximity_dropout_channel=-1,
            tactile_gain=float(rng.uniform(0.98, 1.02)),
            tactile_spike_kpa=0.0,
            action_scale=1.0,
            command_drop_start_step=-1,
            command_drop_steps=0,
            ood_class="ch5_multifinger_positive_axial",
        )
    if profile == "ch5_multifinger_approach_axial_v1":
        # Tactile-only correction data for coarse visual localization error.
        # Keep the anatomical pulse and patient collision geometry fixed, and
        # perturb only the free-space pre-contact landing along the patient
        # axis. Adjacent episodes have the same nuisance draw and opposite
        # landing errors.
        # The custom hand has an asymmetric useful axial footprint: -8/+6 mm
        # are accepted, while -10 mm already misses stable contact. Keep the
        # profile strictly inside the demonstrated contact-establishment
        # envelope.
        approach_axial_mm = (-8.0, 6.0)[int(index) % 2]
        pair_index = int(index) // 2
        rng = np.random.default_rng(int(seed) + pair_index)
        return CollectionScenario(
            scenario_id=f"ch5_multifinger_approach_axial_v1_pair_{pair_index:03d}",
            lateral_offset_mm=0.0,
            axial_offset_mm=0.0,
            approach_axial_offset_mm=approach_axial_mm,
            tactile_noise_std_kpa=float(rng.uniform(0.006, 0.010)),
            heart_rate_bpm=float(rng.uniform(60.0, 82.0)),
            target_force_n=4.0,
            pulse_amplitude_scale=float(rng.uniform(0.88, 1.10)),
            proximity_bias_m=(0.0, 0.0, 0.0, 0.0, 0.0),
            proximity_dropout_channel=-1,
            tactile_gain=float(rng.uniform(0.96, 1.04)),
            tactile_spike_kpa=0.0,
            action_scale=1.0,
            command_drop_start_step=-1,
            command_drop_steps=0,
            ood_class="ch5_multifinger_approach_axial",
        )
    if profile == "ch5_search_recovery_gate_v1":
        # Small paired supplement from the stable-contact closed-loop gate.
        # The current ACT head has sparse first-step evidence for -u and two
        # diagonal quadrants. Keep this inside the demonstrated local workcell
        # and remove execution/noise confounds so the label teaches direction,
        # not nuisance identification. Repeated cycles get independent pairs.
        offsets_uv_mm = (
            (6.0, 0.0),
            (-6.0, 0.0),
            (4.25, 4.25),
            (-4.25, -4.25),
            (4.25, -4.25),
            (-4.25, 4.25),
            (0.0, 6.0),
            (0.0, -6.0),
        )
        lateral_mm, axial_mm = offsets_uv_mm[int(index) % len(offsets_uv_mm)]
        pair_index = int(index) // 2
        rng = np.random.default_rng(int(seed) + pair_index)
        return CollectionScenario(
            scenario_id=f"ch5_search_recovery_gate_v1_pair_{pair_index:03d}",
            lateral_offset_mm=lateral_mm,
            axial_offset_mm=axial_mm,
            tactile_noise_std_kpa=float(rng.uniform(0.006, 0.008)),
            heart_rate_bpm=float(rng.uniform(64.0, 78.0)),
            target_force_n=4.0,
            pulse_amplitude_scale=float(rng.uniform(0.96, 1.04)),
            proximity_bias_m=(0.0, 0.0, 0.0, 0.0, 0.0),
            proximity_dropout_channel=-1,
            tactile_gain=float(rng.uniform(0.98, 1.02)),
            tactile_spike_kpa=0.0,
            action_scale=1.0,
            command_drop_start_step=-1,
            command_drop_steps=0,
            ood_class="ch5_search_recovery_gate",
        )
    if profile == "ch5_multifinger_ood_v1":
        # Sealed custom-hand holdout. These patient-tangent offsets are inside
        # the demonstrated CH5 workcell but do not repeat the training grid.
        # Proximity is intentionally left nominal here because the deployment
        # runner removes that unavailable hardware modality altogether.
        categories = (
            "clean",
            "spatial",
            "tactile",
            "pulse",
            "execution",
            "combined",
        )
        category = categories[int(index) % len(categories)]
        replicate = (int(index) % 30) // len(categories)
        offsets_by_category = {
            "clean": ((0.0, 0.0), (4.0, -5.0), (-4.0, 5.0), (6.0, 3.0), (-6.0, -3.0)),
            "spatial": ((12.0, 5.0), (-12.0, -5.0), (7.0, 12.0), (-7.0, -14.0), (11.0, -7.0)),
            "tactile": ((5.0, 7.0), (-5.0, -7.0), (8.0, -4.0), (-8.0, 4.0), (3.0, -11.0)),
            "pulse": ((6.0, -6.0), (-6.0, 6.0), (3.0, 10.0), (-3.0, -12.0), (9.0, 4.0)),
            "execution": ((10.0, 5.0), (-10.0, -5.0), (5.0, -10.0), (-5.0, 10.0), (11.0, 0.0)),
            "combined": ((11.0, 7.0), (-11.0, -7.0), (7.0, -12.0), (-7.0, 12.0), (12.0, -5.0)),
        }
        lateral_mm, axial_mm = offsets_by_category[category][replicate]
        rng = np.random.default_rng(int(seed) + 104_729 * int(index) + 17)
        tactile_noise = float(rng.uniform(0.007, 0.012))
        pulse_amplitude = float(rng.uniform(0.82, 1.08))
        tactile_gain = float(rng.uniform(0.94, 1.06))
        tactile_spike = 0.0
        action_scale = 1.0
        command_drop_start = -1
        command_drop_steps = 0
        if category in {"tactile", "combined"}:
            tactile_noise = float(rng.uniform(0.014, 0.020))
            tactile_gain = float(rng.uniform(0.88, 1.14))
            tactile_spike = float(rng.uniform(0.14, 0.22))
        if category in {"pulse", "combined"}:
            pulse_amplitude = float(rng.uniform(0.62, 0.78))
        if category in {"execution", "combined"}:
            action_scale = float(rng.uniform(0.92, 1.08))
            command_drop_start = 190 + 20 * (replicate % 4)
            command_drop_steps = 3 + replicate % 3
        return CollectionScenario(
            scenario_id=f"ch5_multifinger_ood_v1_{category}_{int(index):03d}",
            lateral_offset_mm=lateral_mm,
            axial_offset_mm=axial_mm,
            tactile_noise_std_kpa=tactile_noise,
            heart_rate_bpm=float(rng.uniform(54.0, 88.0)),
            target_force_n=4.0,
            pulse_amplitude_scale=pulse_amplitude,
            proximity_bias_m=(0.0, 0.0, 0.0, 0.0, 0.0),
            proximity_dropout_channel=-1,
            tactile_gain=tactile_gain,
            tactile_spike_kpa=tactile_spike,
            action_scale=action_scale,
            command_drop_start_step=command_drop_start,
            command_drop_steps=command_drop_steps,
            ood_class=category,
        )
    if profile in (
        "directional_recovery",
        "directional_recovery_balanced",
        "directional_recovery_blind_v1",
    ):
        # The balanced profile shares every nuisance variable between the two
        # correction directions.  The official Pro right fingertips have an
        # empirically measured ~6 mm asymmetric useful footprint, so equal
        # geometric offsets are not equal observable difficulty.  Preserve
        # the symmetric pilot profile and calibrate only the new profile.
        paired = profile in (
            "directional_recovery_balanced",
            "directional_recovery_blind_v1",
        )
        pair_index = int(index) // 2 if paired else int(index)
        rng = np.random.default_rng(int(seed) + pair_index)
        if profile == "directional_recovery_blind_v1":
            # Sealed post-M3 holdout: unseen interleaved offsets and a modest
            # expansion of every observable nuisance range.  Adjacent
            # negative/positive episodes still share all nuisance variables,
            # so direction is the only within-pair change.
            offsets_mm = (-15.5, 21.5, -18.5, 24.5, -21.5, 27.5)
            noise_range = (0.010, 0.022)
            heart_rate_range = (52.0, 86.0)
            force_range = (4.8, 6.1)
            amplitude_range = (0.45, 0.68)
            proximity_bias_std = 0.0009
            tactile_gain_range = (0.86, 1.18)
            action_scale_range = (0.90, 1.10)
            drop_starts = (175, 200, 225, 250)
        else:
            offsets_mm = (
                (-14.0, 20.0, -17.0, 23.0, -20.0, 26.0)
                if paired
                else (-20.0, 20.0, -17.0, 17.0, -14.0, 14.0)
            )
            noise_range = (0.008, 0.018)
            heart_rate_range = (56.0, 82.0)
            force_range = (5.0, 5.9)
            amplitude_range = (0.48, 0.62)
            proximity_bias_std = 0.0007
            tactile_gain_range = (0.90, 1.12)
            action_scale_range = (0.92, 1.08)
            drop_starts = (180, 205, 230, 255)
        offset_mm = offsets_mm[index % len(offsets_mm)]
        return CollectionScenario(
            scenario_id=f"stage3_{profile}_{index:03d}",
            lateral_offset_mm=float(offset_mm),
            tactile_noise_std_kpa=float(rng.uniform(*noise_range)),
            heart_rate_bpm=float(rng.uniform(*heart_rate_range)),
            target_force_n=float(rng.uniform(*force_range)),
            pulse_amplitude_scale=float(rng.uniform(*amplitude_range)),
            proximity_bias_m=tuple(rng.normal(0.0, proximity_bias_std, 5)),
            proximity_dropout_channel=(
                pair_index % 5
                if (
                    (
                        profile == "directional_recovery_blind_v1"
                        and pair_index % 3 == 2
                    )
                    or (
                        profile != "directional_recovery_blind_v1"
                        and pair_index % 4 == 3
                    )
                )
                else -1
            ),
            tactile_gain=float(rng.uniform(*tactile_gain_range)),
            tactile_spike_kpa=(
                0.18 + 0.04 * (pair_index % 3)
                if profile == "directional_recovery_blind_v1"
                and pair_index % 5 == 4
                else 0.0
            ),
            action_scale=float(rng.uniform(*action_scale_range)),
            command_drop_start_step=drop_starts[
                pair_index % len(drop_starts)
            ],
            command_drop_steps=(5 if profile == "directional_recovery_blind_v1" else 4)
            + (pair_index % 3),
        )
    if profile in {"model_only_ood_v1", "model_only_ood_feasible_v1"}:
        feasible = profile == "model_only_ood_feasible_v1"
        local_index = int(index) % 30
        category = (
            "clean",
            "offset",
            "proximity",
            "tactile",
            "execution",
            "combined",
        )[local_index // 5]
        rng = np.random.default_rng(int(seed) + int(index))
        training_offsets = (-14.0, 20.0, -17.0, 23.0, -20.0, 26.0)
        holdout_offsets = (-15.5, 21.5, -18.5, 24.5, -21.5, 27.5)
        offsets = holdout_offsets if category in {"offset", "combined"} else training_offsets
        offset_mm = offsets[int(index) % len(offsets)]
        scenario = {
            "scenario_id": f"stage3_{profile}_{index:03d}",
            "lateral_offset_mm": float(offset_mm),
            "tactile_noise_std_kpa": 0.008,
            "heart_rate_bpm": 68.0,
            "target_force_n": 5.5,
            "pulse_amplitude_scale": 0.62,
            "proximity_bias_m": (0.0, 0.0, 0.0, 0.0, 0.0),
            "proximity_dropout_channel": -1,
            "tactile_gain": 1.0,
            "tactile_spike_kpa": 0.0,
            "action_scale": 1.0,
            "command_drop_start_step": -1,
            "command_drop_steps": 0,
            "ood_class": category,
        }
        if category in {"proximity", "combined"}:
            scenario["proximity_bias_m"] = tuple(
                rng.normal(0.0, 0.0014, 5)
            )
            scenario["proximity_dropout_channel"] = int(local_index % 5)
        if category in {"tactile", "combined"}:
            scenario["tactile_noise_std_kpa"] = float(
                rng.uniform(0.014, 0.020) if feasible else rng.uniform(0.018, 0.024)
            )
            scenario["heart_rate_bpm"] = float(rng.uniform(52.0, 86.0))
            scenario["pulse_amplitude_scale"] = float(
                rng.uniform(0.58, 0.72) if feasible else rng.uniform(0.44, 0.52)
            )
            scenario["tactile_gain"] = float(
                rng.uniform(0.90, 1.18) if feasible else rng.uniform(0.84, 1.20)
            )
            scenario["tactile_spike_kpa"] = float(
                rng.uniform(0.16, 0.22) if feasible else rng.uniform(0.16, 0.28)
            )
        if category in {"execution", "combined"}:
            scenario["action_scale"] = float(
                rng.uniform(0.92, 1.08) if feasible else rng.uniform(0.88, 1.12)
            )
            scenario["command_drop_start_step"] = 175 + 25 * (local_index % 4)
            scenario["command_drop_steps"] = (
                4 + (local_index % 3) if feasible else 6 + (local_index % 3)
            )
        if category == "combined":
            scenario["target_force_n"] = float(rng.uniform(4.8, 6.1))
        return CollectionScenario(**scenario)
    if profile != "random_recovery":
        raise ValueError(f"unsupported scenario profile: {profile}")
    rng = np.random.default_rng(int(seed) + int(index))
    return CollectionScenario(
        scenario_id=f"stage3_dagger_{index:03d}",
        lateral_offset_mm=float(rng.uniform(-8.0, 8.0)),
        tactile_noise_std_kpa=float(rng.uniform(0.008, 0.024)),
        heart_rate_bpm=float(rng.uniform(55.0, 84.0)),
        target_force_n=float(rng.uniform(4.9, 6.1)),
        pulse_amplitude_scale=float(rng.uniform(0.65, 1.30)),
        action_scale=float(rng.uniform(0.86, 1.12)),
        command_drop_start_step=220 + (index % 3) * 4,
        command_drop_steps=2 + (index % 3),
    )


def build_collection_scenarios(count: int, seed: int) -> list[CollectionScenario]:
    """Create a deterministic scenario sequence shared by regression and capture."""
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = np.random.default_rng(seed)
    return [
        CollectionScenario(
            scenario_id=f"random_{index:06d}",
            lateral_offset_mm=float(rng.uniform(-5.5, 5.5)),
            tactile_noise_std_kpa=float(rng.uniform(0.006, 0.020)),
            heart_rate_bpm=float(rng.uniform(52.0, 86.0)),
            target_force_n=float(rng.uniform(4.8, 6.2)),
            pulse_amplitude_scale=float(rng.uniform(0.60, 1.35)),
            proximity_bias_m=tuple(rng.normal(0.0, 0.0012, 5)),
            proximity_dropout_channel=int(rng.integers(-1, 5)),
            tactile_gain=float(rng.uniform(0.80, 1.30)),
            tactile_spike_kpa=float(rng.uniform(0.0, 0.8)),
            action_scale=float(rng.uniform(0.90, 1.10)),
            command_drop_start_step=220,
            command_drop_steps=int(rng.integers(0, 4)),
        )
        for index in range(count)
    ]


def apply_collection_scenario(backend, scenario: CollectionScenario) -> None:
    """Inject observable-domain variation without exposing truth to the expert."""
    forearm = backend.config["scene"]["patient_forearm"]
    if "from_m" in forearm and "to_m" in forearm:
        axis_xy = np.asarray(forearm["to_m"], dtype=float)[:2] - np.asarray(
            forearm["from_m"], dtype=float
        )[:2]
        axis_xy /= max(float(np.linalg.norm(axis_xy)), 1e-12)
        lateral_xy = np.asarray([-axis_xy[1], axis_xy[0]], dtype=float)
    else:
        # Compatibility for minimal test/fake backends that predate the
        # explicit patient tangent basis. This is the historical Pro frame.
        axis_xy = np.asarray([0.0, 1.0], dtype=float)
        lateral_xy = np.asarray([-1.0, 0.0], dtype=float)
    offset_xy = (
        float(scenario.lateral_offset_mm) * lateral_xy
        + float(scenario.axial_offset_mm) * axis_xy
    ) / 1000.0
    pulse_offset = np.asarray([offset_xy[0], offset_xy[1], 0.0])
    backend.sensors.pulse_point_m += pulse_offset
    backend.config["scene"]["patient_forearm"]["pulse_center_m"] = (
        backend.sensors.pulse_point_m.tolist()
    )
    # Model coarse-localization error by moving the planned free-space landing
    # rather than relocating the anatomical pulse toward a collision boundary.
    # The expert still receives no offset value; it must recover from tactile
    # observations after contact.
    backend.config["task"]["diagnostic_precontact_axial_offset_m"] = float(
        backend.config["task"].get("diagnostic_precontact_axial_offset_m", 0.0)
    ) + float(scenario.approach_axial_offset_mm) / 1000.0
    response = backend.sensors.tactile.sensor_response
    response.config = replace(
        response.config, white_noise_std_kpa=scenario.tactile_noise_std_kpa
    )
    pulse_source = backend.sensors.tactile.pulse_source
    pulse_source.config = replace(
        pulse_source.config, heart_rate_bpm=scenario.heart_rate_bpm
    )
    backend.sensors.tactile.dynamic_amplitude_kpa *= scenario.pulse_amplitude_scale
    backend.config["task"]["pressure_target_force_n_per_finger"] = (
        scenario.target_force_n
    )


def collection_perturbation_scheduler(scenario: CollectionScenario):
    """Build the policy-visible bounded perturbations stored with an episode."""
    from .perturbations import (
        PerturbationEvent,
        PerturbationKind,
        PerturbationScheduler,
    )

    events = [
        PerturbationEvent(
            "proximity_bias",
            PerturbationKind.PROXIMITY_BIAS,
            0,
            1200,
            {"bias_m": list(scenario.proximity_bias_m)},
        ),
        PerturbationEvent(
            "tactile_gain",
            PerturbationKind.TACTILE_GAIN,
            0,
            1200,
            {"gain": float(scenario.tactile_gain)},
        ),
        PerturbationEvent(
            "action_scale",
            PerturbationKind.ACTION_SCALE,
            0,
            1200,
            {"scale": float(scenario.action_scale)},
        ),
    ]
    if scenario.proximity_dropout_channel >= 0:
        events.append(
            PerturbationEvent(
                "proximity_dropout",
                PerturbationKind.PROXIMITY_DROPOUT,
                0,
                1200,
                {"channels": [int(scenario.proximity_dropout_channel)]},
            )
        )
    if scenario.tactile_spike_kpa > 0.0:
        events.append(
            PerturbationEvent(
                "tactile_spike",
                PerturbationKind.TACTILE_SPIKE,
                420,
                421,
                {"amplitude_kpa": float(scenario.tactile_spike_kpa)},
            )
        )
    if scenario.command_drop_steps > 0:
        events.append(
            PerturbationEvent(
                "command_drop",
                PerturbationKind.COMMAND_DROP,
                int(scenario.command_drop_start_step),
                int(
                    scenario.command_drop_start_step
                    + scenario.command_drop_steps
                    - 1
                ),
                {},
            )
        )
    return PerturbationScheduler(events)


def collection_acceptance(result: dict, validation: EpisodeValidation) -> tuple[bool, list[str]]:
    reasons = list(validation.errors)
    if not result.get("success", False):
        reasons.append("expert run failed")
    if not result.get("signal_quality_accepted", False):
        reasons.append("signal quality was not accepted")
    if int(result.get("forbidden_contact_events", 0)) != 0:
        reasons.append("forbidden contact was observed")
    return not reasons, sorted(set(reasons))


def new_collection_manifest(
    target_accepted: int,
    seed: int,
    *,
    source_contract: dict[str, Any] | None = None,
) -> dict:
    manifest = {
        "schema_version": COLLECTION_MANIFEST_SCHEMA_VERSION,
        "target_accepted": int(target_accepted),
        "seed": int(seed),
        "attempts": [],
        "accepted": 0,
        "failed": 0,
        "complete": False,
    }
    if source_contract is not None:
        manifest["source_contract"] = dict(source_contract)
    return manifest


def update_collection_counts(manifest: dict) -> None:
    attempts = manifest.get("attempts", [])
    manifest["accepted"] = sum(bool(item.get("accepted")) for item in attempts)
    manifest["failed"] = len(attempts) - manifest["accepted"]
    manifest["complete"] = manifest["accepted"] >= int(
        manifest["target_accepted"]
    )


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
