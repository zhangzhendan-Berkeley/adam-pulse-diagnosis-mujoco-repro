from __future__ import annotations

"""Five-finger tactile runtime with an exact 2x10 diagnostic compatibility view."""

from dataclasses import dataclass

import numpy as np

from .contact import FULL_HAND_FINGERS, FullHandContactFrame
from .tactile_array import HandleL1DualFingerTactileSensor, TactileFrame
from .tactile_calibration import HandleL1TenChannelCalibration
from .tactile_contact_transfer import (
    compute_orientation_gain,
    compute_preload_gain,
)
from .tactile_layouts import TaxelLayout, handle_l1_layout_for_finger
from .tactile_pulse_field import compute_spatial_gain
from .tactile_sensor_response import SensorResponseConfig, TactileSensorResponse
from .tactile_static_contact import StaticContactArrayModel, StaticContactConfig


DIAGNOSTIC_FINGERS = ("index", "middle")


@dataclass(frozen=True)
class FullHandTactileFrame:
    finger_names: tuple[str, ...]
    channel_counts: np.ndarray
    channel_offsets: np.ndarray
    pressure_kpa: np.ndarray
    static_pressure_kpa: np.ndarray
    dynamic_pressure_kpa: np.ndarray
    taxel_force_n: np.ndarray
    raw_values: np.ndarray
    raw_delta: np.ndarray
    raw_baseline: np.ndarray
    contact: np.ndarray
    total_normal_force_n: np.ndarray
    timestamp_s: float
    diagnostic: TactileFrame

    def finger_slice(self, finger_name: str) -> slice:
        index = self.finger_names.index(str(finger_name))
        start = int(self.channel_offsets[index])
        return slice(start, start + int(self.channel_counts[index]))


class HandleL1FullHandTactileSensor:
    """Add thumb/ring/little sensing without changing index/middle behavior."""

    finger_names = FULL_HAND_FINGERS

    def __init__(
        self,
        diagnostic_sensor: HandleL1DualFingerTactileSensor,
        hardware_calibration: HandleL1TenChannelCalibration,
        *,
        static_contact_config: StaticContactConfig | None = None,
        sensor_response_config: SensorResponseConfig | None = None,
        dynamic_amplitude_kpa: float = 2.0,
        sigma_lateral_m: float = 0.004,
        sigma_axial_m: float = 0.020,
        seed: int = 0,
    ):
        self.diagnostic_sensor = diagnostic_sensor
        self.hardware_calibration = hardware_calibration
        self.layouts: dict[str, TaxelLayout] = {
            name: handle_l1_layout_for_finger(name) for name in self.finger_names
        }
        self.channel_counts = np.asarray(
            [len(self.layouts[name].positions_local_m) for name in self.finger_names],
            dtype=np.int32,
        )
        self.channel_offsets = np.concatenate(
            ([0], np.cumsum(self.channel_counts[:-1]))
        ).astype(np.int32)
        self.auxiliary_names = tuple(
            name for name in self.finger_names if name not in DIAGNOSTIC_FINGERS
        )
        static_config = static_contact_config or StaticContactConfig(
            maximum_force_n=100.0
        )
        response_config = sensor_response_config or SensorResponseConfig()
        self.static_models = {
            name: StaticContactArrayModel(static_config)
            for name in self.auxiliary_names
        }
        self.responses = {
            name: TactileSensorResponse(
                response_config,
                seed=seed + 101 + index,
                shape=(1, int(self.channel_counts[self.finger_names.index(name)])),
            )
            for index, name in enumerate(self.auxiliary_names)
        }
        self.dynamic_amplitude_kpa = float(dynamic_amplitude_kpa)
        self.sigma_lateral_m = float(sigma_lateral_m)
        self.sigma_axial_m = float(sigma_axial_m)
        self._last_time_s: float | None = None
        self._thumb_force_to_raw = np.median(
            hardware_calibration.raw_lut.astype(np.float32), axis=(0, 1, 3)
        )
        self._thumb_baseline = float(np.median(hardware_calibration.baseline_raw))

    def reset(self, seed: int | None = None) -> None:
        self.diagnostic_sensor.reset(seed)
        self._last_time_s = None
        for model in self.static_models.values():
            model.reset()
        for index, response in enumerate(self.responses.values()):
            response.reset(None if seed is None else seed + 101 + index)

    def _ten_channel_raw(
        self, layout: TaxelLayout, taxel_force_n: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        calibration = self.hardware_calibration
        points = np.asarray(
            [
                calibration.local_point_to_supplier(point)
                for point in layout.positions_local_m
            ]
        )
        loaded = calibration.predict_float_many(taxel_force_n, points)
        drop = np.maximum(calibration.baseline_raw[None] - loaded, 0.0).sum(
            axis=0
        )
        raw = np.rint(
            np.clip(
                calibration.baseline_raw - drop,
                calibration.raw_min,
                calibration.raw_max,
            )
        ).astype(np.uint16)
        return raw, calibration.baseline_raw.astype(np.uint16)

    def _thumb_raw(self, taxel_force_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calibration = self.hardware_calibration
        raw = np.interp(
            np.clip(taxel_force_n, 0.0, calibration.maximum_force_n),
            calibration.force_axis_n,
            self._thumb_force_to_raw,
            left=self._thumb_baseline,
            right=float(self._thumb_force_to_raw[-1]),
        )
        baseline = np.full(16, self._thumb_baseline, dtype=np.float32)
        return (
            np.rint(np.clip(raw, calibration.raw_min, calibration.raw_max)).astype(
                np.uint16
            ),
            np.rint(baseline).astype(np.uint16),
        )

    def sample(
        self,
        *,
        time_s: float,
        centers_m: np.ndarray,
        rotations_world_from_tactile: np.ndarray,
        pulse_point_m: np.ndarray,
        pulse_axis_xy: np.ndarray,
        skin_normal: np.ndarray,
        contacts: FullHandContactFrame,
        dt_s: float | None = None,
    ) -> FullHandTactileFrame:
        centers = np.asarray(centers_m, dtype=float)
        rotations = np.asarray(rotations_world_from_tactile, dtype=float)
        if centers.shape != (5, 3) or rotations.shape != (5, 3, 3):
            raise ValueError("full-hand centers/rotations must have shape [5,3]/[5,3,3]")
        dt = (
            1.0 / 60.0
            if dt_s is None and self._last_time_s is None
            else max(float(time_s) - float(self._last_time_s), 1e-6)
            if dt_s is None
            else max(float(dt_s), 1e-6)
        )
        self._last_time_s = float(time_s)
        by_name = dict(zip(contacts.finger_names, contacts.fingers))
        diagnostic_indices = [self.finger_names.index(name) for name in DIAGNOSTIC_FINGERS]
        diagnostic_contacts = [by_name[name] for name in DIAGNOSTIC_FINGERS]
        diagnostic = self.diagnostic_sensor.sample(
            time_s=time_s,
            fingertip_centers_m=centers[diagnostic_indices],
            pulse_point_m=pulse_point_m,
            skin_surface_z_m=np.full(2, float(np.asarray(pulse_point_m)[2])),
            physical_force_n=np.asarray(
                [item.normal_force_n for item in diagnostic_contacts]
            ),
            pulse_axis_xy=pulse_axis_xy,
            dt_s=dt,
            fingertip_rotations=rotations[diagnostic_indices],
            skin_normal=skin_normal,
            physical_contact_points_local_m=tuple(
                item.points_local_m for item in diagnostic_contacts
            ),
            physical_contact_strength=tuple(
                item.point_forces_n for item in diagnostic_contacts
            ),
        )

        total_channels = int(np.sum(self.channel_counts))
        pressure = np.zeros(total_channels, dtype=np.float32)
        static = np.zeros_like(pressure)
        dynamic = np.zeros_like(pressure)
        taxel_force = np.zeros_like(pressure)
        raw = np.zeros(total_channels, dtype=np.uint16)
        baseline = np.zeros(total_channels, dtype=np.uint16)
        contact_flags = np.zeros(5, dtype=bool)
        normal_force = contacts.normal_force_n.astype(np.float32)

        for finger_index, name in enumerate(self.finger_names):
            start = int(self.channel_offsets[finger_index])
            count = int(self.channel_counts[finger_index])
            target = slice(start, start + count)
            if name in DIAGNOSTIC_FINGERS:
                source = DIAGNOSTIC_FINGERS.index(name)
                pressure[target] = diagnostic.arrays_kpa[source]
                static[target] = diagnostic.static_arrays_kpa[source]
                dynamic[target] = diagnostic.dynamic_arrays_kpa[source]
                taxel_force[target] = diagnostic.taxel_force_n[source]
                raw[target] = diagnostic.raw_values[source]
                baseline[target] = diagnostic.raw_baseline
                contact_flags[finger_index] = diagnostic.contact[source]
                continue

            layout = self.layouts[name]
            physical = by_name[name]
            frame = self.static_models[name].sample(
                indentation_m=0.0,
                dt_s=dt,
                taxel_positions_local_m=layout.positions_local_m,
                taxel_area_m2=layout.areas_m2,
                physical_force_n=physical.normal_force_n,
                contact_points_local_m=physical.points_local_m,
                contact_strength=physical.point_forces_n,
            )
            contact_flags[finger_index] = frame.in_contact
            taxel_world = centers[finger_index] + (
                rotations[finger_index] @ layout.positions_local_m.T
            ).T
            spatial = compute_spatial_gain(
                taxel_world[None],
                pulse_point_m,
                pulse_axis_xy,
                self.sigma_lateral_m,
                self.sigma_axial_m,
            )[0]
            preload = compute_preload_gain(
                np.asarray([frame.total_normal_force_n]),
                minimum_force_n=0.20,
                maximum_force_n=20.0,
                rise_width_n=0.10,
                fall_width_n=2.0,
            )[0]
            normal = rotations[finger_index] @ np.asarray([-1.0, 0.0, 0.0])
            orientation = compute_orientation_gain(
                normal[None], np.asarray(skin_normal), exponent=2.0
            )[0]
            pulse = float(self.diagnostic_sensor.pulse_source.normalized(time_s))
            dynamic_values = (
                self.dynamic_amplitude_kpa * pulse * spatial * preload * orientation
            ).astype(np.float32)
            if not frame.in_contact:
                dynamic_values.fill(0.0)
            values = self.responses[name].update(
                (frame.pressure_kpa + dynamic_values)[None],
                dt,
                np.asarray([frame.in_contact]),
            )[0]
            pressure[target] = values
            static[target] = frame.pressure_kpa
            dynamic[target] = dynamic_values
            taxel_force[target] = frame.taxel_force_n
            combined_force = frame.taxel_force_n + (
                dynamic_values * layout.areas_m2 * 1000.0
            )
            raw_values, raw_baseline = (
                self._thumb_raw(combined_force)
                if name == "thumb"
                else self._ten_channel_raw(layout, combined_force)
            )
            raw[target] = raw_values
            baseline[target] = raw_baseline

        return FullHandTactileFrame(
            finger_names=self.finger_names,
            channel_counts=self.channel_counts.copy(),
            channel_offsets=self.channel_offsets.copy(),
            pressure_kpa=pressure,
            static_pressure_kpa=static,
            dynamic_pressure_kpa=dynamic,
            taxel_force_n=taxel_force,
            raw_values=raw,
            raw_delta=np.maximum(
                baseline.astype(np.float32) - raw.astype(np.float32), 0.0
            ),
            raw_baseline=baseline,
            contact=contact_flags,
            total_normal_force_n=normal_force,
            timestamp_s=float(time_s),
            diagnostic=diagnostic,
        )
