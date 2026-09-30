from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tactile_contact_transfer import (
    compute_balance_gain,
    compute_orientation_gain,
    compute_preload_gain,
)
from .tactile_calibration import HandleL1TenChannelCalibration
from .tactile_layouts import handle_l1_ten_channel_layout
from .tactile_pulse_field import compute_spatial_gain
from .tactile_sensor_response import SensorResponseConfig, TactileSensorResponse
from .tactile_static_contact import StaticContactArrayModel, StaticContactConfig


@dataclass(frozen=True)
class PulseSourceConfig:
    """Parameters for a synthetic radial-artery pulse waveform.

    This is a signal-generation model for simulation and is not a physiological
    or diagnostic model.  The second peak approximates the reflected/dicrotic
    component needed to test whether a tactile pipeline preserves waveform shape.
    """

    heart_rate_bpm: float = 72.0
    main_peak_phase: float = 0.18
    main_peak_width: float = 0.055
    reflected_peak_phase: float = 0.43
    reflected_peak_width: float = 0.075
    reflected_peak_ratio: float = 0.34
    notch_phase: float = 0.34
    notch_width: float = 0.035
    notch_ratio: float = 0.10
    respiration_hz: float = 0.24
    respiration_modulation: float = 0.06
    displacement_amplitude_m: float = 0.00045
    dynamic_pressure_kpa: float = 1.4


class RadialPulseSource:
    def __init__(self, config: PulseSourceConfig | None = None):
        self.config = config or PulseSourceConfig()

    @staticmethod
    def _periodic_gaussian(phase: np.ndarray | float, center: float, width: float):
        delta = (np.asarray(phase) - center + 0.5) % 1.0 - 0.5
        return np.exp(-0.5 * np.square(delta / width))

    def normalized(self, time_s: np.ndarray | float):
        cfg = self.config
        phase = (np.asarray(time_s, dtype=float) * cfg.heart_rate_bpm / 60.0) % 1.0
        wave = self._periodic_gaussian(phase, cfg.main_peak_phase, cfg.main_peak_width)
        wave += cfg.reflected_peak_ratio * self._periodic_gaussian(
            phase, cfg.reflected_peak_phase, cfg.reflected_peak_width
        )
        wave -= cfg.notch_ratio * self._periodic_gaussian(
            phase, cfg.notch_phase, cfg.notch_width
        )
        wave = np.maximum(wave, 0.0)
        respiratory = 1.0 + cfg.respiration_modulation * np.sin(
            2.0 * np.pi * cfg.respiration_hz * np.asarray(time_s, dtype=float)
        )
        return np.clip(wave * respiratory, 0.0, 1.25)

    def displacement_m(self, time_s: np.ndarray | float):
        return self.config.displacement_amplitude_m * self.normalized(time_s)

    def pressure_kpa(self, time_s: np.ndarray | float):
        return self.config.dynamic_pressure_kpa * self.normalized(time_s)


@dataclass(frozen=True)
class TactileArrayConfig:
    rows: int = 7
    cols: int = 8
    width_m: float = 0.014
    length_m: float = 0.016
    contact_stiffness_kpa_per_mm: float = 4.0
    maximum_indentation_m: float = 0.0025
    contact_threshold_m: float = 0.00002
    contact_release_gap_m: float = 0.0005
    contact_spread_m: float = 0.0065
    artery_spread_m: float = 0.009
    require_all_fingers_contact_for_output: bool = True
    noise_std_kpa: float = 0.012
    maximum_pressure_kpa: float = 18.0


@dataclass(frozen=True)
class TactileFrame:
    arrays_kpa: np.ndarray
    contact: np.ndarray
    mean_pressure_kpa: np.ndarray
    pulse_normalized: float
    pulse_displacement_m: float
    timestamp_s: float
    static_arrays_kpa: np.ndarray | None = None
    dynamic_arrays_kpa: np.ndarray | None = None
    taxel_force_n: np.ndarray | None = None
    total_normal_force_n: np.ndarray | None = None
    contact_area_m2: np.ndarray | None = None
    contact_weights: np.ndarray | None = None
    indentation_m: np.ndarray | None = None
    spatial_gain: np.ndarray | None = None
    preload_gain: np.ndarray | None = None
    orientation_gain: np.ndarray | None = None
    balance_gain: float | None = None
    source_mode: str = "legacy_indentation"
    raw_values: np.ndarray | None = None
    raw_delta: np.ndarray | None = None
    raw_baseline: np.ndarray | None = None
    calibration_force_clipped: np.ndarray | None = None


class DualFingerTactileSensor:
    """Index/middle fingertip tactile interface with two 7x8 pressure arrays.

    ``physical_force_n`` is the upgrade seam for an Isaac Contact Sensor or a
    real hand driver.  If supplied, its per-finger normal force sets the static
    pressure.  Otherwise a compliant indentation model is used.
    """

    finger_names = ("index", "middle")

    def __init__(
        self,
        config: TactileArrayConfig | None = None,
        pulse_source: RadialPulseSource | None = None,
        seed: int = 0,
    ):
        self.config = config or TactileArrayConfig()
        self.pulse_source = pulse_source or RadialPulseSource()
        self.rng = np.random.default_rng(seed)
        self._contact_latched = np.zeros(2, dtype=bool)
        x = np.linspace(-0.5 * self.config.length_m, 0.5 * self.config.length_m, self.config.cols)
        y = np.linspace(-0.5 * self.config.width_m, 0.5 * self.config.width_m, self.config.rows)
        self.taxel_x_m, self.taxel_y_m = np.meshgrid(x, y)
        self.taxel_area_m2 = self.config.width_m * self.config.length_m / (
            self.config.rows * self.config.cols
        )

    @property
    def shape(self):
        return (2, self.config.rows, self.config.cols)

    def sample(
        self,
        time_s: float,
        fingertip_centers_m,
        pulse_point_m,
        skin_surface_z_m: float | np.ndarray,
        physical_force_n=None,
        pulse_axis_xy=None,
    ) -> TactileFrame:
        centers = np.asarray(fingertip_centers_m, dtype=float)
        if centers.shape != (2, 3):
            raise ValueError(f"fingertip_centers_m must have shape (2, 3), got {centers.shape}")
        pulse_point = np.asarray(pulse_point_m, dtype=float)
        if pulse_point.shape != (3,):
            raise ValueError(f"pulse_point_m must have shape (3,), got {pulse_point.shape}")
        force = None if physical_force_n is None else np.asarray(physical_force_n, dtype=float)
        if force is not None and force.shape != (2,):
            raise ValueError(f"physical_force_n must have shape (2,), got {force.shape}")
        surfaces = np.asarray(skin_surface_z_m, dtype=float)
        if surfaces.ndim == 0:
            surfaces = np.full(2, float(surfaces), dtype=float)
        if surfaces.shape != (2,):
            raise ValueError(f"skin_surface_z_m must be scalar or shape (2,), got {surfaces.shape}")

        pulse = float(self.pulse_source.normalized(time_s))
        displacement = float(self.pulse_source.displacement_m(time_s))
        dynamic_kpa = float(self.pulse_source.pressure_kpa(time_s))
        arrays = np.zeros(self.shape, dtype=np.float32)
        contacts = np.zeros(2, dtype=bool)

        for finger_index, center in enumerate(centers):
            static_indentation = float(surfaces[finger_index] - center[2])
            contacts[finger_index] = bool(
                static_indentation >= self.config.contact_threshold_m
                or (
                    self._contact_latched[finger_index]
                    and static_indentation >= -self.config.contact_release_gap_m
                )
                or (force is not None and force[finger_index] > 0.01)
            )
            self._contact_latched[finger_index] = contacts[finger_index]
            if not contacts[finger_index]:
                continue
            indentation = static_indentation + displacement

            taxel_world_x = center[0] + self.taxel_x_m
            taxel_world_y = center[1] + self.taxel_y_m
            delta_x = taxel_world_x - pulse_point[0]
            delta_y = taxel_world_y - pulse_point[1]
            if pulse_axis_xy is None:
                radial_sq = np.square(delta_x) + np.square(delta_y)
            else:
                axis = np.asarray(pulse_axis_xy, dtype=float)
                axis /= np.linalg.norm(axis)
                longitudinal = delta_x * axis[0] + delta_y * axis[1]
                radial_sq = np.square(delta_x) + np.square(delta_y) - np.square(
                    longitudinal
                )
            contact_center_sq = np.square(self.taxel_x_m) + np.square(self.taxel_y_m)
            contact_profile = np.exp(
                -0.5 * contact_center_sq / self.config.contact_spread_m**2
            )
            artery_profile = np.exp(-0.5 * radial_sq / self.config.artery_spread_m**2)

            if force is None:
                clipped_mm = 1000.0 * np.clip(
                    indentation, 0.0, self.config.maximum_indentation_m
                )
                static_kpa = self.config.contact_stiffness_kpa_per_mm * clipped_mm
            else:
                pad_area_m2 = self.config.width_m * self.config.length_m
                static_kpa = max(0.0, float(force[finger_index])) / pad_area_m2 / 1000.0

            pressure = static_kpa * contact_profile + dynamic_kpa * artery_profile
            pressure += self.rng.normal(0.0, self.config.noise_std_kpa, pressure.shape)
            arrays[finger_index] = np.clip(
                pressure, 0.0, self.config.maximum_pressure_kpa
            ).astype(np.float32)
        if self.config.require_all_fingers_contact_for_output and not contacts.all():
            arrays.fill(0.0)

        return TactileFrame(
            arrays_kpa=arrays,
            contact=contacts,
            mean_pressure_kpa=arrays.mean(axis=(1, 2)),
            pulse_normalized=pulse,
            pulse_displacement_m=displacement,
            timestamp_s=float(time_s),
        )


class HandleL1DualFingerTactileSensor:
    """Interface-faithful index/middle simulation with 10 channels per finger.

    Channel order corresponds to the supplied 10-channel fingertip coordinates
    in ``src.tactile.layouts``. Small deterministic gain differences represent
    channel-to-channel sensitivity variation.
    """

    finger_names = ("index", "middle")
    channels_per_finger = 10

    def __init__(
        self,
        config=None,
        pulse_source=None,
        seed=0,
        static_contact_config: StaticContactConfig | None = None,
        sensor_response_config: SensorResponseConfig | None = None,
        sigma_lateral_m: float = 0.004,
        sigma_axial_m: float = 0.020,
        dynamic_amplitude_kpa: float = 0.30,
        hardware_calibration: HandleL1TenChannelCalibration | None = None,
        preload_minimum_force_n: float = 0.20,
        preload_maximum_force_n: float = 10.0,
        preload_rise_width_n: float = 0.10,
        preload_fall_width_n: float = 0.50,
        orientation_exponent: float = 2.0,
        balance_force_sigma_n: float | None = 2.0,
    ):
        self.config = config or TactileArrayConfig()
        self.pulse_source = pulse_source or RadialPulseSource()
        self.rng = np.random.default_rng(seed)
        self._contact_latched = np.zeros(2, dtype=bool)
        self.channel_gain = np.linspace(0.92, 1.08, self.channels_per_finger)
        self.layouts = tuple(
            handle_l1_ten_channel_layout(name) for name in self.finger_names
        )
        self.static_models = tuple(
            StaticContactArrayModel(static_contact_config) for _ in self.finger_names
        )
        self.sensor_response = TactileSensorResponse(sensor_response_config, seed=seed)
        self.sigma_lateral_m = float(sigma_lateral_m)
        self.sigma_axial_m = float(sigma_axial_m)
        self.dynamic_amplitude_kpa = float(dynamic_amplitude_kpa)
        self.hardware_calibration = hardware_calibration
        self.preload_minimum_force_n = float(preload_minimum_force_n)
        self.preload_maximum_force_n = float(preload_maximum_force_n)
        self.preload_rise_width_n = float(preload_rise_width_n)
        self.preload_fall_width_n = float(preload_fall_width_n)
        self.orientation_exponent = float(orientation_exponent)
        self.balance_force_sigma_n = (
            None
            if balance_force_sigma_n is None
            else float(balance_force_sigma_n)
        )
        self._last_time_s = None

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._contact_latched.fill(False)
        self._last_time_s = None
        for model in self.static_models:
            model.reset()
        self.sensor_response.reset(seed)

    @property
    def shape(self):
        return (2, self.channels_per_finger)

    def sample(
        self,
        time_s,
        fingertip_centers_m,
        pulse_point_m,
        skin_surface_z_m,
        physical_force_n=None,
        pulse_axis_xy=None,
        dt_s=None,
        fingertip_rotations=None,
        skin_normal=None,
        physical_contact_points_local_m=None,
        physical_contact_strength=None,
    ):
        centers = np.asarray(fingertip_centers_m, dtype=float)
        if centers.shape != (2, 3):
            raise ValueError(f"fingertip_centers_m must have shape (2, 3), got {centers.shape}")
        pulse_point = np.asarray(pulse_point_m, dtype=float)
        surfaces = np.asarray(skin_surface_z_m, dtype=float)
        if surfaces.ndim == 0:
            surfaces = np.full(2, float(surfaces), dtype=float)
        if surfaces.shape != (2,):
            raise ValueError(f"skin_surface_z_m must be scalar or shape (2,), got {surfaces.shape}")
        force = None if physical_force_n is None else np.asarray(physical_force_n, dtype=float)
        if force is not None and force.shape != (2,):
            raise ValueError(f"physical_force_n must have shape (2,), got {force.shape}")

        dt = (
            1.0 / 60.0
            if dt_s is None and self._last_time_s is None
            else max(float(time_s) - float(self._last_time_s), 1e-6)
            if dt_s is None
            else max(float(dt_s), 1e-6)
        )
        self._last_time_s = float(time_s)
        pulse = float(self.pulse_source.normalized(time_s))
        displacement = float(self.pulse_source.displacement_m(time_s))
        static_arrays = np.zeros(self.shape, dtype=np.float32)
        taxel_force = np.zeros(self.shape, dtype=np.float32)
        contact_weights = np.zeros(self.shape, dtype=np.float32)
        normal_force = np.zeros(2, dtype=np.float32)
        contact_area = np.zeros(2, dtype=np.float32)
        indentation_values = np.zeros(2, dtype=np.float32)
        contacts = np.zeros(2, dtype=bool)
        if fingertip_rotations is None:
            rotations = np.repeat(np.eye(3)[None], 2, axis=0)
            taxel_world = np.asarray(
                [
                    center
                    + np.column_stack(
                        (
                            self.layouts[index].positions_local_m[:, 1],
                            self.layouts[index].positions_local_m[:, 2],
                            np.zeros(10),
                        )
                    )
                    for index, center in enumerate(centers)
                ]
            )
            fingerpad_normals = np.repeat(
                np.asarray([[0.0, 0.0, -1.0]]), 2, axis=0
            )
        else:
            rotations = np.asarray(fingertip_rotations, dtype=float)
            if rotations.shape != (2, 3, 3):
                raise ValueError("fingertip_rotations must have shape [2, 3, 3]")
            taxel_world = np.asarray(
                [
                    centers[index]
                    + (
                        rotations[index]
                        @ self.layouts[index].positions_local_m.T
                    ).T
                    for index in range(2)
                ]
            )
            fingerpad_normals = np.asarray(
                [rotation @ np.asarray([-1.0, 0.0, 0.0]) for rotation in rotations]
            )
        physical_points = (
            (None, None)
            if physical_contact_points_local_m is None
            else physical_contact_points_local_m
        )
        physical_strength = (
            (None, None)
            if physical_contact_strength is None
            else physical_contact_strength
        )
        for finger_index, center in enumerate(centers):
            static_indentation = float(surfaces[finger_index] - center[2])
            frame = self.static_models[finger_index].sample(
                indentation_m=max(static_indentation, 0.0),
                dt_s=dt,
                taxel_positions_local_m=self.layouts[
                    finger_index
                ].positions_local_m,
                taxel_area_m2=self.layouts[finger_index].areas_m2,
                physical_force_n=(
                    None if force is None else float(force[finger_index])
                ),
                contact_points_local_m=physical_points[finger_index],
                contact_strength=physical_strength[finger_index],
            )
            contacts[finger_index] = frame.in_contact
            static_arrays[finger_index] = frame.pressure_kpa
            taxel_force[finger_index] = frame.taxel_force_n
            contact_weights[finger_index] = frame.contact_weights
            normal_force[finger_index] = frame.total_normal_force_n
            contact_area[finger_index] = frame.contact_area_m2
            indentation_values[finger_index] = frame.indentation_m
        axis = np.asarray([1.0, 0.0] if pulse_axis_xy is None else pulse_axis_xy)
        spatial_gain = compute_spatial_gain(
            taxel_world,
            pulse_point,
            axis,
            self.sigma_lateral_m,
            self.sigma_axial_m,
        )
        preload_gain = compute_preload_gain(
            normal_force,
            minimum_force_n=self.preload_minimum_force_n,
            maximum_force_n=self.preload_maximum_force_n,
            rise_width_n=self.preload_rise_width_n,
            fall_width_n=self.preload_fall_width_n,
        )
        orientation_gain = compute_orientation_gain(
            fingerpad_normals,
            np.asarray([0.0, 0.0, 1.0] if skin_normal is None else skin_normal),
            exponent=self.orientation_exponent,
        )
        balance_gain = (
            1.0
            if self.balance_force_sigma_n is None
            else compute_balance_gain(
                normal_force, sigma_n=self.balance_force_sigma_n
            )
        )
        dynamic_arrays = (
            self.dynamic_amplitude_kpa
            * pulse
            * spatial_gain
            * preload_gain[:, None]
            * orientation_gain[:, None]
            * balance_gain
            * self.channel_gain[None]
        ).astype(np.float32)
        dynamic_arrays[~contacts] = 0.0
        if self.config.require_all_fingers_contact_for_output and not contacts.all():
            dynamic_arrays.fill(0.0)
        arrays = self.sensor_response.update(
            static_arrays + dynamic_arrays, dt, contacts
        )
        raw_values = None
        raw_delta = None
        raw_baseline = None
        calibration_force_clipped = None
        if self.hardware_calibration is not None:
            calibration = self.hardware_calibration
            raw_baseline = calibration.baseline_raw.astype(np.uint16)
            raw_float = np.broadcast_to(
                calibration.baseline_raw, self.shape
            ).astype(np.float32).copy()
            calibration_force_clipped = np.zeros(2, dtype=bool)
            for finger_index in range(2):
                if not contacts[finger_index]:
                    continue
                # The supplied calibration is a local point-load scan. A
                # distributed PhysX contact must therefore be decomposed into
                # local taxel loads before lookup; treating the full finger
                # force as one point load spuriously saturates the 10 N LUT.
                supplier_points = np.asarray(
                    [
                        calibration.local_point_to_supplier(local_point)
                        for local_point in self.layouts[
                            finger_index
                        ].positions_local_m
                    ]
                )
                static_force_n = taxel_force[finger_index].astype(float)
                dynamic_force_n = (
                    dynamic_arrays[finger_index]
                    * self.layouts[finger_index].areas_m2
                    * 1000.0
                ).astype(float)
                calibration_force_clipped[finger_index] = bool(
                    np.any(
                        static_force_n + dynamic_force_n
                        > calibration.maximum_force_n
                    )
                )
                static_raw = calibration.predict_float_many(
                    static_force_n, supplier_points
                )
                loaded_raw = calibration.predict_float_many(
                    static_force_n + dynamic_force_n, supplier_points
                )
                combined_drop = np.maximum(
                    calibration.baseline_raw[None] - static_raw, 0.0
                ).sum(axis=0)
                combined_drop += np.maximum(
                    static_raw - loaded_raw, 0.0
                ).sum(axis=0)
                raw_float[finger_index] = (
                    calibration.baseline_raw - combined_drop
                )
            raw_values = np.rint(
                np.clip(raw_float, calibration.raw_min, calibration.raw_max)
            ).astype(np.uint16)
            raw_delta = np.maximum(
                calibration.baseline_raw[None] - raw_values.astype(np.float32),
                0.0,
            ).astype(np.float32)
        return TactileFrame(
            arrays_kpa=arrays,
            contact=contacts,
            mean_pressure_kpa=arrays.mean(axis=1),
            pulse_normalized=pulse,
            pulse_displacement_m=displacement,
            timestamp_s=float(time_s),
            static_arrays_kpa=static_arrays,
            dynamic_arrays_kpa=dynamic_arrays,
            taxel_force_n=taxel_force,
            total_normal_force_n=normal_force,
            contact_area_m2=contact_area,
            contact_weights=contact_weights,
            indentation_m=indentation_values,
            spatial_gain=spatial_gain,
            preload_gain=preload_gain,
            orientation_gain=orientation_gain,
            balance_gain=balance_gain,
            source_mode="physx" if force is not None else "indentation_fallback",
            raw_values=raw_values,
            raw_delta=raw_delta,
            raw_baseline=raw_baseline,
            calibration_force_clipped=calibration_force_clipped,
        )
