from __future__ import annotations

from pathlib import Path
import copy

import mujoco
import numpy as np

from .contact import MujocoDualFingerContactReader, MujocoFullHandContactReader
from .full_hand_tactile import HandleL1FullHandTactileSensor
from .proximity import MujocoFingertipProximity, ProximityFrame
from .tactile_array import HandleL1DualFingerTactileSensor, TactileFrame
from .tactile_calibration import HandleL1TenChannelCalibration
from .tactile_sensor_response import SensorResponseConfig
from .tactile_static_contact import StaticContactConfig


class MujocoPulseSensorSuite:
    """Synchronized proximity/contact/tactile bridge for the MuJoCo backend."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        calibration_path: Path | str,
        proximity_range_m: float,
        pulse_point_m,
        pulse_axis_xy=(0.0, 1.0),
        seed: int = 0,
    ):
        self.model = model
        self.full_hand_enabled = all(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"handle_l1_{finger}_tactile_pad",
            )
            >= 0
            for finger in ("thumb", "index", "middle", "ring", "little")
        )
        self.contact_reader = (
            MujocoFullHandContactReader(model)
            if self.full_hand_enabled
            else MujocoDualFingerContactReader(model)
        )
        self.proximity_reader = MujocoFingertipProximity(
            model, proximity_range_m
        )
        calibration = HandleL1TenChannelCalibration(calibration_path)
        self.tactile = HandleL1DualFingerTactileSensor(
            hardware_calibration=calibration,
            static_contact_config=StaticContactConfig(
                maximum_force_n=100.0
            ),
            sensor_response_config=SensorResponseConfig(),
            # A strong signal on one finger is allowed; imbalance is useful
            # spatial information rather than a failure condition.
            balance_force_sigma_n=None,
            preload_maximum_force_n=20.0,
            preload_fall_width_n=2.0,
            dynamic_amplitude_kpa=2.0,
            seed=seed,
        )
        self.full_tactile = (
            HandleL1FullHandTactileSensor(
                self.tactile,
                calibration,
                static_contact_config=StaticContactConfig(maximum_force_n=100.0),
                sensor_response_config=SensorResponseConfig(),
                dynamic_amplitude_kpa=2.0,
                seed=seed,
            )
            if self.full_hand_enabled
            else None
        )
        self.last_full_frame = None
        self.pulse_point_m = np.asarray(pulse_point_m, dtype=np.float64)
        self.pulse_axis_xy = np.asarray(pulse_axis_xy, dtype=np.float64)
        self.site_ids = tuple(
            model.site(f"handle_l1_{finger}_tactile_site").id
            for finger in ("index", "middle")
        )
        self.full_site_ids = (
            tuple(
                model.site(f"handle_l1_{finger}_tactile_site").id
                for finger in ("thumb", "index", "middle", "ring", "little")
            )
            if self.full_hand_enabled
            else None
        )

    def reset(self, seed: int | None = None) -> None:
        if self.full_tactile is None:
            self.tactile.reset(seed)
        else:
            self.full_tactile.reset(seed)
        self.last_full_frame = None

    def save_state(self) -> dict:
        snapshot = {
            "last_time_s": self.tactile._last_time_s,
            "contact_latched": self.tactile._contact_latched.copy(),
            "static_previous_indentation_m": np.asarray(
                [
                    model.previous_indentation_m
                    for model in self.tactile.static_models
                ],
                dtype=np.float64,
            ),
            "response_state": self.tactile.sensor_response.state.copy(),
            "response_drift": self.tactile.sensor_response.drift.copy(),
            "response_rng": copy.deepcopy(
                self.tactile.sensor_response.rng.bit_generator.state
            ),
            "tactile_rng": copy.deepcopy(
                self.tactile.rng.bit_generator.state
            ),
        }
        if self.full_tactile is not None:
            snapshot["full_tactile"] = {
                "last_time_s": self.full_tactile._last_time_s,
                "static_previous_indentation_m": {
                    name: model.previous_indentation_m
                    for name, model in self.full_tactile.static_models.items()
                },
                "responses": {
                    name: {
                        "state": response.state.copy(),
                        "drift": response.drift.copy(),
                        "rng": copy.deepcopy(response.rng.bit_generator.state),
                    }
                    for name, response in self.full_tactile.responses.items()
                },
            }
        return snapshot

    def load_state(self, snapshot: dict) -> None:
        self.tactile._last_time_s = snapshot["last_time_s"]
        self.tactile._contact_latched[:] = snapshot["contact_latched"]
        for model, value in zip(
            self.tactile.static_models,
            snapshot["static_previous_indentation_m"],
        ):
            model.previous_indentation_m = float(value)
        self.tactile.sensor_response.state[:] = snapshot["response_state"]
        self.tactile.sensor_response.drift[:] = snapshot["response_drift"]
        self.tactile.sensor_response.rng.bit_generator.state = copy.deepcopy(
            snapshot["response_rng"]
        )
        self.tactile.rng.bit_generator.state = copy.deepcopy(
            snapshot["tactile_rng"]
        )
        if self.full_tactile is not None and "full_tactile" in snapshot:
            full = snapshot["full_tactile"]
            self.full_tactile._last_time_s = full["last_time_s"]
            for name, value in full["static_previous_indentation_m"].items():
                self.full_tactile.static_models[name].previous_indentation_m = float(
                    value
                )
            for name, state in full["responses"].items():
                response = self.full_tactile.responses[name]
                response.state[:] = state["state"]
                response.drift[:] = state["drift"]
                response.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def sample(
        self, data: mujoco.MjData, *, dt_s: float | None = None
    ) -> tuple[ProximityFrame, TactileFrame]:
        timestamp = float(data.time)
        contact = self.contact_reader.read(data, timestamp)
        centers = np.asarray(
            [data.site_xpos[site_id] for site_id in self.site_ids]
        )
        rotations = np.asarray(
            [
                data.site_xmat[site_id].reshape(3, 3)
                for site_id in self.site_ids
            ]
        )
        if self.full_tactile is None:
            tactile = self.tactile.sample(
                time_s=timestamp,
                fingertip_centers_m=centers,
                pulse_point_m=self.pulse_point_m,
                skin_surface_z_m=np.full(2, self.pulse_point_m[2]),
                physical_force_n=contact.normal_force_n,
                pulse_axis_xy=self.pulse_axis_xy,
                dt_s=dt_s,
                fingertip_rotations=rotations,
                skin_normal=np.asarray([0.0, 0.0, 1.0]),
                physical_contact_points_local_m=tuple(
                    finger.points_local_m for finger in contact.fingers
                ),
                physical_contact_strength=tuple(
                    finger.point_forces_n for finger in contact.fingers
                ),
            )
        else:
            assert self.full_site_ids is not None
            full_centers = np.asarray(
                [data.site_xpos[site_id] for site_id in self.full_site_ids]
            )
            full_rotations = np.asarray(
                [
                    data.site_xmat[site_id].reshape(3, 3)
                    for site_id in self.full_site_ids
                ]
            )
            self.last_full_frame = self.full_tactile.sample(
                time_s=timestamp,
                centers_m=full_centers,
                rotations_world_from_tactile=full_rotations,
                pulse_point_m=self.pulse_point_m,
                pulse_axis_xy=self.pulse_axis_xy,
                skin_normal=np.asarray([0.0, 0.0, 1.0]),
                contacts=contact,
                dt_s=dt_s,
            )
            tactile = self.last_full_frame.diagnostic
        proximity = self.proximity_reader.read(data, timestamp)
        if tactile.contact.any():
            distances = proximity.distance_m.copy()
            valid = proximity.valid.copy()
            # A MuJoCo ray whose origin has entered a compliant geom does not
            # report that geom's entry surface. Physical pad contact is the
            # authoritative zero-gap condition for that finger.
            distances[tactile.contact] = 0.0
            valid[tactile.contact] = True
            proximity = ProximityFrame(
                distance_m=distances,
                valid=valid,
                timestamp_s=timestamp,
            )
        return proximity, tactile
