from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


FINGERS = ("index", "middle")
FULL_HAND_FINGERS = ("thumb", "index", "middle", "ring", "little")


@dataclass(frozen=True)
class FingerContact:
    normal_force_n: float
    points_local_m: np.ndarray
    point_forces_n: np.ndarray
    in_contact: bool


@dataclass(frozen=True)
class DualFingerContactFrame:
    fingers: tuple[FingerContact, FingerContact]
    timestamp_s: float

    @property
    def normal_force_n(self) -> np.ndarray:
        return np.asarray(
            [finger.normal_force_n for finger in self.fingers],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class FullHandContactFrame:
    finger_names: tuple[str, ...]
    fingers: tuple[FingerContact, ...]
    timestamp_s: float

    @property
    def normal_force_n(self) -> np.ndarray:
        return np.asarray(
            [finger.normal_force_n for finger in self.fingers],
            dtype=np.float32,
        )

    def select(self, names: tuple[str, ...]) -> tuple[FingerContact, ...]:
        by_name = dict(zip(self.finger_names, self.fingers))
        return tuple(by_name[name] for name in names)


class MujocoDualFingerContactReader:
    """Read only contacts made by the two physical fingertip pad geoms.

    MuJoCo contact positions are transformed into each tactile site's local
    frame. Joint-side distal-link collisions therefore cannot masquerade as a
    taxel hit.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.pad_geom_ids = tuple(
            model.geom(f"handle_l1_{finger}_tactile_pad").id
            for finger in FINGERS
        )
        self.site_ids = tuple(
            model.site(f"handle_l1_{finger}_tactile_site").id
            for finger in FINGERS
        )

    def read(
        self, data: mujoco.MjData, timestamp_s: float
    ) -> DualFingerContactFrame:
        points: list[list[np.ndarray]] = [[], []]
        strengths: list[list[float]] = [[], []]
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            for finger_index, pad_geom_id in enumerate(self.pad_geom_ids):
                if pad_geom_id not in (contact.geom1, contact.geom2):
                    continue
                wrench = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(
                    self.model, data, contact_index, wrench
                )
                normal_force = max(float(wrench[0]), 0.0)
                if normal_force <= 1e-8:
                    continue
                site_id = self.site_ids[finger_index]
                rotation = data.site_xmat[site_id].reshape(3, 3)
                local_point = rotation.T @ (
                    np.asarray(contact.pos, dtype=np.float64)
                    - data.site_xpos[site_id]
                )
                points[finger_index].append(local_point)
                strengths[finger_index].append(normal_force)
        frames = []
        for finger_points, finger_strengths in zip(points, strengths):
            force = np.asarray(finger_strengths, dtype=np.float32)
            local = (
                np.asarray(finger_points, dtype=np.float32).reshape(-1, 3)
                if finger_points
                else np.empty((0, 3), dtype=np.float32)
            )
            frames.append(
                FingerContact(
                    normal_force_n=float(force.sum()),
                    points_local_m=local,
                    point_forces_n=force,
                    in_contact=bool(force.size and force.sum() > 0.01),
                )
            )
        return DualFingerContactFrame(
            fingers=(frames[0], frames[1]),
            timestamp_s=float(timestamp_s),
        )


class MujocoFullHandContactReader:
    """Read contacts from all five continuous custom-hand fingertip shells."""

    def __init__(
        self,
        model: mujoco.MjModel,
        finger_names: tuple[str, ...] = FULL_HAND_FINGERS,
    ):
        self.model = model
        self.finger_names = tuple(finger_names)
        self.pad_geom_ids = tuple(
            model.geom(f"handle_l1_{finger}_tactile_pad").id
            for finger in self.finger_names
        )
        self.site_ids = tuple(
            model.site(f"handle_l1_{finger}_tactile_site").id
            for finger in self.finger_names
        )

    def read(self, data: mujoco.MjData, timestamp_s: float) -> FullHandContactFrame:
        points: list[list[np.ndarray]] = [[] for _ in self.finger_names]
        strengths: list[list[float]] = [[] for _ in self.finger_names]
        geom_to_finger = {
            geom_id: index for index, geom_id in enumerate(self.pad_geom_ids)
        }
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            finger_index = geom_to_finger.get(int(contact.geom1))
            if finger_index is None:
                finger_index = geom_to_finger.get(int(contact.geom2))
            if finger_index is None:
                continue
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, data, contact_index, wrench)
            normal_force = max(float(wrench[0]), 0.0)
            if normal_force <= 1e-8:
                continue
            site_id = self.site_ids[finger_index]
            rotation = data.site_xmat[site_id].reshape(3, 3)
            local_point = rotation.T @ (
                np.asarray(contact.pos, dtype=np.float64)
                - data.site_xpos[site_id]
            )
            points[finger_index].append(local_point)
            strengths[finger_index].append(normal_force)
        frames = []
        for finger_points, finger_strengths in zip(points, strengths):
            force = np.asarray(finger_strengths, dtype=np.float32)
            local = (
                np.asarray(finger_points, dtype=np.float32).reshape(-1, 3)
                if finger_points
                else np.empty((0, 3), dtype=np.float32)
            )
            frames.append(
                FingerContact(
                    normal_force_n=float(force.sum()),
                    points_local_m=local,
                    point_forces_n=force,
                    in_contact=bool(force.size and force.sum() > 0.01),
                )
            )
        return FullHandContactFrame(
            finger_names=self.finger_names,
            fingers=tuple(frames),
            timestamp_s=float(timestamp_s),
        )
