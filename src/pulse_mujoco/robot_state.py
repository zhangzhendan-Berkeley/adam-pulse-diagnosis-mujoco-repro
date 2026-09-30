from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .frames import dual_probe_frame


@dataclass(frozen=True)
class RobotStateFrame:
    joint_names: tuple[str, ...]
    joint_position_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    probe_position_m: np.ndarray
    probe_rotation_world_from_frame: np.ndarray
    probe_linear_velocity_m_s: np.ndarray
    probe_angular_velocity_rad_s: np.ndarray
    timestamp_s: float


class MujocoRobotStateReader:
    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: list[str] | tuple[str, ...],
    ):
        self.model = model
        self.joint_names = tuple(joint_names)
        self.joint_ids = np.asarray(
            [model.joint(name).id for name in self.joint_names], dtype=int
        )
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = model.jnt_dofadr[self.joint_ids]
        self.index_site_id = model.site(
            "handle_l1_index_tactile_site"
        ).id
        self.middle_site_id = model.site(
            "handle_l1_middle_tactile_site"
        ).id

    def read(self, data: mujoco.MjData) -> RobotStateFrame:
        probe = dual_probe_frame(self.model, data)
        velocities = []
        for site_id in (self.index_site_id, self.middle_site_id):
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model,
                data,
                mujoco.mjtObj.mjOBJ_SITE,
                site_id,
                velocity,
                0,
            )
            velocities.append(velocity)
        mean_velocity = np.mean(velocities, axis=0)
        return RobotStateFrame(
            joint_names=self.joint_names,
            joint_position_rad=data.qpos[
                self.qpos_addresses
            ].astype(np.float32, copy=True),
            joint_velocity_rad_s=data.qvel[
                self.dof_addresses
            ].astype(np.float32, copy=True),
            probe_position_m=probe.center.position_m.astype(
                np.float32, copy=True
            ),
            probe_rotation_world_from_frame=(
                probe.center.rotation_world_from_frame.astype(
                    np.float32, copy=True
                )
            ),
            # MuJoCo object velocity layout is angular then linear.
            probe_linear_velocity_m_s=mean_velocity[3:].astype(
                np.float32
            ),
            probe_angular_velocity_rad_s=mean_velocity[:3].astype(
                np.float32
            ),
            timestamp_s=float(data.time),
        )
