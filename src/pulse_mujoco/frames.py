from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


TCP_MOUNT_SITE = "handle_l1_tcp_mount"
INDEX_PAD_SITE = "handle_l1_index_tactile_site"
MIDDLE_PAD_SITE = "handle_l1_middle_tactile_site"
PATIENT_WRIST_SITE = "patient_wrist_frame"
PULSE_REFERENCE_SITE = "pulse_center_reference"


@dataclass(frozen=True)
class FramePose:
    position_m: np.ndarray
    rotation_world_from_frame: np.ndarray


@dataclass(frozen=True)
class DualProbeFrame:
    center: FramePose
    index_pad: FramePose
    middle_pad: FramePose
    skin_facing_normal_world: np.ndarray
    finger_separation_axis_world: np.ndarray


def site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
) -> FramePose:
    site_id = model.site(name).id
    return FramePose(
        position_m=data.site_xpos[site_id].astype(np.float64, copy=True),
        rotation_world_from_frame=data.site_xmat[site_id]
        .reshape(3, 3)
        .astype(np.float64, copy=True),
    )


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def dual_probe_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> DualProbeFrame:
    index = site_pose(model, data, INDEX_PAD_SITE)
    middle = site_pose(model, data, MIDDLE_PAD_SITE)
    center_rotation = _project_rotation(
        index.rotation_world_from_frame
        + middle.rotation_world_from_frame
    )
    separation = middle.position_m - index.position_m
    separation_norm = float(np.linalg.norm(separation))
    if separation_norm > 1e-12:
        separation /= separation_norm
    # In the supplied Handle_L1 frame, tactile-frame -X is the physical
    # outward direction from the finger pad.
    skin_facing_normal = -center_rotation[:, 0]
    return DualProbeFrame(
        center=FramePose(
            position_m=0.5 * (index.position_m + middle.position_m),
            rotation_world_from_frame=center_rotation,
        ),
        index_pad=index,
        middle_pad=middle,
        skin_facing_normal_world=skin_facing_normal,
        finger_separation_axis_world=separation,
    )


def patient_wrist_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> FramePose:
    """Patient convention: +X artery, +Z outward skin normal."""
    return site_pose(model, data, PATIENT_WRIST_SITE)


def pulse_reference_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    return site_pose(model, data, PULSE_REFERENCE_SITE).position_m
