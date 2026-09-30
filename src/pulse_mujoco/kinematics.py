from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .frames import INDEX_PAD_SITE, MIDDLE_PAD_SITE, dual_probe_frame


@dataclass(frozen=True)
class IKSolution:
    joint_positions_rad: np.ndarray
    converged: bool
    iterations: int
    position_error_m: float
    rotation_error_rad: float


class AdamLiteTaskKinematics:
    """Damped-least-squares IK for the Adam Lite left-arm four joints.

    The task frame is the measured midpoint of the two articulated fingertip
    tactile sites. The waist is structurally fixed and no virtual wrist DOF is
    added to the official Lite model.
    """

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
        self.qpos_addresses = np.asarray(
            [model.jnt_qposadr[joint_id] for joint_id in self.joint_ids],
            dtype=int,
        )
        self.dof_addresses = np.asarray(
            [model.jnt_dofadr[joint_id] for joint_id in self.joint_ids],
            dtype=int,
        )
        self.lower = model.jnt_range[self.joint_ids, 0].astype(np.float64)
        self.upper = model.jnt_range[self.joint_ids, 1].astype(np.float64)
        self.site_ids = (
            model.site(INDEX_PAD_SITE).id,
            model.site(MIDDLE_PAD_SITE).id,
        )

    def positions(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_addresses].astype(np.float64, copy=True)

    def set_positions(
        self,
        data: mujoco.MjData,
        joint_positions_rad,
        *,
        lower=None,
        upper=None,
    ) -> None:
        positions = np.asarray(joint_positions_rad, dtype=np.float64)
        if positions.shape != (len(self.joint_names),):
            raise ValueError(
                f"joint_positions_rad must have shape "
                f"({len(self.joint_names)},), got {positions.shape}"
            )
        lower_bound = self.lower if lower is None else np.asarray(lower)
        upper_bound = self.upper if upper is None else np.asarray(upper)
        data.qpos[self.qpos_addresses] = np.clip(
            positions, lower_bound, upper_bound
        )

    def probe_jacobian(
        self,
        data: mujoco.MjData,
    ) -> tuple[np.ndarray, np.ndarray]:
        position_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        rotation_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        scratch_position = np.zeros_like(position_jacobian)
        scratch_rotation = np.zeros_like(rotation_jacobian)
        for site_id in self.site_ids:
            scratch_position.fill(0.0)
            scratch_rotation.fill(0.0)
            mujoco.mj_jacSite(
                self.model,
                data,
                scratch_position,
                scratch_rotation,
                site_id,
            )
            position_jacobian += 0.5 * scratch_position
            rotation_jacobian += 0.5 * scratch_rotation
        return (
            position_jacobian[:, self.dof_addresses],
            rotation_jacobian[:, self.dof_addresses],
        )

    @staticmethod
    def rotation_error(
        current: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        current = np.asarray(current, dtype=np.float64).reshape(3, 3)
        target = np.asarray(target, dtype=np.float64).reshape(3, 3)
        return 0.5 * sum(
            np.cross(current[:, axis], target[:, axis])
            for axis in range(3)
        )

    def solve(
        self,
        data: mujoco.MjData,
        target_position_m,
        target_rotation=None,
        *,
        maximum_iterations: int = 240,
        damping: float = 0.035,
        maximum_joint_step_rad: float = 0.08,
        position_tolerance_m: float = 5e-4,
        rotation_tolerance_rad: float = 0.03,
        rotation_weight: float = 0.30,
        joint_lower_rad=None,
        joint_upper_rad=None,
    ) -> IKSolution:
        target_position = np.asarray(target_position_m, dtype=np.float64)
        if target_position.shape != (3,):
            raise ValueError("target_position_m must have shape (3,)")
        target_rotation_array = (
            None
            if target_rotation is None
            else np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
        )
        lower = (
            self.lower
            if joint_lower_rad is None
            else np.maximum(self.lower, np.asarray(joint_lower_rad, dtype=float))
        )
        upper = (
            self.upper
            if joint_upper_rad is None
            else np.minimum(self.upper, np.asarray(joint_upper_rad, dtype=float))
        )
        if lower.shape != self.lower.shape or upper.shape != self.upper.shape:
            raise ValueError("joint bound overrides must match controlled joints")
        if np.any(lower > upper):
            raise ValueError("joint lower bounds exceed upper bounds")
        self.set_positions(data, self.positions(data), lower=lower, upper=upper)
        converged = False
        position_error_norm = float("inf")
        rotation_error_norm = 0.0
        iterations = 0
        for iterations in range(1, int(maximum_iterations) + 1):
            mujoco.mj_forward(self.model, data)
            probe = dual_probe_frame(self.model, data)
            position_error = target_position - probe.center.position_m
            position_error_norm = float(np.linalg.norm(position_error))
            position_jacobian, rotation_jacobian = self.probe_jacobian(data)
            if target_rotation_array is None:
                error = position_error
                jacobian = position_jacobian
                rotation_error_norm = 0.0
                rotation_ok = True
            else:
                rotation_error = self.rotation_error(
                    probe.center.rotation_world_from_frame,
                    target_rotation_array,
                )
                rotation_error_norm = float(np.linalg.norm(rotation_error))
                error = np.concatenate(
                    (position_error, rotation_weight * rotation_error)
                )
                jacobian = np.vstack(
                    (position_jacobian, rotation_weight * rotation_jacobian)
                )
                rotation_ok = rotation_error_norm <= rotation_tolerance_rad
            if position_error_norm <= position_tolerance_m and rotation_ok:
                converged = True
                break
            regularized = (
                jacobian @ jacobian.T
                + float(damping) ** 2 * np.eye(jacobian.shape[0])
            )
            delta = jacobian.T @ np.linalg.solve(regularized, error)
            delta = np.clip(
                delta,
                -float(maximum_joint_step_rad),
                float(maximum_joint_step_rad),
            )
            current = self.positions(data)
            self.set_positions(
                data, current + delta, lower=lower, upper=upper
            )
        mujoco.mj_forward(self.model, data)
        final_probe = dual_probe_frame(self.model, data)
        position_error_norm = float(
            np.linalg.norm(target_position - final_probe.center.position_m)
        )
        if target_rotation_array is None:
            rotation_error_norm = 0.0
            rotation_ok = True
        else:
            rotation_error_norm = float(
                np.linalg.norm(
                    self.rotation_error(
                        final_probe.center.rotation_world_from_frame,
                        target_rotation_array,
                    )
                )
            )
            rotation_ok = rotation_error_norm <= rotation_tolerance_rad
        converged = bool(
            position_error_norm <= position_tolerance_m and rotation_ok
        )
        return IKSolution(
            joint_positions_rad=self.positions(data),
            converged=converged,
            iterations=iterations,
            position_error_m=position_error_norm,
            rotation_error_rad=rotation_error_norm,
        )
