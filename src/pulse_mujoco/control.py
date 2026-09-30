from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ControlStep:
    maximum_position_error_rad: float
    maximum_velocity_rad_s: float
    maximum_commanded_torque_nm: float


class AdamJointPDController:
    """Torque PD for Adam body motors plus direct hand position control."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.adam_actuators: dict[str, int] = {}
        self.hand_actuators: dict[str, int] = {}
        for actuator_id in range(model.nu):
            actuator_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            joint_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            # MuJoCo ``position`` actuators use an affine bias, while the
            # official Adam body actuators are bias-free ``motor`` actuators.
            # This works for both the suffixed Handle_L1 names and the
            # official Adam Pro hand, whose actuator names equal joint names.
            if model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE:
                self.hand_actuators[joint_name] = actuator_id
            else:
                self.adam_actuators[joint_name] = actuator_id
        self.adam_joint_names = tuple(self.adam_actuators)
        self.qpos_addresses = {
            name: int(model.jnt_qposadr[model.joint(name).id])
            for name in self.adam_joint_names
        }
        self.dof_addresses = {
            name: int(model.jnt_dofadr[model.joint(name).id])
            for name in self.adam_joint_names
        }
        self.target_positions = np.zeros(len(self.adam_joint_names))
        self.hand_targets = {
            name: 0.0 for name in self.hand_actuators
        }
        self.feedforward_torque_nm = {
            name: 0.0 for name in self.adam_actuators
        }
        self.kp = {
            name: self._gain_for_joint(name)[0]
            for name in self.adam_joint_names
        }
        self.kd = {
            name: self._gain_for_joint(name)[1]
            for name in self.adam_joint_names
        }

    @staticmethod
    def _gain_for_joint(name: str) -> tuple[float, float]:
        if name.startswith(("hip", "knee")):
            return 180.0, 12.0
        if name.startswith("ankle"):
            return 45.0, 4.0
        if name == "waistRoll" or name == "waistPitch":
            return 180.0, 10.0
        if name == "waistYaw":
            return 120.0, 7.0
        if "shoulderPitch" in name or "shoulderRoll" in name:
            return 70.0, 5.0
        if "shoulderYaw" in name:
            return 45.0, 3.5
        if "elbow" in name:
            return 40.0, 3.0
        return 30.0, 2.5

    def capture_current_targets(self, data: mujoco.MjData) -> None:
        self.target_positions = np.asarray(
            [data.qpos[self.qpos_addresses[name]] for name in self.adam_joint_names],
            dtype=np.float64,
        )
        for name in self.hand_targets:
            address = int(self.model.jnt_qposadr[self.model.joint(name).id])
            self.hand_targets[name] = float(data.qpos[address])

    def set_adam_targets(self, targets: dict[str, float]) -> None:
        index = {name: i for i, name in enumerate(self.adam_joint_names)}
        for name, value in targets.items():
            if name not in index:
                raise KeyError(f"unknown controlled Adam joint: {name}")
            joint_id = self.model.joint(name).id
            lower, upper = self.model.jnt_range[joint_id]
            self.target_positions[index[name]] = float(
                np.clip(value, lower, upper)
            )

    def set_hand_targets(self, targets: dict[str, float]) -> None:
        for name, value in targets.items():
            if name not in self.hand_targets:
                raise KeyError(f"unknown hand position actuator joint: {name}")
            actuator_id = self.hand_actuators[name]
            lower, upper = self.model.actuator_ctrlrange[actuator_id]
            self.hand_targets[name] = float(np.clip(value, lower, upper))

    def set_adam_feedforward_torques(
        self, torques_nm: dict[str, float]
    ) -> None:
        unknown = set(torques_nm) - set(self.feedforward_torque_nm)
        if unknown:
            raise KeyError(
                f"unknown Adam feedforward joints: {sorted(unknown)}"
            )
        for name in self.feedforward_torque_nm:
            self.feedforward_torque_nm[name] = float(
                torques_nm.get(name, 0.0)
            )

    def apply(self, data: mujoco.MjData) -> ControlStep:
        maximum_error = 0.0
        maximum_velocity = 0.0
        maximum_torque = 0.0
        for target_index, name in enumerate(self.adam_joint_names):
            actuator_id = self.adam_actuators[name]
            qpos_address = self.qpos_addresses[name]
            dof_address = self.dof_addresses[name]
            error = float(
                self.target_positions[target_index]
                - data.qpos[qpos_address]
            )
            velocity = float(data.qvel[dof_address])
            torque = (
                self.kp[name] * error
                - self.kd[name] * velocity
                + float(data.qfrc_bias[dof_address])
                + self.feedforward_torque_nm[name]
            )
            lower, upper = self.model.actuator_ctrlrange[actuator_id]
            torque = float(np.clip(torque, lower, upper))
            data.ctrl[actuator_id] = torque
            maximum_error = max(maximum_error, abs(error))
            maximum_velocity = max(maximum_velocity, abs(velocity))
            maximum_torque = max(maximum_torque, abs(torque))
        for name, target in self.hand_targets.items():
            data.ctrl[self.hand_actuators[name]] = target
        return ControlStep(
            maximum_position_error_rad=maximum_error,
            maximum_velocity_rad_s=maximum_velocity,
            maximum_commanded_torque_nm=maximum_torque,
        )

    def step(
        self,
        data: mujoco.MjData,
        physics_steps: int = 1,
    ) -> ControlStep:
        diagnostics = ControlStep(0.0, 0.0, 0.0)
        for _ in range(int(physics_steps)):
            diagnostics = self.apply(data)
            mujoco.mj_step(self.model, data)
        return diagnostics


# Kept for source compatibility with the Adam Lite baseline and external code.
AdamLiteJointPDController = AdamJointPDController
