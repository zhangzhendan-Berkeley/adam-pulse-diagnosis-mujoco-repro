from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .actions import PulseAction
from .contact import MujocoDualFingerContactReader
from .contact import FULL_HAND_FINGERS
from .config_loader import load_project_config
from .control import AdamJointPDController
from .frames import dual_probe_frame
from .kinematics import AdamLiteTaskKinematics
from .observation import PulseObservation
from .pressure_target import nominal_pressure_to_force_target_n
from .rgbd import MujocoRGBDCamera
from .robot_state import MujocoRobotStateReader
from .sensor_suite import MujocoPulseSensorSuite
from .task import TaskPhase


def _bounded_reference_window(
    physical_lower: np.ndarray,
    physical_upper: np.ndarray,
    reference: np.ndarray,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect a local hold envelope with physical limits safely.

    Torque dynamics may settle a joint a few floating-point units outside its
    nominal range. Clamp that measured reference before constructing the local
    IK window so its lower and upper bounds can never cross.
    """

    lower = np.asarray(physical_lower, dtype=float)
    upper = np.asarray(physical_upper, dtype=float)
    center = np.asarray(reference, dtype=float)
    if lower.shape != upper.shape or center.shape != lower.shape:
        raise ValueError("joint window arrays must have matching shapes")
    if not np.isfinite(center).all() or not np.isfinite(half_width):
        raise ValueError("joint window reference and width must be finite")
    if half_width < 0.0 or np.any(lower > upper):
        raise ValueError("joint window bounds and width must be valid")
    center = np.clip(center, lower, upper)
    bounded_lower = np.maximum(lower, center - float(half_width))
    bounded_upper = np.minimum(upper, center + float(half_width))
    return bounded_lower, bounded_upper, center


def _rejected_ik_fallback_target(
    latched_target: np.ndarray,
    current_position: np.ndarray,
    *,
    pressure_hold_command: bool,
) -> np.ndarray:
    """Choose the Cartesian reference after a rejected IK proposal.

    During pressure hold the target was latched at physical first contact.  If
    a transient IK miss replaces it with the displaced current pose, contact
    load can ratchet the hand across the skin one policy tick at a time.  Keep
    the last safe reference in that case; free-space commands retain the old
    behaviour of rebasing at the actually reached pose.
    """

    target = np.asarray(latched_target, dtype=float)
    current = np.asarray(current_position, dtype=float)
    if target.shape != (3,) or current.shape != (3,):
        raise ValueError("Cartesian fallback targets must have shape (3,)")
    if not np.isfinite(target).all() or not np.isfinite(current).all():
        raise ValueError("Cartesian fallback targets must be finite")
    return (target if pressure_hold_command else current).copy()


def _should_constrain_task_rotation(
    *,
    has_rotation_command: bool,
    task_rotation_hold_enabled: bool,
    pressure_hold_command: bool,
    pressure_recovery_rotation_hold_enabled: bool,
) -> bool:
    """Select explicit, global, or recovery-scoped orientation constraints."""

    return bool(
        has_rotation_command
        or task_rotation_hold_enabled
        or (
            pressure_hold_command
            and pressure_recovery_rotation_hold_enabled
        )
    )


def _clearance_force_scale(
    clearance_m: float, *, hard_m: float, soft_m: float
) -> float:
    """Continuous force-feedforward scale for a forbidden-geometry barrier."""

    values = np.asarray([clearance_m, hard_m, soft_m], dtype=float)
    if not np.isfinite(values).all() or hard_m < 0.0 or soft_m <= hard_m:
        raise ValueError("clearance barrier distances must be finite and ordered")
    return float(np.clip((clearance_m - hard_m) / (soft_m - hard_m), 0.0, 1.0))


def _asymmetric_clearance_filter(
    previous: float,
    target: float,
    *,
    fall_alpha: float,
    rise_alpha: float,
) -> float:
    """Unload quickly near collision and restore pressure without chatter."""

    values = np.asarray(
        [previous, target, fall_alpha, rise_alpha], dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("clearance filter inputs must be finite")
    if not (0.0 <= previous <= 1.0 and 0.0 <= target <= 1.0):
        raise ValueError("clearance filter scales must be in [0, 1]")
    if not (0.0 < fall_alpha <= 1.0 and 0.0 < rise_alpha <= 1.0):
        raise ValueError("clearance filter alphas must be in (0, 1]")
    alpha = fall_alpha if target < previous else rise_alpha
    return float(np.clip(previous + alpha * (target - previous), 0.0, 1.0))


class MujocoPulseDiagnosisBackend:
    """Tick-driven Adam pulse-diagnosis backend.

    Direct qpos writes are restricted to reset/load-state. Runtime arm motion
    is produced by torque PD toward collision-checked IK or named waypoint
    targets.
    """

    def __init__(
        self,
        project_root: Path | str,
        *,
        config_path: Path | str = "configs/alignment.yaml",
        include_rgbd: bool = False,
        rgbd_capture_interval: int = 1,
        policy_rate_hz: float = 20.0,
        maximum_steps: int = 6000,
    ):
        self.root = Path(project_root)
        requested_config = Path(config_path)
        self.config_path = (
            requested_config
            if requested_config.is_absolute()
            else self.root / requested_config
        )
        self.config = load_project_config(self.config_path)
        configured_model = Path(self.config["project"]["generated_model"])
        self.model_path = (
            configured_model
            if configured_model.is_absolute()
            else self.root / configured_model
        )
        self.model = mujoco.MjModel.from_xml_path(
            str(self.model_path.resolve())
        )
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.arm_joint_names = tuple(
            self.config["robot"]["controlled_chain"]
        )
        self.kinematics = AdamLiteTaskKinematics(
            self.model, self.arm_joint_names
        )
        self.controller = AdamJointPDController(self.model)
        self.state_reader = MujocoRobotStateReader(
            self.model, self.arm_joint_names
        )
        sensor_config = self.config["sensors"]
        pulse_center = self.config["scene"]["patient_forearm"][
            "pulse_center_m"
        ]
        self.sensors = MujocoPulseSensorSuite(
            self.model,
            calibration_path=self.root / sensor_config["calibration"],
            proximity_range_m=sensor_config["proximity_range_m"],
            pulse_point_m=pulse_center,
        )
        self.diagnostic_finger_names = tuple(
            self.config["task"].get(
                "diagnostic_fingers", ("index", "middle")
            )
        )
        if (
            not self.diagnostic_finger_names
            or len(set(self.diagnostic_finger_names))
            != len(self.diagnostic_finger_names)
            or any(
                name not in FULL_HAND_FINGERS
                for name in self.diagnostic_finger_names
            )
        ):
            raise ValueError("diagnostic_fingers must be unique full-hand names")
        if any(
            int(self.config["hand"]["channel_counts"][name]) != 10
            for name in self.diagnostic_finger_names
        ):
            raise ValueError(
                "current multi-finger expert requires 10-channel active fingers"
            )
        if (
            self.diagnostic_finger_names != ("index", "middle")
            and not self.sensors.full_hand_enabled
        ):
            raise ValueError("multi-finger diagnosis requires full-hand tactile")
        self._full_finger_index = {
            name: index for index, name in enumerate(FULL_HAND_FINGERS)
        }
        self.pressure_clearance_geom_pairs: list[tuple[int, int]] = []
        for pair in self.config["task"].get(
            "pressure_forbidden_clearance_pairs", []
        ):
            body_id = self.model.body(str(pair["body"])).id
            geom_id = self.model.geom(str(pair["environment_geom"])).id
            start = int(self.model.body_geomadr[body_id])
            count = int(self.model.body_geomnum[body_id])
            for candidate in range(start, start + count):
                if bool(
                    self.model.geom_contype[candidate]
                    or self.model.geom_conaffinity[candidate]
                ):
                    self.pressure_clearance_geom_pairs.append(
                        (candidate, geom_id)
                    )
        self.last_pressure_forbidden_clearance_m = float("inf")
        self.last_pressure_clearance_force_scale = 1.0
        self.contact_reader = MujocoDualFingerContactReader(self.model)
        self.include_rgbd = bool(include_rgbd)
        self.rgbd_capture_interval = int(rgbd_capture_interval)
        if self.rgbd_capture_interval < 1:
            raise ValueError("rgbd_capture_interval must be positive")
        self._rgbd_frame = None
        self._rgbd_observation_count = 0
        self.camera = None
        if self.include_rgbd:
            camera_config = self.config["scene"]["camera"]
            self.camera = MujocoRGBDCamera(
                self.model,
                camera_config["name"],
                camera_config["width"],
                camera_config["height"],
            )
        self.physics_steps_per_policy = max(
            1,
            int(
                round(
                    1.0
                    / (
                        float(policy_rate_hz)
                        * float(self.model.opt.timestep)
                    )
                )
            ),
        )
        self.maximum_steps = int(maximum_steps)
        self.environment_step = 0
        self.phase = TaskPhase.RESET
        self.previous_action = np.zeros(7, dtype=np.float32)
        self.task_target_position = np.zeros(3, dtype=np.float64)
        self.task_target_rotation = np.eye(3, dtype=np.float64)
        self.last_tactile = None
        self.last_tactile_full = None
        self.last_proximity = None
        self.pressure_latched = False
        self.pressure_command_force_n = 0.0
        self.last_pressure_forbidden_clearance_m = float("inf")
        self.last_pressure_clearance_force_scale = 1.0
        self.contact_wrist_reference_rad = None
        self.collection_hold_joint_positions_rad = None
        self.task_rotation_hold_enabled = False
        self.pressure_recovery_rotation_hold_enabled = False
        self.pressure_recovery_joint_hold_positions_rad = None
        self.runtime_forbidden_contact_events = 0
        self.joint_path_preview_rejection_events = 0
        self.contact_wrist_stabilization_enabled = bool(
            self.config["task"].get(
                "contact_wrist_stabilization_enabled", False
            )
        )
        self.last_info: dict = {}

    def _set_joint(self, name: str, value: float) -> None:
        self.data.qpos[
            self.model.jnt_qposadr[self.model.joint(name).id]
        ] = float(value)

    def _set_initial_pose(self) -> None:
        self.data = mujoco.MjData(self.model)
        for name, value in self.config["robot"]["reset_pose_rad"].items():
            self._set_joint(name, value)
        initial_hand_pose = self.config["hand"].get(
            "reset_pose_rad", self.config["hand"]["diagnosis_pose_rad"]
        )
        for name, value in initial_hand_pose.items():
            self._set_joint(name, value)
        for coupling in self.config["hand"].get("position_couplings", []):
            distal = coupling["joint"]
            primary = coupling["source_joint"]
            multiplier = float(coupling["multiplier"])
            primary_value = self.data.qpos[
                self.model.jnt_qposadr[self.model.joint(primary).id]
            ]
            self._set_joint(distal, multiplier * primary_value)
        mujoco.mj_forward(self.model, self.data)

    def reset(
        self, seed: int | None = None
    ) -> tuple[PulseObservation, dict]:
        self._set_initial_pose()
        self.ik_data = mujoco.MjData(self.model)
        self.controller = AdamJointPDController(self.model)
        self.controller.capture_current_targets(self.data)
        self.sensors.reset(seed)
        self.environment_step = 0
        self.phase = TaskPhase.OBSERVE
        self.previous_action.fill(0.0)
        self.pressure_latched = False
        self.pressure_command_force_n = 0.0
        self.last_pressure_forbidden_clearance_m = float("inf")
        self.last_pressure_clearance_force_scale = 1.0
        self.contact_wrist_reference_rad = None
        self.collection_hold_joint_positions_rad = None
        self.task_rotation_hold_enabled = False
        self.pressure_recovery_rotation_hold_enabled = False
        self.pressure_recovery_joint_hold_positions_rad = None
        self.runtime_forbidden_contact_events = 0
        self.joint_path_preview_rejection_events = 0
        self.controller.set_adam_feedforward_torques({})
        self._rgbd_frame = None
        self._rgbd_observation_count = 0
        probe = dual_probe_frame(self.model, self.data)
        self.task_target_position = probe.center.position_m.copy()
        self.task_target_rotation = (
            probe.center.rotation_world_from_frame.copy()
        )
        observation = self._observation()
        info = self._info(ik_converged=True)
        self.last_info = info
        return observation, info

    def _allowed_contact(self, geom1: int, geom2: int) -> bool:
        names = {
            mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom1)
            ),
            mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom2)
            ),
        }
        if "floor" in names:
            return bool(names & {"toe1_left", "toe_right"})
        allowed_fingers = self.config["task"].get(
            "allowed_patient_contact_fingers", ("index", "middle")
        )
        pad = {
            f"handle_l1_{finger}_tactile_pad" for finger in allowed_fingers
        }
        patient = {
            "patient_forearm_skin",
        }
        return bool(names & pad and names & patient)

    def diagnostic_contact(self) -> np.ndarray:
        if self.last_tactile_full is not None:
            return np.asarray(
                [
                    self.last_tactile_full.contact[self._full_finger_index[name]]
                    for name in self.diagnostic_finger_names
                ],
                dtype=bool,
            )
        if self.last_tactile is None:
            return np.zeros(len(self.diagnostic_finger_names), dtype=bool)
        return self.last_tactile.contact.copy()

    def diagnostic_force_n(self) -> np.ndarray:
        if self.last_tactile_full is not None:
            return np.asarray(
                [
                    self.last_tactile_full.total_normal_force_n[
                        self._full_finger_index[name]
                    ]
                    for name in self.diagnostic_finger_names
                ],
                dtype=np.float32,
            )
        if self.last_tactile is None:
            return np.zeros(len(self.diagnostic_finger_names), dtype=np.float32)
        return self.last_tactile.total_normal_force_n.copy()

    def forbidden_contacts(
        self, data: mujoco.MjData | None = None
    ) -> list[dict]:
        data = self.data if data is None else data
        result = []
        for contact in data.contact[: data.ncon]:
            if self._allowed_contact(contact.geom1, contact.geom2):
                continue
            result.append(
                {
                    "geom1": mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        int(contact.geom1),
                    ),
                    "geom2": mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        int(contact.geom2),
                    ),
                    "distance_m": float(contact.dist),
                    "body1": mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(self.model.geom_bodyid[contact.geom1]),
                    ),
                    "body2": mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(self.model.geom_bodyid[contact.geom2]),
                    ),
                }
            )
        return result

    def runtime_forbidden_contacts(self) -> list[dict]:
        """Read and count an actual forbidden contact from live physics.

        Candidate IK collisions are deliberately excluded: this counter is
        the episode safety outcome and therefore only records contacts that
        occurred in ``self.data``.  Every caller stops the current control
        interval immediately when a non-empty result is returned, so one
        physical detection is counted once instead of once per physics tick.
        """
        contacts = self.forbidden_contacts(self.data)
        if contacts:
            self.runtime_forbidden_contact_events += 1
        return contacts

    def preview_forbidden_joint_path(
        self,
        target_positions_rad: np.ndarray,
    ) -> list[dict]:
        """Return forbidden contacts along the kinematic arm-target path.

        A collision-free IK endpoint is not sufficient for a torque-PD arm:
        the live robot can traverse an unsafe intermediate configuration while
        converging to that endpoint.  This preview uses the independent IK
        data buffer, so it neither advances physics nor changes controller,
        sensor, or task state.  It intentionally uses :meth:`_allowed_contact`
        through :meth:`forbidden_contacts`, keeping the preview and runtime
        collision contracts identical.
        """

        target = np.asarray(target_positions_rad, dtype=np.float64)
        if target.shape != (len(self.arm_joint_names),):
            raise ValueError(
                "target_positions_rad must match the controlled arm dimension"
            )
        maximum_joint_step_rad = float(
            self.config["task"].get(
                "joint_path_collision_preview_step_rad", 0.0015
            )
        )
        maximum_samples = int(
            self.config["task"].get(
                "joint_path_collision_preview_maximum_samples", 128
            )
        )
        if maximum_joint_step_rad <= 0.0 or maximum_samples < 1:
            raise ValueError(
                "joint-path collision preview limits must be positive"
            )

        start = self.kinematics.positions(self.data)
        maximum_delta = float(np.max(np.abs(target - start)))
        sample_count = max(
            1, int(np.ceil(maximum_delta / maximum_joint_step_rad))
        )
        if sample_count > maximum_samples:
            return [
                {
                    "preview_rejection": "joint_path_preview_sample_limit",
                    "required_samples": sample_count,
                    "maximum_samples": maximum_samples,
                }
            ]

        # Restore the scratch buffer to live state first.  The kinematic
        # solver may have left it at a prior endpoint, which must not affect
        # the interpolated path samples.
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel.fill(0.0)
        self.ik_data.time = self.data.time
        try:
            for sample_index in range(1, sample_count + 1):
                fraction = sample_index / sample_count
                positions = start + fraction * (target - start)
                self.kinematics.set_positions(self.ik_data, positions)
                mujoco.mj_forward(self.model, self.ik_data)
                contacts = self.forbidden_contacts(self.ik_data)
                if contacts:
                    return [
                        {
                            **contact,
                            "preview_sample_index": sample_index,
                            "preview_sample_count": sample_count,
                            "preview_path_fraction": float(fraction),
                        }
                        for contact in contacts
                    ]
        finally:
            # ``ik_data`` is a scratch buffer, but leave it synchronized with
            # the live state so subsequent IK always starts from a known pose.
            self.ik_data.qpos[:] = self.data.qpos
            self.ik_data.qvel.fill(0.0)
            self.ik_data.time = self.data.time
            mujoco.mj_forward(self.model, self.ik_data)
        return []

    def _observation(self) -> PulseObservation:
        proximity, tactile = self.sensors.sample(self.data, dt_s=None)
        self.last_proximity = proximity
        self.last_tactile = tactile
        self.last_tactile_full = self.sensors.last_full_frame
        state = self.state_reader.read(self.data)
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            quaternion,
            state.probe_rotation_world_from_frame.astype(
                np.float64
            ).reshape(-1),
        )
        rgb = depth = None
        frame = self._rgbd_frame_for_observation()
        if frame is not None:
            rgb, depth = frame.rgb, frame.depth_m
        return PulseObservation(
            timestamp_s=float(self.data.time),
            joint_position_rad=state.joint_position_rad,
            joint_velocity_rad_s=state.joint_velocity_rad_s,
            tcp_pose=np.concatenate(
                (state.probe_position_m, quaternion.astype(np.float32))
            ),
            tcp_velocity=np.concatenate(
                (
                    state.probe_linear_velocity_m_s,
                    state.probe_angular_velocity_rad_s,
                )
            ),
            proximity_m=proximity.fused_distance_m,
            proximity_valid=proximity.fused_valid,
            tactile_kpa=tactile.arrays_kpa,
            tactile_raw=tactile.raw_values,
            tactile_raw_delta=tactile.raw_delta,
            contact=tactile.contact,
            task_phase=int(self.phase),
            previous_action=self.previous_action,
            rgb=rgb,
            depth_m=depth,
        )

    def _rgbd_frame_for_observation(self):
        if self.camera is None:
            return None
        if (
            self._rgbd_frame is None
            or self._rgbd_observation_count % self.rgbd_capture_interval == 0
        ):
            self._rgbd_frame = self.camera.capture(self.data)
        self._rgbd_observation_count += 1
        return self._rgbd_frame

    def _info(self, *, ik_converged: bool) -> dict:
        force = self.diagnostic_force_n()
        forbidden = self.forbidden_contacts()
        info = {
            "simulator": "mujoco",
            "simulator_version": mujoco.__version__,
            "collision_safe": not forbidden,
            "forbidden_contacts": forbidden,
            "ik_converged": bool(ik_converged),
            "normal_force_n": force.copy(),
            "diagnostic_finger_names": list(self.diagnostic_finger_names),
            "task_phase": int(self.phase),
            "target_probe_position_m": self.task_target_position.copy(),
            "pressure_forbidden_clearance_m": float(
                self.last_pressure_forbidden_clearance_m
            ),
            "pressure_clearance_force_scale": float(
                self.last_pressure_clearance_force_scale
            ),
            "joint_path_preview_rejection_events": int(
                self.joint_path_preview_rejection_events
            ),
            "proximity_per_finger_m": (
                None
                if self.last_proximity is None
                else self.last_proximity.distance_m.copy()
            ),
            "proximity_per_finger_valid": (
                None
                if self.last_proximity is None
                else self.last_proximity.valid.copy()
            ),
        }
        if self.last_tactile_full is not None:
            full = self.last_tactile_full
            info["tactile_full_56"] = {
                "finger_names": list(full.finger_names),
                "channel_counts": full.channel_counts.copy(),
                "channel_offsets": full.channel_offsets.copy(),
                "pressure_kpa": full.pressure_kpa.copy(),
                "dynamic_pressure_kpa": full.dynamic_pressure_kpa.copy(),
                "raw_values": full.raw_values.copy(),
                "raw_delta": full.raw_delta.copy(),
                "contact": full.contact.copy(),
                "total_normal_force_n": full.total_normal_force_n.copy(),
            }
        return info

    def apply_action(
        self, action: PulseAction
    ) -> tuple[PulseObservation, float, bool, bool, dict]:
        before = dual_probe_frame(self.model, self.data)
        contacts_before = self.diagnostic_contact().astype(bool).copy()
        pressure_latched_before = bool(self.pressure_latched)
        recovery_hold_before = bool(
            self.pressure_recovery_rotation_hold_enabled
            and self.pressure_recovery_joint_hold_positions_rad is not None
        )
        feedforward_norm_before = float(
            np.linalg.norm(
                np.asarray(
                    list(self.controller.feedforward_torque_nm.values()),
                    dtype=float,
                )
            )
            if self.controller.feedforward_torque_nm
            else 0.0
        )
        pressure_contact_lost = False
        requested_phase = (
            self.phase
            if action.phase_command is None
            else TaskPhase(action.phase_command)
        )
        force_window = self.config["task"]
        diagnostic_force = self.diagnostic_force_n()
        diagnostic_contact = self.diagnostic_contact()
        active_finger_count = len(self.diagnostic_finger_names)
        zero_motion_pressure_hold = bool(
            action.target_pressure_kpa > 0.0
            and np.linalg.norm(action.delta_tcp_xyz_m) <= 1e-10
            and np.linalg.norm(action.delta_tcp_rpy_rad) <= 1e-10
        )
        force_in_collection_window = bool(
            self.last_tactile is not None
            and np.all(
                diagnostic_force
                >= float(force_window["pressure_minimum_force_n_per_finger"])
            )
            and np.all(
                diagnostic_force
                <= float(force_window["pressure_maximum_force_n_per_finger"])
            )
        )
        requested_force_n = active_finger_count * nominal_pressure_to_force_target_n(
            action.target_pressure_kpa,
            self.config["sensors"]["tactile_taxel_area_m2"],
            self.config["hand"]["channel_counts"]["index"],
        )
        pressure_command_settled = bool(
            self.pressure_command_force_n >= requested_force_n - 1e-6
        )
        stationary_collection = bool(
            requested_phase
            in {TaskPhase.PULSE_SEARCH, TaskPhase.SIGNAL_CAPTURE}
            and action.target_pressure_kpa > 0.0
            and np.linalg.norm(action.delta_tcp_xyz_m) <= 1e-10
            and np.linalg.norm(action.delta_tcp_rpy_rad) <= 1e-10
            and self.last_tactile is not None
            and diagnostic_contact.all()
            and force_in_collection_window
            and pressure_command_settled
        )
        if stationary_collection:
            if self.collection_hold_joint_positions_rad is None:
                controller_index = {
                    name: index
                    for index, name in enumerate(
                        self.controller.adam_joint_names
                    )
                }
                self.collection_hold_joint_positions_rad = np.asarray(
                    [
                        self.controller.target_positions[
                            controller_index[name]
                        ]
                        for name in self.arm_joint_names
                    ],
                    dtype=np.float64,
                )
        else:
            self.collection_hold_joint_positions_rad = None
        contact_motion_phase = self.phase in {
            TaskPhase.PRE_CONTACT,
            TaskPhase.CONTACT_ESTABLISHMENT,
            TaskPhase.PRESSURE_REGULATION,
            TaskPhase.PULSE_SEARCH,
            TaskPhase.SIGNAL_CAPTURE,
        }
        if (
            self.contact_wrist_stabilization_enabled
            and contact_motion_phase
            and self.contact_wrist_reference_rad is None
        ):
            current_arm = self.kinematics.positions(self.data)
            self.contact_wrist_reference_rad = current_arm[-3:].copy()
        self.task_target_position += action.delta_tcp_xyz_m
        workspace_min = np.asarray(
            self.config["task"].get(
                "cartesian_workspace_min_m", [0.30, -0.35, 1.08]
            ),
            dtype=np.float64,
        )
        workspace_max = np.asarray(
            self.config["task"].get(
                "cartesian_workspace_max_m", [0.72, 0.55, 1.55]
            ),
            dtype=np.float64,
        )
        if (
            workspace_min.shape != (3,)
            or workspace_max.shape != (3,)
            or not np.isfinite(workspace_min).all()
            or not np.isfinite(workspace_max).all()
            or np.any(workspace_min >= workspace_max)
        ):
            raise ValueError("Cartesian workspace bounds must be finite XYZ limits")
        self.task_target_position[:] = np.clip(
            self.task_target_position,
            workspace_min,
            workspace_max,
        )
        if np.linalg.norm(action.delta_tcp_rpy_rad) > 0.0:
            self.task_target_rotation = (
                self.task_target_rotation
                @ Rotation.from_euler(
                    "xyz", action.delta_tcp_rpy_rad
                ).as_matrix()
            )
        if action.target_pressure_kpa <= 0.0:
            self.pressure_latched = False
            self.pressure_recovery_rotation_hold_enabled = False
            self.pressure_recovery_joint_hold_positions_rad = None
            self.pressure_command_force_n = 0.0
            self.controller.set_adam_feedforward_torques({})
        elif (
            self.last_tactile is not None
            and diagnostic_contact.any()
            and (
                diagnostic_contact.all()
                or not zero_motion_pressure_hold
            )
        ):
            if not self.pressure_latched:
                # The final approach may have accumulated an unreachable
                # target below the compliant skin. Start force regulation
                # from the physically reached contact pose instead. The
                # orientation must be latched at the same instant; retaining
                # the old pre-contact rotation can pull a newly seated hand
                # back toward the table during pressure regulation.
                self.task_target_position = before.center.position_m.copy()
                self.task_target_rotation = (
                    before.center.rotation_world_from_frame.copy()
                )
            self.pressure_latched = True
        elif zero_motion_pressure_hold and self.pressure_latched:
            # Pressure feed-forward is only valid while every diagnostic pad
            # remains seated. A transient partial/no-contact state used to
            # leave the old torque and redundant-arm target armed, allowing a
            # free-space oscillation before the search guard could observe it.
            pressure_contact_lost = bool(
                self.last_tactile is not None and not diagnostic_contact.all()
            )
            self.pressure_latched = False
            self.pressure_recovery_rotation_hold_enabled = False
            self.pressure_recovery_joint_hold_positions_rad = None
            self.pressure_command_force_n = 0.0
            self.controller.set_adam_feedforward_torques({})
            self.task_target_position = before.center.position_m.copy()
            self.task_target_rotation = (
                before.center.rotation_world_from_frame.copy()
            )
            current_positions = self.kinematics.positions(self.data)
            self.controller.set_adam_targets(
                dict(zip(self.arm_joint_names, current_positions))
            )
            self.collection_hold_joint_positions_rad = None
        if (
            self.last_tactile is not None
            and action.target_pressure_kpa > 0.0
            and self.pressure_latched
            and (
                diagnostic_contact.all()
                or not zero_motion_pressure_hold
            )
        ):
            # target_pressure_kpa is defined per active fingertip. Each pad
            # uses a stable nominal 1 cm^2 area, so 1 kPa corresponds to
            # 0.1 N per finger. Apply the summed target through J^T while PD
            # holds the contact pose. A bounded ramp prevents impact spikes.
            sensor_config = self.config["sensors"]
            target_force_per_finger_n = (
                nominal_pressure_to_force_target_n(
                    action.target_pressure_kpa,
                    sensor_config["tactile_taxel_area_m2"],
                    self.config["hand"]["channel_counts"]["index"],
                )
            )
            target_total_force_n = active_finger_count * target_force_per_finger_n
            force_step_n = float(
                self.config["task"].get(
                    "pressure_force_unload_n_per_policy_step",
                    self.config["task"][
                        "pressure_force_ramp_n_per_policy_step"
                    ],
                )
                if target_total_force_n < self.pressure_command_force_n
                else self.config["task"][
                    "pressure_force_ramp_n_per_policy_step"
                ]
            )
            self.pressure_command_force_n += float(
                np.clip(
                    target_total_force_n
                    - self.pressure_command_force_n,
                    -force_step_n,
                    force_step_n,
                )
            )
            position_jacobian, _ = self.kinematics.probe_jacobian(
                self.data
            )
            force_direction = before.skin_facing_normal_world.copy()
            force_direction[:2] *= float(
                self.config["task"].get(
                    "pressure_force_lateral_scale", 1.0
                )
            )
            force_direction /= max(
                float(np.linalg.norm(force_direction)), 1e-12
            )
            guarded_command_force_n = max(
                0.0,
                self.pressure_command_force_n
                - float(
                    self.config["task"].get(
                        "pressure_force_guard_gain", 0.0
                    )
                )
                * max(
                    0.0,
                    float(
                        np.max(diagnostic_force)
                    )
                    - float(
                        self.config["task"].get(
                            "pressure_force_guard_soft_n", np.inf
                        )
                    ),
                ),
            )
            self.last_pressure_forbidden_clearance_m = (
                self._minimum_pressure_forbidden_clearance_m()
            )
            clearance_soft_m = float(
                self.config["task"].get(
                    "pressure_forbidden_clearance_soft_m", 0.0
                )
            )
            clearance_hard_m = float(
                self.config["task"].get(
                    "pressure_forbidden_clearance_hard_m", 0.0
                )
            )
            target_clearance_scale = (
                _clearance_force_scale(
                    self.last_pressure_forbidden_clearance_m,
                    hard_m=clearance_hard_m,
                    soft_m=clearance_soft_m,
                )
                if self.pressure_clearance_geom_pairs
                and clearance_soft_m > clearance_hard_m
                else 1.0
            )
            self.last_pressure_clearance_force_scale = (
                _asymmetric_clearance_filter(
                    self.last_pressure_clearance_force_scale,
                    target_clearance_scale,
                    fall_alpha=float(
                        self.config["task"].get(
                            "pressure_clearance_force_scale_fall_alpha", 1.0
                        )
                    ),
                    rise_alpha=float(
                        self.config["task"].get(
                            "pressure_clearance_force_scale_rise_alpha", 1.0
                        )
                    ),
                )
            )
            guarded_command_force_n *= (
                self.last_pressure_clearance_force_scale
            )
            task_force_world_n = (
                force_direction
                * guarded_command_force_n
                * float(
                    self.config["task"].get(
                        "pressure_feedforward_scale", 1.0
                    )
                )
            )
            xy_error_m = (
                self.task_target_position[:2]
                - before.center.position_m[:2]
            )
            xy_restoring_force_n = np.clip(
                float(
                    self.config["task"].get(
                        "pressure_xy_hold_stiffness_n_per_m", 0.0
                    )
                )
                * xy_error_m,
                -float(
                    self.config["task"].get(
                        "pressure_xy_hold_maximum_force_n", 0.0
                    )
                ),
                float(
                    self.config["task"].get(
                        "pressure_xy_hold_maximum_force_n", 0.0
                    )
                ),
            )
            task_force_world_n[:2] += xy_restoring_force_n
            arm_torque_nm = position_jacobian.T @ task_force_world_n
            self.controller.set_adam_feedforward_torques(
                dict(zip(self.arm_joint_names, arm_torque_nm))
            )
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel.fill(0.0)
        self.ik_data.time = self.data.time
        mujoco.mj_forward(self.model, self.ik_data)
        ik_start_positions = self.kinematics.positions(self.ik_data)
        ik_start_probe = dual_probe_frame(self.model, self.ik_data).center
        ik_start_position_error_m = float(
            np.linalg.norm(
                self.task_target_position - ik_start_probe.position_m
            )
        )
        has_rotation_command = np.linalg.norm(action.delta_tcp_rpy_rad) > 0.0
        stabilize_approach = bool(
            self.contact_wrist_stabilization_enabled
            and self.phase
            in {TaskPhase.PRE_CONTACT, TaskPhase.CONTACT_ESTABLISHMENT}
            and action.target_pressure_kpa <= 0.0
            and not self.pressure_latched
        )
        pressure_hold_command = bool(
            contact_motion_phase
            and action.target_pressure_kpa > 0.0
            and np.linalg.norm(action.delta_tcp_xyz_m) <= 1e-10
            and not has_rotation_command
        )
        if (
            pressure_hold_command
            and self.last_tactile is not None
            and diagnostic_contact.all()
        ):
            pressure_kp = float(
                self.config["task"].get(
                    "pressure_wrist_stiffness",
                    self.config["task"].get("contact_wrist_stiffness", 12.0),
                )
            )
            pressure_kd = float(
                self.config["task"].get(
                    "pressure_wrist_damping",
                    self.config["task"].get("contact_wrist_damping", 1.2),
                )
            )
            for name in self.arm_joint_names[-3:]:
                self.controller.kp[name] = pressure_kp
                self.controller.kd[name] = pressure_kd
            pressure_gain_scale = float(
                self.config["task"].get(
                    "pressure_arm_gain_scale", 1.0
                )
            )
            if pressure_gain_scale != 1.0:
                for name in self.arm_joint_names[:-3]:
                    nominal_kp, nominal_kd = self.controller._gain_for_joint(
                        name
                    )
                    self.controller.kp[name] = (
                        nominal_kp * pressure_gain_scale
                    )
                    self.controller.kd[name] = (
                        nominal_kd * pressure_gain_scale
                    )
        elif self.contact_wrist_stabilization_enabled and contact_motion_phase:
            self.set_contact_wrist_damping(True)
        stabilize_contact_solution = bool(
            stabilize_approach or pressure_hold_command
        )
        stabilize_contact_wrist = bool(
            self.contact_wrist_stabilization_enabled
            and contact_motion_phase
            and self.contact_wrist_reference_rad is not None
            and not has_rotation_command
        )
        constrain_task_rotation = _should_constrain_task_rotation(
            has_rotation_command=has_rotation_command,
            task_rotation_hold_enabled=self.task_rotation_hold_enabled,
            pressure_hold_command=pressure_hold_command,
            pressure_recovery_rotation_hold_enabled=(
                self.pressure_recovery_rotation_hold_enabled
            ),
        )
        target_rotation = (
            self.task_target_rotation if constrain_task_rotation else None
        )
        joint_lower = self.kinematics.lower.copy()
        joint_upper = self.kinematics.upper.copy()
        m5_fine_approach_corridor = bool(
            self.phase == TaskPhase.FINE_APPROACH
            and self.config["task"].get(
                "m5_cartesian_fine_approach_enabled", False
            )
        )
        if m5_fine_approach_corridor:
            current_positions = self.kinematics.positions(self.data)
            arm_step = float(
                self.config["task"].get(
                    "m5_fine_approach_ik_joint_step_rad", 0.006
                )
            )
            wrist_step = float(
                self.config["task"].get(
                    "m5_fine_approach_ik_wrist_step_rad", 0.004
                )
            )
            if arm_step <= 0.0 or wrist_step <= 0.0:
                raise ValueError(
                    "M5 fine-approach IK corridor limits must be positive"
                )
            corridor = np.full_like(current_positions, arm_step)
            corridor[-3:] = wrist_step
            joint_lower = np.maximum(
                joint_lower, current_positions - corridor
            )
            joint_upper = np.minimum(
                joint_upper, current_positions + corridor
            )
        if (
            stabilize_contact_wrist
        ):
            wrist_limit = min(
                np.deg2rad(
                    float(
                        self.config["task"].get("wrist_motion_limit_deg", 15.0)
                    )
                ),
                float(
                    self.config["task"].get(
                        "approach_wrist_hold_tolerance_rad", 0.002
                    )
                ),
            )
            wrist_lower, wrist_upper, wrist_reference = (
                _bounded_reference_window(
                    joint_lower[-3:],
                    joint_upper[-3:],
                    self.contact_wrist_reference_rad,
                    wrist_limit,
                )
            )
            joint_lower[-3:] = wrist_lower
            joint_upper[-3:] = wrist_upper
            self.contact_wrist_reference_rad = wrist_reference
        solution = self.kinematics.solve(
            self.ik_data,
            self.task_target_position,
            target_rotation=target_rotation,
            maximum_iterations=int(
                self.config["task"].get(
                    "contact_ik_maximum_iterations", 120
                )
                if stabilize_contact_solution
                else 120
            ),
            damping=(
                float(
                    self.config["task"].get(
                        "m5_fine_approach_ik_damping", 0.10
                    )
                )
                if m5_fine_approach_corridor
                else 0.025
            ),
            maximum_joint_step_rad=0.015,
            position_tolerance_m=5e-4,
            rotation_tolerance_rad=(
                float(
                    self.config["task"].get(
                        "m5_fine_approach_ik_rotation_tolerance_rad",
                        0.015,
                    )
                )
                if m5_fine_approach_corridor
                else 0.002
                if has_rotation_command
                else (0.02 if self.task_rotation_hold_enabled else 0.10)
            ),
            rotation_weight=(
                float(
                    self.config["task"].get(
                        "m5_fine_approach_ik_rotation_weight", 0.08
                    )
                )
                if m5_fine_approach_corridor
                else 0.25
                if has_rotation_command
                else (0.12 if self.task_rotation_hold_enabled else 0.08)
            ),
            joint_lower_rad=joint_lower,
            joint_upper_rad=joint_upper,
        )
        candidate_forbidden = self.forbidden_contacts(self.ik_data)
        maximum_ik_position_error_m = float(
            self.config["task"].get(
                "m5_fine_approach_maximum_ik_position_error_m", 0.006
            )
            if m5_fine_approach_corridor
            else 0.003
        )
        accepted = bool(
            solution.position_error_m <= maximum_ik_position_error_m
            and not candidate_forbidden
        )
        if stationary_collection:
            candidate_forbidden = self.forbidden_contacts(self.data)
            accepted = not candidate_forbidden
        swept_candidate_forbidden: list[dict] = []
        pressure_hold_target_retained = False
        if accepted:
            commanded_positions = (
                self.collection_hold_joint_positions_rad.copy()
                if stationary_collection
                else self.pressure_recovery_joint_hold_positions_rad.copy()
                if (
                    pressure_hold_command
                    and self.pressure_recovery_joint_hold_positions_rad
                    is not None
                )
                else solution.joint_positions_rad.copy()
            )
            retaining_recovery_preload = bool(
                pressure_hold_command
                and self.pressure_recovery_joint_hold_positions_rad is not None
            )
            if stabilize_contact_solution and not retaining_recovery_preload:
                current_positions = self.kinematics.positions(self.data)
                seating_second_pad = bool(
                    has_rotation_command
                    and self.last_tactile is not None
                    and diagnostic_contact.any()
                    and not diagnostic_contact.all()
                )
                pressure_hold = pressure_hold_command
                joint_step_key = (
                    "pressure_joint_target_step_rad"
                    if pressure_hold
                    else (
                        "dual_contact_joint_target_step_rad"
                        if seating_second_pad
                        else "approach_joint_target_step_rad"
                    )
                )
                wrist_step_key = (
                    "pressure_wrist_target_step_rad"
                    if pressure_hold
                    else (
                        "dual_contact_wrist_target_step_rad"
                        if seating_second_pad
                        else "approach_wrist_target_step_rad"
                    )
                )
                maximum_delta = float(
                    self.config["task"].get(
                        joint_step_key,
                        0.025,
                    )
                )
                wrist_delta = float(
                    self.config["task"].get(
                        wrist_step_key,
                        0.010,
                    )
                )
                limits = np.full_like(commanded_positions, maximum_delta)
                limits[-3:] = wrist_delta
                commanded_positions = current_positions + np.clip(
                    commanded_positions - current_positions, -limits, limits
                )
            swept_candidate_forbidden = self.preview_forbidden_joint_path(
                commanded_positions
            )
            accepted = not swept_candidate_forbidden
            if accepted:
                self.controller.set_adam_targets(
                    dict(
                        zip(
                            self.arm_joint_names,
                            commanded_positions,
                        )
                    )
                )
        if not accepted:
            if swept_candidate_forbidden:
                self.joint_path_preview_rejection_events += 1
            # A rejected target must not leave the previous controller target
            # armed.  Otherwise the torque-PD controller continues chasing a
            # stale target during the rejected policy interval, which can move
            # the hand far from the measured safe pose without a new action.
            recovery_target_available = bool(
                pressure_hold_command
                and self.phase == TaskPhase.PULSE_SEARCH
                and self.pressure_recovery_joint_hold_positions_rad is not None
            )
            hold_positions = (
                self.pressure_recovery_joint_hold_positions_rad.copy()
                if recovery_target_available
                else self.kinematics.positions(self.data)
            )
            self.controller.set_adam_targets(
                dict(zip(self.arm_joint_names, hold_positions))
            )
            if not pressure_hold_command:
                self.controller.set_adam_feedforward_torques({})
            self.task_target_position = _rejected_ik_fallback_target(
                self.task_target_position,
                before.center.position_m,
                pressure_hold_command=pressure_hold_command,
            )
            pressure_hold_target_retained = bool(pressure_hold_command)
            if not pressure_hold_command:
                self.task_target_rotation = (
                    before.center.rotation_world_from_frame.copy()
                )

        runtime_forbidden = []
        for _ in range(self.physics_steps_per_policy):
            self.controller.step(self.data)
            runtime_forbidden = self.runtime_forbidden_contacts()
            if runtime_forbidden:
                break
        if action.phase_command is not None:
            self.phase = TaskPhase(action.phase_command)
        after = dual_probe_frame(self.model, self.data)
        contacts_after = self.diagnostic_contact().astype(bool).copy()
        # A pressure hold can lose contact during the physics integration even
        # when all pads were seated at the beginning of the action.  Report
        # that post-step transition immediately so the expert can enter its
        # contact-preserving recovery on the next policy tick instead of
        # continuing to push a now-free hand until pressure timeout.
        if (
            pressure_hold_command
            and self.pressure_latched
            and not contacts_after.all()
        ):
            pressure_contact_lost = True
        recovery_target_delta_rad = None
        if self.pressure_recovery_joint_hold_positions_rad is not None:
            current_targets = np.asarray(
                [
                    self.controller.target_positions[
                        self.controller.adam_joint_names.index(name)
                    ]
                    for name in self.arm_joint_names
                ],
                dtype=float,
            )
            recovery_target_delta_rad = float(
                np.linalg.norm(
                    current_targets
                    - self.pressure_recovery_joint_hold_positions_rad
                )
            )
        rotation_delta = Rotation.from_matrix(
            before.center.rotation_world_from_frame
        ).inv() * Rotation.from_matrix(
            after.center.rotation_world_from_frame
        )
        executed = np.concatenate(
            (
                after.center.position_m - before.center.position_m,
                rotation_delta.as_euler("xyz"),
                np.asarray([action.target_pressure_kpa]),
            )
        ).astype(np.float32)
        self.previous_action = executed
        self.environment_step += 1
        observation = self._observation()
        info = self._info(ik_converged=accepted)
        info.update(
            {
                "executed_action": executed,
                "ik_solver_converged": bool(solution.converged),
                "ik_position_error_m": float(solution.position_error_m),
                "maximum_ik_position_error_m": maximum_ik_position_error_m,
                "ik_rotation_error_rad": float(solution.rotation_error_rad),
                "ik_iterations": int(solution.iterations),
                "ik_start_position_error_m": ik_start_position_error_m,
                "ik_maximum_joint_delta_rad": float(
                    np.max(
                        np.abs(
                            solution.joint_positions_rad
                            - ik_start_positions
                        )
                    )
                ),
                "candidate_forbidden_contacts": candidate_forbidden,
                "swept_candidate_forbidden_contacts": (
                    swept_candidate_forbidden
                ),
                "joint_path_preview_rejected": bool(
                    swept_candidate_forbidden
                ),
                "joint_path_preview_rejection_events": int(
                    self.joint_path_preview_rejection_events
                ),
                "runtime_forbidden_contacts": runtime_forbidden,
                "action_accepted": accepted,
                "pressure_command_force_n": float(
                    self.pressure_command_force_n
                ),
                "collection_pose_locked": bool(stationary_collection),
                "pressure_hold_target_retained_on_ik_rejection": bool(
                    pressure_hold_target_retained
                ),
                "pressure_recovery_joint_hold_active": bool(
                    pressure_hold_command
                    and self.pressure_recovery_joint_hold_positions_rad
                    is not None
                ),
                "pressure_contact_lost": bool(pressure_contact_lost),
                # Recovery diagnostics are observational only. They make the
                # first pressure tick after seating auditable without exposing
                # privileged state to the policy observation.
                "diagnostic_contacts_before": contacts_before,
                "diagnostic_contacts_after": contacts_after,
                "diagnostic_contact_mask_changed": bool(
                    not np.array_equal(contacts_before, contacts_after)
                ),
                "pressure_latched_before": pressure_latched_before,
                "pressure_latched_after": bool(self.pressure_latched),
                "pressure_recovery_hold_active_before": recovery_hold_before,
                "pressure_recovery_hold_active_after": bool(
                    self.pressure_recovery_rotation_hold_enabled
                    and self.pressure_recovery_joint_hold_positions_rad
                    is not None
                ),
                "pressure_feedforward_torque_norm_nm_before": (
                    feedforward_norm_before
                ),
                "pressure_feedforward_torque_norm_nm_after": float(
                    np.linalg.norm(
                        np.asarray(
                            list(
                                self.controller.feedforward_torque_nm.values()
                            ),
                            dtype=float,
                        )
                    )
                    if self.controller.feedforward_torque_nm
                    else 0.0
                ),
                "pressure_recovery_target_delta_rad": recovery_target_delta_rad,
            }
        )
        self.last_info = info
        reward = -float(
            np.linalg.norm(
                after.center.position_m - self.task_target_position
            )
        )
        terminated = self.phase in (TaskPhase.SUCCESS, TaskPhase.FAILURE)
        truncated = self.environment_step >= self.maximum_steps
        return observation, reward, terminated, truncated, info

    def _minimum_pressure_forbidden_clearance_m(self) -> float:
        if not self.pressure_clearance_geom_pairs:
            return float("inf")
        maximum_query_m = float(
            self.config["task"].get(
                "pressure_forbidden_clearance_query_m", 0.010
            )
        )
        from_to = np.empty(6, dtype=np.float64)
        return min(
            float(
                mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    geom1,
                    geom2,
                    maximum_query_m,
                    from_to,
                )
            )
            for geom1, geom2 in self.pressure_clearance_geom_pairs
        )

    def move_to_joint_pose(
        self,
        pose: dict[str, float],
        *,
        duration_s: float = 2.0,
        settle_s: float = 0.5,
        on_control_step: Callable[[], None] | None = None,
        track_tcp_action: bool = False,
    ) -> list[np.ndarray]:
        """Execute a smooth torque-controlled named waypoint."""
        start = {
            name: float(
                self.data.qpos[
                    self.model.jnt_qposadr[self.model.joint(name).id]
                ]
            )
            for name in self.arm_joint_names
        }
        steps = max(
            1,
            int(
                round(
                    float(duration_s)
                    / (
                        self.physics_steps_per_policy
                        * float(self.model.opt.timestep)
                    )
                )
            ),
        )
        trace = []
        for index in range(1, steps + 1):
            before = (
                dual_probe_frame(self.model, self.data).center
                if track_tcp_action
                else None
            )
            u = index / steps
            smooth = u * u * (3.0 - 2.0 * u)
            target = {
                name: (1.0 - smooth) * start[name]
                + smooth * float(pose[name])
                for name in self.arm_joint_names
            }
            self.controller.set_adam_targets(target)
            for _ in range(self.physics_steps_per_policy):
                self.controller.step(self.data)
                forbidden = self.runtime_forbidden_contacts()
                if forbidden:
                    raise RuntimeError(
                        f"forbidden contact during waypoint: {forbidden}"
                    )
            trace.append(
                dual_probe_frame(
                    self.model, self.data
                ).center.position_m.copy()
            )
            if before is not None:
                after = dual_probe_frame(self.model, self.data).center
                rotation_delta = Rotation.from_matrix(
                    before.rotation_world_from_frame
                ).inv() * Rotation.from_matrix(
                    after.rotation_world_from_frame
                )
                self.previous_action = np.concatenate(
                    (
                        after.position_m - before.position_m,
                        rotation_delta.as_euler("xyz"),
                        np.zeros(1),
                    )
                ).astype(np.float32)
            if on_control_step is not None:
                on_control_step()
        self.controller.set_adam_targets(
            {name: float(pose[name]) for name in self.arm_joint_names}
        )
        settle_steps = max(
            0,
            int(
                round(
                    float(settle_s)
                    / (
                        self.physics_steps_per_policy
                        * float(self.model.opt.timestep)
                    )
                )
            ),
        )
        for _ in range(settle_steps):
            before = (
                dual_probe_frame(self.model, self.data).center
                if track_tcp_action
                else None
            )
            for _ in range(self.physics_steps_per_policy):
                self.controller.step(self.data)
                forbidden = self.runtime_forbidden_contacts()
                if forbidden:
                    raise RuntimeError(
                        f"forbidden contact while settling: {forbidden}"
                    )
            trace.append(
                dual_probe_frame(
                    self.model, self.data
                ).center.position_m.copy()
            )
            if before is not None:
                after = dual_probe_frame(self.model, self.data).center
                rotation_delta = Rotation.from_matrix(
                    before.rotation_world_from_frame
                ).inv() * Rotation.from_matrix(
                    after.rotation_world_from_frame
                )
                self.previous_action = np.concatenate(
                    (
                        after.position_m - before.position_m,
                        rotation_delta.as_euler("xyz"),
                        np.zeros(1),
                    )
                ).astype(np.float32)
            if on_control_step is not None:
                on_control_step()
        probe = dual_probe_frame(self.model, self.data)
        self.task_target_position = probe.center.position_m.copy()
        self.task_target_rotation = (
            probe.center.rotation_world_from_frame.copy()
        )
        return trace

    def move_to_joint_pose_path_controlled(
        self,
        pose: dict[str, float],
        *,
        duration_s: float,
        settle_s: float,
        progress_selector: Callable[[np.ndarray], tuple[float, dict | None]],
        on_control_step: Callable[[np.ndarray, dict | None], None],
    ) -> list[np.ndarray]:
        """Move along a named joint path with policy-controlled progress.

        The geometric path remains the same smooth, collision-checked joint
        waypoint used by the expert. The selector may only choose a positive
        progress scale up to one for the next tick, so it cannot leave the
        path or make the waypoint faster than the audited reference.
        """
        if duration_s <= 0.0 or settle_s < 0.0:
            raise ValueError("path-controlled waypoint durations are invalid")
        start = {
            name: float(
                self.data.qpos[
                    self.model.jnt_qposadr[self.model.joint(name).id]
                ]
            )
            for name in self.arm_joint_names
        }
        policy_period_s = (
            self.physics_steps_per_policy * float(self.model.opt.timestep)
        )
        nominal_steps = max(1, int(round(float(duration_s) / policy_period_s)))
        maximum_steps = nominal_steps * 4
        progress = 0.0
        trace: list[np.ndarray] = []
        step_count = 0

        while progress < 1.0 - 1e-12:
            if step_count >= maximum_steps:
                raise RuntimeError("path-controlled joint waypoint timed out")
            before = dual_probe_frame(self.model, self.data).center
            current_smooth = progress * progress * (3.0 - 2.0 * progress)
            current_reference = {
                name: (1.0 - current_smooth) * start[name]
                + current_smooth * float(pose[name])
                for name in self.arm_joint_names
            }
            nominal_progress = min(1.0, progress + 1.0 / nominal_steps)
            nominal_smooth = nominal_progress * nominal_progress * (
                3.0 - 2.0 * nominal_progress
            )
            nominal_target = {
                name: (1.0 - nominal_smooth) * start[name]
                + nominal_smooth * float(pose[name])
                for name in self.arm_joint_names
            }
            current_position, current_rotation = self.probe_pose_for_joint_pose(
                current_reference
            )
            target_position, target_rotation = self.probe_pose_for_joint_pose(
                nominal_target
            )
            nominal_rotation = Rotation.from_matrix(
                current_rotation
            ).inv() * Rotation.from_matrix(target_rotation)
            nominal_action = np.concatenate(
                (
                    target_position - current_position,
                    nominal_rotation.as_euler("xyz"),
                    np.zeros(1),
                )
            ).astype(np.float32)
            progress_scale, control_metadata = progress_selector(nominal_action)
            progress_scale = float(progress_scale)
            if not np.isfinite(progress_scale) or not 0.0 < progress_scale <= 1.0:
                raise ValueError(
                    "path-controlled waypoint progress must be in (0, 1]"
                )

            progress = min(1.0, progress + progress_scale / nominal_steps)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            target = {
                name: (1.0 - smooth) * start[name]
                + smooth * float(pose[name])
                for name in self.arm_joint_names
            }
            self.controller.set_adam_targets(target)
            for _ in range(self.physics_steps_per_policy):
                self.controller.step(self.data)
                forbidden = self.runtime_forbidden_contacts()
                if forbidden:
                    raise RuntimeError(
                        "forbidden contact during path-controlled waypoint: "
                        f"{forbidden}"
                    )
            after = dual_probe_frame(self.model, self.data).center
            rotation_delta = Rotation.from_matrix(
                before.rotation_world_from_frame
            ).inv() * Rotation.from_matrix(after.rotation_world_from_frame)
            executed_action = np.concatenate(
                (
                    after.position_m - before.position_m,
                    rotation_delta.as_euler("xyz"),
                    np.zeros(1),
                )
            ).astype(np.float32)
            self.previous_action = executed_action.copy()
            trace.append(after.position_m.copy())
            on_control_step(executed_action, control_metadata)
            step_count += 1

        self.controller.set_adam_targets(
            {name: float(pose[name]) for name in self.arm_joint_names}
        )
        settle_steps = max(0, int(round(float(settle_s) / policy_period_s)))
        for _ in range(settle_steps):
            before = dual_probe_frame(self.model, self.data).center
            for _ in range(self.physics_steps_per_policy):
                self.controller.step(self.data)
                forbidden = self.runtime_forbidden_contacts()
                if forbidden:
                    raise RuntimeError(
                        "forbidden contact while settling path-controlled waypoint: "
                        f"{forbidden}"
                    )
            after = dual_probe_frame(self.model, self.data).center
            rotation_delta = Rotation.from_matrix(
                before.rotation_world_from_frame
            ).inv() * Rotation.from_matrix(after.rotation_world_from_frame)
            executed_action = np.concatenate(
                (
                    after.position_m - before.position_m,
                    rotation_delta.as_euler("xyz"),
                    np.zeros(1),
                )
            ).astype(np.float32)
            self.previous_action = executed_action.copy()
            trace.append(after.position_m.copy())
            on_control_step(executed_action, None)

        probe = dual_probe_frame(self.model, self.data)
        self.task_target_position = probe.center.position_m.copy()
        self.task_target_rotation = probe.center.rotation_world_from_frame.copy()
        return trace

    def probe_pose_for_joint_pose(
        self, pose: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a named arm pose without mutating live simulation state."""

        missing = [name for name in self.arm_joint_names if name not in pose]
        if missing:
            raise ValueError(
                f"joint pose is missing controlled-arm joints: {missing}"
            )
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel.fill(0.0)
        for name in self.arm_joint_names:
            address = self.model.jnt_qposadr[self.model.joint(name).id]
            self.ik_data.qpos[address] = float(pose[name])
        mujoco.mj_forward(self.model, self.ik_data)
        probe = dual_probe_frame(self.model, self.ik_data).center
        return (
            probe.position_m.copy(),
            probe.rotation_world_from_frame.copy(),
        )

    def move_hand_to_diagnosis_pose(
        self,
        *,
        duration_s: float | None = None,
        settle_s: float | None = None,
        on_control_step: Callable[[], None] | None = None,
    ) -> None:
        """Close the unused fingers smoothly while holding the arm still."""
        hand = self.config["hand"]
        target = dict(hand["diagnosis_pose_rad"])
        duration_s = float(
            hand.get("diagnosis_transition_duration_s", 1.0)
            if duration_s is None
            else duration_s
        )
        settle_s = float(
            hand.get("diagnosis_transition_settle_s", 0.2)
            if settle_s is None
            else settle_s
        )
        start = {
            name: float(
                self.data.qpos[
                    self.model.jnt_qposadr[self.model.joint(name).id]
                ]
            )
            for name in target
        }
        policy_dt = (
            self.physics_steps_per_policy * float(self.model.opt.timestep)
        )
        steps = max(1, int(round(duration_s / policy_dt)))
        for index in range(1, steps + 1):
            u = index / steps
            smooth = u * u * (3.0 - 2.0 * u)
            self.controller.set_hand_targets(
                {
                    name: (1.0 - smooth) * start[name]
                    + smooth * float(target[name])
                    for name in target
                }
            )
            for _ in range(self.physics_steps_per_policy):
                self.controller.step(self.data)
                forbidden = self.runtime_forbidden_contacts()
                if forbidden:
                    raise RuntimeError(
                        "forbidden contact while forming diagnosis hand pose: "
                        f"{forbidden}"
                    )
            if on_control_step is not None:
                on_control_step()
        self.controller.set_hand_targets(target)
        for _ in range(max(0, int(round(settle_s / policy_dt)))):
            for _ in range(self.physics_steps_per_policy):
                self.controller.step(self.data)
                forbidden = self.runtime_forbidden_contacts()
                if forbidden:
                    raise RuntimeError(
                        "forbidden contact while settling diagnosis hand pose: "
                        f"{forbidden}"
                    )
            if on_control_step is not None:
                on_control_step()

    def set_contact_wrist_damping(self, enabled: bool) -> None:
        """Increase wrist damping only after the collision-safe approach."""
        if not self.contact_wrist_stabilization_enabled:
            return
        contact_kp = float(
            self.config["task"].get("contact_wrist_stiffness", 12.0)
        )
        contact_kd = float(
            self.config["task"].get("contact_wrist_damping", 1.2)
        )
        for name in self.arm_joint_names[:-3]:
            nominal_kp, nominal_kd = self.controller._gain_for_joint(name)
            self.controller.kp[name] = nominal_kp
            self.controller.kd[name] = nominal_kd
        for name in self.arm_joint_names[-3:]:
            nominal_kp, nominal_kd = self.controller._gain_for_joint(name)
            self.controller.kp[name] = contact_kp if enabled else nominal_kp
            self.controller.kd[name] = contact_kd if enabled else nominal_kd

    def arm_pressure_recovery_rotation_hold(self) -> None:
        """Preserve the seated recovery target while pressure ramps back in."""

        probe = dual_probe_frame(self.model, self.data).center
        self.task_target_position = probe.position_m.copy()
        self.task_target_rotation = probe.rotation_world_from_frame.copy()
        self.pressure_recovery_rotation_hold_enabled = True
        controller_index = {
            name: index
            for index, name in enumerate(self.controller.adam_joint_names)
        }
        self.pressure_recovery_joint_hold_positions_rad = np.asarray(
            [
                self.controller.target_positions[controller_index[name]]
                for name in self.arm_joint_names
            ],
            dtype=np.float64,
        )

    def clear_pressure_recovery_hold(self) -> None:
        self.pressure_recovery_rotation_hold_enabled = False
        self.pressure_recovery_joint_hold_positions_rad = None

    def emergency_retreat(self) -> PulseObservation:
        self.phase = TaskPhase.RETREAT
        for pose in reversed(self.safe_waypoint_poses()):
            self.move_to_joint_pose(pose, duration_s=1.0)
        return self._observation()

    def safe_waypoint_poses(self) -> list[dict[str, float]]:
        """Return the configured collision-safe path with legacy fallback."""
        robot = self.config["robot"]
        if "safe_waypoint_poses_rad" in robot:
            poses = robot["safe_waypoint_poses_rad"]
            if not poses:
                raise ValueError("safe_waypoint_poses_rad cannot be empty")
            return [dict(pose) for pose in poses]
        return [dict(robot["safe_waypoint_pose_rad"])]

    def get_privileged_state(self) -> dict:
        return {
            "pulse_center_m": np.asarray(
                self.config["scene"]["patient_forearm"][
                    "pulse_center_m"
                ],
                dtype=np.float64,
            ),
            "pulse_reference": (
                None
                if self.last_tactile is None
                else float(self.last_tactile.pulse_normalized)
            ),
        }

    def save_state(self) -> dict:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "act": self.data.act.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
            "controller_targets": self.controller.target_positions.copy(),
            "hand_targets": copy.deepcopy(self.controller.hand_targets),
            "task_target_position": self.task_target_position.copy(),
            "task_target_rotation": self.task_target_rotation.copy(),
            "environment_step": self.environment_step,
            "phase": int(self.phase),
            "previous_action": self.previous_action.copy(),
            "pressure_latched": bool(self.pressure_latched),
            "pressure_command_force_n": float(
                self.pressure_command_force_n
            ),
            "runtime_forbidden_contact_events": int(
                self.runtime_forbidden_contact_events
            ),
            "joint_path_preview_rejection_events": int(
                self.joint_path_preview_rejection_events
            ),
            "task_rotation_hold_enabled": bool(
                self.task_rotation_hold_enabled
            ),
            "pressure_recovery_rotation_hold_enabled": bool(
                self.pressure_recovery_rotation_hold_enabled
            ),
            "pressure_recovery_joint_hold_positions_rad": (
                None
                if self.pressure_recovery_joint_hold_positions_rad is None
                else self.pressure_recovery_joint_hold_positions_rad.copy()
            ),
            "collection_hold_joint_positions_rad": (
                None
                if self.collection_hold_joint_positions_rad is None
                else self.collection_hold_joint_positions_rad.copy()
            ),
            "controller_feedforward_torque_nm": copy.deepcopy(
                self.controller.feedforward_torque_nm
            ),
            "sensor_state": self.sensors.save_state(),
        }

    def load_state(self, snapshot: dict) -> None:
        self.data.qpos[:] = snapshot["qpos"]
        self.data.qvel[:] = snapshot["qvel"]
        self.data.act[:] = snapshot["act"]
        self.data.ctrl[:] = snapshot["ctrl"]
        self.data.time = float(snapshot["time"])
        self.controller.target_positions[:] = snapshot[
            "controller_targets"
        ]
        self.controller.hand_targets = copy.deepcopy(
            snapshot["hand_targets"]
        )
        self.task_target_position[:] = snapshot["task_target_position"]
        self.task_target_rotation[:] = snapshot["task_target_rotation"]
        self.environment_step = int(snapshot["environment_step"])
        self.phase = TaskPhase(snapshot["phase"])
        self.previous_action[:] = snapshot["previous_action"]
        self.pressure_latched = bool(snapshot["pressure_latched"])
        self.pressure_command_force_n = float(
            snapshot["pressure_command_force_n"]
        )
        self.runtime_forbidden_contact_events = int(
            snapshot.get("runtime_forbidden_contact_events", 0)
        )
        self.joint_path_preview_rejection_events = int(
            snapshot.get("joint_path_preview_rejection_events", 0)
        )
        self.task_rotation_hold_enabled = bool(
            snapshot.get("task_rotation_hold_enabled", False)
        )
        self.pressure_recovery_rotation_hold_enabled = bool(
            snapshot.get("pressure_recovery_rotation_hold_enabled", False)
        )
        pressure_recovery_joint_hold = snapshot.get(
            "pressure_recovery_joint_hold_positions_rad"
        )
        self.pressure_recovery_joint_hold_positions_rad = (
            None
            if pressure_recovery_joint_hold is None
            else np.asarray(
                pressure_recovery_joint_hold, dtype=np.float64
            ).copy()
        )
        collection_hold = snapshot.get("collection_hold_joint_positions_rad")
        self.collection_hold_joint_positions_rad = (
            None
            if collection_hold is None
            else np.asarray(collection_hold, dtype=np.float64).copy()
        )
        self.controller.set_adam_feedforward_torques(
            snapshot["controller_feedforward_torque_nm"]
        )
        self.sensors.load_state(snapshot["sensor_state"])
        self._rgbd_frame = None
        self._rgbd_observation_count = 0
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
