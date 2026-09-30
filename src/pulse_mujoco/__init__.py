"""Adam Pro right-hand pulse-diagnosis environment implemented on MuJoCo."""

import os


def configure_headless_mujoco_gl() -> None:
    """Choose EGL before any MuJoCo import when no display server is present."""

    display_available = bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if not display_available:
        # Preserve an explicit caller choice such as ``osmesa``.  EGL is the
        # verified Ubuntu GPU-server default and must be selected before the
        # modules below import mujoco.
        os.environ.setdefault("MUJOCO_GL", "egl")


configure_headless_mujoco_gl()

from .model_info import ModelInventory, inventory_model
from .control import AdamJointPDController, AdamLiteJointPDController, ControlStep
from .frames import (
    DualProbeFrame,
    FramePose,
    dual_probe_frame,
    patient_wrist_frame,
    pulse_reference_position,
)
from .kinematics import AdamLiteTaskKinematics, IKSolution
from .contact import (
    DualFingerContactFrame,
    FingerContact,
    FullHandContactFrame,
    MujocoDualFingerContactReader,
    MujocoFullHandContactReader,
)
from .proximity import MujocoFingertipProximity, ProximityFrame
from .tactile_array import (
    HandleL1DualFingerTactileSensor,
    PulseSourceConfig,
    RadialPulseSource,
    TactileFrame,
)
from .full_hand_tactile import (
    FullHandTactileFrame,
    HandleL1FullHandTactileSensor,
)
from .tactile_calibration import HandleL1TenChannelCalibration
from .tactile_layouts import (
    SIXTEEN_CHANNEL_COORDS_MM,
    TEN_CHANNEL_COORDS_MM,
    TaxelLayout,
    handle_l1_layout_for_finger,
    handle_l1_sixteen_channel_layout,
    handle_l1_ten_channel_layout,
)
from .sensor_suite import MujocoPulseSensorSuite
from .rgbd import MujocoRGBDCamera, RGBDFrame
from .robot_state import MujocoRobotStateReader, RobotStateFrame
from .actions import HandMode, PulseAction
from .observation import PulseObservation
from .safety import SafetyLimits, SafetySupervisor
from .task import TaskPhase
from .backend import MujocoPulseDiagnosisBackend
from .environment import PulseDiagnosisEnv
from .phase_gated_policy import (
    PhaseGatedPolicyConfig,
    PhaseGatedPolicyController,
    PolicyControlDecision,
    observable_diagnostic_contact,
    pressure_regulation_effective_uncertainty,
    project_contact_establishment_action,
    project_pressure_regulation_action,
    project_stable_capture_action,
    project_joint_waypoint_progress,
    project_search_translation,
    review_search_segment_direction,
)
from .pressure_target import (
    force_target_to_nominal_pressure_kpa,
    nominal_pressure_to_force_target_n,
)
from .pulse_search_motion import (
    PulseSearchState,
    SurfaceFrame,
    best_finite_score_index,
    build_surface_frame,
    clip_target_to_search_radius,
    unique_position_count,
    validate_three_stage_transitions,
)
from .spatial_guidance import (
    PulseSpatialEstimate,
    estimate_pulse_center_mm,
)
from .expert import ExpertRun, MujocoPulseExpert
from .pulse_signal_quality import (
    PULSE_DIAGNOSIS_QUALITY_GATE_SCHEMA_VERSION,
    PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS,
    PULSE_SIGNAL_QUALITY_SCHEMA_VERSION,
    PULSE_TACTILE_COVERAGE_OBSERVABLE_FIELDS,
    PulseSignalQuality,
    PulseSignalQualityThresholds,
    PulseTactileCoverageThresholds,
    despike_tactile_window,
    evaluate_pulse_signal_quality,
    pulse_spatial_direction_is_reliable,
    pulse_diagnosis_quality_gate_from_metrics,
    pulse_signal_quality_gate_from_metrics,
    pulse_tactile_coverage_gate_from_metrics,
    should_extend_pulse_quality_confirmation,
)
from .episode_data import (
    EPISODE_SCHEMA_VERSION,
    SUPPORTED_EPISODE_SCHEMA_VERSIONS,
    EpisodeReplayBackend,
    MujocoEpisodeRecorder,
    validate_episode,
)
from .dataset_index import (
    DATASET_INDEX_SCHEMA_VERSION,
    build_dataset_index,
    episode_split_counts,
    write_dataset_index,
)
from .dataset_overlay import (
    TRAIN_ONLY_OVERLAY_SCHEMA_VERSION,
    build_train_only_dataset_overlay,
    supplement_episode_alias,
    train_only_splits,
)
from .perturbations import (
    PerturbationEvent,
    PerturbationKind,
    PerturbationScheduler,
)
from .data_collection import (
    COLLECTION_MANIFEST_SCHEMA_VERSION,
    CollectionScenario,
    build_stage3_collection_scenario,
    apply_collection_scenario,
    build_collection_scenarios,
    collection_acceptance,
    collection_perturbation_scheduler,
    new_collection_manifest,
    update_collection_counts,
    write_json_atomic,
)
from .pure_policy_audit import (
    PURE_POLICY_MODE,
    REQUIRED_EPISODE_FIELDS,
    audit_pure_policy_summary,
)
from .phase_takeover_audit import audit_phase_takeover_summary
from .m5_transition_overlay import project_m5_action_to_safety
from .tactile_normalization import (
    FULL56_CHANNEL_COUNTS,
    FULL56_FINGER_NAMES,
    FULL56_OFFSETS,
    TACTILE_NORMALIZATION_SCHEMA_VERSION,
    TactileNormalization,
    fit_tactile_normalization,
    load_tactile_normalization,
    save_tactile_normalization,
)

__all__ = [
    "AdamLiteJointPDController",
    "AdamJointPDController",
    "AdamLiteTaskKinematics",
    "ControlStep",
    "configure_headless_mujoco_gl",
    "DualProbeFrame",
    "DualFingerContactFrame",
    "FullHandContactFrame",
    "FullHandTactileFrame",
    "FingerContact",
    "FramePose",
    "IKSolution",
    "ModelInventory",
    "MujocoDualFingerContactReader",
    "MujocoFullHandContactReader",
    "MujocoFingertipProximity",
    "MujocoPulseSensorSuite",
    "MujocoPulseDiagnosisBackend",
    "MujocoRGBDCamera",
    "MujocoRobotStateReader",
    "ProximityFrame",
    "HandleL1DualFingerTactileSensor",
    "HandleL1FullHandTactileSensor",
    "HandleL1TenChannelCalibration",
    "HandMode",
    "PulseSourceConfig",
    "PulseAction",
    "PulseObservation",
    "PulseDiagnosisEnv",
    "PhaseGatedPolicyConfig",
    "PhaseGatedPolicyController",
    "PolicyControlDecision",
    "observable_diagnostic_contact",
    "pressure_regulation_effective_uncertainty",
    "project_contact_establishment_action",
    "project_pressure_regulation_action",
    "project_stable_capture_action",
    "project_joint_waypoint_progress",
    "project_search_translation",
    "review_search_segment_direction",
    "RadialPulseSource",
    "RGBDFrame",
    "RobotStateFrame",
    "SafetyLimits",
    "SafetySupervisor",
    "TaskPhase",
    "TEN_CHANNEL_COORDS_MM",
    "SIXTEEN_CHANNEL_COORDS_MM",
    "TactileFrame",
    "TaxelLayout",
    "dual_probe_frame",
    "inventory_model",
    "patient_wrist_frame",
    "pulse_reference_position",
    "handle_l1_ten_channel_layout",
    "handle_l1_sixteen_channel_layout",
    "handle_l1_layout_for_finger",
    "force_target_to_nominal_pressure_kpa",
    "nominal_pressure_to_force_target_n",
    "PulseSearchState",
    "SurfaceFrame",
    "PulseSpatialEstimate",
    "best_finite_score_index",
    "build_surface_frame",
    "clip_target_to_search_radius",
    "estimate_pulse_center_mm",
    "unique_position_count",
    "validate_three_stage_transitions",
    "ExpertRun",
    "MujocoPulseExpert",
    "PULSE_SIGNAL_QUALITY_OBSERVABLE_FIELDS",
    "PULSE_SIGNAL_QUALITY_SCHEMA_VERSION",
    "PulseSignalQuality",
    "PulseSignalQualityThresholds",
    "despike_tactile_window",
    "evaluate_pulse_signal_quality",
    "pulse_spatial_direction_is_reliable",
    "pulse_signal_quality_gate_from_metrics",
    "should_extend_pulse_quality_confirmation",
    "EPISODE_SCHEMA_VERSION",
    "SUPPORTED_EPISODE_SCHEMA_VERSIONS",
    "EpisodeReplayBackend",
    "MujocoEpisodeRecorder",
    "validate_episode",
    "DATASET_INDEX_SCHEMA_VERSION",
    "build_dataset_index",
    "episode_split_counts",
    "write_dataset_index",
    "TRAIN_ONLY_OVERLAY_SCHEMA_VERSION",
    "build_train_only_dataset_overlay",
    "supplement_episode_alias",
    "train_only_splits",
    "PerturbationEvent",
    "PerturbationKind",
    "PerturbationScheduler",
    "COLLECTION_MANIFEST_SCHEMA_VERSION",
    "CollectionScenario",
    "apply_collection_scenario",
    "build_collection_scenarios",
    "collection_acceptance",
    "collection_perturbation_scheduler",
    "new_collection_manifest",
    "update_collection_counts",
    "write_json_atomic",
    "PURE_POLICY_MODE",
    "REQUIRED_EPISODE_FIELDS",
    "audit_pure_policy_summary",
    "audit_phase_takeover_summary",
    "project_m5_action_to_safety",
    "FULL56_CHANNEL_COUNTS",
    "FULL56_FINGER_NAMES",
    "FULL56_OFFSETS",
    "TACTILE_NORMALIZATION_SCHEMA_VERSION",
    "TactileNormalization",
    "fit_tactile_normalization",
    "load_tactile_normalization",
    "save_tactile_normalization",
]
