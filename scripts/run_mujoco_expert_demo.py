from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import mujoco
import numpy as np

from pulse_mujoco import (
    MujocoPulseDiagnosisBackend,
    MujocoPulseExpert,
    PulseDiagnosisEnv,
    TEN_CHANNEL_COORDS_MM,
    TaskPhase,
)
from pulse_mujoco.tactile_visual_layout import (
    CUSTOM_HAND_LARGE_RECTS,
    CUSTOM_HAND_LONGITUDINAL_COMPENSATION_MM,
    CUSTOM_HAND_PLACEMENTS,
)
from pulse_mujoco.perturbations import (
    PerturbationEvent,
    PerturbationKind,
    PerturbationScheduler,
)
from pulse_world_vla.search_segment_runtime import (
    SearchSegmentShadowHook,
    load_search_segment_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPS = 20.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.65,
    color=(245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


class DemoRecorder:
    def __init__(
        self,
        backend: MujocoPulseDiagnosisBackend,
        output_dir: Path,
    ):
        self.backend = backend
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keyframe_dir = self.output_dir / "keyframes"
        self.keyframe_dir.mkdir(exist_ok=True)
        self.renderer = mujoco.Renderer(
            backend.model, height=720, width=1280
        )
        self.global_camera = mujoco.MjvCamera()
        self.global_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Fixed right-side overview, matched in MuJoCo scene coordinates to
        # the accepted Isaac demo composition.
        self.global_camera.lookat[:] = [0.32, -0.08, 1.12]
        self.global_camera.distance = 1.55
        self.global_camera.azimuth = 225.0
        self.global_camera.elevation = -12.0
        self.local_camera = mujoco.MjvCamera()
        self.local_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Fixed high oblique fingertip view. MuJoCo elevation has the opposite
        # sign to the Isaac camera convention used by the source preset.
        self.local_camera.lookat[:] = [0.535, -0.181, 1.12]
        self.local_camera.distance = 0.54
        self.local_camera.azimuth = 312.3
        self.local_camera.elevation = -46.9
        self.scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.scene_option)
        # Sites are simulation/debug primitives. The embedded tactile and
        # proximity layers must not appear in delivery footage.
        self.scene_option.sitegroup[:] = 0
        self.scene_option.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
        ] = False
        self.scene_option.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
        ] = False
        self.scene_option.flags[
            mujoco.mjtVisFlag.mjVIS_RANGEFINDER
        ] = False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.front_path = output_dir / "front_view_1x.mp4"
        self.local_path = output_dir / "fingertip_view_1x.mp4"
        self.composite_path = output_dir / "expert_demo_1080p_1x.mp4"
        self.front_writer = cv2.VideoWriter(
            str(self.front_path), fourcc, FPS, (1280, 720)
        )
        self.local_writer = cv2.VideoWriter(
            str(self.local_path), fourcc, FPS, (1280, 720)
        )
        self.composite_writer = cv2.VideoWriter(
            str(self.composite_path), fourcc, FPS, (1920, 1080)
        )
        if not all(
            writer.isOpened()
            for writer in (
                self.front_writer,
                self.local_writer,
                self.composite_writer,
            )
        ):
            raise RuntimeError("failed to open one or more MP4 writers")
        self.time_s: list[float] = []
        self.diagnostic_finger_names = tuple(backend.diagnostic_finger_names)
        self.signal_by_finger = {
            name: [] for name in self.diagnostic_finger_names
        }
        self.metadata: list[dict] = []
        self.frame_index = 0
        self.last_label = None
        self.expert: MujocoPulseExpert | None = None
        self._idw = self._build_idw_weights(280, 160)
        self.custom_right_hand = "custom_right_hand" in backend.model_path.name
        hand_path = PROJECT_ROOT / "assets/visuals" / (
            "handle_l1_right_palm.png"
            if self.custom_right_hand
            else "adam_pro_right_palm_aligned_v2.png"
        )
        self.hand_cutout = cv2.imread(str(hand_path), cv2.IMREAD_UNCHANGED)
        if self.hand_cutout is None or self.hand_cutout.shape[2] != 4:
            raise RuntimeError(f"expected RGBA right-hand artwork: {hand_path}")
        opaque = cv2.findNonZero(
            (self.hand_cutout[:, :, 3] > 0).astype(np.uint8)
        )
        if opaque is None:
            raise RuntimeError(f"right-hand artwork is transparent: {hand_path}")
        x, y, width, height = cv2.boundingRect(opaque)
        self.hand_cutout = self.hand_cutout[
            y : y + height, x : x + width
        ]
        self.hand_artwork_path = hand_path

    @staticmethod
    def _build_idw_weights(
        width: int,
        height: int,
        *,
        longitudinal_shift_mm: float = 0.0,
    ) -> np.ndarray:
        coords = TEN_CHANNEL_COORDS_MM.astype(float)
        # The fingertip artwork points upward. In the official Pro distal-link
        # frame, supplier longitudinal 0 mm is distal and 25 mm is proximal.
        x = np.linspace(-1.0, 26.0, height)
        y = np.linspace(-1.0, 5.0, width)
        # Sample the unshifted sensor field at the inverse visual offset. A
        # positive display shift therefore moves the same physical peak down
        # toward the proximal/base end of the drawn fingertip.
        grid_x, grid_y = np.meshgrid(
            x - float(longitudinal_shift_mm), y, indexing="ij"
        )
        distance_sq = (
            (grid_x[..., None] - coords[:, 0]) ** 2
            + (grid_y[..., None] - coords[:, 1]) ** 2
        )
        weights = 1.0 / np.maximum(distance_sq, 0.20**2)
        weights /= weights.sum(axis=-1, keepdims=True)
        return weights.astype(np.float32)

    def _render(self, camera) -> np.ndarray:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(
            self.backend.data,
            camera=camera,
            scene_option=self.scene_option,
        )
        rgb = self.renderer.render().copy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _annotate_camera(
        self,
        image: np.ndarray,
        title: str,
        observation: dict,
        info: dict,
        label: str,
    ) -> np.ndarray:
        result = image.copy()
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (1280, 94), (15, 20, 24), -1)
        cv2.addWeighted(overlay, 0.70, result, 0.30, 0.0, result)
        phase = TaskPhase(int(observation["task_phase"])).name
        force = np.asarray(
            info.get(
                "normal_force_n",
                np.zeros(len(self.diagnostic_finger_names)),
            )
        )
        contact = self.backend.diagnostic_contact()
        initials = "/".join(name[0].upper() for name in self.diagnostic_finger_names)
        _text(result, title, (22, 30), scale=0.75, thickness=2)
        _text(
            result,
            f"t={observation['timestamp_s']:.2f}s  phase={phase}  motion={label}",
            (22, 58),
            scale=0.55,
        )
        _text(
            result,
            f"contact {initials}="
            + "/".join(str(int(value)) for value in contact)
            + f"  force {initials}="
            + "/".join(f"{value:.2f}" for value in force)
            + " N",
            (22, 84),
            scale=0.55,
        )
        return result

    def _waveform_panel(self, observation: dict, label: str) -> np.ndarray:
        panel = np.full((540, 960, 3), (246, 248, 250), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (959, 539), (95, 105, 112), 2)
        _text(
            panel,
            "Contact-gated dynamic tactile pulse",
            (28, 38),
            scale=0.72,
            color=(30, 35, 40),
            thickness=2,
        )
        left, top, right, bottom = 72, 78, 920, 430
        cv2.rectangle(panel, (left, top), (right, bottom), (220, 224, 228), 1)
        for fraction in np.linspace(0.0, 1.0, 5):
            y = int(top + fraction * (bottom - top))
            cv2.line(panel, (left, y), (right, y), (226, 229, 232), 1)
        window = 180
        times = np.asarray(self.time_s[-window:], dtype=float)
        signals = (
            np.asarray(
                [
                    self.signal_by_finger[name][-window:]
                    for name in self.diagnostic_finger_names
                ]
            )
            if self.time_s
            else np.zeros((len(self.diagnostic_finger_names), 0))
        )
        if len(times) >= 2:
            x = left + (
                (times - times[0])
                / max(float(times[-1] - times[0]), 1e-6)
                * (right - left)
            )
            ymax = max(0.35, float(np.percentile(signals, 99)) * 1.20)
            colors = (
                (40, 95, 225),
                (35, 175, 70),
                (210, 105, 35),
                (165, 60, 185),
                (40, 165, 195),
            )
            for values, color in zip(signals, colors):
                y = bottom - np.clip(values / ymax, 0.0, 1.0) * (
                    bottom - top
                )
                points = np.column_stack((x, y)).astype(np.int32)
                if len(points) >= 2:
                    cv2.polylines(
                        panel, [points], False, color, 2, cv2.LINE_AA
                    )
            _text(
                panel,
                f"window={times[-1] - times[0]:.1f}s  scale=0..{ymax:.2f} kPa",
                (left, 466),
                scale=0.48,
                color=(55, 60, 65),
            )
        legend_colors = (
            (40, 95, 225),
            (35, 175, 70),
            (210, 105, 35),
            (165, 60, 185),
            (40, 165, 195),
        )
        legend_x = 610
        for name, color in zip(self.diagnostic_finger_names, legend_colors):
            _text(
                panel,
                name.upper(),
                (legend_x, 38),
                scale=0.46,
                color=color,
                thickness=2,
            )
            legend_x += 96
        if self.expert is not None and self.expert.search_samples:
            samples = self.expert.search_samples
            text = "search: " + "  ".join(
                f"P{s['point_index']}={s['pulse_score_kpa']:.3f}"
                for s in samples
            )
            _text(
                panel,
                text[:84],
                (left, 510),
                scale=0.45,
                color=(55, 60, 65),
            )
        return panel

    def _draw_fingertip_field(
        self,
        panel: np.ndarray,
        *,
        left: int,
        top: int,
        width: int,
        height: int,
        values: np.ndarray,
        label: str | None = None,
        show_channels: bool = True,
        longitudinal_shift_mm: float = 0.0,
    ) -> None:
        # Positive shift moves the displayed sensor pattern toward the
        # proximal/base direction.  The raw channel order remains unchanged.
        if abs(float(longitudinal_shift_mm)) > 1e-9:
            shifted_idw = self._build_idw_weights(
                280,
                160,
                longitudinal_shift_mm=float(longitudinal_shift_mm),
            )
        else:
            shifted_idw = self._idw
        interpolated = np.tensordot(
            shifted_idw,
            np.asarray(values, dtype=np.float32),
            axes=([2], [0]),
        )
        normalized = np.uint8(
            np.clip(interpolated / 1.6, 0.0, 1.0) * 255.0
        )
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        colored = cv2.resize(
            colored, (width, height), interpolation=cv2.INTER_CUBIC
        )
        radius = width // 2
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (0, radius), (width - 1, height - 1), 255, -1)
        cv2.ellipse(
            mask,
            (width // 2, radius),
            (radius, radius),
            0,
            180,
            360,
            255,
            -1,
            cv2.LINE_AA,
        )
        region = panel[top : top + height, left : left + width]
        alpha = (mask.astype(np.float32) / 255.0 * 0.92)[:, :, None]
        region[:] = np.round(
            region.astype(np.float32) * (1.0 - alpha)
            + colored.astype(np.float32) * alpha
        ).astype(np.uint8)
        cv2.ellipse(
            panel,
            (left + width // 2, top + radius),
            (radius, radius),
            0,
            180,
            360,
            (65, 65, 65),
            1 if width < 50 else 2,
            cv2.LINE_AA,
        )
        cv2.line(
            panel,
            (left, top + radius),
            (left, top + height - 1),
            (65, 65, 65),
            1 if width < 50 else 2,
            cv2.LINE_AA,
        )
        cv2.line(
            panel,
            (left + width - 1, top + radius),
            (left + width - 1, top + height - 1),
            (65, 65, 65),
            1 if width < 50 else 2,
            cv2.LINE_AA,
        )
        if label is not None:
            _text(
                panel,
                label,
                (left + 8, top - 10),
                scale=0.52,
                color=(45, 45, 45),
                thickness=2,
            )
        if not show_channels:
            return
        for channel, (longitudinal, lateral) in enumerate(
            TEN_CHANNEL_COORDS_MM, start=1
        ):
            px = int(left + float(lateral) / 4.0 * (width - 14) + 7)
            py = int(
                top
                + (float(longitudinal) + float(longitudinal_shift_mm)) / 25.0
                * (height - 10)
                + 5
            )
            cv2.circle(panel, (px, py), 6, (250, 250, 250), -1, cv2.LINE_AA)
            cv2.circle(panel, (px, py), 6, (35, 35, 35), 1, cv2.LINE_AA)
            cv2.putText(
                panel,
                str(channel),
                (px - 4, py + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.22,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )

    def _heatmap_panel(self, observation: dict, info: dict) -> np.ndarray:
        panel = np.full((540, 960, 3), (246, 248, 250), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (959, 539), (95, 105, 112), 2)
        hand_name = (
            "custom right hand" if self.custom_right_hand else "Adam Pro right hand"
        )
        _text(
            panel,
            f"Live tactile map on {hand_name}",
            (28, 38),
            scale=0.68,
            color=(30, 35, 40),
            thickness=2,
        )
        _text(
            panel,
            "Active fingertip placement with anatomical height compensation",
            (28, 65),
            scale=0.43,
            color=(80, 85, 90),
        )
        dynamic, contact, force, raw = self._diagnostic_tactile()
        dynamic = np.maximum(dynamic, 0.0)
        dynamic[~contact] = 0.0

        hand_height = 395
        hand_width = int(
            self.hand_cutout.shape[1] / self.hand_cutout.shape[0] * hand_height
        )
        hand_left, hand_top = 26, 86
        hand_rgba = cv2.resize(
            self.hand_cutout,
            (hand_width, hand_height),
            interpolation=cv2.INTER_AREA,
        )
        alpha = hand_rgba[:, :, 3:4].astype(np.float32) / 255.0
        region = panel[
            hand_top : hand_top + hand_height,
            hand_left : hand_left + hand_width,
        ]
        region[:] = np.round(
            region.astype(np.float32) * (1.0 - alpha)
            + hand_rgba[:, :, :3].astype(np.float32) * alpha
        ).astype(np.uint8)

        # Palm-facing right hand: middle is left of index. The custom-hand
        # coordinates come from the mirrored, transparent Handle L1 artwork;
        # the official-hand coordinates preserve the frozen M6 layout.
        if self.custom_right_hand:
            placements = CUSTOM_HAND_PLACEMENTS
        else:
            placements = {
                "middle": (0.423, 0.040),
                "index": (0.610, 0.100),
            }
        by_name = {
            name: dynamic[index]
            for index, name in enumerate(self.diagnostic_finger_names)
        }
        for name in self.diagnostic_finger_names:
            x_fraction, y_fraction = placements[name]
            center_x = int(hand_left + x_fraction * hand_width)
            top_y = int(hand_top + y_fraction * hand_height)
            compensation_mm = float(
                CUSTOM_HAND_LONGITUDINAL_COMPENSATION_MM.get(name, 0.0)
            )
            self._draw_fingertip_field(
                panel,
                left=center_x - 13,
                top=top_y,
                width=26,
                height=76,
                values=by_name[name],
                show_channels=False,
                longitudinal_shift_mm=compensation_mm,
            )
        large_rectangles = (
            CUSTOM_HAND_LARGE_RECTS
            if self.custom_right_hand
            else {
                "middle": (585, 146, 120, 168),
                "index": (760, 214, 120, 168),
            }
        )
        for name in ("ring", "middle", "index"):
            if name not in by_name:
                continue
            left, top, width, height = large_rectangles[name]
            self._draw_fingertip_field(
                panel,
                left=left,
                top=top,
                width=width,
                height=height,
                values=by_name[name],
                label=f"{name.title()} fingertip",
                longitudinal_shift_mm=float(
                    CUSTOM_HAND_LONGITUDINAL_COMPENSATION_MM.get(name, 0.0)
                )
                if self.custom_right_hand
                else 0.0,
            )
        bar = np.tile(np.arange(256, dtype=np.uint8), (24, 1))
        bar = cv2.applyColorMap(bar, cv2.COLORMAP_TURBO)
        bar = cv2.resize(bar, (380, 20))
        panel[412:432, 430:890] = cv2.resize(bar, (460, 20))
        _text(
            panel,
            "0 kPa",
            (430, 456),
            scale=0.42,
            color=(50, 55, 60),
        )
        _text(
            panel,
            "1.6 kPa",
            (835, 456),
            scale=0.42,
            color=(50, 55, 60),
        )
        _text(
            panel,
            "dynamic pulse amplitude",
            (595, 456),
            scale=0.42,
            color=(50, 55, 60),
        )
        initials = "/".join(name[0].upper() for name in self.diagnostic_finger_names)
        _text(
            panel,
            f"Force {initials}: " + " / ".join(f"{value:.2f}" for value in force) + " N",
            (430, 493),
            scale=0.42,
            color=(35, 40, 45),
            thickness=2,
        )
        _text(
            panel,
            f"ADC min {initials}: " + " / ".join(str(int(row.min())) for row in raw),
            (430, 523),
            scale=0.38,
            color=(35, 40, 45),
        )
        return panel

    def _diagnostic_tactile(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        full = self.backend.last_tactile_full
        if full is not None:
            dynamic = []
            raw = []
            contact = []
            force = []
            for name in self.diagnostic_finger_names:
                target = full.finger_slice(name)
                dynamic.append(full.dynamic_pressure_kpa[target])
                raw.append(full.raw_values[target])
                finger_index = full.finger_names.index(name)
                contact.append(full.contact[finger_index])
                force.append(full.total_normal_force_n[finger_index])
            return (
                np.asarray(dynamic, dtype=np.float32),
                np.asarray(contact, dtype=bool),
                np.asarray(force, dtype=np.float32),
                np.asarray(raw, dtype=np.uint16),
            )
        tactile = self.backend.last_tactile
        if tactile is None:
            count = len(self.diagnostic_finger_names)
            return (
                np.zeros((count, 10), dtype=np.float32),
                np.zeros(count, dtype=bool),
                np.zeros(count, dtype=np.float32),
                np.zeros((count, 10), dtype=np.uint16),
            )
        return (
            tactile.dynamic_arrays_kpa.copy(),
            tactile.contact.copy(),
            tactile.total_normal_force_n.copy(),
            tactile.raw_values.copy(),
        )

    def __call__(
        self,
        backend,
        observation: dict,
        info: dict,
        label: str,
    ) -> None:
        del backend
        dynamic, contact, force, _raw = self._diagnostic_tactile()
        self.time_s.append(float(observation["timestamp_s"]))
        for index, name in enumerate(self.diagnostic_finger_names):
            self.signal_by_finger[name].append(float(dynamic[index].mean()))
        probe = np.asarray(observation["tcp_pose"][:3])
        front = self._annotate_camera(
            self._render(self.global_camera),
            "GLOBAL VIEW",
            observation,
            info,
            label,
        )
        local = self._annotate_camera(
            self._render(self.local_camera),
            "FINGERTIP VIEW",
            observation,
            info,
            label,
        )
        self.front_writer.write(front)
        self.local_writer.write(local)
        composite = np.empty((1080, 1920, 3), dtype=np.uint8)
        composite[:540, :960] = cv2.resize(front, (960, 540))
        composite[:540, 960:] = cv2.resize(local, (960, 540))
        composite[540:, :960] = self._waveform_panel(
            observation, label
        )
        composite[540:, 960:] = self._heatmap_panel(observation, info)
        self.composite_writer.write(composite)
        self.metadata.append(
            {
                "frame": self.frame_index,
                "time_s": float(observation["timestamp_s"]),
                "phase": TaskPhase(
                    int(observation["task_phase"])
                ).name,
                "motion": label,
                "diagnostic_finger_names": list(self.diagnostic_finger_names),
                "contact": contact.tolist(),
                "force_n": force.tolist(),
                "probe_position_m": probe.tolist(),
            }
        )
        if label != self.last_label:
            safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", label)
            cv2.imwrite(
                str(
                    self.keyframe_dir
                    / f"{self.frame_index:04d}_{safe}.png"
                ),
                composite,
            )
            self.last_label = label
        self.frame_index += 1

    def close(self) -> None:
        self.front_writer.release()
        self.local_writer.release()
        self.composite_writer.release()
        self.renderer.close()
        with (self.output_dir / "frame_metadata.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for record in self.metadata:
                stream.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/alignment_pro_custom_right_hand_multifinger_elbow_out_safe_v7_expert_fast_contact_contactfirst.yaml",
    )
    parser.add_argument(
        "--search-strategy",
        choices=("directional_gradient", "signal_quality_early_stop"),
        default=None,
        help="override configs/alignment.yaml task.pulse_search_strategy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/expert_demo/review_v1",
    )
    parser.add_argument("--pulse-offset-x-mm", type=float, default=0.0)
    parser.add_argument("--pulse-offset-y-mm", type=float, default=0.0)
    parser.add_argument(
        "--disable-proximity",
        action="store_true",
        help="record with all simulated proximity channels invalid",
    )
    parser.add_argument(
        "--segment-checkpoint",
        type=Path,
        help="optional frozen search-segment checkpoint used during the demo",
    )
    parser.add_argument(
        "--segment-policy-mode",
        choices=("disabled", "shadow", "execute_reviewed", "execute_model_only"),
        default="disabled",
        help=(
            "model authority for the pulse-search displacement only; "
            "execute_model_only does not replace approach/pressure/safety control"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cuda",),
        default="cuda",
        help="CUDA is mandatory whenever the optional learned search policy is used.",
    )
    args = parser.parse_args()
    if args.segment_policy_mode == "disabled" and args.segment_checkpoint is not None:
        parser.error("--segment-checkpoint requires an enabled segment policy mode")
    if args.segment_policy_mode != "disabled":
        if args.segment_checkpoint is None:
            parser.error("enabled segment policy requires --segment-checkpoint")
        if not args.segment_checkpoint.exists():
            parser.error(f"segment checkpoint does not exist: {args.segment_checkpoint}")
    backend = MujocoPulseDiagnosisBackend(
        PROJECT_ROOT,
        config_path=args.config,
        include_rgbd=False,
        maximum_steps=9000,
    )
    pulse_offset_m = np.asarray(
        [args.pulse_offset_x_mm, args.pulse_offset_y_mm], dtype=float
    ) / 1000.0
    backend.sensors.pulse_point_m[:2] += pulse_offset_m
    backend.config["scene"]["patient_forearm"]["pulse_center_m"] = (
        backend.sensors.pulse_point_m.tolist()
    )
    perturbations = None
    if args.disable_proximity:
        perturbations = PerturbationScheduler(
            (
                PerturbationEvent(
                    "proximity_hardware_absent",
                    PerturbationKind.PROXIMITY_DROPOUT,
                    0,
                    20_000,
                    {"channels": [0, 1, 2, 3, 4]},
                    recoverable=False,
                ),
            )
        )
    env = PulseDiagnosisEnv(backend, perturbation_scheduler=perturbations)
    recorder = DemoRecorder(backend, args.output_dir)
    segment_hook = None
    step_callback = recorder
    if args.segment_policy_mode != "disabled":
        segment_hook = SearchSegmentShadowHook(
            load_search_segment_runtime(
                args.segment_checkpoint,
                device=None if args.device == "auto" else args.device,
            ),
            downstream=recorder,
        )
        step_callback = segment_hook
    expert = MujocoPulseExpert(
        env,
        on_step=step_callback,
        search_strategy=args.search_strategy,
        search_segment_selector=(
            None if segment_hook is None else segment_hook.propose
        ),
        search_segment_policy_mode=args.segment_policy_mode,
    )
    recorder.expert = expert
    try:
        run = expert.run(seed=args.seed)
    finally:
        recorder.close()
    model_path = backend.model_path.resolve()
    run.result.update(
        {
            "mujoco_version": mujoco.__version__,
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "expert_version": (
                "mujoco_signal_quality_v1"
                if run.result["search_strategy"] == "signal_quality_early_stop"
                else "mujoco_directional_v1"
            ),
            "sensor_model_version": (
                "handle_l1_right_full56_empirical_10n_v1"
                if recorder.custom_right_hand
                else "adam_pro_right_custom_tactile_empirical_10n_v1"
            ),
            "migration_version": (
                "mujoco_adam_pro_custom_right_hand_v1"
                if recorder.custom_right_hand
                else "mujoco_adam_pro_official_right_hand_v1"
            ),
            "pulse_offset_xy_mm": [
                float(args.pulse_offset_x_mm),
                float(args.pulse_offset_y_mm),
            ],
            "proximity_hardware_present": not args.disable_proximity,
            "search_segment_policy_mode": args.segment_policy_mode,
            "search_segment_checkpoint": (
                None
                if args.segment_checkpoint is None
                else str(args.segment_checkpoint.resolve())
            ),
            "search_segment_policy_runtime": (
                None if segment_hook is None else segment_hook.summary()
            ),
            "hand_artwork_path": str(recorder.hand_artwork_path.resolve()),
            "videos": {
                "front": recorder.front_path.name,
                "fingertip": recorder.local_path.name,
                "composite": recorder.composite_path.name,
            },
            "keyframes": recorder.keyframe_dir.name,
            "video_fps": FPS,
            "video_playback_rate": "1x",
            "video_frame_count": recorder.frame_index,
        }
    )
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(run.result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "tactile_trace.npz",
        time_s=run.tactile_time_s,
        tactile_kpa=run.tactile_kpa,
        tactile_dynamic_kpa=run.tactile_dynamic_kpa,
        tactile_raw=run.tactile_raw,
        tactile_raw_delta=run.tactile_raw_delta,
        force_n=run.force_n,
        contact=run.contact,
        probe_position_m=run.probe_position_m,
        phase=run.phase,
        tactile_full_56_kpa=run.tactile_full_56_kpa,
        tactile_full_56_dynamic_kpa=run.tactile_full_56_dynamic_kpa,
        tactile_full_56_raw=run.tactile_full_56_raw,
        full_hand_force_n=run.full_hand_force_n,
        full_hand_contact=run.full_hand_contact,
        full_hand_finger_names=np.asarray(
            ["thumb", "index", "middle", "ring", "little"]
        ),
        diagnostic_finger_names=np.asarray(recorder.diagnostic_finger_names),
    )
    print(json.dumps(run.result, ensure_ascii=False, indent=2))
    env.close()
    if not run.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
