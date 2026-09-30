from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


EXPECTED_VIDEOS = {
    "front": (1280, 720),
    "fingertip": (1280, 720),
    "composite": (1920, 1080),
}

SEARCH_TRANSITION = [
    "LIFT_FROM_CURRENT",
    "VERIFY_CONTACT_RELEASE",
    "TRANSLATE_ABOVE_SURFACE",
    "VERIFY_TARGET_REACHED",
    "DESCEND_TO_NEW_POINT",
    "ESTABLISH_DUAL_CONTACT",
    "REGULATE_PRESSURE",
]


def _contains_subsequence(values: list[str], expected: list[str]) -> bool:
    position = 0
    for value in values:
        if value == expected[position]:
            position += 1
            if position == len(expected):
                return True
    return False


def _validate_video(
    path: Path,
    expected_size: tuple[int, int],
    expected_frames: int,
    expected_fps: float,
) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError(f"cannot open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    assert (width, height) == expected_size, (path, width, height)
    assert frames == expected_frames, (path, frames, expected_frames)
    assert abs(fps - expected_fps) <= 0.05, (path, fps, expected_fps)
    sampled_std = []
    for index in np.linspace(0, frames - 1, 7, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        assert ok and frame is not None, (path, index)
        sampled_std.append(float(frame.std()))
    capture.release()
    assert min(sampled_std) >= 5.0, (path, sampled_std)
    return {
        "path": str(path.resolve()),
        "size": [width, height],
        "frames": frames,
        "fps": fps,
        "duration_s": frames / fps,
        "minimum_sampled_pixel_std": min(sampled_std),
    }


def validate(output_dir: Path) -> dict:
    result = json.loads((output_dir / "result.json").read_text())
    assert result["success"]
    assert result["final_phase"] == "success"
    assert result["search_strategy"] in {
        "directional_gradient",
        "signal_quality_early_stop",
    }
    assert not result["uses_privileged_pulse_guidance"]
    assert not result["fixed_trajectory_replay"]
    if result["search_strategy"] == "directional_gradient":
        assert result["three_stage_search_success"]
        assert result["unique_search_positions"] >= 3
        assert _contains_subsequence(result["search_states"], SEARCH_TRANSITION)
        assert result["evaluation_lateral_error_mm"] <= 2.0
    else:
        assert result["signal_quality_schema_version"] == (
            "pulse_signal_quality_v1"
        )
        assert result["signal_quality_accepted"]
        assert result["selected_signal_quality"]["accepted"]
        assert result["search_stop_reason"] == "signal_quality_satisfied"
        if result["unique_search_positions"] > 1:
            assert result["three_stage_search_success"]
            assert _contains_subsequence(
                result["search_states"], SEARCH_TRANSITION
            )
    assert result["capture"]["duration_s"] >= 6.0
    assert result["capture"]["contact_ratio"] >= 0.95
    assert 1.0 <= result["capture"]["dominant_frequency_hz"] <= 1.4
    assert result["capture"]["mean_dynamic_pulse_peak_to_peak_kpa"] >= 0.2
    # The sensor calibration covers 0--10 N.  The active task contract may
    # deliberately retain a lower software ceiling (9 N for the CH5 custom
    # hand, 8 N for older demos), so validate against the recorded contract
    # rather than a stale hard-coded two-finger limit.
    maximum_force_limit_n = float(
        result.get("signal_quality_thresholds", {}).get(
            "maximum_force_n", 8.0
        )
    )
    assert 0.0 < maximum_force_limit_n <= 10.0
    assert max(result["maximum_force_n_per_finger"]) <= maximum_force_limit_n
    assert result["forbidden_contact_events"] == 0

    trace = np.load(output_dir / "tactile_trace.npz")
    frame_count = int(result["video_frame_count"])
    if "diagnostic_finger_names" in trace.files:
        diagnostic_finger_count = int(
            np.asarray(trace["diagnostic_finger_names"]).size
        )
    else:
        # Backward compatibility for old two-finger artifacts.
        diagnostic_finger_count = int(trace["tactile_kpa"].shape[1])
    assert diagnostic_finger_count >= 2
    diagnostic_shape = (frame_count, diagnostic_finger_count, 10)
    assert trace["tactile_kpa"].shape == diagnostic_shape
    assert trace["tactile_raw"].shape == diagnostic_shape
    assert trace["contact"].shape == (frame_count, diagnostic_finger_count)
    if "tactile_full_56_kpa" in trace.files:
        assert trace["tactile_full_56_kpa"].shape == (frame_count, 56)
        assert trace["tactile_full_56_raw"].shape == (frame_count, 56)
        assert trace["full_hand_force_n"].shape == (frame_count, 5)
        assert trace["full_hand_contact"].shape == (frame_count, 5)

    video_results = {}
    for name, expected_size in EXPECTED_VIDEOS.items():
        video_path = Path(result["videos"][name])
        if not video_path.is_absolute():
            video_path = output_dir / video_path
        elif not video_path.exists():
            video_path = output_dir / video_path.name
        video_results[name] = _validate_video(
            video_path,
            expected_size,
            frame_count,
            float(result["video_fps"]),
        )

    keyframes = list((output_dir / "keyframes").glob("*.png"))
    required_keyframes = [
        "contact_approach",
        "search_sample",
        "signal_capture",
        "retreat_lift",
        "success",
    ]
    if result["unique_search_positions"] > 1:
        required_keyframes.extend(
            ["search_lift", "search_translate", "search_descend"]
        )
    for label in required_keyframes:
        assert any(path.stem.endswith(label) for path in keyframes), label

    return {
        "valid": True,
        "output_dir": str(output_dir.resolve()),
        "search_positions": result["unique_search_positions"],
        "lateral_error_mm": result["evaluation_lateral_error_mm"],
        "dominant_frequency_hz": result["capture"]["dominant_frequency_hz"],
        "pulse_peak_to_peak_kpa": result["capture"][
            "mean_dynamic_pulse_peak_to_peak_kpa"
        ],
        "diagnostic_finger_count": diagnostic_finger_count,
        "maximum_force_limit_n": maximum_force_limit_n,
        "videos": video_results,
        "keyframe_count": len(keyframes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
