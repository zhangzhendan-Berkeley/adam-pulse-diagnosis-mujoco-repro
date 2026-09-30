from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from pulse_mujoco import (
    COLLECTION_MANIFEST_SCHEMA_VERSION,
    MujocoEpisodeRecorder,
    MujocoPulseDiagnosisBackend,
    MujocoPulseExpert,
    PulseDiagnosisEnv,
    apply_collection_scenario,
    build_collection_scenarios,
    collection_acceptance,
    collection_perturbation_scheduler,
    new_collection_manifest,
    update_collection_counts,
    validate_episode,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "安全接近患者腕部并采集满足质量标准的脉搏信号"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _load_manifest(
    path: Path,
    target: int,
    seed: int,
    resume: bool,
    source_contract: dict,
) -> dict:
    if path.exists():
        if not resume:
            raise FileExistsError(f"collection manifest already exists: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != COLLECTION_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported collection manifest schema")
        if int(manifest["seed"]) != seed:
            raise ValueError("resume seed does not match collection manifest")
        if int(manifest["target_accepted"]) != target:
            raise ValueError("resume target does not match collection manifest")
        if manifest.get("source_contract") != source_contract:
            raise ValueError(
                "resume source contract does not match collection manifest"
            )
        update_collection_counts(manifest)
        return manifest
    return new_collection_manifest(
        target, seed, source_contract=source_contract
    )


def _quarantine_untracked_episode(output: Path, episode_dir: Path) -> Path:
    """Preserve a directory left between mkdir and atomic manifest commit."""
    failed_root = output / "failed_runs"
    failed_root.mkdir(exist_ok=True)
    destination = failed_root / f"orphan_{episode_dir.name}"
    suffix = 1
    while destination.exists():
        destination = failed_root / f"orphan_{episode_dir.name}_{suffix:02d}"
        suffix += 1
    shutil.move(str(episode_dir), destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect resumable randomized MuJoCo expert episodes.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/alignment_pro_custom_right_hand_multifinger_elbow_out_safe_v7_expert_fast_contact_contactfirst.yaml",
    )
    parser.add_argument("--target-accepted", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--capture-interval", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--record-width", type=int, default=320)
    parser.add_argument("--record-height", type=int, default=240)
    parser.add_argument("--maximum-attempts", type=int)
    parser.add_argument(
        "--search-strategy",
        choices=("directional_gradient", "signal_quality_early_stop"),
        default="signal_quality_early_stop",
    )
    args = parser.parse_args()
    if args.target_accepted < 1:
        parser.error("--target-accepted must be positive")
    if min(args.record_width, args.record_height) < 1:
        parser.error("--record-width and --record-height must be positive")
    maximum_attempts = args.maximum_attempts or max(2 * args.target_accepted, args.target_accepted + 10)
    if maximum_attempts < args.target_accepted:
        parser.error("--maximum-attempts must be at least --target-accepted")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    probe_backend = MujocoPulseDiagnosisBackend(
        ROOT, config_path=config_path, include_rgbd=False, maximum_steps=1
    )
    source_contract = {
        "robot_model": str(probe_backend.config["project"]["robot_model"]),
        "hand_side": str(
            probe_backend.config["hand"].get(
                "diagnosis_side",
                probe_backend.config["hand"].get("side", "unknown"),
            )
        ),
        "model_sha256": _sha256(probe_backend.model_path),
        "config_sha256": _sha256(config_path),
        "arm_joint_dim": len(probe_backend.arm_joint_names),
    }
    if args.search_strategy != "signal_quality_early_stop":
        source_contract["expert_search_strategy"] = args.search_strategy
    probe_backend.close()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "collection_manifest.json"
    manifest = _load_manifest(
        manifest_path,
        args.target_accepted,
        args.seed,
        args.resume,
        source_contract,
    )
    write_json_atomic(manifest_path, manifest)
    scenarios = build_collection_scenarios(maximum_attempts, args.seed)
    hashes = dict(source_contract)

    attempt_index = len(manifest["attempts"])
    while manifest["accepted"] < args.target_accepted and attempt_index < maximum_attempts:
        scenario = scenarios[attempt_index]
        episode_id = f"{attempt_index:06d}"
        episode_dir = args.output / f"episode_{episode_id}"
        if episode_dir.exists():
            if not args.resume:
                raise FileExistsError(f"untracked episode blocks collection: {episode_dir}")
            quarantined = _quarantine_untracked_episode(args.output, episode_dir)
            print(f"quarantined untracked episode directory: {quarantined}", flush=True)
        started = time.monotonic()
        recorder = MujocoEpisodeRecorder(
            args.output,
            episode_id,
            INSTRUCTION,
            capture_interval=args.capture_interval,
            jpeg_quality=args.jpeg_quality,
            record_size=(args.record_width, args.record_height),
        )
        backend = None
        run = None
        operational_error = None
        try:
            backend = MujocoPulseDiagnosisBackend(
                ROOT,
                config_path=config_path,
                include_rgbd=True,
                maximum_steps=8000,
            )
            apply_collection_scenario(backend, scenario)
            env = PulseDiagnosisEnv(
                backend,
                perturbation_scheduler=collection_perturbation_scheduler(
                    scenario
                ),
            )
            run = MujocoPulseExpert(
                env, on_step=recorder, search_strategy=args.search_strategy
            ).run(seed=args.seed + attempt_index)
            result = run.result
            calibration_path = ROOT / backend.config["sensors"]["calibration"]
            recorder.close(
                result,
                extra={
                    "random_seed": args.seed + attempt_index,
                    "collection_seed": args.seed,
                    "scenario": scenario.as_dict(),
                    "patient_profile": "mujoco_randomized_forearm_v1",
                    "pulse_profile": "radial_pulse_randomized_v1",
                    "split_group": f"scenario:{scenario.scenario_id}",
                    "config_path": str(config_path),
                    "controlled_chain": list(backend.arm_joint_names),
                    **hashes,
                    "calibration_sha256": _sha256(calibration_path),
                    "physics_dt_s": float(backend.model.opt.timestep),
                    "policy_rate_hz": 20.0,
                    "collection_search_strategy": args.search_strategy,
                },
            )
        except Exception as error:
            operational_error = f"{type(error).__name__}: {error}"
            result = {
                "success": False,
                "signal_quality_accepted": False,
                "forbidden_contact_events": 0,
                "failure": operational_error,
            }
            if recorder.records and not (episode_dir / "episode.json").exists():
                recorder.close(result, extra={"random_seed": args.seed + attempt_index, "scenario": scenario.as_dict()})
        finally:
            if backend is not None:
                backend.close()

        validation = validate_episode(episode_dir) if (episode_dir / "episode.json").exists() else None
        if validation is None:
            accepted, reasons = False, [operational_error or "episode was not finalized"]
        else:
            accepted, reasons = collection_acceptance(result, validation)
        if not accepted and episode_dir.exists():
            failed_root = args.output / "failed_runs"
            failed_root.mkdir(exist_ok=True)
            destination = failed_root / episode_dir.name
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(episode_dir), destination)
            stored_path = destination.relative_to(args.output)
        else:
            stored_path = episode_dir.relative_to(args.output)
        quality = result.get("selected_signal_quality") or {}
        entry = {
            "attempt": attempt_index,
            "episode": str(stored_path),
            "accepted": accepted,
            "scenario": scenario.as_dict(),
            "failure": result.get("failure") or operational_error,
            "rejection_reasons": reasons,
            "frames": 0 if validation is None else validation.frames,
            "bytes": _directory_bytes(args.output / stored_path),
            "simulation_time_s": result.get("simulation_time_s"),
            "signal_quality": quality,
            "forbidden_contact_events": int(result.get("forbidden_contact_events", 0)),
            "search_strategy": result.get("search_strategy"),
        }
        manifest["attempts"].append(entry)
        update_collection_counts(manifest)
        write_json_atomic(manifest_path, manifest)
        print(json.dumps({"accepted": manifest["accepted"], "failed": manifest["failed"], **entry}, ensure_ascii=False), flush=True)
        attempt_index += 1

    if not manifest["complete"]:
        raise SystemExit(f"accepted {manifest['accepted']}/{args.target_accepted} after {maximum_attempts} attempts")


if __name__ == "__main__":
    main()
