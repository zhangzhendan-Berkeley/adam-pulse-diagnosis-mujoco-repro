from pathlib import Path
import re

import mujoco
import numpy as np

from pulse_mujoco import MujocoPulseDiagnosisBackend


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/alignment_pro_custom_right_hand_multifinger_elbow_out_safe_v7_expert_fast_contact_contactfirst.yaml"
MODEL = ROOT / "generated/adam_pro_custom_right_hand_elbow_out_safe_v5_pulse.xml"


def test_scene_meshes_are_local_and_model_loads():
    text = MODEL.read_text(encoding="utf-8")
    for reference in re.findall(r'file="([^"]+)"', text):
        assert (MODEL.parent / reference).is_file(), reference
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert model.ngeom > 100


def test_contact_first_expert_has_full_hand_tactile_contract():
    backend = MujocoPulseDiagnosisBackend(
        ROOT, config_path=CONFIG, include_rgbd=False, maximum_steps=1
    )
    try:
        assert backend.model_path == MODEL
        assert backend.diagnostic_finger_names == ("index", "middle", "ring")
        backend.reset(seed=11)
        assert np.asarray(backend._info(ik_converged=True)["tactile_full_56"]["pressure_kpa"]).size == 56
        assert backend.config["task"]["pressure_stable_without_full_command"]
    finally:
        backend.close()
