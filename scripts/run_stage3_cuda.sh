#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$project_root/.venv_stage3/bin/python"

if [[ ! -x "$python_bin" ]]; then
  echo "Stage 3 Python runtime not found: $python_bin" >&2
  exit 2
fi

cd "$project_root"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" - "$python_bin" <<'PY'
from __future__ import annotations

import os
from pathlib import Path
import sys

expected_python = Path(sys.argv[1]).resolve()
actual_python = Path(sys.executable).resolve()
if actual_python != expected_python:
    raise SystemExit(
        f"wrong Python runtime: expected {expected_python}, got {actual_python}"
    )

import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not visible in .venv_stage3. Stop here; do not fall back to CPU."
    )

if os.environ.get("MUJOCO_GL") != "egl":
    raise SystemExit("headless Stage 3 requires MUJOCO_GL=egl")

import mujoco
import numpy as np

xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="1 1 0.1" rgba="0.2 0.3 0.4 1"/>
    <body pos="0 0 0.15">
      <geom type="sphere" size="0.15" rgba="0.8 0.2 0.1 1"/>
    </body>
    <camera name="smoke" pos="0 -1.5 0.8" xyaxes="1 0 0 0 0.45 0.89"/>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=64, width=64)
try:
    renderer.update_scene(data, camera="smoke")
    image = renderer.render()
finally:
    renderer.close()

if image.shape != (64, 64, 3) or not np.any(image):
    raise SystemExit(f"MuJoCo EGL smoke render is invalid: shape={image.shape}")

print(
    "Stage 3 runtime OK: "
    f"python={sys.executable}, torch={torch.__version__}, "
    f"gpu={torch.cuda.get_device_name(0)}, MUJOCO_GL={os.environ['MUJOCO_GL']}, "
    f"render_shape={image.shape}"
)
PY

if [[ $# -eq 0 || "${1:-}" == "--check" ]]; then
  exit 0
fi

exec "$python_bin" "$@"
