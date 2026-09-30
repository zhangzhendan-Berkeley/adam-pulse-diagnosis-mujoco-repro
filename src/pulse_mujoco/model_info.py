from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco


def _names(model: mujoco.MjModel, count: int, object_type) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, object_type, index) or ""
        for index in range(count)
    )


@dataclass(frozen=True)
class ModelInventory:
    source: str
    mujoco_version: str
    nq: int
    nv: int
    nu: int
    nbody: int
    ngeom: int
    nsensor: int
    ncam: int
    joints: tuple[str, ...]
    actuators: tuple[str, ...]
    bodies: tuple[str, ...]
    sensors: tuple[str, ...]
    cameras: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def inventory_model(path: str | Path) -> ModelInventory:
    source = Path(path).resolve()
    model = mujoco.MjModel.from_xml_path(str(source))
    return ModelInventory(
        source=str(source),
        mujoco_version=mujoco.__version__,
        nq=int(model.nq),
        nv=int(model.nv),
        nu=int(model.nu),
        nbody=int(model.nbody),
        ngeom=int(model.ngeom),
        nsensor=int(model.nsensor),
        ncam=int(model.ncam),
        joints=_names(
            model, model.njnt, mujoco.mjtObj.mjOBJ_JOINT
        ),
        actuators=_names(
            model, model.nu, mujoco.mjtObj.mjOBJ_ACTUATOR
        ),
        bodies=_names(
            model, model.nbody, mujoco.mjtObj.mjOBJ_BODY
        ),
        sensors=_names(
            model, model.nsensor, mujoco.mjtObj.mjOBJ_SENSOR
        ),
        cameras=_names(
            model, model.ncam, mujoco.mjtObj.mjOBJ_CAMERA
        ),
    )

