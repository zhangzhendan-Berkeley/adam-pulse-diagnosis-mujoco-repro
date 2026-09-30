from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    timestamp_s: float
    camera_name: str


class MujocoRGBDCamera:
    """Metric RGB-D camera with explicit calibration and timestamp."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str,
        width: int,
        height: int,
    ):
        self.model = model
        self.camera_name = str(camera_name)
        self.camera_id = model.camera(self.camera_name).id
        self.width = int(width)
        self.height = int(height)
        self.renderer = mujoco.Renderer(
            model, width=self.width, height=self.height
        )
        self.intrinsics = self._intrinsics(
            self.width,
            self.height,
            float(model.cam_fovy[self.camera_id]),
        )

    @staticmethod
    def _intrinsics(
        width: int, height: int, vertical_fov_deg: float
    ) -> np.ndarray:
        fy = 0.5 * float(height) / np.tan(
            0.5 * np.deg2rad(float(vertical_fov_deg))
        )
        return np.asarray(
            [
                [fy, 0.0, (float(width) - 1.0) * 0.5],
                [0.0, fy, (float(height) - 1.0) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def extrinsics(self, data: mujoco.MjData) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = data.cam_xmat[self.camera_id].reshape(3, 3)
        transform[:3, 3] = data.cam_xpos[self.camera_id]
        return transform

    def capture(self, data: mujoco.MjData) -> RGBDFrame:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        depth = self.renderer.render().copy().astype(np.float32)
        self.renderer.disable_depth_rendering()
        return RGBDFrame(
            rgb=rgb,
            depth_m=depth,
            intrinsics=self.intrinsics.copy(),
            world_from_camera=self.extrinsics(data),
            timestamp_s=float(data.time),
            camera_name=self.camera_name,
        )

    def close(self) -> None:
        self.renderer.close()
