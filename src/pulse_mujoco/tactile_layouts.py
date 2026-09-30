from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Channel order and local fingertip coordinates copied from
# incoming/触觉传感信息.txt. Units were confirmed by the user as millimetres.
TEN_CHANNEL_COORDS_MM = np.asarray(
    [
        [0.0, 2.0],
        [5.0, 0.0],
        [5.0, 4.0],
        [10.0, 0.0],
        [10.0, 4.0],
        [15.0, 0.0],
        [15.0, 4.0],
        [20.0, 0.0],
        [20.0, 4.0],
        [25.0, 2.0],
    ],
    dtype=np.float32,
)

SIXTEEN_CHANNEL_COORDS_MM = np.asarray(
    [
        [0.0, 2.0],
        [0.0, 6.0],
        [5.0, 0.0],
        [5.0, 4.0],
        [5.0, 8.0],
        [10.0, 0.0],
        [10.0, 4.0],
        [10.0, 8.0],
        [15.0, 0.0],
        [15.0, 4.0],
        [15.0, 8.0],
        [20.0, 0.0],
        [20.0, 4.0],
        [20.0, 8.0],
        [25.0, 2.0],
        [25.0, 6.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class TaxelLayout:
    """Provisional SI-unit geometry using supplied channel centre coordinates.

    The supplier data contains channel centres but no individual sensitive
    areas. Equal 10 mm² effective areas are therefore an explicit development
    assumption, not a claimed hardware specification.
    """

    positions_local_m: np.ndarray
    areas_m2: np.ndarray
    finger_name: str


def handle_l1_ten_channel_layout(finger_name: str) -> TaxelLayout:
    coordinates = TEN_CHANNEL_COORDS_MM.astype(np.float64) * 1e-3
    coordinates -= np.mean(coordinates, axis=0, keepdims=True)
    # Tactile frame local -X is the pad outward normal. The supplier's first
    # coordinate spans 0..25 mm and therefore follows the finger longitudinal
    # direction (local Z); its second coordinate spans the pad width (local Y).
    positions = np.zeros((10, 3), dtype=np.float32)
    positions[:, 1] = coordinates[:, 1]
    positions[:, 2] = coordinates[:, 0]
    return TaxelLayout(
        positions_local_m=positions,
        areas_m2=np.full(10, 10.0e-6, dtype=np.float32),
        finger_name=str(finger_name),
    )


def handle_l1_sixteen_channel_layout(finger_name: str = "thumb") -> TaxelLayout:
    coordinates = SIXTEEN_CHANNEL_COORDS_MM.astype(np.float64) * 1e-3
    coordinates -= np.mean(coordinates, axis=0, keepdims=True)
    positions = np.zeros((16, 3), dtype=np.float32)
    positions[:, 1] = coordinates[:, 1]
    positions[:, 2] = coordinates[:, 0]
    return TaxelLayout(
        positions_local_m=positions,
        areas_m2=np.full(16, 10.0e-6, dtype=np.float32),
        finger_name=str(finger_name),
    )


def handle_l1_layout_for_finger(finger_name: str) -> TaxelLayout:
    name = str(finger_name)
    return (
        handle_l1_sixteen_channel_layout(name)
        if name == "thumb"
        else handle_l1_ten_channel_layout(name)
    )
