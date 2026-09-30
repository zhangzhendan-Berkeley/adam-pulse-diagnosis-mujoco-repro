from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CalibrationBuildReport:
    parsed_rows: int
    valid_loaded_rows: int
    rejected_positive_or_zero_fz: int
    rejected_px38: int
    baseline_raw: np.ndarray
    material_exponent: float
    material_coefficient_n_per_m_power: float


def _isotonic_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Unweighted pool-adjacent-violators fit constrained to decrease."""
    source = np.asarray(values, dtype=float)
    means: list[float] = []
    counts: list[int] = []
    for value in -source:
        means.append(float(value))
        counts.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            count = counts[-2] + counts[-1]
            mean = (
                means[-2] * counts[-2] + means[-1] * counts[-1]
            ) / count
            means[-2:] = [mean]
            counts[-2:] = [count]
    fitted = np.concatenate(
        [np.full(count, -mean, dtype=float) for mean, count in zip(means, counts)]
    )
    return fitted


def _load_force_sensor_rows(path: Path) -> np.ndarray:
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) != 20:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No 20-column calibration rows found in {path}")
    return np.asarray(rows, dtype=np.float64)


def fit_material_force_curve(path: Path) -> tuple[float, float]:
    """Fit F = coefficient * indentation_m**exponent to the 1 cm cube data."""
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) != 10:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue
    data = np.asarray(rows, dtype=float)
    if data.shape[0] < 10:
        raise ValueError("Insufficient material compression samples")
    force_n = -data[:, 3]
    z_mm = data[:, 9]
    zero_z_mm = float(z_mm[np.argmin(np.maximum(force_n, 0.0))])
    indentation_m = np.maximum(z_mm - zero_z_mm, 0.0) * 1e-3
    valid = (force_n > 0.05) & (indentation_m > 0.0)
    exponent, log_coefficient = np.polyfit(
        np.log(indentation_m[valid]), np.log(force_n[valid]), 1
    )
    return float(exponent), float(np.exp(log_coefficient))


def build_handle_l1_10ch_lut(
    source_path: Path,
    material_path: Path,
    output_path: Path,
    force_step_n: float = 0.05,
    maximum_force_n: float = 10.0,
) -> CalibrationBuildReport:
    """Build a monotonic empirical 10-channel lookup table."""
    data = _load_force_sensor_rows(source_path)
    fz = data[:, 3]
    raw = data[:, 7:17]
    px = data[:, 17]
    py = data[:, 18]
    baseline_mask = np.abs(fz) <= 0.02
    baseline = np.median(raw[baseline_mask], axis=0)
    loaded = (fz < -0.02) & (-fz <= maximum_force_n + 0.2)
    valid = loaded & (px != 38.0)
    force_n = -fz
    grid_x = np.arange(39.0, 68.0, 1.0)
    grid_y = np.arange(16.0, 23.0, 1.0)
    force_axis = np.arange(
        0.0, maximum_force_n + 0.5 * force_step_n, force_step_n
    )
    lut = np.empty(
        (len(grid_x), len(grid_y), len(force_axis), 10), dtype=np.uint16
    )
    for x_index, x_value in enumerate(grid_x):
        for y_index, y_value in enumerate(grid_y):
            cell = valid & (px == x_value) & (py == y_value)
            cell_force = force_n[cell]
            cell_raw = raw[cell]
            if not np.any(cell):
                lut[x_index, y_index] = np.broadcast_to(
                    np.rint(baseline).astype(np.uint16), (len(force_axis), 10)
                )
                continue
            bin_index = np.clip(
                np.rint(cell_force / force_step_n).astype(int),
                0,
                len(force_axis) - 1,
            )
            unique_bins = np.unique(bin_index)
            sample_force = np.concatenate(
                ([0.0], force_axis[unique_bins])
            )
            sample_raw = np.vstack(
                (
                    baseline,
                    np.asarray(
                        [
                            np.median(cell_raw[bin_index == index], axis=0)
                            for index in unique_bins
                        ]
                    ),
                )
            )
            for channel in range(10):
                curve = np.interp(
                    force_axis,
                    sample_force,
                    sample_raw[:, channel],
                    left=baseline[channel],
                    right=sample_raw[-1, channel],
                )
                curve = np.minimum(curve, baseline[channel])
                curve = _isotonic_nonincreasing(curve)
                curve[0] = baseline[channel]
                lut[x_index, y_index, :, channel] = np.rint(
                    np.clip(curve, 0.0, 4096.0)
                ).astype(np.uint16)
    exponent, coefficient = fit_material_force_curve(material_path)
    digest = sha256(Path(source_path).read_bytes()).hexdigest()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        raw_lut=lut,
        grid_x=grid_x.astype(np.float32),
        grid_y=grid_y.astype(np.float32),
        force_axis_n=force_axis.astype(np.float32),
        baseline_raw=np.rint(baseline).astype(np.uint16),
        supplier_to_machine_x_offset_mm=np.float32(40.0),
        supplier_to_machine_y_intercept_mm=np.float32(21.0),
        supplier_to_machine_y_sign=np.float32(-1.0),
        raw_min=np.uint16(0),
        raw_max=np.uint16(4096),
        material_exponent=np.float64(exponent),
        material_coefficient_n_per_m_power=np.float64(coefficient),
        source_sha256=np.asarray(digest),
        source_rows=np.int64(len(data)),
        valid_loaded_rows=np.int64(np.count_nonzero(valid)),
    )
    return CalibrationBuildReport(
        parsed_rows=len(data),
        valid_loaded_rows=int(np.count_nonzero(valid)),
        rejected_positive_or_zero_fz=int(np.count_nonzero(~loaded)),
        rejected_px38=int(np.count_nonzero(loaded & (px == 38.0))),
        baseline_raw=np.rint(baseline).astype(np.uint16),
        material_exponent=exponent,
        material_coefficient_n_per_m_power=coefficient,
    )


class HandleL1TenChannelCalibration:
    """Runtime interpolation of the empirical 10 N calibration table."""

    def __init__(self, artifact_path: Path | str):
        artifact = np.load(Path(artifact_path), allow_pickle=False)
        self.raw_lut = artifact["raw_lut"]
        self.grid_x = artifact["grid_x"].astype(float)
        self.grid_y = artifact["grid_y"].astype(float)
        self.force_axis_n = artifact["force_axis_n"].astype(float)
        self.baseline_raw = artifact["baseline_raw"].astype(np.float32)
        self.x_offset_mm = float(artifact["supplier_to_machine_x_offset_mm"])
        self.y_intercept_mm = float(
            artifact["supplier_to_machine_y_intercept_mm"]
        )
        self.y_sign = float(artifact["supplier_to_machine_y_sign"])
        self.raw_min = int(artifact["raw_min"])
        self.raw_max = int(artifact["raw_max"])
        self.material_exponent = float(artifact["material_exponent"])
        self.material_coefficient_n_per_m_power = float(
            artifact["material_coefficient_n_per_m_power"]
        )

    @property
    def maximum_force_n(self) -> float:
        return float(self.force_axis_n[-1])

    def supplier_to_machine(self, supplier_xy_mm) -> np.ndarray:
        point = np.asarray(supplier_xy_mm, dtype=float)
        if point.shape != (2,):
            raise ValueError("supplier_xy_mm must have shape [2]")
        return np.asarray(
            [
                point[0] + self.x_offset_mm,
                self.y_intercept_mm + self.y_sign * point[1],
            ]
        )

    @staticmethod
    def local_point_to_supplier(local_point_m) -> np.ndarray:
        point = np.asarray(local_point_m, dtype=float)
        if point.shape != (3,):
            raise ValueError("local_point_m must have shape [3]")
        # Supplied channel coordinates span local tactile-frame Y/Z.
        return np.asarray([12.5 + 1000.0 * point[1], 2.0 + 1000.0 * point[2]])

    @staticmethod
    def _bounds(axis: np.ndarray, value: float) -> tuple[int, int, float]:
        clipped = float(np.clip(value, axis[0], axis[-1]))
        upper = int(np.searchsorted(axis, clipped, side="right"))
        upper = min(max(upper, 1), len(axis) - 1)
        lower = upper - 1
        denominator = max(float(axis[upper] - axis[lower]), 1e-12)
        fraction = (clipped - float(axis[lower])) / denominator
        return lower, upper, float(fraction)

    def predict_float(self, force_n: float, supplier_xy_mm) -> np.ndarray:
        machine = self.supplier_to_machine(supplier_xy_mm)
        x0, x1, tx = self._bounds(self.grid_x, machine[0])
        y0, y1, ty = self._bounds(self.grid_y, machine[1])
        f0, f1, tf = self._bounds(
            self.force_axis_n, float(np.clip(force_n, 0.0, self.maximum_force_n))
        )
        corners = self.raw_lut[
            np.ix_([x0, x1], [y0, y1], [f0, f1], np.arange(10))
        ].astype(np.float32)
        along_force = corners[:, :, 0] * (1.0 - tf) + corners[:, :, 1] * tf
        along_y = along_force[:, 0] * (1.0 - ty) + along_force[:, 1] * ty
        value = along_y[0] * (1.0 - tx) + along_y[1] * tx
        return np.clip(value, self.raw_min, self.raw_max).astype(np.float32)

    def predict(self, force_n: float, supplier_xy_mm) -> np.ndarray:
        return np.rint(self.predict_float(force_n, supplier_xy_mm)).astype(np.uint16)

    def predict_float_many(
        self, force_n: np.ndarray, supplier_xy_mm: np.ndarray
    ) -> np.ndarray:
        """Vectorized trilinear interpolation for multiple local point loads."""
        forces = np.asarray(force_n, dtype=float).reshape(-1)
        points = np.asarray(supplier_xy_mm, dtype=float)
        if points.shape != (len(forces), 2):
            raise ValueError(
                "supplier_xy_mm must have shape [N, 2] matching force_n"
            )
        machine_x = points[:, 0] + self.x_offset_mm
        machine_y = self.y_intercept_mm + self.y_sign * points[:, 1]

        def bounds_many(axis: np.ndarray, values: np.ndarray):
            clipped = np.clip(values, axis[0], axis[-1])
            upper = np.searchsorted(axis, clipped, side="right")
            upper = np.clip(upper, 1, len(axis) - 1)
            lower = upper - 1
            fraction = (clipped - axis[lower]) / np.maximum(
                axis[upper] - axis[lower], 1e-12
            )
            return lower, upper, fraction.astype(np.float32)

        x0, x1, tx = bounds_many(self.grid_x, machine_x)
        y0, y1, ty = bounds_many(self.grid_y, machine_y)
        f0, f1, tf = bounds_many(
            self.force_axis_n,
            np.clip(forces, 0.0, self.maximum_force_n),
        )

        def at(x_index, y_index, force_index):
            return self.raw_lut[x_index, y_index, force_index].astype(
                np.float32
            )

        c000 = at(x0, y0, f0)
        c001 = at(x0, y0, f1)
        c010 = at(x0, y1, f0)
        c011 = at(x0, y1, f1)
        c100 = at(x1, y0, f0)
        c101 = at(x1, y0, f1)
        c110 = at(x1, y1, f0)
        c111 = at(x1, y1, f1)
        tf = tf[:, None]
        ty = ty[:, None]
        tx = tx[:, None]
        c00 = c000 * (1.0 - tf) + c001 * tf
        c01 = c010 * (1.0 - tf) + c011 * tf
        c10 = c100 * (1.0 - tf) + c101 * tf
        c11 = c110 * (1.0 - tf) + c111 * tf
        c0 = c00 * (1.0 - ty) + c01 * ty
        c1 = c10 * (1.0 - ty) + c11 * ty
        return np.clip(
            c0 * (1.0 - tx) + c1 * tx, self.raw_min, self.raw_max
        ).astype(np.float32)
