from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SegmentAdapterPrediction:
    delta_uv_m: tuple[float,float]
    raw_xy_m: tuple[float,float]
    clipped: bool
    radius_m: float

class WorldPolicySegmentAdapter:
    """Fail-closed projection from world-policy translation to bounded search delta.

    The policy output is already in patient/world Cartesian coordinates. We map
    x/y to lateral/axial and bound the vector before any caller can execute it.
    """
    schema_version = "adam_pulse_world_policy_segment_adapter_v1"
    def __init__(self, max_radius_m: float = 0.004, gain: float = 1.0):
        if not 0.0 < max_radius_m <= 0.004: raise ValueError("radius must be in (0,4mm]")
        self.max_radius_m=float(max_radius_m); self.gain=float(gain)
    def project(self, action7: np.ndarray) -> SegmentAdapterPrediction:
        a=np.asarray(action7,dtype=np.float64).reshape(-1)
        if a.size < 2 or not np.isfinite(a[:2]).all(): raise ValueError("invalid policy action")
        raw=np.asarray([a[0],a[1]],dtype=np.float64)*self.gain
        norm=float(np.linalg.norm(raw)); clipped=norm>self.max_radius_m
        out=raw if not clipped else raw*(self.max_radius_m/norm)
        return SegmentAdapterPrediction(tuple(out.tolist()),tuple(raw.tolist()),clipped,float(np.linalg.norm(out)))
