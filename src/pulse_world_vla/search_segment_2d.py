from __future__ import annotations

"""Small explicit 2-D segment baseline used before adapting World-VLA."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


OBSERVATION_FEATURE_FIELDS = (
    "tactile",
    "robot_state",
    "action",
    "task_phase",
)
PROXIMITY_FEATURE_FIELDS = ("proximity", "proximity_valid")
FEATURE_VERSION_LEGACY = "v1_resized_linear_order"
FEATURE_VERSION_TAXEL_CENTROID = "v2_explicit_taxel_centroid"
FEATURE_VERSION_TAXEL_PULSE_PROFILE = "v3_explicit_taxel_pulse_profile"
COMPACT_SPATIAL_FEATURE_CONTRACT = (
    "dynamic_taxel_centroid_normalized_pulse_profile_validity_40d_v1"
)
FEATURE_VERSIONS = (
    FEATURE_VERSION_LEGACY,
    FEATURE_VERSION_TAXEL_CENTROID,
    FEATURE_VERSION_TAXEL_PULSE_PROFILE,
)

# Supplier channel coordinates from pulse_mujoco.tactile_layouts, centered and
# normalized independently along the 25 mm longitudinal and 4 mm width axes.
TEN_CHANNEL_COORDS_NORMALIZED = (
    (-1.0, 0.0),
    (-0.6, -1.0),
    (-0.6, 1.0),
    (-0.2, -1.0),
    (-0.2, 1.0),
    (0.2, -1.0),
    (0.2, 1.0),
    (0.6, -1.0),
    (0.6, 1.0),
    (1.0, 0.0),
)


def observation_feature_fields(*, include_proximity: bool = False) -> tuple[str, ...]:
    """Return the explicit deployable allowlist for one CH5 model.

    Proximity is opt-in for backward-compatible ablations.  The custom-hand
    deployment contract defaults to tactile plus robot/action state because
    the physical Adam Pro currently has no proximity hardware.
    """

    if include_proximity:
        return (
            "tactile",
            *PROXIMITY_FEATURE_FIELDS,
            "robot_state",
            "action",
            "task_phase",
        )
    return OBSERVATION_FEATURE_FIELDS


@dataclass(frozen=True)
class SearchSegment2DOutput:
    delta_uv_normalized: torch.Tensor
    axis_direction_logits: torch.Tensor | None = None
    axis_magnitude_normalized: torch.Tensor | None = None


@dataclass(frozen=True)
class HierarchicalSearchSegment2DOutput:
    delta_uv_normalized: torch.Tensor
    tail_route: torch.Tensor
    coarse_delta_uv_normalized: torch.Tensor
    tail_delta_uv_normalized: torch.Tensor


OUTPUT_CONTRACT_DIRECT = "direct_bounded_uv"
OUTPUT_CONTRACT_DISCRETE_AXIS_MAGNITUDE = "discrete_axis_magnitude"
OUTPUT_CONTRACTS = (
    OUTPUT_CONTRACT_DIRECT,
    OUTPUT_CONTRACT_DISCRETE_AXIS_MAGNITUDE,
)


def bound_delta_uv_normalized(delta_uv: torch.Tensor) -> torch.Tensor:
    """Project a batched 2-D proposal onto the closed unit disk."""

    if delta_uv.shape[-1] != 2:
        raise ValueError("2-D segment proposals must end in dimension 2")
    norm = torch.linalg.vector_norm(delta_uv, dim=-1, keepdim=True)
    return delta_uv / norm.clamp_min(1.0)


def multifinger_feature_frame_dim(
    *,
    finger_count: int = 3,
    state_dim: int = 15,
    include_proximity: bool = False,
    feature_version: str = FEATURE_VERSION_LEGACY,
) -> int:
    # Per finger: mean, std, static within-pad gradient, adjacent change,
    # adjacent-change fraction, full-window pulse amplitude, pulse-energy
    # fraction and dynamic within-pad gradient. Then shared proximity,
    # state/action, tactile validity and task phase.
    if feature_version not in FEATURE_VERSIONS:
        raise ValueError(f"unsupported tactile feature version: {feature_version}")
    proximity_dim = 10 if include_proximity else 0
    topology_dim = 0
    if feature_version in {
        FEATURE_VERSION_TAXEL_CENTROID,
        FEATURE_VERSION_TAXEL_PULSE_PROFILE,
    }:
        topology_dim += 4 * int(finger_count)
    if feature_version == FEATURE_VERSION_TAXEL_PULSE_PROFILE:
        topology_dim += 10 * int(finger_count)
    return (
        8 * int(finger_count)
        + topology_dim
        + proximity_dim
        + int(state_dim)
        + 7
        + 1
        + 1
    )


def _weighted_taxel_center(
    values: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    weights = (values - values.amin(dim=-1, keepdim=True)).clamp_min(0.0)
    denominator = weights.sum(dim=-1, keepdim=True)
    center = torch.sum(weights.unsqueeze(-1) * coordinates, dim=-2)
    return torch.where(
        denominator > 1e-6,
        center / denominator.clamp_min(1e-6),
        torch.zeros_like(center),
    )


def multifinger_feature_frames(
    observation: Mapping[str, torch.Tensor],
    *,
    finger_count: int = 3,
    include_proximity: bool = False,
    feature_version: str = FEATURE_VERSION_LEGACY,
) -> torch.Tensor:
    """Build topology-aware CH5 features from the active tactile image.

    The active view originates from a ``[finger, 10]`` array. Splitting the
    resized image into ordered horizontal bands preserves index/middle/ring
    identity and therefore the observable patient-axis cue.
    """

    if feature_version not in FEATURE_VERSIONS:
        raise ValueError(f"unsupported tactile feature version: {feature_version}")
    fields = observation_feature_fields(include_proximity=include_proximity)
    missing = [field for field in fields if field not in observation]
    if missing:
        raise KeyError(f"missing 2-D segment observation fields: {missing}")
    tactile = observation["tactile"][:, :, 0].float()
    if tactile.ndim != 4:
        raise ValueError("tactile must have shape [batch,horizon,1,height,width]")
    batch, horizon, height, width = tactile.shape
    count = int(finger_count)
    if count < 2 or height < count:
        raise ValueError("active tactile height cannot represent requested fingers")
    edges = torch.linspace(0, height, count + 1, device=tactile.device)
    edges = edges.round().long()
    means, standard_deviations, gradients, changes, channel_profiles = (
        [], [], [], [], []
    )
    for index in range(count):
        band = tactile[..., edges[index] : edges[index + 1], :]
        means.append(band.mean(dim=(-1, -2)))
        standard_deviations.append(
            band.flatten(-2).std(dim=-1, unbiased=False)
        )
        gradients.append(
            band[..., width // 2 :].mean(dim=(-1, -2))
            - band[..., : width // 2].mean(dim=(-1, -2))
        )
        change = torch.zeros(batch, horizon, device=tactile.device)
        if horizon > 1:
            change[:, 1:] = (
                band[:, 1:] - band[:, :-1]
            ).abs().mean(dim=(-1, -2))
        changes.append(change)
        center_row = band[..., band.shape[-2] // 2, :]
        profile = F.interpolate(
            center_row.reshape(batch * horizon, 1, width),
            size=10,
            mode="linear",
            align_corners=True,
        ).reshape(batch, horizon, 10)
        channel_profiles.append(profile)
    finger_mean = torch.stack(means, dim=-1)
    finger_std = torch.stack(standard_deviations, dim=-1)
    finger_gradient = torch.stack(gradients, dim=-1)
    finger_change = torch.stack(changes, dim=-1)
    positive_energy = finger_change + finger_std
    energy_fraction = positive_energy / positive_energy.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    pulse_image = (
        torch.quantile(tactile, 0.95, dim=1)
        - torch.quantile(tactile, 0.05, dim=1)
    )
    pulse_amplitudes, pulse_gradients = [], []
    for index in range(count):
        band = pulse_image[..., edges[index] : edges[index + 1], :]
        pulse_amplitudes.append(band.mean(dim=(-1, -2)))
        pulse_gradients.append(
            band[..., width // 2 :].mean(dim=(-1, -2))
            - band[..., : width // 2].mean(dim=(-1, -2))
        )
    pulse_amplitude = torch.stack(pulse_amplitudes, dim=-1)
    pulse_energy_fraction = pulse_amplitude / pulse_amplitude.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    pulse_gradient = torch.stack(pulse_gradients, dim=-1)
    pulse_amplitude = pulse_amplitude[:, None].expand(-1, horizon, -1)
    pulse_energy_fraction = pulse_energy_fraction[:, None].expand(
        -1, horizon, -1
    )
    pulse_gradient = pulse_gradient[:, None].expand(-1, horizon, -1)
    phase = observation["task_phase"].float().unsqueeze(-1) / 13.0
    parts = [
        finger_mean,
        finger_std,
        finger_gradient,
        finger_change,
        energy_fraction,
        pulse_amplitude,
        pulse_energy_fraction,
        pulse_gradient,
    ]
    if feature_version in {
        FEATURE_VERSION_TAXEL_CENTROID,
        FEATURE_VERSION_TAXEL_PULSE_PROFILE,
    }:
        profiles = torch.stack(channel_profiles, dim=2)
        coordinates = torch.as_tensor(
            TEN_CHANNEL_COORDS_NORMALIZED,
            dtype=profiles.dtype,
            device=profiles.device,
        )
        static_center = _weighted_taxel_center(profiles, coordinates)
        pulse_channels = (
            torch.quantile(profiles, 0.95, dim=1)
            - torch.quantile(profiles, 0.05, dim=1)
        )
        dynamic_center = _weighted_taxel_center(
            pulse_channels, coordinates
        )[:, None].expand(-1, horizon, -1, -1)
        parts.extend(
            (
                static_center[..., 0],
                static_center[..., 1],
                dynamic_center[..., 0],
                dynamic_center[..., 1],
            )
        )
        if feature_version == FEATURE_VERSION_TAXEL_PULSE_PROFILE:
            pulse_profile = pulse_channels.reshape(batch, count * 10)
            parts.append(
                pulse_profile[:, None].expand(-1, horizon, -1)
            )
    if include_proximity:
        parts.extend(
            (
                observation["proximity"].float(),
                observation["proximity_valid"].float(),
            )
        )
    parts.extend(
        (
            observation["robot_state"].float(),
            observation["action"].float(),
            observation.get(
                "tactile_valid",
                torch.ones(batch, horizon, device=tactile.device),
            ).float().unsqueeze(-1),
            phase,
        )
    )
    features = torch.cat(parts, dim=-1)
    expected = multifinger_feature_frame_dim(
        finger_count=count,
        state_dim=observation["robot_state"].shape[-1],
        include_proximity=include_proximity,
        feature_version=feature_version,
    )
    if features.shape != (batch, horizon, expected):
        raise AssertionError(
            f"unexpected CH5 feature shape {tuple(features.shape)} != "
            f"{(batch, horizon, expected)}"
        )
    return features


class ACTSearchSegment2DPolicy(nn.Module):
    """ACT-style single-query policy for one bounded patient-tangent move."""

    def __init__(
        self,
        *,
        feature_dim: int,
        model_dim: int = 96,
        layers: int = 2,
        heads: int = 4,
        maximum_horizon: int = 4,
        include_proximity: bool = False,
        feature_version: str = FEATURE_VERSION_LEGACY,
        output_contract: str = OUTPUT_CONTRACT_DIRECT,
        axis_adapter_dim: int = 0,
        axis_adapter_input: str = "summary",
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.maximum_horizon = int(maximum_horizon)
        self.include_proximity = bool(include_proximity)
        if feature_version not in FEATURE_VERSIONS:
            raise ValueError(f"unsupported tactile feature version: {feature_version}")
        self.feature_version = str(feature_version)
        if output_contract not in OUTPUT_CONTRACTS:
            raise ValueError(f"unsupported 2-D output contract: {output_contract}")
        self.output_contract = str(output_contract)
        self.frame = nn.Sequential(
            nn.Linear(self.feature_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.SiLU(),
        )
        self.position = nn.Parameter(
            torch.zeros(1, self.maximum_horizon, model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.query = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.cross_attention = nn.MultiheadAttention(
            model_dim, heads, batch_first=True, dropout=0.0
        )
        head_layers: list[nn.Module] = [
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
        ]
        if self.output_contract == OUTPUT_CONTRACT_DIRECT:
            head_layers.append(nn.Linear(model_dim, 2))
        self.head = nn.Sequential(*head_layers)
        if self.output_contract == OUTPUT_CONTRACT_DISCRETE_AXIS_MAGNITUDE:
            self.axis_direction_head = nn.Linear(model_dim, 2 * 3)
            self.axis_magnitude_head = nn.Linear(model_dim, 2)
            self.axis_adapter_dim = int(axis_adapter_dim)
            if self.axis_adapter_dim < 0:
                raise ValueError("axis_adapter_dim must be non-negative")
            if axis_adapter_input not in {"summary", "full_window"}:
                raise ValueError(
                    "axis_adapter_input must be 'summary' or 'full_window'"
                )
            self.axis_adapter_input = str(axis_adapter_input)
            if self.axis_adapter_dim:
                adapter_input_dim = model_dim + (
                    3 * self.feature_dim
                    if self.axis_adapter_input == "summary"
                    else self.maximum_horizon * self.feature_dim
                )
                self.axis_adapter = nn.Sequential(
                    nn.LayerNorm(adapter_input_dim),
                    nn.Linear(adapter_input_dim, self.axis_adapter_dim),
                    nn.SiLU(),
                    nn.Linear(self.axis_adapter_dim, 8),
                )
                nn.init.zeros_(self.axis_adapter[-1].weight)
                nn.init.zeros_(self.axis_adapter[-1].bias)

    def forward(self, features: torch.Tensor) -> SearchSegment2DOutput:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                "features must have shape [batch,horizon,feature_dim]"
            )
        if features.shape[1] > self.maximum_horizon:
            raise ValueError("feature history exceeds ACT maximum horizon")
        memory = self.encoder(
            self.frame(features) + self.position[:, : features.shape[1]]
        )
        query = self.query.expand(features.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            query, memory, memory, need_weights=False
        )
        hidden = self.head(attended[:, 0])
        if self.output_contract == OUTPUT_CONTRACT_DIRECT:
            return SearchSegment2DOutput(
                delta_uv_normalized=bound_delta_uv_normalized(
                    torch.tanh(hidden)
                )
            )
        direction_logits = self.axis_direction_head(hidden).reshape(-1, 2, 3)
        magnitude_raw = self.axis_magnitude_head(hidden)
        if getattr(self, "axis_adapter_dim", 0):
            # The inherited temporal encoder can collapse late recovery states
            # onto old large-axis states.  Retain a narrow observable bypass
            # using only statistics of the same policy-visible feature window.
            if self.axis_adapter_input == "summary":
                adapter_features = torch.cat(
                    (
                        features.mean(dim=1),
                        features.std(dim=1, unbiased=False),
                        features[:, -1],
                    ),
                    dim=-1,
                )
            else:
                if features.shape[1] != self.maximum_horizon:
                    raise ValueError(
                        "full_window axis adapter requires maximum history"
                    )
                adapter_features = features.flatten(1)
            adapter_input = torch.cat((hidden, adapter_features), dim=-1)
            correction = self.axis_adapter(adapter_input)
            direction_logits = direction_logits + correction[:, :6].reshape(
                -1, 2, 3
            )
            magnitude_raw = magnitude_raw + correction[:, 6:]
        # The magnitude head is sign-invariant by contract.  ``abs(tanh)``
        # keeps it bounded while allowing an old signed direct projection to
        # seed this head without changing its pre-migration magnitude.
        magnitude = torch.abs(torch.tanh(magnitude_raw))
        if self.training:
            probabilities = torch.softmax(direction_logits, dim=-1)
            signed_direction = probabilities[..., 2] - probabilities[..., 0]
        else:
            direction_class = direction_logits.argmax(dim=-1)
            signed_direction = (direction_class - 1).to(magnitude.dtype)
        return SearchSegment2DOutput(
            delta_uv_normalized=bound_delta_uv_normalized(
                signed_direction * magnitude
            ),
            axis_direction_logits=direction_logits,
            axis_magnitude_normalized=magnitude,
        )


class HierarchicalSearchSegment2DPolicy(nn.Module):
    """Route only observable late recovery states to a sub-mm specialist.

    The coarse policy remains byte-for-byte frozen.  Routing uses the public
    search-point index and the coarse policy's own bounded prediction; neither
    pulse truth nor a teacher label is available online.  A small calibrated
    magnitude bias applies only after the discrete tail policy has selected a
    direction.
    """

    def __init__(
        self,
        coarse: nn.Module,
        tail: nn.Module,
        *,
        unconditional_tail_point_index: int,
        conditional_tail_point_index: int,
        conditional_coarse_norm_max_normalized: float,
        tail_magnitude_bias_normalized: float,
    ) -> None:
        super().__init__()
        self.coarse = coarse
        self.tail = tail
        self.unconditional_tail_point_index = int(
            unconditional_tail_point_index
        )
        self.conditional_tail_point_index = int(conditional_tail_point_index)
        self.conditional_coarse_norm_max_normalized = float(
            conditional_coarse_norm_max_normalized
        )
        self.tail_magnitude_bias_normalized = float(
            tail_magnitude_bias_normalized
        )
        if self.unconditional_tail_point_index < 0:
            raise ValueError("unconditional tail point index must be non-negative")
        if not 0 <= self.conditional_tail_point_index <= (
            self.unconditional_tail_point_index
        ):
            raise ValueError("conditional tail point index is out of range")
        if not 0.0 < self.conditional_coarse_norm_max_normalized <= 1.0:
            raise ValueError("conditional coarse norm threshold must be in (0, 1]")
        if not 0.0 <= self.tail_magnitude_bias_normalized <= 1.0:
            raise ValueError("tail magnitude bias must be in [0, 1]")

    def forward(
        self, features: torch.Tensor, point_index: torch.Tensor
    ) -> HierarchicalSearchSegment2DOutput:
        points = torch.as_tensor(point_index, device=features.device).long()
        if points.ndim != 1 or points.shape[0] != features.shape[0]:
            raise ValueError("point_index must have shape [batch]")
        coarse = self.coarse(features).delta_uv_normalized
        tail = self.tail(features).delta_uv_normalized
        coarse_norm = torch.linalg.vector_norm(coarse, dim=-1)
        route = (points >= self.unconditional_tail_point_index) | (
            (points >= self.conditional_tail_point_index)
            & (coarse_norm <= self.conditional_coarse_norm_max_normalized)
        )
        tail_norm = torch.linalg.vector_norm(tail, dim=-1, keepdim=True)
        biased_norm = (tail_norm + self.tail_magnitude_bias_normalized).clamp_max(
            1.0
        )
        calibrated_tail = torch.where(
            tail_norm > 1e-12,
            tail * (biased_norm / tail_norm.clamp_min(1e-12)),
            tail,
        )
        selected = torch.where(route[:, None], calibrated_tail, coarse)
        return HierarchicalSearchSegment2DOutput(
            delta_uv_normalized=bound_delta_uv_normalized(selected),
            tail_route=route,
            coarse_delta_uv_normalized=coarse,
            tail_delta_uv_normalized=calibrated_tail,
        )


def ridge_feature_vectors(
    observation: Mapping[str, torch.Tensor],
    *,
    include_proximity: bool = False,
) -> torch.Tensor:
    """Return a compact physiology-focused vector for the ridge baseline.

    The first 15 features summarize per-frame tactile pressure/change over the
    whole window. The final nine are the full-window pulse amplitude, energy
    fraction and within-pad dynamic gradient. Robot pose/action are omitted on
    purpose: they cannot reveal an unknown pulse center and encouraged
    nuisance memorization in the 38-episode pilot.
    """

    features = multifinger_feature_frames(
        observation, include_proximity=include_proximity
    )
    if include_proximity:
        raise ValueError("the no-proximity ridge baseline forbids proximity input")
    return torch.cat((features[:, :, :15].mean(dim=1), features[:, -1, 15:24]), dim=-1)


def compact_spatial_feature_vectors(
    observation: Mapping[str, torch.Tensor],
    *,
    finger_count: int = 3,
) -> torch.Tensor:
    """Return nuisance-resistant tactile topology features for late search.

    The vector contains only policy-visible tactile evidence: six dynamic
    taxel-centroid coordinates, 30 per-finger pulse-profile values normalized
    within each finger, three pulse-validity flags and one completed-window
    tactile-validity ratio.  Absolute pulse gain, robot state, action history,
    proximity, pulse truth and teacher labels are intentionally excluded.
    """

    if int(finger_count) != 3:
        raise ValueError("the compact spatial contract currently requires three fingers")
    features = multifinger_feature_frames(
        observation,
        finger_count=finger_count,
        include_proximity=False,
        feature_version=FEATURE_VERSION_TAXEL_PULSE_PROFILE,
    )
    # Full-window dynamic topology is repeated along the horizon by the v3
    # extractor, so reading the final frame preserves the completed window.
    final = features[:, -1]
    dynamic_centers = final[:, 30:36]
    pulse_profile = final[:, 36:66].reshape(-1, finger_count, 10)
    profile_energy = pulse_profile.sum(dim=-1, keepdim=True)
    normalized_profile = torch.where(
        profile_energy > 1e-6,
        pulse_profile / profile_energy.clamp_min(1e-6),
        torch.zeros_like(pulse_profile),
    )
    pulse_valid = (profile_energy[..., 0] > 1e-6).float()
    tactile_valid = observation.get("tactile_valid")
    if tactile_valid is None:
        window_validity = torch.ones(
            len(features), 1, dtype=features.dtype, device=features.device
        )
    else:
        window_validity = tactile_valid.float().mean(dim=1, keepdim=True)
    result = torch.cat(
        (
            dynamic_centers,
            normalized_profile.reshape(len(features), -1),
            pulse_valid,
            window_validity,
        ),
        dim=-1,
    )
    if result.shape != (len(features), 40):
        raise AssertionError(f"unexpected compact spatial shape: {tuple(result.shape)}")
    return result


class RidgeSearchSegment2DPolicy(nn.Module):
    """Closed-form bounded linear baseline with frozen normalization."""

    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        scale = torch.as_tensor(feature_scale, dtype=torch.float32)
        matrix = torch.as_tensor(weights, dtype=torch.float32)
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("ridge feature mean/scale must be matching vectors")
        if matrix.shape != (len(mean), 2):
            raise ValueError("ridge weights must have shape [feature_dim,2]")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        self.register_buffer("weights", matrix)

    def forward(self, features: torch.Tensor) -> SearchSegment2DOutput:
        if features.ndim != 2 or features.shape[1] != len(self.feature_mean):
            raise ValueError("ridge features have an incompatible shape")
        normalized = (features - self.feature_mean) / self.feature_scale
        return SearchSegment2DOutput(
            delta_uv_normalized=bound_delta_uv_normalized(
                normalized @ self.weights
            )
        )


class CompactTactileSpatialPolicy(nn.Module):
    """Frozen linear axis classifier for strict-center late search states."""

    def __init__(
        self,
        *,
        feature_indices: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        axis_class_weights: torch.Tensor,
        step_radius_normalized: float,
    ) -> None:
        super().__init__()
        indices = torch.as_tensor(feature_indices, dtype=torch.long)
        mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        scale = torch.as_tensor(feature_scale, dtype=torch.float32)
        weights = torch.as_tensor(axis_class_weights, dtype=torch.float32)
        if indices.ndim != 1 or mean.shape != indices.shape or scale.shape != indices.shape:
            raise ValueError("compact spatial feature metadata must be matching vectors")
        if weights.shape != (2, len(indices) + 1, 3):
            raise ValueError("compact spatial weights must have shape [2,D+1,3]")
        radius = float(step_radius_normalized)
        if not 0.0 < radius <= 1.0:
            raise ValueError("compact spatial step radius must be in (0,1]")
        self.register_buffer("feature_indices", indices)
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale.clamp_min(1e-6))
        self.register_buffer("axis_class_weights", weights)
        self.step_radius_normalized = radius

    def forward(self, features: torch.Tensor) -> SearchSegment2DOutput:
        if features.ndim != 2 or features.shape[1] != 40:
            raise ValueError("compact spatial features must have shape [batch,40]")
        selected = features.index_select(1, self.feature_indices)
        normalized = (selected - self.feature_mean) / self.feature_scale
        design = torch.cat(
            (normalized, torch.ones(len(features), 1, device=features.device)), dim=1
        )
        scores = torch.einsum("bd,adc->bac", design, self.axis_class_weights)
        direction = scores.argmax(dim=-1).to(features.dtype) - 1.0
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        delta = torch.where(
            norm > 0.0,
            direction * (self.step_radius_normalized / norm.clamp_min(1.0)),
            torch.zeros_like(direction),
        )
        # Last four contract values are three pulse-valid flags and completed
        # tactile-window validity. A missing tactile modality cannot request a
        # motion even if classifier biases are non-zero.
        valid = torch.all(features[:, 36:39] > 0.5, dim=1) & (
            features[:, 39] > 0.999
        )
        delta = torch.where(valid[:, None], delta, torch.zeros_like(delta))
        return SearchSegment2DOutput(
            delta_uv_normalized=bound_delta_uv_normalized(delta)
        )


def flatten_segment_observation(
    observation: Mapping[str, torch.Tensor],
    *,
    include_proximity: bool = False,
) -> torch.Tensor:
    """Flatten only the frozen deployable allowlist fields used by the MLP."""

    fields = observation_feature_fields(include_proximity=include_proximity)
    missing = [field for field in fields if field not in observation]
    if missing:
        raise KeyError(f"missing 2-D segment observation fields: {missing}")
    batch_size = int(observation["tactile"].shape[0])
    values = []
    for field in fields:
        tensor = observation[field]
        if field in {"proximity_valid"}:
            tensor = tensor.float()
        elif field == "task_phase":
            tensor = tensor.float() / 13.0
        else:
            tensor = tensor.float()
        values.append(tensor.reshape(batch_size, -1))
    return torch.cat(values, dim=-1)


class SearchSegment2DMLP(nn.Module):
    """Predict stop plus a bounded patient-tangent ``[du, dv]`` segment."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.stop_head = nn.Linear(self.hidden_dim, 1)
        self.delta_uv_head = nn.Linear(self.hidden_dim, 2)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"features must have shape [batch,{self.input_dim}], got "
                f"{tuple(features.shape)}"
            )
        hidden = self.backbone(features)
        return {
            "stop_logit": self.stop_head(hidden).squeeze(-1),
            "delta_uv_normalized": bound_delta_uv_normalized(
                torch.tanh(self.delta_uv_head(hidden))
            ),
        }


def segment_2d_loss(
    prediction: Mapping[str, torch.Tensor],
    *,
    stop: torch.Tensor,
    delta_uv_normalized: torch.Tensor,
) -> dict[str, torch.Tensor]:
    stop_loss = F.binary_cross_entropy_with_logits(
        prediction["stop_logit"], stop.float()
    )
    delta_loss = F.mse_loss(
        prediction["delta_uv_normalized"], delta_uv_normalized.float()
    )
    return {
        "loss": stop_loss + delta_loss,
        "stop_loss": stop_loss,
        "delta_uv_loss": delta_loss,
    }
