"""CUDA-only preflight for learned-policy production entry points.

Unit tests may deliberately instantiate a model on CPU, but collection,
training, and formal evaluation must never silently downgrade when a session
temporarily loses its NVIDIA device mapping.
"""

from __future__ import annotations

import torch


def require_cuda(*, purpose: str) -> torch.device:
    """Return the CUDA device or fail with an actionable session-level error."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for "
            f"{purpose}, but PyTorch cannot see an NVIDIA device in this session. "
            "Do not rerun this workload on CPU. Reconnect or wait for the GPU "
            "device mapping, then verify with: "
            "python -c 'import torch; print(torch.cuda.is_available(), "
            "torch.cuda.get_device_name(0))'"
        )
    return torch.device("cuda")
