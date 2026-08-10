from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import gaussforge
import torch

from .model import Gaussians


def _value(cloud: Any, *names: str) -> Any | None:
    """Read a GaussForge field from either its mapping or object representation."""
    for name in names:
        if isinstance(cloud, Mapping) and name in cloud:
            return cloud[name]
        value = getattr(cloud, name, None)
        if value is not None:
            return value
    return None


def _tensor(value: Any, *, columns: int, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if result.ndim != 2 or result.shape[1] != columns:
        raise ValueError(f"GaussForge {name} must have shape (N, {columns})")
    if not torch.isfinite(result).all():
        raise ValueError(f"GaussForge {name} contains non-finite values")
    return result.contiguous()


def load_gaussians(payload: bytes, default_scale: float = 0.01) -> Gaussians:
    """Parse an in-memory PLY through GaussForge and normalize its tensors."""
    if not math.isfinite(default_scale) or default_scale <= 0:
        raise ValueError("default_scale must be a positive finite number")
    if not payload:
        raise ValueError("PLY payload is empty")

    cloud = gaussforge.load_ply(payload)
    positions = _value(cloud, "positions", "means", "xyz", "centers")
    if positions is None:
        raise ValueError("GaussForge result has no positions/means/xyz field")
    centers = _tensor(positions, columns=3, name="positions")
    count = len(centers)

    scales_value = _value(cloud, "scales", "scaling")
    log_scales_value = _value(cloud, "log_scales", "log_scaling")
    if scales_value is not None:
        scales = _tensor(scales_value, columns=3, name="scales")
    elif log_scales_value is not None:
        scales = torch.exp(
            _tensor(log_scales_value, columns=3, name="log_scales").clamp(-20.0, 20.0)
        )
    else:
        scales = torch.full((count, 3), default_scale, dtype=torch.float32)

    rotations_value = _value(cloud, "rotations", "quaternions", "quats")
    if rotations_value is None:
        rotations = torch.zeros((count, 4), dtype=torch.float32)
        rotations[:, 0] = 1.0
    else:
        rotations = _tensor(rotations_value, columns=4, name="rotations")
        rotations = torch.nn.functional.normalize(rotations, dim=1, eps=1e-12)

    if len(scales) != count or len(rotations) != count:
        raise ValueError("GaussForge Gaussian property counts do not match")
    if torch.any(scales <= 0):
        raise ValueError("GaussForge scales must be positive")
    return Gaussians(centers, scales.contiguous(), rotations.contiguous())
