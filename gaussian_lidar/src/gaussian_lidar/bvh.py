from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .model import Gaussians


@dataclass(frozen=True, slots=True)
class FlatBvh:
    bounds_min: torch.Tensor
    bounds_max: torch.Tensor
    metadata: torch.Tensor
    primitive_indices: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.bounds_min, self.bounds_max, self.metadata, self.primitive_indices

    def to(self, device: torch.device | str) -> FlatBvh:
        return FlatBvh(*(tensor.to(device).contiguous() for tensor in self.tensors()))


def build_bvh(gaussians: Gaussians, sigma_extent: float = 3.0, leaf_size: int = 8) -> FlatBvh:
    """Build a median-split BVH using PyTorch CPU operations."""
    if leaf_size < 1:
        raise ValueError("leaf_size must be positive")
    if not math.isfinite(sigma_extent) or sigma_extent <= 0:
        raise ValueError("sigma_extent must be a positive finite number")
    if gaussians.centers.device.type != "cpu":
        raise ValueError("BVH construction expects CPU tensors")
    count = len(gaussians.centers)
    if count == 0:
        raise ValueError("cannot build a BVH for an empty PLY")
    radii = gaussians.scales.amax(dim=1, keepdim=True) * sigma_extent
    primitive_min = gaussians.centers - radii
    primitive_max = gaussians.centers + radii
    node_min: list[torch.Tensor] = []
    node_max: list[torch.Tensor] = []
    metadata: list[list[int]] = []
    ordered: list[int] = []

    def add_node(indices: torch.Tensor) -> int:
        node = len(metadata)
        node_min.append(primitive_min[indices].amin(dim=0))
        node_max.append(primitive_max[indices].amax(dim=0))
        metadata.append([0, 0, 0, 0])
        if len(indices) <= leaf_size:
            start = len(ordered)
            ordered.extend(indices.tolist())
            metadata[node] = [0, 0, start, len(indices)]
            return node
        selected_centers = gaussians.centers[indices]
        extent = selected_centers.amax(dim=0) - selected_centers.amin(dim=0)
        axis = int(torch.argmax(extent))
        order = torch.argsort(gaussians.centers[indices, axis], stable=True)
        sorted_indices = indices[order]
        middle = len(sorted_indices) // 2
        left = add_node(sorted_indices[:middle])
        right = add_node(sorted_indices[middle:])
        metadata[node] = [left, right, 0, 0]
        return node

    add_node(torch.arange(count, dtype=torch.int64))
    return FlatBvh(
        torch.stack(node_min).to(torch.float32),
        torch.stack(node_max).to(torch.float32),
        torch.tensor(metadata, dtype=torch.int32),
        torch.tensor(ordered, dtype=torch.int32),
    )
