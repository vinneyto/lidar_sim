from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Gaussians:
    centers: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor

    def __post_init__(self) -> None:
        count = len(self.centers)
        if self.centers.shape != (count, 3):
            raise ValueError("centers must have shape (N, 3)")
        if self.scales.shape != (count, 3):
            raise ValueError("scales must have shape (N, 3)")
        if self.rotations.shape != (count, 4):
            raise ValueError("rotations must have shape (N, 4)")
        if any(tensor.dtype != torch.float32 for tensor in self.tensors()):
            raise ValueError("Gaussian tensors must use torch.float32")

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.centers, self.scales, self.rotations

    def to(self, device: torch.device | str) -> Gaussians:
        return Gaussians(*(tensor.to(device).contiguous() for tensor in self.tensors()))


@dataclass(frozen=True, slots=True)
class ScanRequest:
    position: torch.Tensor
    quaternion: torch.Tensor
    width: int
    height: int
    horizontal_fov_deg: float
    vertical_fov_deg: float
    max_distance: float
