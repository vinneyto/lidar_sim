from __future__ import annotations

from dataclasses import dataclass, replace
import torch


@dataclass(frozen=True)
class GaussianCloud:
    """Gaussians in normalized form; quaternions use (w, x, y, z)."""

    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor | None = None

    def __post_init__(self) -> None:
        n = self.means.shape[0]
        if self.means.shape != (n, 3) or self.scales.shape != (n, 3):
            raise ValueError("means and scales must have shape [N, 3]")
        if self.rotations.shape != (n, 4) or self.opacities.shape != (n,):
            raise ValueError("rotations/opacities must have shapes [N, 4]/[N]")
        tensors = (self.means, self.scales, self.rotations, self.opacities)
        if len({x.device for x in tensors}) != 1:
            raise ValueError("all tensors must share a device")
        if bool((self.scales <= 0).any()) or bool(((self.opacities < 0) | (self.opacities > 1)).any()):
            raise ValueError("scales must be positive and opacities must be in [0, 1]")

    def to(self, device: torch.device | str) -> "GaussianCloud":
        values = {k: (v.to(device) if v is not None else None) for k, v in self.__dict__.items()}
        return replace(self, **values)

    def normalized(self) -> "GaussianCloud":
        return replace(self, rotations=torch.nn.functional.normalize(self.rotations, dim=-1))
