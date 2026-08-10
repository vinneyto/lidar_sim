from __future__ import annotations

import importlib.resources
import platform
from dataclasses import dataclass, field
from typing import Any

import torch

from .bvh import FlatBvh
from .model import Gaussians


@dataclass(slots=True)
class MetalTracer:
    """Run the packaged Metal kernel directly against PyTorch MPS tensors."""

    gaussians: Gaussians
    bvh: FlatBvh
    sigma_extent: float = 3.0
    device: torch.device = field(init=False)
    _kernel: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if platform.system() != "Darwin" or not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS with Metal support is required")
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("this PyTorch build does not provide torch.mps.compile_shader")
        self.device = torch.device("mps")
        self.gaussians = self.gaussians.to(self.device)
        self.bvh = self.bvh.to(self.device)
        source = (
            importlib.resources.files("gaussian_lidar.kernels").joinpath("lidar.metal").read_text()
        )
        library = torch.mps.compile_shader(source)
        self._kernel = library.trace_rays

    def trace(
        self, origins: torch.Tensor, directions: torch.Tensor, max_distance: float
    ) -> torch.Tensor:
        if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins and directions must both have shape (N, 3)")
        if origins.device != self.device or directions.device != self.device:
            raise ValueError("ray tensors must reside on the tracer MPS device")
        count = len(origins)
        hits = torch.empty((count, 4), dtype=torch.float32, device=self.device)
        constants = (
            torch.tensor([count], dtype=torch.int32, device=self.device),
            torch.tensor([max_distance], dtype=torch.float32, device=self.device),
            torch.tensor([self.sigma_extent], dtype=torch.float32, device=self.device),
        )
        # compile_shader exposes each Metal entry point as a callable PyTorch custom op.
        self._kernel(
            origins.contiguous(),
            directions.contiguous(),
            *self.bvh.tensors(),
            *self.gaussians.tensors(),
            hits,
            *constants,
            threads=(count, 1, 1),
        )
        torch.mps.synchronize()
        return hits
