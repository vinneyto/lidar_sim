from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from pathlib import Path

import torch
from course_3dgs import GaussianData

from .gaussian_cloud import GaussianCloud


_SH_C0 = 0.28209479177387814
_MAX_SH_LEVELS = 4


@dataclass(frozen=True)
class GaussianPlyData:
    """Canonical 3DGS PLY data adapted for Rerun and the course Metal renderer.

    PLY files store ``rot_0..3`` as scalar-first ``(w, x, y, z)``. This class
    converts them once at the file boundary and keeps ``rotations_xyzw`` in the
    ``(x, y, z, w)`` order used by Rerun and by the course renderer geometry.
    The legacy LiDAR-facing :class:`GaussianCloud` remains scalar-first and is
    produced explicitly by :meth:`to_gaussian_cloud`.
    """

    positions: torch.Tensor
    scale_raw: torch.Tensor
    rotations_xyzw: torch.Tensor
    opacity_raw: torch.Tensor
    f_dc: torch.Tensor | None
    f_rest: torch.Tensor | None

    def __post_init__(self) -> None:
        n = self.positions.shape[0]
        if self.positions.shape != (n, 3) or self.scale_raw.shape != (n, 3):
            raise ValueError("positions and scale_raw must have shape [N, 3]")
        if self.rotations_xyzw.shape != (n, 4):
            raise ValueError("rotations_xyzw must have shape [N, 4]")
        if self.opacity_raw.shape != (n,):
            raise ValueError("opacity_raw must have shape [N]")
        if self.f_dc is not None and self.f_dc.shape != (n, 3):
            raise ValueError("f_dc must have shape [N, 3]")
        if self.f_rest is not None:
            if self.f_dc is None:
                raise ValueError("f_rest requires f_dc")
            if self.f_rest.ndim != 2 or self.f_rest.shape[0] != n:
                raise ValueError("f_rest must have shape [N, 3 * (K - 1)]")
            if self.f_rest.shape[1] % 3 != 0:
                raise ValueError("f_rest width must be divisible by 3")

        tensors = [
            self.positions,
            self.scale_raw,
            self.rotations_xyzw,
            self.opacity_raw,
        ]
        if self.f_dc is not None:
            tensors.append(self.f_dc)
        if self.f_rest is not None:
            tensors.append(self.f_rest)
        if any(tensor.dtype != torch.float32 for tensor in tensors):
            raise TypeError("PLY tensors must use float32")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all PLY tensors must live on the same device")

        _ = self.sh_levels

    @classmethod
    def from_ply(cls, path: str | Path) -> "GaussianPlyData":
        """Load a canonical Inria 3DGS PLY without truncating SH coefficients."""
        from plyfile import PlyData

        vertex = PlyData.read(str(path))["vertex"].data
        names = set(vertex.dtype.names or ())
        required = {
            "x",
            "y",
            "z",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
            "opacity",
        }
        missing = required - names
        if missing:
            raise ValueError(
                f"not a canonical 3DGS PLY; missing properties: {sorted(missing)}"
            )

        def tensor(columns: list[str]) -> torch.Tensor:
            return torch.stack(
                [torch.from_numpy(vertex[column].copy()) for column in columns],
                dim=-1,
            ).float()

        dc_names = [f"f_dc_{index}" for index in range(3)]
        present_dc = [name in names for name in dc_names]
        if any(present_dc) and not all(present_dc):
            raise ValueError("f_dc must contain exactly f_dc_0, f_dc_1, and f_dc_2")
        f_dc = tensor(dc_names) if all(present_dc) else None

        rest_indices = sorted(
            int(name.removeprefix("f_rest_"))
            for name in names
            if name.startswith("f_rest_") and name.removeprefix("f_rest_").isdigit()
        )
        if rest_indices and rest_indices != list(range(rest_indices[-1] + 1)):
            raise ValueError("f_rest_* properties must be contiguous from f_rest_0")
        rest_names = [f"f_rest_{index}" for index in rest_indices]
        f_rest = tensor(rest_names) if rest_names else None

        rotations_wxyz = tensor(["rot_0", "rot_1", "rot_2", "rot_3"])
        rotations_xyzw = rotations_wxyz[:, [1, 2, 3, 0]]
        rotations_xyzw = torch.nn.functional.normalize(rotations_xyzw, dim=-1)

        return cls(
            positions=tensor(["x", "y", "z"]),
            scale_raw=tensor(["scale_0", "scale_1", "scale_2"]),
            rotations_xyzw=rotations_xyzw.contiguous(),
            opacity_raw=torch.from_numpy(vertex["opacity"].copy()).float(),
            f_dc=f_dc,
            f_rest=f_rest,
        )

    @property
    def sh_levels(self) -> int:
        """Return all complete SH levels present in the PLY (1..4)."""
        if self.f_dc is None:
            return 1
        rest_per_channel = 0 if self.f_rest is None else self.f_rest.shape[1] // 3
        coefficient_count = rest_per_channel + 1
        levels = isqrt(coefficient_count)
        if levels * levels != coefficient_count:
            raise ValueError("f_rest does not contain a complete number of SH levels")
        if levels > _MAX_SH_LEVELS:
            raise ValueError(
                f"PLY contains {levels} SH levels; the Metal renderer supports "
                f"at most {_MAX_SH_LEVELS}"
            )
        return levels

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.scale_raw).clamp_min(1e-6)

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_raw)

    @property
    def colors(self) -> torch.Tensor | None:
        """Return the canonical view-independent DC color used by the PLY."""
        if self.f_dc is None:
            return None
        return (0.5 + _SH_C0 * self.f_dc).clamp(0, 1)

    def to_gaussian_cloud(self) -> GaussianCloud:
        """Adapt to the LiDAR-facing scalar-first ``GaussianCloud`` contract."""
        rotations_wxyz = self.rotations_xyzw[:, [3, 0, 1, 2]]
        return GaussianCloud(
            means=self.positions,
            scales=self.scales,
            rotations=rotations_wxyz.contiguous(),
            opacities=self.opacities,
            colors=self.colors,
        )

    def to_renderer_data(
        self,
        device: torch.device | str = "cpu",
    ) -> GaussianData:
        """Pass canonical PLY SH coefficients to the renderer unchanged."""
        if self.f_dc is None:
            raise ValueError("camera rendering requires f_dc_0, f_dc_1, and f_dc_2")

        device = torch.device(device)
        positions = self.positions.to(device=device, dtype=torch.float32)
        scale_raw = self.scale_raw.to(device=device, dtype=torch.float32)
        rotations_xyzw = self.rotations_xyzw.to(device=device, dtype=torch.float32)
        opacity_raw = self.opacity_raw.to(device=device, dtype=torch.float32)
        f_dc = self.f_dc.to(device=device, dtype=torch.float32)
        f_rest = (
            self.f_rest.to(device=device, dtype=torch.float32)
            if self.f_rest is not None
            else None
        )

        sigma = _build_covariance_xyzw(scale_raw, rotations_xyzw)
        return GaussianData.from_flat_tensors(
            positions=positions,
            f_dc=f_dc,
            f_rest=f_rest,
            opacity_raw=opacity_raw,
            sigma=sigma,
            sh_levels=self.sh_levels,
        )


def _build_covariance_xyzw(
    scale_raw: torch.Tensor,
    rotations_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Build covariance directly from the xyzw convention used by course_3dgs."""
    scale = torch.exp(scale_raw).clamp_min(1e-6)
    q = rotations_xyzw / (
        torch.linalg.norm(rotations_xyzw, dim=-1, keepdim=True) + 1e-9
    )
    x, y, z, w = q.unbind(dim=-1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    scale_matrix = torch.diag_embed(scale)
    return rotation @ scale_matrix @ scale_matrix @ rotation.transpose(1, 2)


def load_gaussian_ply(path: str | Path) -> GaussianCloud:
    """Load canonical 3DGS PLY into the legacy LiDAR-facing data structure."""
    return GaussianPlyData.from_ply(path).to_gaussian_cloud()
