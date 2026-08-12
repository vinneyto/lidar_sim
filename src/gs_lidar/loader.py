from pathlib import Path
from typing import Any

import torch

from .gaussian_cloud import GaussianCloud

_SH_C0 = 0.28209479177387814


def _tensor(value: Any, columns: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().clone().float()
    if columns is not None and (tensor.ndim != 2 or tensor.shape[1] != columns):
        raise ValueError(f"expected an [N,{columns}] gsply field, got {list(tensor.shape)}")
    return tensor


def _sh0_to_rgb(sh0: Any) -> torch.Tensor:
    """Convert degree-zero spherical-harmonic coefficients to display RGB."""
    return (0.5 + _SH_C0 * _tensor(sh0, 3)).clamp(0, 1)


def load_gaussian_scene(path: str | Path) -> GaussianCloud:
    """Load any Gaussian-splat scene format supported by gsply.

    gsply presents PLY, SOG, SPLAT and its other supported encodings through a
    normalized representation: scales are linear, opacity is in ``[0, 1]``,
    and rotations are scalar-first quaternions.  No format-specific decoding
    belongs in the LiDAR simulator.
    """
    import gsply

    source = Path(path)
    # Pass ``device`` explicitly so type checkers select gsply's file-loading
    # overload instead of its in-place ``load(path, gstensor, ...)`` overload.
    scene = gsply.load(source, device="cpu")

    means = _tensor(scene.means, 3)
    scales = _tensor(scene.scales, 3)
    rotations = _tensor(scene.quats, 4)
    opacities = _tensor(scene.opacities).squeeze(-1)
    if opacities.ndim != 1:
        raise ValueError(f"expected an [N] gsply opacity field, got {list(opacities.shape)}")

    colors = None
    if scene.sh0 is not None:
        # GaussianCloud stores RGB for Rerun, whereas GSTensor stores the
        # degree-zero spherical-harmonic coefficient. This does not affect
        # LiDAR tracing; colors are visualization-only metadata.
        colors = _sh0_to_rgb(scene.sh0)

    count = means.shape[0]
    if any(value.shape[0] != count for value in (scales, rotations, opacities)):
        raise ValueError("gsply scene fields have inconsistent Gaussian counts")
    if colors is not None and colors.shape[0] != count:
        raise ValueError("gsply color field has an inconsistent Gaussian count")
    return GaussianCloud(means, scales, rotations, opacities, colors).normalized()


def load_gaussian_ply(path: str | Path) -> GaussianCloud:
    """Backward-compatible alias; prefer :func:`load_gaussian_scene`."""
    return load_gaussian_scene(path)
