from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .gaussian_cloud import GaussianCloud


def _field(scene: Any, *names: str) -> Any:
    """Return a gsply scene field across its mapping and object representations."""
    for name in names:
        if isinstance(scene, Mapping) and name in scene:
            return scene[name]
        if hasattr(scene, name):
            return getattr(scene, name)
    raise ValueError(f"gsply scene is missing the field {names[0]!r}")


def _tensor(value: Any, columns: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().clone().float()
    if columns is not None and (tensor.ndim != 2 or tensor.shape[1] != columns):
        raise ValueError(f"expected an [N,{columns}] gsply field, got {list(tensor.shape)}")
    return tensor


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

    means = _tensor(_field(scene, "means", "positions"), 3)
    scales = _tensor(_field(scene, "scales"), 3)
    rotations = _tensor(_field(scene, "rotations", "quaternions"), 4)
    opacities = _tensor(_field(scene, "opacities", "opacity")).squeeze(-1)
    if opacities.ndim != 1:
        raise ValueError(f"expected an [N] gsply opacity field, got {list(opacities.shape)}")

    colors_value = None
    for name in ("colors", "sh0"):
        if (isinstance(scene, Mapping) and name in scene) or hasattr(scene, name):
            colors_value = _field(scene, name)
            break
    colors = None if colors_value is None else _tensor(colors_value, 3)

    count = means.shape[0]
    if any(value.shape[0] != count for value in (scales, rotations, opacities)):
        raise ValueError("gsply scene fields have inconsistent Gaussian counts")
    if colors is not None and colors.shape[0] != count:
        raise ValueError("gsply color field has an inconsistent Gaussian count")
    return GaussianCloud(means, scales, rotations, opacities, colors).normalized()


def load_gaussian_ply(path: str | Path) -> GaussianCloud:
    """Backward-compatible alias; prefer :func:`load_gaussian_scene`."""
    return load_gaussian_scene(path)
