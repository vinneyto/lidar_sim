from pathlib import Path
import torch
from .gaussian_cloud import GaussianCloud


def load_gaussian_ply(path: str | Path) -> GaussianCloud:
    """Load canonical Inria 3DGS PLY (scale logits, opacity logits, scalar-first rotation)."""
    from plyfile import PlyData
    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity"}
    missing = required - names
    if missing:
        raise ValueError(f"not a canonical 3DGS PLY; missing properties: {sorted(missing)}")
    def tensor(columns: list[str]) -> torch.Tensor:
        # plyfile exposes structured NumPy data; conversion is confined to this boundary.
        return torch.stack([torch.from_numpy(vertex[c].copy()) for c in columns], -1).float()
    means = tensor(["x", "y", "z"])
    scales = tensor(["scale_0", "scale_1", "scale_2"]).exp()
    rotations = tensor(["rot_0", "rot_1", "rot_2", "rot_3"])
    opacities = torch.from_numpy(vertex["opacity"].copy()).float().sigmoid()
    colors = None
    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        colors = (0.5 + 0.28209479177387814*tensor(["f_dc_0", "f_dc_1", "f_dc_2"])).clamp(0, 1)
    return GaussianCloud(means, scales, rotations, opacities, colors).normalized()
