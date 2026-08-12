from dataclasses import dataclass
import math
import torch
from .gaussian_geometry import quaternion_to_matrix


@dataclass(frozen=True)
class LidarPose:
    position: torch.Tensor
    orientation: torch.Tensor  # (w, x, y, z)

    def to(self, device: torch.device | str) -> "LidarPose":
        return LidarPose(self.position.to(device), self.orientation.to(device))


@dataclass(frozen=True)
class LidarConfig:
    azimuth_min: float = -math.pi
    azimuth_max: float = math.pi
    elevation_min: float = -math.pi / 3
    elevation_max: float = math.pi / 3
    azimuth_samples: int = 360
    elevation_samples: int = 64
    near: float = 0.1
    far: float = 100.0
    gaussian_sigma_cutoff: float = 3.0
    accumulated_alpha_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.azimuth_samples < 1 or self.elevation_samples < 1 or self.near < 0 or self.far <= self.near:
            raise ValueError("invalid sample count or range")
        if not 0 < self.accumulated_alpha_threshold <= 1:
            raise ValueError("alpha threshold must be in (0, 1]")


def generate_rays(pose: LidarPose, config: LidarConfig) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = pose.position.device, pose.position.dtype
    az = torch.arange(config.azimuth_samples, device=device, dtype=dtype)
    az = config.azimuth_min + az * ((config.azimuth_max-config.azimuth_min)/config.azimuth_samples)
    el = torch.linspace(config.elevation_min, config.elevation_max, config.elevation_samples, device=device, dtype=dtype)
    eg, ag = torch.meshgrid(el, az, indexing="ij")
    local = torch.stack((torch.cos(eg)*torch.cos(ag), torch.cos(eg)*torch.sin(ag), torch.sin(eg)), -1)
    world = local.reshape(-1, 3) @ quaternion_to_matrix(pose.orientation).T
    return pose.position.expand_as(world), world
