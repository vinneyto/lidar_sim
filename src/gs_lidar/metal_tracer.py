from importlib.resources import files

import torch

from .bvh import FlatBVH
from .gaussian_cloud import GaussianCloud
from .lidar import LidarConfig, LidarPose, generate_rays
from .scan import LidarScan


class MetalLidarTracer:
    """One custom-Metal thread per ray, using PyTorch MPS buffers directly."""

    def __init__(self) -> None:
        if not torch.backends.mps.is_available() or not hasattr(
            torch.mps, "compile_shader"
        ):
            raise RuntimeError("MPS and torch.mps.compile_shader are required")
        source = files("gs_lidar.metal").joinpath("lidar_trace.metal").read_text()
        self.kernel = torch.mps.compile_shader(source).lidar_trace

    def trace(
        self, scene: GaussianCloud, pose: LidarPose, config: LidarConfig, bvh: FlatBVH
    ) -> LidarScan:
        if scene.means.device.type != "mps" or bvh.bbox_min.device.type != "mps":
            raise ValueError("scene and BVH must already reside on MPS")
        origins, directions = generate_rays(pose, config)
        n = origins.shape[0]
        ranges = torch.empty(n, device="mps")
        points = torch.empty((n, 3), device="mps")
        hits = torch.empty(n, dtype=torch.uint8, device="mps")
        ids = torch.empty(n, dtype=torch.int32, device="mps")
        alpha = torch.empty(n, device="mps")
        candidate_overflow = torch.zeros(1, dtype=torch.int32, device="mps")
        stack_overflow = torch.zeros(1, dtype=torch.int32, device="mps")

        def scalar(value: float) -> torch.Tensor:
            return torch.tensor(value, dtype=torch.float32, device="mps")

        args = (
            scene.means.contiguous(),
            scene.scales.contiguous(),
            scene.rotations.contiguous(),
            scene.opacities.contiguous(),
            bvh.bbox_min,
            bvh.bbox_max,
            bvh.left_child,
            bvh.right_child,
            bvh.first_primitive,
            bvh.primitive_count,
            bvh.primitive_indices,
            origins,
            directions,
            ranges,
            points,
            hits,
            ids,
            alpha,
            candidate_overflow,
            stack_overflow,
            scalar(config.near),
            scalar(config.far),
            scalar(config.gaussian_sigma_cutoff**2),
            scalar(config.accumulated_alpha_threshold),
        )
        self.kernel(*args, threads=n)
        shape = (config.elevation_samples, config.azimuth_samples)
        return LidarScan(
            ranges.reshape(shape),
            points.reshape(shape + (3,)),
            hits.bool().reshape(shape),
            ids.reshape(shape),
            alpha.reshape(shape),
            candidate_overflow,
            stack_overflow,
        )
