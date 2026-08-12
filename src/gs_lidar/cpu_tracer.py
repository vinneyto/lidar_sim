import torch
from .bvh import FlatBVH, bvh_candidates
from .gaussian_cloud import GaussianCloud
from .gaussian_geometry import ray_gaussian_peaks
from .lidar import LidarConfig, LidarPose, generate_rays
from .scan import LidarScan


class CpuLidarTracer:
    def __init__(self, use_bvh: bool = True):
        self.use_bvh = use_bvh

    def trace(self, scene: GaussianCloud, pose: LidarPose, config: LidarConfig,
              bvh: FlatBVH | None = None) -> LidarScan:
        if scene.means.device.type != "cpu":
            raise ValueError("CpuLidarTracer requires CPU tensors")
        if self.use_bvh and bvh is None:
            raise ValueError("BVH tracer requires a BVH")
        origins, directions = generate_rays(pose, config)
        count = origins.shape[0]
        ranges = torch.full((count,), torch.inf, dtype=origins.dtype)
        points = torch.full((count, 3), torch.nan, dtype=origins.dtype)
        hit = torch.zeros(count, dtype=torch.bool)
        ids = torch.full((count,), -1, dtype=torch.int64)
        accumulated = torch.zeros(count, dtype=origins.dtype)
        all_ids = torch.arange(scene.means.shape[0])
        for ray in range(count):
            candidates = bvh_candidates(bvh, origins[ray], directions[ray], config.near, config.far) if self.use_bvh else all_ids
            if candidates.numel() == 0:
                continue
            t, q = ray_gaussian_peaks(origins[ray:ray+1], directions[ray:ray+1], scene.means[candidates],
                                      scene.scales[candidates], scene.rotations[candidates])
            t, q = t[0], q[0]
            valid = (t >= config.near) & (t <= config.far) & (q <= config.gaussian_sigma_cutoff**2)
            selected = candidates[valid]
            order = torch.argsort(t[valid], stable=True)
            alpha = scene.opacities[selected] * torch.exp(-0.5*q[valid])
            a = torch.zeros((), dtype=origins.dtype)
            for j in order:
                a = a + (1-a)*alpha[j]
                accumulated[ray] = a
                if a >= config.accumulated_alpha_threshold:
                    distance = t[valid][j]
                    ranges[ray], points[ray], hit[ray], ids[ray] = distance, origins[ray]+distance*directions[ray], True, selected[j]
                    break
        shape = (config.elevation_samples, config.azimuth_samples)
        return LidarScan(ranges.reshape(shape), points.reshape(shape+(3,)), hit.reshape(shape), ids.reshape(shape), accumulated.reshape(shape))
