from .bvh import FlatBVH
from .cpu_tracer import CpuLidarTracer
from .gaussian_cloud import GaussianCloud
from .lidar import LidarConfig, LidarPose
from .metal_tracer import MetalLidarTracer


class LidarSimulator:
    def __init__(self, scene: GaussianCloud, bvh: FlatBVH, backend: str | None = None):
        self.scene, self.bvh = scene, bvh
        backend = backend or ("metal" if scene.means.device.type == "mps" else "cpu")
        self.tracer = MetalLidarTracer() if backend == "metal" else CpuLidarTracer(use_bvh=True)

    def scan(self, pose: LidarPose, config: LidarConfig):
        return self.tracer.trace(self.scene, pose, config, self.bvh)
