from .gaussian_cloud import GaussianCloud
from .lidar import LidarPose
from .scan import LidarScan


def visualize(scene: GaussianCloud, pose: LidarPose, scan: LidarScan) -> None:
    """The only boundary where tensors are intentionally copied to CPU/NumPy."""
    import rerun as rr

    rr.init("gs-lidar", spawn=True)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    colors = (
        to_numpy(scene.colors * 255).astype("uint8")
        if scene.colors is not None
        else [160, 160, 200]
    )
    # Points are a stable fallback across Rerun versions; simulation is unaffected.
    rr.log(
        "scene/gaussians",
        rr.Points3D(
            to_numpy(scene.means),
            colors=colors,
            radii=to_numpy(scene.scales.mean(-1)),
        ),
    )
    rr.log(
        "lidar/position",
        rr.Points3D(to_numpy(pose.position[None]), colors=[255, 180, 0], radii=0.08),
    )
    rr.log(
        "lidar/returns",
        rr.Points3D(to_numpy(scan.valid_points()), colors=[0, 255, 120], radii=0.025),
    )
