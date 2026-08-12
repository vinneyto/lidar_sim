import torch

from .gaussian_cloud import GaussianCloud
from .lidar import LidarPose
from .scan import LidarScan


def visualize(scene: GaussianCloud, pose: LidarPose, scan: LidarScan) -> None:
    """The only boundary where tensors are intentionally copied to CPU/NumPy."""
    import rerun as rr

    rr.init("gs-lidar", spawn=True)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    rgb = (
        scene.colors
        if scene.colors is not None
        else scene.means.new_tensor([160, 160, 200]) / 255
    )
    rgb = rgb.expand(scene.means.shape[0], 3)
    rgba = to_numpy(
        torch.cat((rgb, scene.opacities[:, None]), dim=-1) * 255
    ).astype("uint8")

    # Rerun expects quaternions in (x, y, z, w) order.
    quaternions = scene.rotations[:, [1, 2, 3, 0]]
    rr.log(
        "scene/gaussians",
        rr.GaussianSplats3D(
            means=to_numpy(scene.means),
            scales=to_numpy(scene.scales),
            quaternions=to_numpy(quaternions),
            colors=rgba,
        ),
    )
    rr.log(
        "lidar/position",
        rr.Points3D(to_numpy(pose.position[None]), colors=[255, 180, 0], radii=0.08),
    )
    rr.log(
        "lidar/returns",
        rr.Points3D(to_numpy(scan.valid_points()), colors=[0, 255, 120], radii=0.01),
    )
