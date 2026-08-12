from typing import Literal

import torch

from .gaussian_cloud import GaussianCloud
from .lidar import LidarPose
from .scan import LidarScan


UpAxis = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def visualize(
    scene: GaussianCloud,
    pose: LidarPose,
    scan: LidarScan,
    *,
    up_axis: UpAxis = "+Y",
) -> None:
    """Visualize a scan with a configurable camera up axis.

    ``up_axis`` changes Rerun's world-coordinate convention, which controls the
    turntable camera without modifying the scene coordinates. The simulator's
    native convention is ``+Z`` up, while the example model uses ``+Y`` up.
    """
    import rerun as rr

    rr.init("gs-lidar", spawn=True)

    # Without an explicit view origin Rerun creates its 3D view at "/".  The
    # ViewCoordinates logged below live at "world", so a root view falls back
    # to Z-up turntable controls even though its child entities are annotated
    # as Y-up.  Anchor the view at the same entity as the coordinate convention.
    rr.send_blueprint(
        rr.blueprint.Blueprint(rr.blueprint.Spatial3DView(origin="world"))
    )

    view_coordinates = {
        "+X": rr.ViewCoordinates.RIGHT_HAND_X_UP,
        "-X": rr.ViewCoordinates.RIGHT_HAND_X_DOWN,
        "+Y": rr.ViewCoordinates.RIGHT_HAND_Y_UP,
        "-Y": rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
        "+Z": rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        "-Z": rr.ViewCoordinates.RIGHT_HAND_Z_DOWN,
    }
    if up_axis not in view_coordinates:
        raise ValueError("up_axis must be one of +X, -X, +Y, -Y, +Z, or -Z")
    rr.log("world", view_coordinates[up_axis], static=True)

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
        "world/scene/gaussians",
        rr.GaussianSplats3D(
            to_numpy(scene.means),
            scales=to_numpy(scene.scales),
            quaternions=to_numpy(quaternions),
            colors=rgba,
        ),
    )
    rr.log(
        "world/lidar/position",
        rr.Points3D(to_numpy(pose.position[None]), colors=[255, 180, 0], radii=0.08),
    )
    rr.log(
        "world/lidar/returns",
        rr.Points3D(to_numpy(scan.valid_points()), colors=[0, 255, 120], radii=0.01),
    )
