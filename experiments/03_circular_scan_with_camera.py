"""Render a tangential 3DGS camera while it follows a circular trajectory.

Edit the constants below, then run:

    uv run python experiments/03_circular_scan_with_camera.py

The one-meter orbit is centered at the coordinate origin and lies in the XZ
plane. Rerun shows the 3DGS scene and a moving camera-frustum pyramid on the
left, and the SH-free Metal camera render on the right. The camera looks along
the orbit tangent, with its local +Y axis pointing toward world +Y.
"""

import math
import time
from pathlib import Path

import torch

from gs_lidar import (
    CameraIntrinsics,
    GaussianCloud,
    MetalGaussianRenderer,
    load_gaussian_ply,
)

# -----------------------------------------------------------------------------
# Experiment configuration — edit these values before running the script.
# -----------------------------------------------------------------------------

PLY_PATH = Path("mug.ply")

ORBIT_CENTER = (0.0, 0.0, 0.0)
ORBIT_RADIUS = 1.0
ANGULAR_VELOCITY_DEGREES_PER_SECOND = 5.0
STEP_SECONDS = 0.1
NUMBER_OF_STEPS = 360
RING_SAMPLES = 256

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FX = 500.0
CAMERA_FY = 500.0
CAMERA_NEAR = 0.1
CAMERA_FAR = 10.0
CAMERA_FRUSTUM_DEPTH = 0.35

# Rerun turntable rotation axis. This reconstructed model uses "+Y"
# (and some exports may need "-Y").
RERUN_UP_AXIS = "+Y"


def load_scene(path: Path) -> GaussianCloud:
    """Load a canonical 3DGS PLY scene."""
    if not path.is_file():
        raise FileNotFoundError(
            f"PLY scene not found: {path}. Set PLY_PATH at the top of this file."
        )
    return load_gaussian_ply(path)


def select_device() -> torch.device:
    """Return the MPS device required by the Metal renderer."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("This experiment requires an available PyTorch MPS backend")
    return torch.device("mps")


def orbit_position(device: torch.device, angle_radians: float) -> torch.Tensor:
    """Return a point on the fixed one-meter XZ-plane orbit."""
    center = torch.tensor(ORBIT_CENTER, dtype=torch.float32, device=device)
    offset = center.new_tensor(
        [
            ORBIT_RADIUS * math.cos(angle_radians),
            0.0,
            ORBIT_RADIUS * math.sin(angle_radians),
        ]
    )
    return center + offset


def orbit_points() -> torch.Tensor:
    """Create a closed polyline describing the fixed camera orbit."""
    center = torch.tensor(ORBIT_CENTER, dtype=torch.float32)
    angles = torch.linspace(0, 2 * math.pi, RING_SAMPLES + 1)
    points = center.expand(RING_SAMPLES + 1, 3).clone()
    points[:, 0] += ORBIT_RADIUS * torch.cos(angles)
    points[:, 2] += ORBIT_RADIUS * torch.sin(angles)
    return points


def camera_c2w(device: torch.device, angle_radians: float) -> torch.Tensor:
    """Create a tangent-facing camera pose with +Y up and +Z forward."""
    position = orbit_position(device, angle_radians)
    forward = position.new_tensor(
        [-math.sin(angle_radians), 0.0, math.cos(angle_radians)]
    )
    up = position.new_tensor([0.0, 1.0, 0.0])
    right = torch.linalg.cross(up, forward)

    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return c2w


def create_camera_intrinsics() -> CameraIntrinsics:
    """Create the fixed camera used for rendering and the frustum pyramid."""
    return CameraIntrinsics(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fx=CAMERA_FX,
        fy=CAMERA_FY,
        cx=CAMERA_WIDTH / 2,
        cy=CAMERA_HEIGHT / 2,
        near=CAMERA_NEAR,
        far=CAMERA_FAR,
    )


def camera_frustum_line_strips(
    c2w: torch.Tensor,
    intrinsics: CameraIntrinsics,
    depth: float = CAMERA_FRUSTUM_DEPTH,
) -> list[torch.Tensor]:
    """Return the rectangular base and four sides of a camera-frustum pyramid."""
    if depth <= 0:
        raise ValueError("camera frustum depth must be positive")

    x_left = -intrinsics.cx / intrinsics.fx * depth
    x_right = (intrinsics.width - intrinsics.cx) / intrinsics.fx * depth
    y_top = intrinsics.cy / intrinsics.fy * depth
    y_bottom = -(intrinsics.height - intrinsics.cy) / intrinsics.fy * depth
    corners_camera = c2w.new_tensor(
        [
            [x_left, y_top, depth],
            [x_right, y_top, depth],
            [x_right, y_bottom, depth],
            [x_left, y_bottom, depth],
        ]
    )

    rotation = c2w[:3, :3]
    origin = c2w[:3, 3]
    corners_world = corners_camera @ rotation.T + origin
    base = torch.cat((corners_world, corners_world[:1]), dim=0)
    sides = [torch.stack((origin, corner)) for corner in corners_world]
    return [base, *sides]


def initialize_rerun(scene: GaussianCloud) -> None:
    """Open Rerun and log the immutable scene and circular trajectory."""
    import rerun as rr

    rr.init("3dgs-circular-camera", spawn=True)
    rr.send_blueprint(
        rr.blueprint.Blueprint(
            rr.blueprint.Horizontal(
                rr.blueprint.Spatial3DView(origin="world"),
                rr.blueprint.Spatial2DView(origin="camera"),
            )
        )
    )
    coordinates = {
        "+X": rr.ViewCoordinates.RIGHT_HAND_X_UP,
        "-X": rr.ViewCoordinates.RIGHT_HAND_X_DOWN,
        "+Y": rr.ViewCoordinates.RIGHT_HAND_Y_UP,
        "-Y": rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
        "+Z": rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        "-Z": rr.ViewCoordinates.RIGHT_HAND_Z_DOWN,
    }
    if RERUN_UP_AXIS not in coordinates:
        raise ValueError("RERUN_UP_AXIS must be one of +X, -X, +Y, -Y, +Z, or -Z")
    rr.log("world", coordinates[RERUN_UP_AXIS], static=True)

    def numpy(tensor: torch.Tensor):
        return tensor.detach().cpu().numpy()

    rgb = scene.colors
    if rgb is None:
        rgb = scene.means.new_tensor([160, 160, 200]) / 255
    rgba = torch.cat(
        (rgb.expand(scene.means.shape[0], 3), scene.opacities[:, None]), dim=-1
    )
    rr.log(
        "world/scene/gaussians",
        rr.GaussianSplats3D(
            numpy(scene.means),
            scales=numpy(scene.scales),
            quaternions=numpy(scene.rotations[:, [1, 2, 3, 0]]),
            colors=numpy(rgba * 255).astype("uint8"),
        ),
        static=True,
    )
    rr.log(
        "world/camera/orbit",
        rr.LineStrips3D(
            [numpy(orbit_points())],
            colors=[120, 170, 255],
            radii=0.01,
        ),
        static=True,
    )


def log_frame(
    step: int,
    c2w: torch.Tensor,
    intrinsics: CameraIntrinsics,
    image: torch.Tensor,
) -> None:
    """Record the moving camera-frustum pyramid and synchronized image."""
    import rerun as rr

    rr.set_time("frame", sequence=step)
    line_strips = [
        strip.detach().cpu().numpy()
        for strip in camera_frustum_line_strips(c2w, intrinsics)
    ]
    rr.log(
        "world/camera/frustum",
        rr.LineStrips3D(
            line_strips,
            colors=[80, 160, 255],
            radii=0.01,
        ),
    )
    image_u8 = image.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    rr.log("camera/render", rr.Image(image_u8))


def run_experiment() -> None:
    """Render the moving camera at every point on its circular orbit."""
    if ORBIT_RADIUS <= 0 or STEP_SECONDS <= 0 or NUMBER_OF_STEPS < 1:
        raise ValueError("orbit radius, step duration, and step count must be positive")

    device = select_device()
    scene_cpu = load_scene(PLY_PATH)
    initialize_rerun(scene_cpu)

    scene = scene_cpu.to(device)
    renderer = MetalGaussianRenderer(scene)
    camera_intrinsics = create_camera_intrinsics()
    angular_velocity = math.radians(ANGULAR_VELOCITY_DEGREES_PER_SECOND)

    for step in range(NUMBER_OF_STEPS):
        angle = angular_velocity * STEP_SECONDS * step
        c2w = camera_c2w(device, angle)
        image = renderer.render(c2w, camera_intrinsics)
        log_frame(step, c2w, camera_intrinsics, image)
        print(f"Rendered frame {step + 1}/{NUMBER_OF_STEPS}")
        if step + 1 < NUMBER_OF_STEPS:
            time.sleep(STEP_SECONDS)


if __name__ == "__main__":
    run_experiment()
