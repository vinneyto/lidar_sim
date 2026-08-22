"""Render a tangential 3DGS camera while it follows a circular trajectory.

Edit the constants below, then run:

    uv run python experiments/03_circular_scan_with_camera.py

The one-meter orbit is centered at the coordinate origin and lies in the XZ
plane. Rerun shows the 3DGS scene and a moving camera-frustum pyramid on the
left, and the spherical-harmonic Metal camera render on the right. The camera
always looks toward the orbit center and 15 degrees downward, with a stable
roll-free up direction.
"""

import math
import time
from pathlib import Path

import torch
from course_3dgs import GaussianData, MetalRenderer

from gs_lidar import CameraIntrinsics, GaussianPlyData

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

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960
CAMERA_FX = 1000.0
CAMERA_FY = 1000.0
CAMERA_NEAR = 0.1
CAMERA_FAR = 10.0
CAMERA_FRUSTUM_DEPTH = 0.35
CAMERA_DOWNWARD_PITCH_DEGREES = 15.0

# Rerun turntable rotation axis. This reconstructed model uses "+Y"
# (and some exports may need "-Y").
RERUN_UP_AXIS = "+Y"


def load_scene(path: Path) -> GaussianPlyData:
    """Load the canonical 3DGS PLY scene used by this camera experiment."""
    if not path.is_file():
        raise FileNotFoundError(
            f"PLY scene not found: {path}. Set PLY_PATH at the top of this file."
        )
    return GaussianPlyData.from_ply(path)


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
    """Create a roll-free camera looking toward the orbit center and downward."""
    position = orbit_position(device, angle_radians)
    orbit_center = position.new_tensor(ORBIT_CENTER)
    horizontal_forward = orbit_center - position
    horizontal_forward[1] = 0.0
    horizontal_forward = horizontal_forward / torch.linalg.norm(horizontal_forward)
    world_up = position.new_tensor([0.0, 1.0, 0.0])
    downward_pitch = math.radians(CAMERA_DOWNWARD_PITCH_DEGREES)
    forward = (
        math.cos(downward_pitch) * horizontal_forward
        - math.sin(downward_pitch) * world_up
    )
    right = torch.linalg.cross(world_up, forward)
    right = right / torch.linalg.norm(right)
    up = torch.linalg.cross(forward, right)

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


def create_metal_renderer(
    data: GaussianData,
    intrinsics: CameraIntrinsics,
) -> MetalRenderer:
    """Create the reusable course_3dgs renderer for this fixed camera model."""
    # The experiment exposes a conventional +Y-up camera. Image rows grow
    # downward, so the course renderer receives a negative fy just like the
    # previous lidar_sim-local copy did.
    return MetalRenderer(
        data,
        H=intrinsics.height,
        W=intrinsics.width,
        fx=intrinsics.fx,
        fy=-intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
        near=intrinsics.near,
        far=intrinsics.far,
    )


def print_color_diagnostics(
    scene: GaussianPlyData,
    renderer_data: GaussianData,
) -> None:
    """Compare canonical PLY DC colors with effective course renderer colors."""
    canonical = scene.colors
    if canonical is None:
        print("Color debug: PLY has no f_dc color coefficients")
        return

    # sh_levels=1 is view-independent, so an identity camera is sufficient to
    # exercise the same sigmoid color convention used by the Metal setup kernel.
    c2w = torch.eye(
        4,
        dtype=torch.float32,
        device=renderer_data.positions.device,
    )
    effective = renderer_data.evaluate_color(c2w).detach().cpu()
    canonical = canonical.detach().cpu()

    canonical_mean = canonical.mean(dim=0)
    renderer_mean = effective.mean(dim=0)
    max_abs_difference = (canonical - effective).abs().max().item()

    print(
        "Color debug (DC only): "
        f"canonical mean RGB={canonical_mean.tolist()}, "
        f"renderer mean RGB={renderer_mean.tolist()}, "
        f"canonical range=[{canonical.min().item():.4f}, {canonical.max().item():.4f}], "
        f"renderer range=[{effective.min().item():.4f}, {effective.max().item():.4f}], "
        f"max |difference|={max_abs_difference:.6g}"
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


def initialize_rerun(scene: GaussianPlyData) -> None:
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
        rgb = scene.positions.new_tensor([160, 160, 200]) / 255
    rgba = torch.cat(
        (rgb.expand(scene.positions.shape[0], 3), scene.opacities[:, None]), dim=-1
    )
    rr.log(
        "world/scene/gaussians",
        rr.GaussianSplats3D(
            numpy(scene.positions),
            scales=numpy(scene.scales),
            quaternions=numpy(scene.rotations_xyzw),
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
    scene = load_scene(PLY_PATH)
    initialize_rerun(scene)

    camera_intrinsics = create_camera_intrinsics()
    renderer_data = scene.to_renderer_data(device)
    renderer = create_metal_renderer(renderer_data, camera_intrinsics)
    angular_velocity = math.radians(ANGULAR_VELOCITY_DEGREES_PER_SECOND)

    print(
        f"Loaded {renderer_data.num_gaussians:,} Gaussians; "
        f"PLY contains {scene.sh_levels} SH level(s), "
        f"renderer currently uses {renderer_data.sh_levels} level(s) "
        f"({renderer_data.sh_coefficient_count} coefficients/channel)"
    )
    print_color_diagnostics(scene, renderer_data)

    for step in range(NUMBER_OF_STEPS):
        angle = angular_velocity * STEP_SECONDS * step
        c2w = camera_c2w(device, angle)
        render_started_at = time.perf_counter()
        image = renderer.render(c2w)
        # course_3dgs intentionally leaves synchronization to the caller. We
        # synchronize here so the printed timing measures completed GPU work.
        torch.mps.synchronize()
        render_seconds = time.perf_counter() - render_started_at
        log_frame(step, c2w, camera_intrinsics, image)
        print(
            f"Rendered frame {step + 1}/{NUMBER_OF_STEPS} "
            f"in {render_seconds * 1000:.2f} ms (Metal 3DGS)"
        )
        if step + 1 < NUMBER_OF_STEPS:
            time.sleep(STEP_SECONDS)


if __name__ == "__main__":
    run_experiment()
