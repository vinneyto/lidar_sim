"""Render a 3DGS camera on a circular trajectory and overlay Metal DoG features.

Run with:

    uv run python experiments/03_circular_scan_with_camera.py

The detector mirrors the detector stage used by the Kornia comparison:
MultiResolutionDetector + BlobDoGSingle(1.0, 1.6), with PassLAF orientation.
"""

import math
import time
from pathlib import Path

import torch
from course_3dgs import GaussianData, MetalRenderer

from gs_lidar import CameraIntrinsics, GaussianPlyData
from metal_sift import MetalSiftDetector, SiftFeatures

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

SIFT_FEATURE_COUNT = 768
SIFT_POINT_RADIUS = 3.0
SIFT_DEBUG = True

RERUN_UP_AXIS = "+Y"


def load_scene(path: Path) -> GaussianPlyData:
    if not path.is_file():
        raise FileNotFoundError(
            f"PLY scene not found: {path}. Set PLY_PATH at the top of this file."
        )
    return GaussianPlyData.from_ply(path)


def select_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("This experiment requires an available PyTorch MPS backend")
    return torch.device("mps")


def orbit_position(device: torch.device, angle_radians: float) -> torch.Tensor:
    center = torch.tensor(ORBIT_CENTER, dtype=torch.float32, device=device)
    return center + center.new_tensor(
        [
            ORBIT_RADIUS * math.cos(angle_radians),
            0.0,
            ORBIT_RADIUS * math.sin(angle_radians),
        ]
    )


def orbit_points() -> torch.Tensor:
    center = torch.tensor(ORBIT_CENTER, dtype=torch.float32)
    angles = torch.linspace(0, 2 * math.pi, RING_SAMPLES + 1)
    points = center.expand(RING_SAMPLES + 1, 3).clone()
    points[:, 0] += ORBIT_RADIUS * torch.cos(angles)
    points[:, 2] += ORBIT_RADIUS * torch.sin(angles)
    return points


def camera_c2w(device: torch.device, angle_radians: float) -> torch.Tensor:
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
        color_mode="canonical_3dgs",
    )


def camera_frustum_line_strips(
    c2w: torch.Tensor,
    intrinsics: CameraIntrinsics,
    depth: float = CAMERA_FRUSTUM_DEPTH,
) -> list[torch.Tensor]:
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


def feature_labels(features: SiftFeatures) -> list[str]:
    responses = features.responses.detach().cpu().tolist()
    scales = features.scales.detach().cpu().tolist()
    return [
        f"#{index} response={response:.4g} scale={scale:.2f}"
        for index, (response, scale) in enumerate(zip(responses, scales))
    ]


def sift_debug_suffix(features: SiftFeatures) -> str:
    stats = features.debug
    if stats is None:
        return ""
    level_counts = "/".join(str(value) for value in stats.per_level_candidate_counts)
    counts = (
        f", nms-candidates={stats.candidate_count} "
        f"[{level_counts}], preselected={stats.pyramid_preselected_count}"
    )
    if stats.first_rejected_abs_response is None:
        return (
            f"{counts}, weakest response={stats.cutoff_abs_response:.6f}, "
            "global top-k limit not reached"
        )
    return (
        f"{counts}, cutoff response={stats.cutoff_abs_response:.6f}, "
        f"next={stats.first_rejected_abs_response:.6f}, "
        f"gap={stats.boundary_gap:.2e}, "
        f"near-cutoff={stats.near_cutoff_count}/{stats.selected_count} "
        f"({stats.near_cutoff_fraction:.1%})"
    )


def log_frame(
    step: int,
    c2w: torch.Tensor,
    intrinsics: CameraIntrinsics,
    image: torch.Tensor,
    features: SiftFeatures,
) -> None:
    import rerun as rr

    rr.set_time("frame", sequence=step)
    rr.log(
        "world/camera/frustum",
        rr.LineStrips3D(
            [
                strip.detach().cpu().numpy()
                for strip in camera_frustum_line_strips(c2w, intrinsics)
            ],
            colors=[80, 160, 255],
            radii=0.01,
        ),
    )
    image_u8 = image.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    rr.log("camera/render", rr.Image(image_u8))
    rr.log(
        "camera/features/metal_sift",
        rr.Points2D(
            features.keypoints_xy.detach().cpu().numpy(),
            radii=SIFT_POINT_RADIUS,
            colors=[0, 255, 0],
            labels=feature_labels(features),
            show_labels=False,
            keypoint_ids=list(range(features.count)),
            draw_order=10.0,
        ),
    )


def run_experiment() -> None:
    if ORBIT_RADIUS <= 0 or STEP_SECONDS <= 0 or NUMBER_OF_STEPS < 1:
        raise ValueError("orbit radius, step duration, and step count must be positive")

    device = select_device()
    scene = load_scene(PLY_PATH)
    initialize_rerun(scene)

    camera_intrinsics = create_camera_intrinsics()
    renderer_data = scene.to_renderer_data(device)
    renderer = create_metal_renderer(renderer_data, camera_intrinsics)
    feature_detector = MetalSiftDetector(
        num_features=SIFT_FEATURE_COUNT,
        debug=SIFT_DEBUG,
    )
    angular_velocity = math.radians(ANGULAR_VELOCITY_DEGREES_PER_SECOND)

    print(
        f"Loaded {renderer_data.num_gaussians:,} Gaussians with "
        f"{renderer_data.sh_levels} SH level(s) "
        f"({renderer_data.sh_coefficient_count} coefficients/channel), "
        "color mode=canonical_3dgs"
    )

    for step in range(NUMBER_OF_STEPS):
        angle = angular_velocity * STEP_SECONDS * step
        c2w = camera_c2w(device, angle)

        render_started_at = time.perf_counter()
        image = renderer.render(c2w)
        torch.mps.synchronize()
        render_seconds = time.perf_counter() - render_started_at

        sift_started_at = time.perf_counter()
        features = feature_detector.detect(image)
        torch.mps.synchronize()
        sift_seconds = time.perf_counter() - sift_started_at

        log_frame(step, c2w, camera_intrinsics, image, features)
        print(
            f"Rendered frame {step + 1}/{NUMBER_OF_STEPS} "
            f"in {render_seconds * 1000:.2f} ms (Metal 3DGS), "
            f"detected {features.count} SIFT keypoints "
            f"in {sift_seconds * 1000:.2f} ms (Kornia-style custom Metal)"
            f"{sift_debug_suffix(features)}"
        )
        if step + 1 < NUMBER_OF_STEPS:
            time.sleep(STEP_SECONDS)


if __name__ == "__main__":
    run_experiment()
