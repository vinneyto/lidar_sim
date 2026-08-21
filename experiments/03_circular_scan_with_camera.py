"""Run circular LiDAR scans with a co-located tangential RGB camera.

Edit the constants below, then run:

    uv run python experiments/03_circular_scan_with_camera.py

The one-meter orbit is centered at the coordinate origin and lies in the XZ
plane. Rerun shows the LiDAR scene on the left and the SH-free Metal 3DGS
camera render on the right. The camera looks along the orbit tangent, with its
local +Y axis pointing toward world +Y.
"""

import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

from gs_lidar import (
    CameraIntrinsics,
    FlatBVH,
    GaussianCloud,
    LidarConfig,
    LidarPose,
    LidarScan,
    LidarSimulator,
    MetalGaussianRenderer,
    build_bvh,
    gaussian_aabbs,
    load_gaussian_ply,
)

# -----------------------------------------------------------------------------
# Experiment configuration — edit these values before running the script.
# -----------------------------------------------------------------------------

PLY_PATH = Path("mug.ply")
BACKEND = "metal"  # "cpu" or "metal"

AZIMUTH_SAMPLES = 360
ELEVATION_SAMPLES = 64
NEAR = 0.1
FAR = 100.0
SIGMA_CUTOFF = 3.0
ALPHA_THRESHOLD = 0.5

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
DEBUG_LOG_PATH = Path("circular_scan_with_camera_debug.jsonl")

# Rerun turntable rotation axis. Canonical simulator coordinates use "+Z";
# this reconstructed model uses "+Y" (and some exports may need "-Y").
RERUN_UP_AXIS = "+Y"


def load_scene(path: Path) -> GaussianCloud:
    """Load a canonical 3DGS PLY scene."""
    if not path.is_file():
        raise FileNotFoundError(
            f"PLY scene not found: {path}. Set PLY_PATH at the top of this file."
        )
    return load_gaussian_ply(path)


def create_bvh(scene: GaussianCloud) -> FlatBVH:
    """Build the finite-support Gaussian BVH once on the CPU."""
    bbox_min, bbox_max = gaussian_aabbs(
        scene.means, scene.scales, scene.rotations, SIGMA_CUTOFF
    )
    return build_bvh(bbox_min, bbox_max)


def select_device(backend: str) -> torch.device:
    """Validate the selected backend and return its torch device."""
    if backend == "cpu":
        return torch.device("cpu")
    if backend != "metal":
        raise ValueError("BACKEND must be either 'cpu' or 'metal'")
    if not torch.backends.mps.is_available():
        raise RuntimeError("BACKEND='metal' requires an available PyTorch MPS backend")
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
    """Create a closed polyline describing the fixed sensor orbit."""
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
    """Create the fixed camera requested for the circular experiment."""
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


def lidar_orientation(device: torch.device) -> torch.Tensor:
    """Align the scanner's elevation axis with this scene's +Y up axis.

    Ray generation is natively Z-up. A -90 degree rotation around X maps the
    local XY azimuth plane onto the scene's XZ tabletop plane and local +Z onto
    world +Y. Without this rotation, the orbit and the scan pattern disagree
    about which direction is up, producing alternating misses and dense bands.
    """
    half_angle = -math.pi / 4
    return torch.tensor(
        [math.cos(half_angle), math.sin(half_angle), 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )


def create_config() -> LidarConfig:
    return LidarConfig(
        azimuth_samples=AZIMUTH_SAMPLES,
        elevation_samples=ELEVATION_SAMPLES,
        near=NEAR,
        far=FAR,
        gaussian_sigma_cutoff=SIGMA_CUTOFF,
        accumulated_alpha_threshold=ALPHA_THRESHOLD,
    )


def initialize_rerun(scene: GaussianCloud) -> None:
    """Open Rerun and log the immutable scene and circular trajectory."""
    import rerun as rr

    rr.init("gs-lidar-circular-scan-with-camera", spawn=True)
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
        "world/lidar/orbit",
        rr.LineStrips3D(
            [numpy(orbit_points())],
            colors=[120, 170, 255],
            radii=0.01,
        ),
        static=True,
    )


def log_scan(
    step: int,
    pose: LidarPose,
    points: torch.Tensor,
    c2w: torch.Tensor,
    image: torch.Tensor,
) -> None:
    """Record synchronized LiDAR and camera outputs on the scan timeline."""
    import rerun as rr

    # Rerun 0.23+ uses the unified set_time API. Passing an integer sequence
    # value gives every scan its own frame on the timeline.
    rr.set_time("scan", sequence=step)
    rr.log(
        "world/lidar/position",
        rr.Points3D(
            pose.position.detach().cpu().numpy()[None],
            colors=[255, 180, 0],
            radii=0.08,
        ),
    )
    rr.log(
        "world/lidar/returns",
        rr.Points3D(points.detach().cpu().numpy(), colors=[0, 255, 120], radii=0.01),
    )
    camera_origin = c2w[:3, 3].detach().cpu().numpy()[None]
    camera_forward = c2w[:3, 2].detach().cpu().numpy()[None]
    rr.log(
        "world/camera/forward",
        rr.Arrows3D(
            origins=camera_origin,
            vectors=camera_forward,
            colors=[80, 160, 255],
            radii=0.015,
        ),
    )
    image_u8 = image.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    rr.log("camera/render", rr.Image(image_u8))


def tensor_digest(tensor: torch.Tensor) -> str:
    """Return a compact, deterministic fingerprint of a CPU tensor."""
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()[:16]


def scan_diagnostics(
    step: int,
    pose: LidarPose,
    scan: LidarScan,
    previous: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Collect enough CPU-side data to diagnose unstable repeated scans."""
    hit_mask = scan.hit_mask.detach().cpu()
    ranges = scan.ranges.detach().cpu()
    gaussian_ids = scan.gaussian_ids.detach().cpu()
    alpha = scan.accumulated_alpha.detach().cpu()
    valid_ranges = ranges[hit_mask]
    valid_alpha = alpha[torch.isfinite(alpha)]

    record: dict[str, object] = {
        "step": step,
        "position": pose.position.detach().cpu().tolist(),
        "orientation_wxyz": pose.orientation.detach().cpu().tolist(),
        "hits": int(hit_mask.sum()),
        "candidate_overflows": (
            int(scan.candidate_overflow_count.detach().cpu().item())
            if scan.candidate_overflow_count is not None
            else None
        ),
        "stack_overflows": (
            int(scan.bvh_stack_overflow_count.detach().cpu().item())
            if scan.bvh_stack_overflow_count is not None
            else None
        ),
        "hit_mask_digest": tensor_digest(hit_mask),
        "range_digest": tensor_digest(ranges),
        "gaussian_id_digest": tensor_digest(gaussian_ids),
        "hits_per_elevation_row": hit_mask.sum(dim=1).tolist(),
        "range_min_mean_max": (
            [
                float(valid_ranges.min()),
                float(valid_ranges.mean()),
                float(valid_ranges.max()),
            ]
            if valid_ranges.numel()
            else None
        ),
        "alpha_min_mean_max": (
            [
                float(valid_alpha.min()),
                float(valid_alpha.mean()),
                float(valid_alpha.max()),
            ]
            if valid_alpha.numel()
            else None
        ),
    }
    current = {"hit_mask": hit_mask, "ranges": ranges, "ids": gaussian_ids}
    if previous is not None:
        common_hits = hit_mask & previous["hit_mask"]
        record["changed_hit_masks"] = int((hit_mask != previous["hit_mask"]).sum())
        record["changed_gaussian_ids"] = int((gaussian_ids != previous["ids"]).sum())
        record["common_hit_range_abs_max"] = (
            float((ranges[common_hits] - previous["ranges"][common_hits]).abs().max())
            if common_hits.any()
            else None
        )
    return record, current


def write_debug_header(scene: GaussianCloud, config: LidarConfig) -> None:
    """Start a machine-readable diagnostic log that can be shared verbatim."""
    header = {
        "kind": "configuration",
        "backend": BACKEND,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "gaussians": scene.means.shape[0],
        "azimuth_samples": config.azimuth_samples,
        "elevation_samples": config.elevation_samples,
        "near": config.near,
        "far": config.far,
        "sigma_cutoff": config.gaussian_sigma_cutoff,
        "alpha_threshold": config.accumulated_alpha_threshold,
        "camera_width": CAMERA_WIDTH,
        "camera_height": CAMERA_HEIGHT,
        "camera_fx": CAMERA_FX,
        "camera_fy": CAMERA_FY,
        "camera_near": CAMERA_NEAR,
        "camera_far": CAMERA_FAR,
    }
    DEBUG_LOG_PATH.write_text(json.dumps(header) + "\n")


def run_experiment() -> None:
    """Build the static scene once, then scan at every point on the orbit."""
    if ORBIT_RADIUS <= 0 or STEP_SECONDS <= 0 or NUMBER_OF_STEPS < 1:
        raise ValueError("orbit radius, step duration, and step count must be positive")
    if BACKEND != "metal":
        raise ValueError("the camera renderer requires BACKEND='metal'")

    device = select_device(BACKEND)
    scene_cpu = load_scene(PLY_PATH)
    bvh_cpu = create_bvh(scene_cpu)
    initialize_rerun(scene_cpu)

    scene, bvh = scene_cpu.to(device), bvh_cpu.to(device)
    simulator = LidarSimulator(scene, bvh, BACKEND)
    renderer = MetalGaussianRenderer(scene)
    config = create_config()
    camera_intrinsics = create_camera_intrinsics()
    write_debug_header(scene_cpu, config)
    orientation = lidar_orientation(device)
    angular_velocity = math.radians(ANGULAR_VELOCITY_DEGREES_PER_SECOND)
    previous_diagnostics = None

    for step in range(NUMBER_OF_STEPS):
        angle = angular_velocity * STEP_SECONDS * step
        pose = LidarPose(orbit_position(device, angle), orientation)
        scan = simulator.scan(pose, config)
        c2w = camera_c2w(device, angle)
        image = renderer.render(c2w, camera_intrinsics)
        diagnostics, previous_diagnostics = scan_diagnostics(
            step, pose, scan, previous_diagnostics
        )
        with DEBUG_LOG_PATH.open("a") as debug_log:
            debug_log.write(json.dumps(diagnostics) + "\n")
        log_scan(step, pose, scan.valid_points(), c2w, image)
        print(
            f"Scan {step + 1}/{NUMBER_OF_STEPS}: {diagnostics['hits']} hits; "
            f"candidate_overflows={diagnostics['candidate_overflows']}; "
            f"stack_overflows={diagnostics['stack_overflows']}; "
            f"mask={diagnostics['hit_mask_digest']}; "
            f"ranges={diagnostics['range_digest']}"
        )
        if step + 1 < NUMBER_OF_STEPS:
            time.sleep(STEP_SECONDS)

    print(f"Diagnostic log: {DEBUG_LOG_PATH.resolve()}")


if __name__ == "__main__":
    run_experiment()
