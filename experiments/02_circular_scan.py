"""Run repeated LiDAR scans while the sensor moves around the scene.

Edit the constants below, then run:

    uv run python experiments/02_circular_scan.py

The one-meter orbit is centered at the coordinate origin and lies in the XZ
plane. Every pose and its return cloud is recorded on Rerun's
``scan`` timeline, so the timeline controls can be used to inspect the motion.
"""

import math
import time
from pathlib import Path

import torch

from gs_lidar import (
    FlatBVH,
    GaussianCloud,
    LidarConfig,
    LidarPose,
    LidarSimulator,
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

    rr.init("gs-lidar-circular-scan", spawn=True)
    rr.send_blueprint(rr.blueprint.Blueprint(rr.blueprint.Spatial3DView(origin="world")))
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


def log_scan(step: int, pose: LidarPose, points: torch.Tensor) -> None:
    """Record one moving sensor pose and return cloud on the scan timeline."""
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
        rr.Points3D(
            points.detach().cpu().numpy(), colors=[0, 255, 120], radii=0.01
        ),
    )


def run_experiment() -> None:
    """Build the static scene once, then scan at every point on the orbit."""
    if ORBIT_RADIUS <= 0 or STEP_SECONDS <= 0 or NUMBER_OF_STEPS < 1:
        raise ValueError("orbit radius, step duration, and step count must be positive")

    device = select_device(BACKEND)
    scene_cpu = load_scene(PLY_PATH)
    bvh_cpu = create_bvh(scene_cpu)
    initialize_rerun(scene_cpu)

    scene, bvh = scene_cpu.to(device), bvh_cpu.to(device)
    simulator = LidarSimulator(scene, bvh, BACKEND)
    config = create_config()
    orientation = lidar_orientation(device)
    angular_velocity = math.radians(ANGULAR_VELOCITY_DEGREES_PER_SECOND)

    for step in range(NUMBER_OF_STEPS):
        angle = angular_velocity * STEP_SECONDS * step
        pose = LidarPose(orbit_position(device, angle), orientation)
        scan = simulator.scan(pose, config)
        if device.type == "mps":
            torch.mps.synchronize()
        log_scan(step, pose, scan.valid_points())
        print(f"Scan {step + 1}/{NUMBER_OF_STEPS}: {int(scan.hit_mask.sum().cpu())} hits")
        if step + 1 < NUMBER_OF_STEPS:
            time.sleep(STEP_SECONDS)


if __name__ == "__main__":
    run_experiment()
