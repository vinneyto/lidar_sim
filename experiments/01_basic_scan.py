"""Run a basic virtual LiDAR experiment.

Edit the constants in the configuration section below, then run:

    uv run python experiments/01_basic_scan.py

Angles are specified in degrees. The pose uses intrinsic Z-Y-X
(yaw-pitch-roll) rotations in the +X forward, +Y left, +Z up frame.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from gs_lidar import (
    FlatBVH,
    GaussianCloud,
    LidarConfig,
    LidarPose,
    LidarScan,
    LidarSimulator,
    build_bvh,
    gaussian_aabbs,
    load_gaussian_ply,
)
from gs_lidar.rerun_viewer import visualize

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

LIDAR_POSITION = (0.0, 0.0, 0.0)
LIDAR_YAW_DEGREES = 0.0
LIDAR_PITCH_DEGREES = 0.0
LIDAR_ROLL_DEGREES = 0.0

# Rerun turntable up direction. Canonical simulator coordinates use "+Z";
# common camera/reconstruction coordinates may instead need "+Y" or "-Y".
RERUN_UP_AXIS = "+Z"
RERUN_AXIS_LENGTH = 1.0


@dataclass(frozen=True)
class ExperimentTimings:
    ply_load_seconds: float
    bvh_build_seconds: float
    device_upload_seconds: float
    trace_seconds: float


def load_scene(path: Path) -> GaussianCloud:
    """Load and normalize a canonical 3DGS PLY scene."""
    if not path.is_file():
        raise FileNotFoundError(
            f"PLY scene not found: {path}. Set PLY_PATH at the top of this file."
        )
    return load_gaussian_ply(path)


def create_bvh(scene: GaussianCloud, sigma_cutoff: float) -> FlatBVH:
    """Build the finite-support Gaussian BVH once on the CPU."""
    bbox_min, bbox_max = gaussian_aabbs(
        scene.means,
        scene.scales,
        scene.rotations,
        sigma_cutoff,
    )
    return build_bvh(bbox_min, bbox_max)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply scalar-first (w, x, y, z) quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def create_orientation(yaw: float, pitch: float, roll: float) -> torch.Tensor:
    """Create an intrinsic Z-Y-X orientation from angles in degrees."""
    yaw_half, pitch_half, roll_half = (
        math.radians(value) / 2.0 for value in (yaw, pitch, roll)
    )
    yaw_rotation = torch.tensor([math.cos(yaw_half), 0.0, 0.0, math.sin(yaw_half)])
    pitch_rotation = torch.tensor(
        [math.cos(pitch_half), 0.0, math.sin(pitch_half), 0.0]
    )
    roll_rotation = torch.tensor([math.cos(roll_half), math.sin(roll_half), 0.0, 0.0])
    return quaternion_multiply(
        quaternion_multiply(yaw_rotation, pitch_rotation), roll_rotation
    ).float()


def create_lidar_pose(device: torch.device) -> LidarPose:
    """Create the sensor pose directly on the simulation device."""
    orientation = create_orientation(
        LIDAR_YAW_DEGREES,
        LIDAR_PITCH_DEGREES,
        LIDAR_ROLL_DEGREES,
    )
    return LidarPose(
        position=torch.tensor(LIDAR_POSITION, dtype=torch.float32, device=device),
        orientation=orientation.to(device),
    )


def create_lidar_config() -> LidarConfig:
    """Create the angular and opacity configuration for this experiment."""
    return LidarConfig(
        azimuth_samples=AZIMUTH_SAMPLES,
        elevation_samples=ELEVATION_SAMPLES,
        near=NEAR,
        far=FAR,
        gaussian_sigma_cutoff=SIGMA_CUTOFF,
        accumulated_alpha_threshold=ALPHA_THRESHOLD,
    )


def select_device(backend: str) -> torch.device:
    """Validate the selected backend and return its torch device."""
    if backend == "cpu":
        return torch.device("cpu")
    if backend != "metal":
        raise ValueError("BACKEND must be either 'cpu' or 'metal'")
    if not torch.backends.mps.is_available():
        raise RuntimeError("BACKEND='metal' requires an available PyTorch MPS backend")
    return torch.device("mps")


def upload_static_data(
    scene: GaussianCloud, bvh: FlatBVH, device: torch.device
) -> tuple[GaussianCloud, FlatBVH]:
    """Move the immutable scene and BVH to the target device exactly once."""
    return scene.to(device), bvh.to(device)


def run_scan(
    scene: GaussianCloud,
    bvh: FlatBVH,
    pose: LidarPose,
    config: LidarConfig,
) -> LidarScan:
    """Run one scan while keeping all computational outputs on their device."""
    simulator = LidarSimulator(scene, bvh, BACKEND)
    return simulator.scan(pose, config)


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous MPS work before recording benchmark timings."""
    if device.type == "mps":
        torch.mps.synchronize()


def print_statistics(
    scene: GaussianCloud,
    bvh: FlatBVH,
    scan: LidarScan,
    timings: ExperimentTimings,
) -> None:
    """Print scene, scan, overflow, and stage timing statistics."""
    hit_count = int(scan.hit_mask.sum().cpu())
    ray_count = scan.hit_mask.numel()
    hit_percentage = 100.0 * hit_count / ray_count
    mrays_per_second = ray_count / max(timings.trace_seconds, 1e-9) / 1e6

    print(f"Gaussians: {scene.means.shape[0]}")
    print(f"BVH nodes: {bvh.bbox_min.shape[0]}")
    print(f"Rays: {ray_count} ({ELEVATION_SAMPLES} x {AZIMUTH_SAMPLES})")
    print(f"Hits: {hit_count} ({hit_percentage:.1f}%)")
    print(f"PLY load: {timings.ply_load_seconds:.3f} s")
    print(f"BVH build: {timings.bvh_build_seconds:.3f} s")
    print(f"Device upload: {timings.device_upload_seconds:.3f} s")
    print(f"Trace: {timings.trace_seconds:.3f} s ({mrays_per_second:.3f} MRays/s)")

    candidate_overflow_count = scan.candidate_overflow_count
    stack_overflow_count = scan.bvh_stack_overflow_count
    if candidate_overflow_count is not None and stack_overflow_count is not None:
        candidate_overflows = int(candidate_overflow_count.cpu())
        stack_overflows = int(stack_overflow_count.cpu())
        print(f"Candidate overflows: {candidate_overflows}")
        print(f"BVH stack overflows: {stack_overflows}")


def run_experiment() -> None:
    """Execute the complete load → BVH → upload → scan → visualize pipeline."""
    device = select_device(BACKEND)

    started = time.perf_counter()
    scene = load_scene(PLY_PATH)
    ply_loaded = time.perf_counter()

    bvh = create_bvh(scene, SIGMA_CUTOFF)
    bvh_built = time.perf_counter()

    scene, bvh = upload_static_data(scene, bvh, device)
    pose = create_lidar_pose(device)
    config = create_lidar_config()
    synchronize(device)
    data_uploaded = time.perf_counter()

    scan = run_scan(scene, bvh, pose, config)
    synchronize(device)
    scan_finished = time.perf_counter()

    timings = ExperimentTimings(
        ply_load_seconds=ply_loaded - started,
        bvh_build_seconds=bvh_built - ply_loaded,
        device_upload_seconds=data_uploaded - bvh_built,
        trace_seconds=scan_finished - data_uploaded,
    )
    print_statistics(scene, bvh, scan, timings)
    visualize(
        scene,
        pose,
        scan,
        up_axis=RERUN_UP_AXIS,
        axis_length=RERUN_AXIS_LENGTH,
    )


if __name__ == "__main__":
    run_experiment()
