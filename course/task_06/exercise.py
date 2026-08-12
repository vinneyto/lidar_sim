"""Run repeated LiDAR scans while the sensor moves around the scene.

Edit the constants below, then run:

    uv run python experiments/02_circular_scan.py

The one-meter orbit is centered at the coordinate origin and lies in the XZ
plane. Every pose and its return cloud is recorded on Rerun's
``scan`` timeline, so the timeline controls can be used to inspect the motion.
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
    FlatBVH,
    GaussianCloud,
    LidarConfig,
    LidarPose,
    LidarScan,
    LidarSimulator,
    generate_rays,
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
DEBUG_LOG_PATH = Path("circular_scan_debug.jsonl")

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
    # TODO: откройте Rerun и запишите статические гауссианы и орбиту
    raise NotImplementedError


def main() -> None:
    scene = load_scene(PLY_PATH)
    initialize_rerun(scene)
    print(f"В Rerun отправлено гауссиан: {scene.means.shape[0]}")
    print(f"В Rerun отправлена орбита из {orbit_points().shape[0]} точек")


if __name__ == "__main__":
    main()
