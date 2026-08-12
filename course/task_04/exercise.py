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
    load_gaussian_scene,
)

# -----------------------------------------------------------------------------
# Experiment configuration — edit these values before running the script.
# -----------------------------------------------------------------------------

SCENE_PATH = Path("scene.sog")
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
    """Load a Gaussian splat scene."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Gaussian scene not found: {path}. Set SCENE_PATH at the top of this file."
        )
    return load_gaussian_scene(path)


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
    # TODO: верните позицию на XZ-окружности
    raise NotImplementedError


def orbit_points() -> torch.Tensor:
    """Create a closed polyline describing the fixed sensor orbit."""
    center = torch.tensor(ORBIT_CENTER, dtype=torch.float32)
    angles = torch.linspace(0, 2 * math.pi, RING_SAMPLES + 1)
    points = center.expand(RING_SAMPLES + 1, 3).clone()
    points[:, 0] += ORBIT_RADIUS * torch.cos(angles)
    points[:, 2] += ORBIT_RADIUS * torch.sin(angles)
    return points


def main() -> None:
    device = select_device(BACKEND)
    samples = torch.stack([orbit_position(device, a) for a in (0, math.pi / 2, math.pi)])
    print("Три позиции орбиты:\n", samples.cpu())
    print(f"Точек полилинии: {orbit_points().shape[0]}; замкнута: {torch.allclose(orbit_points()[0], orbit_points()[-1])}")


if __name__ == "__main__":
    main()
