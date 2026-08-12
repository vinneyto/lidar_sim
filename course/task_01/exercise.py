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
    # TODO: проверьте путь и загрузите canonical 3DGS PLY
    raise NotImplementedError


def main() -> None:
    scene = load_scene(PLY_PATH)
    print(f"Загружено гауссиан: {scene.means.shape[0]}")
    print(f"Диапазон opacity: {scene.opacities.min():.4f} .. {scene.opacities.max():.4f}")


if __name__ == "__main__":
    main()
