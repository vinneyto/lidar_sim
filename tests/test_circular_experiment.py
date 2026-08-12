import importlib.util
import math
from pathlib import Path

import torch


EXPERIMENT_PATH = (
    Path(__file__).parents[1] / "experiments" / "02_circular_scan.py"
)
SPEC = importlib.util.spec_from_file_location("circular_scan", EXPERIMENT_PATH)
assert SPEC is not None and SPEC.loader is not None
circular_scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(circular_scan)


def test_orbit_stays_in_xz_plane_around_scene_center():
    center = torch.tensor([2.0, -3.0, 4.0])
    radius = 2.5

    positions = torch.stack(
        [
            circular_scan.orbit_position(center, radius, angle)
            for angle in (0, math.pi / 2, math.pi, 3 * math.pi / 2)
        ]
    )

    torch.testing.assert_close(positions[:, 1], center[1].expand(4))
    torch.testing.assert_close(
        torch.linalg.vector_norm(positions[:, [0, 2]] - center[[0, 2]], dim=1),
        torch.full((4,), radius),
    )


def test_orbit_polyline_is_closed():
    points = circular_scan.orbit_points(torch.zeros(3), radius=3.0, samples=16)

    assert points.shape == (17, 3)
    torch.testing.assert_close(points[0], points[-1], atol=1e-6, rtol=0)
