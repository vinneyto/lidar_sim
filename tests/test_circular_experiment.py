import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from gs_lidar import LidarConfig, LidarPose, generate_rays


EXPERIMENT_PATH = (
    Path(__file__).parents[1] / "experiments" / "02_circular_scan.py"
)
SPEC = importlib.util.spec_from_file_location("circular_scan", EXPERIMENT_PATH)
assert SPEC is not None and SPEC.loader is not None
circular_scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(circular_scan)


def test_orbit_is_one_meter_xz_circle_around_origin():
    positions = torch.stack(
        [
            circular_scan.orbit_position(torch.device("cpu"), angle)
            for angle in (0, math.pi / 2, math.pi, 3 * math.pi / 2)
        ]
    )

    assert circular_scan.ORBIT_CENTER == (0.0, 0.0, 0.0)
    assert circular_scan.ORBIT_RADIUS == 1.0
    torch.testing.assert_close(positions[:, 1], torch.zeros(4))
    torch.testing.assert_close(
        torch.linalg.vector_norm(positions[:, [0, 2]], dim=1),
        torch.ones(4),
    )


def test_orbit_polyline_is_closed():
    points = circular_scan.orbit_points()

    assert points.shape == (circular_scan.RING_SAMPLES + 1, 3)
    torch.testing.assert_close(points[0], points[-1], atol=1e-6, rtol=0)


def test_lidar_azimuth_plane_matches_xz_orbit_plane():
    pose = LidarPose(
        torch.zeros(3), circular_scan.lidar_orientation(torch.device("cpu"))
    )
    config = LidarConfig(
        azimuth_min=0,
        azimuth_max=2 * math.pi,
        elevation_min=0,
        elevation_max=0,
        azimuth_samples=4,
        elevation_samples=1,
    )

    _, directions = generate_rays(pose, config)

    torch.testing.assert_close(directions[:, 1], torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        directions,
        torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [-1.0, 0, 0], [0, 0, 1.0]]),
        atol=1e-6,
        rtol=0,
    )


def test_log_scan_uses_current_rerun_timeline_api(monkeypatch):
    times = []
    logged = []

    class Points3D:
        def __init__(self, points, **kwargs):
            self.points = points
            self.kwargs = kwargs

    rerun = SimpleNamespace(
        Points3D=Points3D,
        set_time=lambda timeline, *, sequence: times.append((timeline, sequence)),
        log=lambda path, value: logged.append((path, value)),
    )
    monkeypatch.setitem(sys.modules, "rerun", rerun)

    circular_scan.log_scan(
        7,
        circular_scan.LidarPose(
            torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 0.0, 0.0, 0.0])
        ),
        torch.tensor([[4.0, 5.0, 6.0]]),
    )

    assert times == [("scan", 7)]
    assert [path for path, _ in logged] == [
        "world/lidar/position",
        "world/lidar/returns",
    ]
