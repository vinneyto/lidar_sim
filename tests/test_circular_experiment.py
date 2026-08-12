import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_scan_diagnostics_reports_changes_and_overflows():
    pose = circular_scan.LidarPose(
        torch.zeros(3), torch.tensor([1.0, 0.0, 0.0, 0.0])
    )
    first = SimpleNamespace(
        hit_mask=torch.tensor([[True, False]]),
        ranges=torch.tensor([[1.0, float("inf")]]),
        gaussian_ids=torch.tensor([[3, -1]]),
        accumulated_alpha=torch.tensor([[0.6, 0.1]]),
        candidate_overflow_count=torch.tensor([2]),
        bvh_stack_overflow_count=torch.tensor([1]),
    )
    second = SimpleNamespace(
        hit_mask=torch.tensor([[True, True]]),
        ranges=torch.tensor([[1.25, 2.0]]),
        gaussian_ids=torch.tensor([[4, 5]]),
        accumulated_alpha=torch.tensor([[0.7, 0.8]]),
        candidate_overflow_count=torch.tensor([0]),
        bvh_stack_overflow_count=torch.tensor([0]),
    )

    first_record, previous = circular_scan.scan_diagnostics(0, pose, first, None)
    second_record, _ = circular_scan.scan_diagnostics(1, pose, second, previous)

    assert first_record["candidate_overflows"] == 2
    assert first_record["stack_overflows"] == 1
    assert first_record["hits_per_elevation_row"] == [1]
    assert second_record["changed_hit_masks"] == 1
    assert second_record["changed_gaussian_ids"] == 2
    assert second_record["common_hit_range_abs_max"] == pytest.approx(0.25)
