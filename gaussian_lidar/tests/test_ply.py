from types import SimpleNamespace

import torch

from gaussian_lidar.ply import load_gaussians


def test_load_gaussforge_cloud(monkeypatch) -> None:
    cloud = SimpleNamespace(
        positions=[[1, 2, 3], [4, 5, 6]],
        log_scales=[[0, 0, 0], [0, 0, 0]],
        quaternions=[[2, 0, 0, 0], [1, 0, 0, 0]],
    )
    received = []
    monkeypatch.setattr(
        "gaussian_lidar.ply.gaussforge.load_ply", lambda value: received.append(value) or cloud
    )

    points = load_gaussians(b"ply\n...")

    assert received == [b"ply\n..."]
    torch.testing.assert_close(points.centers, torch.tensor([[1, 2, 3], [4, 5, 6.0]]))
    torch.testing.assert_close(points.scales, torch.ones((2, 3)))
    torch.testing.assert_close(points.rotations, torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0.0]]))


def test_plain_point_cloud_gets_default_properties(monkeypatch) -> None:
    monkeypatch.setattr(
        "gaussian_lidar.ply.gaussforge.load_ply", lambda value: {"xyz": [[1, 2, 3]]}
    )

    points = load_gaussians(b"ply", default_scale=0.25)

    torch.testing.assert_close(points.scales, torch.full((1, 3), 0.25))
    torch.testing.assert_close(points.rotations, torch.tensor([[1, 0, 0, 0.0]]))
