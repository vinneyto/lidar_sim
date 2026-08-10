from types import SimpleNamespace

import torch

from gaussian_lidar.ply import load_gaussians


def test_load_gaussforge_cloud(monkeypatch, tmp_path) -> None:
    path = tmp_path / "cloud.ply"
    path.touch()
    cloud = SimpleNamespace(
        positions=[[1, 2, 3], [4, 5, 6]],
        log_scales=[[0, 0, 0], [0, 0, 0]],
        quaternions=[[2, 0, 0, 0], [1, 0, 0, 0]],
    )
    monkeypatch.setattr("gaussian_lidar.ply.gaussforge.load_ply", lambda value: cloud)

    points = load_gaussians(path)

    torch.testing.assert_close(points.centers, torch.tensor([[1, 2, 3], [4, 5, 6.0]]))
    torch.testing.assert_close(points.scales, torch.ones((2, 3)))
    torch.testing.assert_close(points.rotations, torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0.0]]))


def test_plain_point_cloud_gets_default_properties(monkeypatch, tmp_path) -> None:
    path = tmp_path / "points.ply"
    path.touch()
    monkeypatch.setattr(
        "gaussian_lidar.ply.gaussforge.load_ply", lambda value: {"xyz": [[1, 2, 3]]}
    )

    points = load_gaussians(path, default_scale=0.25)

    torch.testing.assert_close(points.scales, torch.full((1, 3), 0.25))
    torch.testing.assert_close(points.rotations, torch.tensor([[1, 0, 0, 0.0]]))
