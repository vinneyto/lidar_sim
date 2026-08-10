import json

import pytest
import torch

from gaussian_lidar.bvh import build_bvh
from gaussian_lidar.model import Gaussians
from gaussian_lidar.rays import make_rays, parse_scan_request
from gaussian_lidar.service import LidarService


def scene() -> Gaussians:
    return Gaussians(
        torch.tensor([[0, 0, 0], [10, 1, 2], [-3, 2, 1]], dtype=torch.float32),
        torch.ones((3, 3), dtype=torch.float32),
        torch.tensor([[1, 0, 0, 0]] * 3, dtype=torch.float32),
    )


def test_bvh_contains_every_primitive() -> None:
    bvh = build_bvh(scene(), sigma_extent=2, leaf_size=1)
    assert len(bvh.metadata) == 5
    assert sorted(bvh.primitive_indices.tolist()) == [0, 1, 2]
    assert torch.all(torch.tensor([11, 3, 4]) < bvh.bounds_max[0] + 1e-5)
    assert torch.all(bvh.bounds_min[0] - 1e-5 < torch.tensor([-5, -2, -2]))


def test_identity_pose_generates_forward_ray() -> None:
    request = parse_scan_request(
        {
            "position": [1, 2, 3],
            "scan": {"width": 1, "height": 1, "horizontal_fov_deg": 0, "vertical_fov_deg": 0},
        }
    )
    origins, directions = make_rays(request)
    torch.testing.assert_close(origins, torch.tensor([[1, 2, 3]], dtype=torch.float32))
    torch.testing.assert_close(directions, torch.tensor([[1, 0, 0]], dtype=torch.float32))


def test_rejects_excessive_scan() -> None:
    with pytest.raises(ValueError, match="at most"):
        parse_scan_request({"position": [0, 0, 0], "scan": {"width": 5000, "height": 5000}})


@pytest.mark.asyncio
async def test_service_returns_only_hits() -> None:
    class FakeTracer:
        device = torch.device("cpu")

        def trace(self, origins, directions, max_distance):
            return torch.tensor([[1, 2, 3, 4], [torch.nan] * 4], dtype=torch.float32)

    response = json.loads(
        await LidarService(FakeTracer()).handle(
            json.dumps({"position": [0, 0, 0], "scan": {"width": 2, "height": 1}})
        )
    )
    assert response == {
        "ok": True,
        "width": 2,
        "height": 1,
        "points": [[1.0, 2.0, 3.0]],
        "distances": [4.0],
        "ray_indices": [0],
    }
