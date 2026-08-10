from __future__ import annotations

import math

import torch

from .model import ScanRequest


def parse_scan_request(data: dict[str, object]) -> ScanRequest:
    if not isinstance(data, dict):
        raise ValueError("request must be a JSON object")
    sensor = data.get("sensor", data)
    scan = data.get("scan", {})
    if not isinstance(sensor, dict) or not isinstance(scan, dict):
        raise ValueError("sensor and scan must be objects")
    position = torch.as_tensor(sensor.get("position"), dtype=torch.float32)
    quaternion = torch.as_tensor(sensor.get("quaternion", [1, 0, 0, 0]), dtype=torch.float32)
    if position.shape != (3,) or not torch.isfinite(position).all():
        raise ValueError("position must be an array of three finite numbers")
    if quaternion.shape != (4,) or not torch.isfinite(quaternion).all():
        raise ValueError("quaternion must be an array [w, x, y, z]")
    norm = torch.linalg.vector_norm(quaternion)
    if norm < 1e-12:
        raise ValueError("quaternion must be non-zero")
    width, height = int(scan.get("width", 1024)), int(scan.get("height", 64))
    if width < 1 or height < 1 or width * height > 4_194_304:
        raise ValueError("scan dimensions must be positive and contain at most 4194304 rays")
    horizontal_fov = float(scan.get("horizontal_fov_deg", 360.0))
    vertical_fov = float(scan.get("vertical_fov_deg", 30.0))
    max_distance = float(scan.get("max_distance", 200.0))
    if not (0.0 <= horizontal_fov <= 360.0 and 0.0 <= vertical_fov <= 180.0):
        raise ValueError("FOV must be finite and within 0..360 horizontal and 0..180 vertical")
    if not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be a positive finite number")
    return ScanRequest(
        position, quaternion / norm, width, height, horizontal_fov, vertical_fov, max_distance
    )


def make_rays(
    request: ScanRequest, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create row-major Torch ray tensors, with +X as the sensor forward axis."""
    dtype = torch.float32
    horizontal_fov = torch.deg2rad(torch.tensor(request.horizontal_fov_deg, dtype=dtype))
    vertical_fov = torch.deg2rad(torch.tensor(request.vertical_fov_deg, dtype=dtype))
    yaw = torch.linspace(-horizontal_fov / 2, horizontal_fov / 2, request.width + 1, dtype=dtype)[
        :-1
    ]
    if request.height == 1:
        pitch = torch.zeros(1, dtype=dtype)
    else:
        pitch = torch.linspace(-vertical_fov / 2, vertical_fov / 2, request.height, dtype=dtype)
    pitch_grid, yaw_grid = torch.meshgrid(pitch, yaw, indexing="ij")
    local = torch.stack(
        (
            torch.cos(pitch_grid) * torch.cos(yaw_grid),
            torch.cos(pitch_grid) * torch.sin(yaw_grid),
            torch.sin(pitch_grid),
        ),
        dim=-1,
    )
    w, x, y, z = request.quaternion
    rotation = torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        )
    )
    directions = (local.reshape(-1, 3) @ rotation.T).to(device).contiguous()
    origins = request.position.to(device).expand_as(directions).contiguous()
    return origins, directions
