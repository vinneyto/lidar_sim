from __future__ import annotations

import asyncio
import json
from typing import Protocol

import torch

from .rays import make_rays, parse_scan_request


class Tracer(Protocol):
    device: torch.device

    def trace(
        self, origins: torch.Tensor, directions: torch.Tensor, max_distance: float
    ) -> torch.Tensor: ...


class LidarService:
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    async def handle(self, payload: str | bytes) -> str:
        try:
            data = json.loads(payload)
            request = parse_scan_request(data)
            origins, directions = make_rays(request, self._tracer.device)
            hits = await asyncio.to_thread(
                self._tracer.trace, origins, directions, request.max_distance
            )
            hits = hits.to("cpu")
            valid = torch.isfinite(hits[:, 3])
            indices = torch.nonzero(valid, as_tuple=False).flatten()
            return json.dumps(
                {
                    "ok": True,
                    "width": request.width,
                    "height": request.height,
                    "points": hits[valid, :3].tolist(),
                    "distances": hits[valid, 3].tolist(),
                    "ray_indices": indices.tolist(),
                },
                separators=(",", ":"),
            )
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return json.dumps({"ok": False, "error": str(error)}, separators=(",", ":"))
