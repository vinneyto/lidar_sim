from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging

from .bvh import build_bvh
from .download import download_ply
from .metal import MetalTracer
from .ply import load_gaussians
from .service import LidarService
from .transports import serve_rabbitmq, serve_websocket


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Emulate a LiDAR over a Gaussian PLY scene")
    result.add_argument("ply_url", help="HTTP(S) URL of a Gaussian Splatting PLY")
    result.add_argument("--transport", choices=("websocket", "rabbitmq"), default="websocket")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--rabbitmq-url", default="amqp://guest:guest@localhost/")
    result.add_argument("--queue", default="gaussian-lidar.scan")
    result.add_argument("--default-scale", type=float, default=0.01)
    result.add_argument("--sigma-extent", type=float, default=3.0)
    result.add_argument("--leaf-size", type=int, default=8)
    result.add_argument("--download-timeout", type=float, default=60.0)
    result.add_argument("--max-download-mib", type=int, default=2048)
    result.add_argument("--verbose", action="store_true")
    return result


async def run(args: argparse.Namespace) -> None:
    logging.info("Downloading PLY from %s", args.ply_url)
    payload = await download_ply(
        args.ply_url,
        timeout=args.download_timeout,
        max_bytes=args.max_download_mib * 1024 * 1024,
    )
    logging.info("Downloaded %.1f MiB; parsing with GaussForge", len(payload) / 1024 / 1024)
    gaussians = load_gaussians(payload, args.default_scale)
    logging.info("Loaded %d Gaussians; building BVH", len(gaussians.centers))
    bvh = build_bvh(gaussians, args.sigma_extent, args.leaf_size)
    logging.info("Built BVH with %d nodes", len(bvh.metadata))
    service = LidarService(MetalTracer(gaussians, bvh, args.sigma_extent))
    if args.transport == "websocket":
        await serve_websocket(service, args.host, args.port)
    else:
        await serve_rabbitmq(service, args.rabbitmq_url, args.queue)


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
