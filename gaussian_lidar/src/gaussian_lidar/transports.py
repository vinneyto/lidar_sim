from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path

from websockets.datastructures import Headers
from websockets.http11 import Response

from .service import LidarService

LOG = logging.getLogger(__name__)


async def serve_websocket(
    service: LidarService, host: str, port: int, ply_path: Path
) -> None:
    from websockets.asyncio.server import serve

    frontend = Path(__file__).with_name("frontend")

    async def process_request(connection, request) -> Response | None:
        path = request.path.split("?", 1)[0]
        if path == "/ws":
            return None
        if path == "/scene.ply":
            file_path = ply_path
        elif path == "/":
            file_path = frontend / "index.html"
        elif path.startswith("/assets/"):
            file_path = frontend / path.removeprefix("/")
        else:
            return Response(404, "Not Found", Headers(), b"Not Found\n")

        try:
            body = await asyncio.to_thread(file_path.read_bytes)
        except FileNotFoundError:
            message = b"Frontend is not built. Run `npm run build` in frontend/.\n"
            return Response(404, "Not Found", Headers(), message)
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        headers = Headers(
            {
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "Cache-Control": "no-cache",
            }
        )
        return Response(200, "OK", headers, body)

    async def connection(websocket) -> None:
        async for message in websocket:
            await websocket.send(await service.handle(message))

    LOG.info("WebSocket listening on ws://%s:%d", host, port)
    async with serve(
        connection,
        host,
        port,
        max_size=16 * 1024 * 1024,
        process_request=process_request,
    ):
        await asyncio.Future()


async def serve_rabbitmq(service: LidarService, url: str, queue_name: str) -> None:
    try:
        import aio_pika
    except ImportError as error:
        raise RuntimeError("RabbitMQ transport requires: uv sync --extra rabbitmq") from error

    connection = await aio_pika.connect_robust(url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await channel.declare_queue(queue_name, durable=True)
    LOG.info("RabbitMQ consuming queue %s", queue_name)

    async def consume(message) -> None:
        async with message.process():
            response = await service.handle(message.body)
            if message.reply_to:
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        response.encode(),
                        correlation_id=message.correlation_id,
                        content_type="application/json",
                    ),
                    routing_key=message.reply_to,
                )

    await queue.consume(consume)
    try:
        await asyncio.Future()
    finally:
        await connection.close()
