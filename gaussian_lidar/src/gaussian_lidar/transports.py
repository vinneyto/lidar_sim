from __future__ import annotations

import asyncio
import logging

from .service import LidarService

LOG = logging.getLogger(__name__)


async def serve_websocket(service: LidarService, host: str, port: int) -> None:
    from websockets.asyncio.server import serve

    async def connection(websocket) -> None:
        async for message in websocket:
            await websocket.send(await service.handle(message))

    LOG.info("WebSocket listening on ws://%s:%d", host, port)
    async with serve(connection, host, port, max_size=16 * 1024 * 1024):
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
