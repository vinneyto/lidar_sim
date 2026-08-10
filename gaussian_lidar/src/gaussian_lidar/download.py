from __future__ import annotations

from urllib.parse import urlparse

import httpx


async def download_ply(
    url: str,
    *,
    timeout: float = 60.0,
    max_bytes: int = 2_000_000_000,
    transport: httpx.AsyncBaseTransport | None = None,
) -> bytes:
    """Download an HTTP(S) PLY into memory with a hard response-size limit."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("PLY URL must be an absolute http:// or https:// URL")
    if timeout <= 0:
        raise ValueError("download timeout must be positive")
    if max_bytes <= 0:
        raise ValueError("max download size must be positive")

    timeout_config = httpx.Timeout(timeout)
    async with (
        httpx.AsyncClient(
            follow_redirects=True, timeout=timeout_config, transport=transport
        ) as client,
        client.stream("GET", url, headers={"Accept": "application/octet-stream"}) as response,
    ):
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        if content_length is not None and int(content_length) > max_bytes:
            raise ValueError(f"PLY exceeds the {max_bytes}-byte download limit")
        payload = bytearray()
        async for chunk in response.aiter_bytes():
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise ValueError(f"PLY exceeds the {max_bytes}-byte download limit")
    if not payload:
        raise ValueError("downloaded PLY is empty")
    return bytes(payload)
