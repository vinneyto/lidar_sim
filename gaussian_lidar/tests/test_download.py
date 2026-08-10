import httpx
import pytest

from gaussian_lidar.download import download_ply


@pytest.mark.asyncio
async def test_downloads_ply_into_memory() -> None:
    async def response(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "application/octet-stream"
        return httpx.Response(200, content=b"ply\nbody", request=request)

    payload = await download_ply(
        "https://assets.example/scene.ply", transport=httpx.MockTransport(response)
    )

    assert payload == b"ply\nbody"


@pytest.mark.asyncio
async def test_rejects_response_over_limit() -> None:
    async def response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"too large", request=request)

    with pytest.raises(ValueError, match="download limit"):
        await download_ply(
            "https://assets.example/scene.ply",
            max_bytes=3,
            transport=httpx.MockTransport(response),
        )


@pytest.mark.asyncio
async def test_rejects_non_http_url() -> None:
    with pytest.raises(ValueError, match="http"):
        await download_ply("file:///tmp/scene.ply")
