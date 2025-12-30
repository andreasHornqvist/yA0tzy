"""Tests for the /capabilities endpoint (E13.2S5)."""

from __future__ import annotations

import asyncio
import json


def test_capabilities_with_reload_callback() -> None:
    """GET /capabilities returns hot_reload=true when callback is set."""
    from yatzy_az.server.metrics import MetricsSnapshot
    from yatzy_az.server.metrics_server import ReloadResult, start_metrics_server

    async def run_test() -> None:
        def dummy_snapshot() -> MetricsSnapshot:
            return MetricsSnapshot()

        def dummy_reload(model_id: str, path: str) -> ReloadResult:
            return ReloadResult(ok=True)

        # Start server on ephemeral port.
        server = await start_metrics_server(
            bind="127.0.0.1:0",
            get_snapshot=dummy_snapshot,
            reload_callback=dummy_reload,
        )
        sock = server.sockets[0]
        port = sock.getsockname()[1]

        # Make HTTP request using raw sockets.
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /capabilities HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        response = await reader.read(4096)
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

        # Parse response.
        lines = response.decode().split("\r\n")
        assert lines[0] == "HTTP/1.1 200 OK"
        body = lines[-1]
        data = json.loads(body)
        assert data["version"] == "1"
        assert data["hot_reload"] is True

    asyncio.run(run_test())


def test_capabilities_without_reload_callback() -> None:
    """GET /capabilities returns hot_reload=false when no callback."""
    from yatzy_az.server.metrics import MetricsSnapshot
    from yatzy_az.server.metrics_server import start_metrics_server

    async def run_test() -> None:
        def dummy_snapshot() -> MetricsSnapshot:
            return MetricsSnapshot()

        # Start server on ephemeral port without reload callback.
        server = await start_metrics_server(
            bind="127.0.0.1:0",
            get_snapshot=dummy_snapshot,
            reload_callback=None,
        )
        sock = server.sockets[0]
        port = sock.getsockname()[1]

        # Make HTTP request.
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /capabilities HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        response = await reader.read(4096)
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

        # Parse response.
        lines = response.decode().split("\r\n")
        assert lines[0] == "HTTP/1.1 200 OK"
        body = lines[-1]
        data = json.loads(body)
        assert data["version"] == "1"
        assert data["hot_reload"] is False

    asyncio.run(run_test())

