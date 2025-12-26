"""Asyncio HTTP server that exposes Prometheus metrics at GET /metrics."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from .metrics import MetricsSnapshot, format_prometheus


def parse_metrics_bind(bind: str) -> tuple[str, int]:
    host, port_s = bind.rsplit(":", 1)
    return (host, int(port_s))


async def start_metrics_server(
    bind: str, get_snapshot: Callable[[], MetricsSnapshot]
) -> asyncio.AbstractServer:
    host, port = parse_metrics_bind(bind)

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            # e.g. "GET /metrics HTTP/1.1"
            parts = line.decode("ascii", errors="ignore").strip().split()
            if len(parts) < 2:
                await _respond(writer, 400, b"bad request\n")
                return
            method, path = parts[0], parts[1]

            # Drain headers until empty line.
            while True:
                h = await reader.readline()
                if not h or h in (b"\r\n", b"\n"):
                    break

            if method != "GET":
                await _respond(writer, 405, b"method not allowed\n")
                return
            if path != "/metrics":
                await _respond(writer, 404, b"not found\n")
                return

            body = format_prometheus(get_snapshot()).encode("utf-8")
            await _respond(
                writer,
                200,
                body,
                content_type=b"text/plain; version=0.0.4; charset=utf-8",
            )
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    return await asyncio.start_server(handler, host, port)


async def _respond(
    writer: asyncio.StreamWriter,
    status: int,
    body: bytes,
    *,
    content_type: bytes = b"text/plain; charset=utf-8",
) -> None:
    reason = {
        200: b"OK",
        400: b"Bad Request",
        404: b"Not Found",
        405: b"Method Not Allowed",
    }.get(status, b"OK")
    hdr = (
        b"HTTP/1.1 "
        + str(status).encode("ascii")
        + b" "
        + reason
        + b"\r\n"
        + b"Content-Type: "
        + content_type
        + b"\r\n"
        + b"Content-Length: "
        + str(len(body)).encode("ascii")
        + b"\r\n"
        + b"Connection: close\r\n"
        + b"\r\n"
    )
    writer.write(hdr + body)
    await writer.drain()
