"""Asyncio HTTP server that exposes Prometheus metrics at GET /metrics and POST /reload."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass

from .metrics import MetricsSnapshot, format_prometheus


def parse_metrics_bind(bind: str) -> tuple[str, int]:
    host, port_s = bind.rsplit(":", 1)
    return (host, int(port_s))


@dataclass(slots=True)
class ReloadRequest:
    """Request payload for POST /reload."""

    model_id: str  # "best" or "cand"
    path: str  # absolute path to checkpoint


@dataclass(slots=True)
class ReloadResult:
    """Result of a reload operation."""

    ok: bool
    error: str | None = None


# Callback signature: (model_id, path) -> ReloadResult
ReloadCallback = Callable[[str, str], ReloadResult]


async def start_metrics_server(
    bind: str,
    get_snapshot: Callable[[], MetricsSnapshot],
    reload_callback: ReloadCallback | None = None,
) -> asyncio.AbstractServer:
    """Start the metrics/control HTTP server.

    Args:
        bind: Host:port to bind (e.g. "127.0.0.1:9100").
        get_snapshot: Callback to get current metrics snapshot.
        reload_callback: Optional callback for POST /reload (E13.2S4).
    """
    host, port = parse_metrics_bind(bind)

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            # e.g. "GET /metrics HTTP/1.1" or "POST /reload HTTP/1.1"
            parts = line.decode("ascii", errors="ignore").strip().split()
            if len(parts) < 2:
                await _respond(writer, 400, b'{"error":"bad request"}\n')
                return
            method, path = parts[0], parts[1]

            # Read headers to get Content-Length.
            content_length = 0
            while True:
                h = await reader.readline()
                if not h or h in (b"\r\n", b"\n"):
                    break
                h_str = h.decode("ascii", errors="ignore").strip().lower()
                if h_str.startswith("content-length:"):
                    try:
                        content_length = int(h_str.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            # Route request.
            if method == "GET" and path == "/metrics":
                body = format_prometheus(get_snapshot()).encode("utf-8")
                await _respond(
                    writer,
                    200,
                    body,
                    content_type=b"text/plain; version=0.0.4; charset=utf-8",
                )
            elif method == "GET" and path == "/capabilities":
                # E13.2S5: Return server capabilities for TUI preflight check.
                caps = {
                    "version": "1",
                    "hot_reload": reload_callback is not None,
                }
                await _respond(
                    writer,
                    200,
                    (json.dumps(caps) + "\n").encode(),
                    content_type=b"application/json",
                )
            elif method == "POST" and path == "/reload":
                await _handle_reload(reader, writer, content_length, reload_callback)
            elif method == "GET":
                await _respond(writer, 404, b'{"error":"not found"}\n')
            else:
                await _respond(writer, 405, b'{"error":"method not allowed"}\n')
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    return await asyncio.start_server(handler, host, port)


async def _handle_reload(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    content_length: int,
    reload_callback: ReloadCallback | None,
) -> None:
    """Handle POST /reload request (E13.2S4)."""
    if reload_callback is None:
        await _respond(
            writer,
            501,
            b'{"error":"reload not supported (no callback)"}\n',
            content_type=b"application/json",
        )
        return

    # Read request body.
    if content_length <= 0:
        await _respond(
            writer,
            400,
            b'{"error":"missing request body"}\n',
            content_type=b"application/json",
        )
        return

    body_bytes = await reader.read(content_length)
    try:
        data = json.loads(body_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        await _respond(
            writer,
            400,
            f'{{"error":"invalid JSON: {e}"}}\n'.encode(),
            content_type=b"application/json",
        )
        return

    model_id = data.get("model_id")
    path = data.get("path")

    if model_id not in ("best", "cand"):
        await _respond(
            writer,
            400,
            b'{"error":"model_id must be \\"best\\" or \\"cand\\""}\n',
            content_type=b"application/json",
        )
        return
    if not isinstance(path, str) or not path:
        await _respond(
            writer,
            400,
            b'{"error":"path must be a non-empty string"}\n',
            content_type=b"application/json",
        )
        return

    # Execute reload.
    result = reload_callback(model_id, path)
    if result.ok:
        await _respond(
            writer,
            200,
            b'{"ok":true}\n',
            content_type=b"application/json",
        )
    else:
        err_msg = result.error or "unknown error"
        await _respond(
            writer,
            500,
            f'{{"ok":false,"error":"{_escape_json(err_msg)}"}}\n'.encode(),
            content_type=b"application/json",
        )


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
        500: b"Internal Server Error",
        501: b"Not Implemented",
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


def _escape_json(s: str) -> str:
    """Escape a string for embedding in JSON."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
