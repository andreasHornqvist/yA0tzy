"""Asyncio UDS inference server with dynamic batching."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import pathlib
import signal
import time
from dataclasses import dataclass

from .batcher import Batcher
from .metrics import MetricsSnapshot, now_s
from .metrics_server import start_metrics_server
from .model import build_model
from .protocol_v1 import (
    DecodeError,
    InferResponseV1,
    decode_request_v1,
    encode_response_v1,
    read_frame,
    write_frame,
)


def _parse_bind(bind: str) -> tuple[str, str]:
    if bind.startswith("unix://"):
        return ("unix", bind[len("unix://") :])
    if bind.startswith("tcp://"):
        return ("tcp", bind[len("tcp://") :])
    raise ValueError(f"unsupported bind scheme: {bind}")


@dataclass(slots=True)
class ServerConfig:
    bind: str
    device: str
    max_batch: int
    max_wait_us: int
    print_stats_every_s: float
    best_id: int
    cand_id: int
    best_spec: str
    cand_spec: str
    metrics_bind: str
    metrics_disable: bool


async def _handle_conn(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, batcher: Batcher
) -> None:
    try:
        while True:
            payload = await read_frame(reader)

            try:
                req = decode_request_v1(payload)
            except DecodeError:
                # Protocol violation: close the connection.
                break

            resp: InferResponseV1 = await batcher.enqueue(req)
            out_payload = encode_response_v1(resp)
            await write_frame(writer, out_payload)
    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def serve(config: ServerConfig) -> None:
    scheme, addr = _parse_bind(config.bind)

    start_s = now_s()
    model_by_id = {
        int(config.best_id): build_model(config.best_spec, device=config.device),
        int(config.cand_id): build_model(config.cand_spec, device=config.device),
    }
    batcher = Batcher(model_by_id, max_batch=config.max_batch, max_wait_us=config.max_wait_us)
    batcher_task = asyncio.create_task(batcher.run())

    def snapshot() -> MetricsSnapshot:
        return MetricsSnapshot(
            now_s=now_s(),
            start_s=start_s,
            queue_depth=batcher.queue_depth,
            batcher=batcher.stats,
        )

    stop_ev = asyncio.Event()

    def _stop() -> None:
        stop_ev.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _stop)

    sock_path: pathlib.Path | None = None
    if scheme == "unix":
        sock_path = pathlib.Path(addr)
        if sock_path.exists():
            sock_path.unlink()
        sock_path.parent.mkdir(parents=True, exist_ok=True)
        server = await asyncio.start_unix_server(
            lambda r, w: _handle_conn(r, w, batcher), path=str(sock_path)
        )
        # Restrict perms a bit (best-effort).
        with contextlib.suppress(Exception):
            os.chmod(sock_path, 0o600)
    else:
        host, port_s = addr.rsplit(":", 1)
        server = await asyncio.start_server(
            lambda r, w: _handle_conn(r, w, batcher), host, int(port_s)
        )

    try:
        async with server:
            metrics_srv = None
            if not config.metrics_disable:
                metrics_srv = await start_metrics_server(config.metrics_bind, snapshot)
            stats_task = asyncio.create_task(_print_stats_loop(batcher, config.print_stats_every_s))
            try:
                await stop_ev.wait()
            finally:
                stats_task.cancel()
                with contextlib.suppress(Exception):
                    await stats_task
                if metrics_srv is not None:
                    metrics_srv.close()
                    with contextlib.suppress(Exception):
                        await metrics_srv.wait_closed()
    finally:
        if sock_path is not None:
            with contextlib.suppress(FileNotFoundError):
                sock_path.unlink()

    batcher.stop()
    with contextlib.suppress(Exception):
        await batcher_task


async def _print_stats_loop(batcher: Batcher, every_s: float) -> None:
    if every_s <= 0:
        return
    last_t = time.monotonic()
    last_req = 0
    while True:
        await asyncio.sleep(every_s)
        now = time.monotonic()
        dt = max(1e-9, now - last_t)
        total = batcher.stats.requests_total
        rps = (total - last_req) / dt
        last_t = now
        last_req = total
        # Simple, human-friendly log.
        print(
            f"[infer-server] req_total={total} rps={rps:.1f} "
            f"batches={batcher.stats.batches_total} max_batch_seen={batcher.stats.max_batch_seen}"
        )
        # Per-model stats.
        for model_id, ms in sorted(batcher.stats.by_model.items()):
            print(
                f"  [model {model_id}] req_total={ms.requests_total} batches={ms.batches_total} "
                f"max_batch_seen={ms.max_batch_seen}"
            )


def add_args(p: argparse.ArgumentParser) -> None:
    # Note: `--bind` is defined in the top-level CLI for help grouping.
    p.add_argument("--max-batch", type=int, default=256, help="Batch flush size threshold")
    p.add_argument("--max-wait-us", type=int, default=2000, help="Max queue wait before flush")
    p.add_argument(
        "--best",
        default="dummy",
        help="Best model spec (dummy, dummy:0.1, or path:/abs/to/best.pt)",
    )
    p.add_argument(
        "--cand",
        default="dummy",
        help="Candidate model spec (dummy, dummy:-0.1, or path:/abs/to/candidate.pt)",
    )
    p.add_argument("--best-id", type=int, default=0, help="model_id for best")
    p.add_argument("--cand-id", type=int, default=1, help="model_id for candidate")
    p.add_argument(
        "--metrics-bind",
        default="127.0.0.1:18080",
        help="Metrics HTTP bind host:port",
    )
    p.add_argument("--metrics-disable", action="store_true", help="Disable metrics endpoint")
    p.add_argument(
        "--print-stats-every-s",
        type=float,
        default=2.0,
        help="Print periodic throughput/batch stats (0 disables)",
    )


def run_from_args(args: argparse.Namespace) -> int:
    cfg = ServerConfig(
        bind=args.bind,
        device=args.device,
        max_batch=args.max_batch,
        max_wait_us=args.max_wait_us,
        print_stats_every_s=args.print_stats_every_s,
        best_id=args.best_id,
        cand_id=args.cand_id,
        best_spec=args.best,
        cand_spec=args.cand,
        metrics_bind=args.metrics_bind,
        metrics_disable=args.metrics_disable,
    )
    asyncio.run(serve(cfg))
    return 0
