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
import json as _json

from .batcher import Batcher
from .metrics import MetricsSnapshot, now_s
from .metrics_server import ReloadResult, start_metrics_server
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
    torch_threads: int | None
    torch_interop_threads: int | None


def _apply_torch_thread_settings(cfg: ServerConfig) -> None:
    if cfg.torch_threads is None and cfg.torch_interop_threads is None:
        return
    import torch

    if cfg.torch_threads is not None:
        torch.set_num_threads(int(cfg.torch_threads))
    if cfg.torch_interop_threads is not None:
        torch.set_num_interop_threads(int(cfg.torch_interop_threads))

    # region agent log
    try:
        import json as _json
        with open(
            "/Users/andreashornqvist/code/yA0tzy/.cursor/debug.log",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                _json.dumps(
                    {
                        "timestamp": int(time.time() * 1000),
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H_latency",
                        "location": "python/yatzy_az/server/server.py:_apply_torch_thread_settings",
                        "message": "torch thread settings applied",
                        "data": {
                            "torch_threads": cfg.torch_threads,
                            "torch_interop_threads": cfg.torch_interop_threads,
                            "torch_get_num_threads": int(torch.get_num_threads()),
                            "torch_get_num_interop_threads": int(torch.get_num_interop_threads()),
                        },
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # endregion agent log


async def _handle_conn(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, batcher: Batcher
) -> None:
    # Allow pipelining multiple in-flight requests per connection.
    # The Rust client can have many concurrent tickets; if we await each response here,
    # we serialize the entire connection and the batcher will never form batches > 1.
    inflight_limit = 4096
    inflight_sem = asyncio.Semaphore(inflight_limit)
    out_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=inflight_limit)
    pending: set[asyncio.Task[None]] = set()

    async def _writer_loop() -> None:
        while True:
            payload = await out_q.get()
            if payload == b"":
                return
            await write_frame(writer, payload)

    async def _handle_one(req) -> None:
        try:
            resp: InferResponseV1 = await batcher.enqueue(req)

            # region agent log
            try:
                with open(
                    "/Users/andreashornqvist/code/yA0tzy/.cursor/debug.log",
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        _json.dumps(
                            {
                                "timestamp": int(time.time() * 1000),
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "H1",
                                "location": "python/yatzy_az/server/server.py:_handle_conn",
                                "message": "enqueue completed; queueing response for writer loop",
                                "data": {
                                    "request_id": int(resp.request_id),
                                    "queue_depth": int(batcher.queue_depth),
                                    "out_q": int(out_q.qsize()),
                                },
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # endregion agent log

            await out_q.put(encode_response_v1(resp))
        finally:
            inflight_sem.release()

    writer_task = asyncio.create_task(_writer_loop())
    try:
        while True:
            payload = await read_frame(reader)

            try:
                req = decode_request_v1(payload)
            except DecodeError:
                # Protocol violation: close the connection.
                break

            # region agent log
            try:
                with open(
                    "/Users/andreashornqvist/code/yA0tzy/.cursor/debug.log",
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        _json.dumps(
                            {
                                "timestamp": int(time.time() * 1000),
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "H1",
                                "location": "python/yatzy_az/server/server.py:_handle_conn",
                                "message": "decoded request; scheduling enqueue task (pipelined)",
                                "data": {
                                    "request_id": int(req.request_id),
                                    "model_id": int(req.model_id),
                                    "features_len": len(req.features),
                                    "legal_len": len(req.legal_mask),
                                    "queue_depth": int(batcher.queue_depth),
                                    "pending_tasks": len(pending),
                                },
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # endregion agent log

            await inflight_sem.acquire()
            t = asyncio.create_task(_handle_one(req))
            pending.add(t)
            t.add_done_callback(lambda tt: pending.discard(tt))
    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            for t in list(pending):
                t.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*pending)
            with contextlib.suppress(Exception):
                await out_q.put(b"")
            writer_task.cancel()
            with contextlib.suppress(Exception):
                await writer_task
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
            reloads_total=batcher.reloads_total,
        )

    def reload_model(model_id_str: str, path: str) -> ReloadResult:
        """Hot-reload a model (E13.2S4)."""
        try:
            # Map model_id string to numeric ID.
            if model_id_str == "best":
                numeric_id = int(config.best_id)
            elif model_id_str == "cand":
                numeric_id = int(config.cand_id)
            else:
                return ReloadResult(ok=False, error=f"unknown model_id: {model_id_str}")

            # Build new model from checkpoint.
            new_model = build_model(f"path:{path}", device=config.device)

            # Atomically swap the model.
            batcher.replace_model(numeric_id, new_model)

            return ReloadResult(ok=True)
        except Exception as e:  # noqa: BLE001
            return ReloadResult(ok=False, error=str(e))

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
                def capabilities() -> dict:
                    return {
                        "version": "2",
                        "hot_reload": True,
                        "pid": os.getpid(),
                        "bind": config.bind,
                        "metrics_bind": config.metrics_bind,
                        "device": config.device,
                        "max_batch": int(config.max_batch),
                        "max_wait_us": int(config.max_wait_us),
                    }

                metrics_srv = await start_metrics_server(
                    config.metrics_bind,
                    snapshot,
                    reload_callback=reload_model,
                    capabilities_callback=capabilities,
                    shutdown_callback=_stop,
                )
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
        "--torch-threads",
        type=int,
        default=None,
        help="Optional: torch intra-op threads (CPU perf stability).",
    )
    p.add_argument(
        "--torch-interop-threads",
        type=int,
        default=None,
        help="Optional: torch inter-op threads (CPU perf stability).",
    )
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
        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
    )
    _apply_torch_thread_settings(cfg)
    asyncio.run(serve(cfg))
    return 0
