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
from .debug_log import emit as _dbg_emit
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

# Debug-mode log sampling (keep evidence while avoiding log-volume perf cliffs).
# We sample hot-path per-request logs by request_id bitmask.
_DBG_REQ_SAMPLE_MASK = 0x3FF  # 1 / 1024

def _dbg_sample_req(request_id: int) -> bool:
    return (int(request_id) & _DBG_REQ_SAMPLE_MASK) == 0


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
        _dbg_emit(
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
    except Exception:
        pass
    # endregion agent log


async def _handle_conn(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, batcher: Batcher
) -> None:
    # Bounded inflight to prevent queueing tail explosions.
    # Must match Rust-side global inflight budget (64 per client * ~10 workers = 640 max).
    # We use a tighter bound per-connection to enable backpressure.
    INGRESS_MAX = 128
    inflight_sem = asyncio.Semaphore(INGRESS_MAX)
    out_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=INGRESS_MAX)

    # Avoid per-request task churn: use a fixed worker pool that awaits `batcher.enqueue`.
    # Reduced from 256 to 32 to lower scheduling overhead and contention.
    WORKER_COUNT = 32
    req_q: asyncio.Queue[object] = asyncio.Queue(maxsize=INGRESS_MAX)
    worker_tasks: list[asyncio.Task[None]] = []

    # region agent log
    # Sampled per-request timing: decode->worker_start and decode->enqueue_done.
    # Keep extremely low-volume (1/1024 by request_id) to avoid perturbing perf.
    t_dec_by_id: dict[int, float] = {}
    # endregion agent log

    async def _writer_loop() -> None:
        while True:
            payload = await out_q.get()
            if payload == b"":
                return
            await write_frame(writer, payload)

    async def _worker_loop(worker_id: int) -> None:
        while True:
            item = await req_q.get()
            if item is None:
                return
            req = item
            assert isinstance(req, object)
            try:
                # region agent log
                try:
                    rid = int(req.request_id)  # type: ignore[attr-defined]
                    t_dec = t_dec_by_id.pop(rid, None)
                    if t_dec is not None:
                        dt_to_worker_ms = (time.monotonic() - t_dec) * 1000.0
                        _dbg_emit(
                            {
                                "timestamp": int(time.time() * 1000),
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "H_conn_sched",
                                "location": "python/yatzy_az/server/server.py:_worker_loop",
                                "message": "worker picked request",
                                "data": {
                                    "request_id": rid,
                                    "worker_id": int(worker_id),
                                    "dt_decode_to_worker_ms": dt_to_worker_ms,
                                    "queue_depth": int(batcher.queue_depth),
                                    "req_q": int(req_q.qsize()),
                                    "out_q": int(out_q.qsize()),
                                },
                            }
                        )
                        # put back the decode timestamp so we can also log decode->enqueue_done below
                        t_dec_by_id[rid] = t_dec
                except Exception:
                    pass
                # endregion agent log

                resp: InferResponseV1 = await batcher.enqueue(req)  # type: ignore[arg-type]

                # region agent log
                try:
                    rid = int(resp.request_id)
                    t_dec = t_dec_by_id.pop(rid, None)
                    if t_dec is not None:
                        dt_to_done_ms = (time.monotonic() - t_dec) * 1000.0
                        _dbg_emit(
                            {
                                "timestamp": int(time.time() * 1000),
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "H_conn_sched",
                                "location": "python/yatzy_az/server/server.py:_worker_loop",
                                "message": "enqueue completed",
                                "data": {
                                    "request_id": rid,
                                    "worker_id": int(worker_id),
                                    "dt_decode_to_enqueue_done_ms": dt_to_done_ms,
                                    "queue_depth": int(batcher.queue_depth),
                                    "req_q": int(req_q.qsize()),
                                    "out_q": int(out_q.qsize()),
                                },
                            }
                        )
                except Exception:
                    pass
                # endregion agent log

                await out_q.put(encode_response_v1(resp))
            finally:
                inflight_sem.release()

    writer_task = asyncio.create_task(_writer_loop())
    # Spawn a bounded number of workers; concurrency is still limited by `inflight_sem`.
    # Reduced from 256 to WORKER_COUNT (32) to lower asyncio task scheduling overhead.
    for wid in range(WORKER_COUNT):
        worker_tasks.append(asyncio.create_task(_worker_loop(wid)))
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
                if _dbg_sample_req(int(req.request_id)):
                    # Record decode time for sampled requests (used by worker timing logs).
                    t_dec_by_id[int(req.request_id)] = time.monotonic()
                    _dbg_emit(
                        {
                            "timestamp": int(time.time() * 1000),
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H1",
                            "location": "python/yatzy_az/server/server.py:_handle_conn",
                            "message": "decoded request; enqueued to worker pool (pipelined)",
                            "data": {
                                "request_id": int(req.request_id),
                                "model_id": int(req.model_id),
                                "features_len": len(req.features),
                                "legal_len": len(req.legal_mask),
                                "queue_depth": int(batcher.queue_depth),
                                "req_q": int(req_q.qsize()),
                            },
                        }
                    )
            except Exception:
                pass
            # endregion agent log

            await inflight_sem.acquire()
            await req_q.put(req)
    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            # Stop workers.
            for _ in worker_tasks:
                with contextlib.suppress(Exception):
                    await req_q.put(None)
            for t in worker_tasks:
                t.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*worker_tasks)
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
