"""Dynamic batching core (PRD §9.1)."""

from __future__ import annotations

import asyncio
import collections
import time
import json as _json
from dataclasses import dataclass
from typing import Final

import numpy as np

from .model import Model
from .protocol_v1 import (
    ACTION_SPACE_A,
    FEATURE_LEN_V1,
    InferRequestPacked,
    InferResponsePacked,
)
from .debug_log import emit as _dbg_emit, enabled as _dbg_enabled


@dataclass(slots=True)
class _Queued:
    req: InferRequestPacked
    fut: asyncio.Future[InferResponsePacked]
    t0: float


@dataclass(slots=True)
class ModelStats:
    requests_total: int = 0
    batches_total: int = 0
    batch_hist: dict[int, int] | None = None
    max_batch_seen: int = 0
    # Flush reasons: {"full": n, "deadline": n, "forced": n}
    flush_reason_total: dict[str, int] | None = None
    # Per-batch timing/queue-wait histograms (cheap, per-process).
    batch_queue_wait_us: "PromHistogram" | None = None
    batch_build_ms: "PromHistogram" | None = None
    batch_forward_ms: "PromHistogram" | None = None
    batch_post_ms: "PromHistogram" | None = None
    # Utilization proxies (EMA in [0,1]).
    batch_underfill_frac: float = 0.0
    batch_full_frac: float = 0.0

    def __post_init__(self) -> None:
        if self.batch_hist is None:
            self.batch_hist = {}
        if self.flush_reason_total is None:
            self.flush_reason_total = {}


@dataclass(slots=True)
class BatcherStats:
    requests_total: int = 0
    batches_total: int = 0
    batch_hist: dict[int, int] | None = None
    max_batch_seen: int = 0
    by_model: dict[int, ModelStats] | None = None

    def __post_init__(self) -> None:
        if self.batch_hist is None:
            self.batch_hist = {}
        if self.by_model is None:
            self.by_model = {}


# Fixed buckets for cheap Prometheus histograms. Keep bucket counts small (<= ~16).
_HIST_QUEUE_WAIT_US = (
    100.0,
    200.0,
    500.0,
    1_000.0,
    2_000.0,
    5_000.0,
    10_000.0,
    20_000.0,
    50_000.0,
    100_000.0,
    200_000.0,
    500_000.0,
    1_000_000.0,
)
_HIST_BUILD_MS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
_HIST_FORWARD_MS = (
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
)
_HIST_POST_MS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


@dataclass(slots=True)
class PromHistogram:
    """Tiny fixed-bucket histogram for Prometheus exposition."""

    buckets: tuple[float, ...]  # finite bucket upper bounds (le)
    counts: list[int]
    overflow: int = 0
    count: int = 0
    sum: float = 0.0

    @classmethod
    def with_buckets(cls, buckets: tuple[float, ...]) -> "PromHistogram":
        return cls(buckets=tuple(float(x) for x in buckets), counts=[0 for _ in buckets])

    def observe(self, x: float) -> None:
        xf = float(x)
        if not np.isfinite(xf):
            return
        self.count += 1
        self.sum += xf
        for i, le in enumerate(self.buckets):
            if xf <= le:
                self.counts[i] += 1
                return
        self.overflow += 1


def _ema_update(prev: float, x: float, alpha: float) -> float:
    a = float(alpha)
    if a <= 0.0:
        return float(prev)
    if a >= 1.0:
        return float(x)
    return float(prev) * (1.0 - a) + float(x) * a


class Batcher:
    def __init__(self, model_by_id: dict[int, Model], *, max_batch: int, max_wait_us: int) -> None:
        if max_batch <= 0:
            raise ValueError("max_batch must be > 0")
        if max_wait_us <= 0:
            raise ValueError("max_wait_us must be > 0")
        self._models = model_by_id
        self._max_batch: Final[int] = max_batch
        self._max_wait_s: Final[float] = max_wait_us / 1_000_000.0
        self._q: asyncio.Queue[_Queued] = asyncio.Queue()
        # Pending items staged by model_id. This allows per-model batching so we don't
        # form a global batch that then gets split into smaller sub-batches (common in gating).
        self._pending_by_model: dict[int, collections.deque[_Queued]] = {}
        self._pending_total: int = 0
        # region agent log
        self._first_batch_logged: set[int] | None = set() if _dbg_enabled() else None
        # endregion agent log
        self._stats = BatcherStats()
        self._stop = asyncio.Event()
        self._reloads_total: int = 0
        # Debug log sampling counter (avoid huge log volume affecting perf).
        self._dbg_batch_ctr: int = 0

    @property
    def stats(self) -> BatcherStats:
        return self._stats

    @property
    def queue_depth(self) -> int:
        # Include both ingress queue and staged per-model pending.
        return int(self._q.qsize()) + int(self._pending_total)

    @property
    def reloads_total(self) -> int:
        return self._reloads_total

    def replace_model(self, model_id: int, new_model: Model) -> None:
        """Atomically replace the model for the given ID (E13.2S4).

        Safe because Python dict assignment is atomic under the GIL.
        In-flight batches using the old model will complete; new requests use the new model.
        """
        self._models[model_id] = new_model
        self._reloads_total += 1

    async def enqueue(self, req: InferRequestPacked) -> InferResponsePacked:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[InferResponsePacked] = loop.create_future()
        await self._q.put(_Queued(req=req, fut=fut, t0=time.monotonic()))
        return await fut

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                first = await asyncio.wait_for(self._q.get(), timeout=0.1)
            except TimeoutError:
                # If we already have staged work, keep making progress by flushing
                # any model whose oldest item has exceeded max_wait.
                if self._pending_total > 0:
                    now = time.monotonic()
                    oldest_deadline = None
                    oldest_mid = None
                    for mid, dq in self._pending_by_model.items():
                        if not dq:
                            continue
                        dl = dq[0].t0 + self._max_wait_s
                        if oldest_deadline is None or dl < oldest_deadline:
                            oldest_deadline = dl
                            oldest_mid = mid
                    if oldest_deadline is not None and oldest_mid is not None and (oldest_deadline - now) <= 0:
                        self._flush_one(sample=False, force_model_id=oldest_mid, reason="deadline")
                continue

            self._stage(first)
            # Opportunistically drain already-queued work without awaiting.
            # This reduces per-item scheduling overhead and helps form larger batches
            # when max_wait is small (e.g. 1000us) and QPS is high.
            self._drain_nowait()

            # Keep pulling until we can flush a full batch for some model, or until the
            # oldest pending request (across all models) hits its deadline.
            while self._pending_total > 0 and not self._stop.is_set():
                # If any model already has a full batch, flush it immediately.
                full_model_id = None
                for mid, dq in self._pending_by_model.items():
                    if len(dq) >= self._max_batch:
                        full_model_id = mid
                        break
                if full_model_id is not None:
                    self._flush_one(sample=True, force_model_id=full_model_id, reason="full")
                    continue

                # Compute remaining time until the oldest pending item reaches max_wait.
                now = time.monotonic()
                oldest_deadline = None
                oldest_mid = None
                for mid, dq in self._pending_by_model.items():
                    if not dq:
                        continue
                    dl = dq[0].t0 + self._max_wait_s
                    if oldest_deadline is None or dl < oldest_deadline:
                        oldest_deadline = dl
                        oldest_mid = mid
                if oldest_deadline is None or oldest_mid is None:
                    break

                remaining = oldest_deadline - now
                if remaining <= 0:
                    self._flush_one(sample=True, force_model_id=oldest_mid, reason="deadline")
                    continue

                try:
                    item = await asyncio.wait_for(self._q.get(), timeout=remaining)
                    self._stage(item)
                    # If the queue is already populated, drain it quickly without extra awaits.
                    self._drain_nowait()
                except TimeoutError:
                    # Oldest item reached deadline; flush that model.
                    self._flush_one(sample=True, force_model_id=oldest_mid, reason="deadline")
                    continue

    def stop(self) -> None:
        self._stop.set()

    def _stage(self, item: _Queued) -> None:
        dq = self._pending_by_model.get(int(item.req.model_id))
        if dq is None:
            dq = collections.deque()
            self._pending_by_model[int(item.req.model_id)] = dq
        dq.append(item)
        self._pending_total += 1

    def _drain_nowait(self, *, max_items: int = 4096) -> None:
        # Drain up to max_items queued requests without awaiting.
        # Safe: asyncio.Queue.get_nowait() is O(1).
        for _ in range(max_items):
            try:
                item = self._q.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._stage(item)

    def _flush_one(
        self, *, sample: bool, force_model_id: int | None = None, reason: str = "forced"
    ) -> None:
        """Flush one per-model batch.\n\n        If force_model_id is provided, we flush that model (if any pending).\n        Otherwise, we flush the model whose oldest item has waited the longest.\n        """
        if self._pending_total <= 0:
            return

        # Pick model to flush.
        if force_model_id is not None:
            model_id = int(force_model_id)
            dq = self._pending_by_model.get(model_id)
            if not dq:
                return
        else:
            now = time.monotonic()
            model_id = None
            best_age = -1.0
            for mid, dq2 in self._pending_by_model.items():
                if not dq2:
                    continue
                age = now - dq2[0].t0
                if age > best_age:
                    best_age = age
                    model_id = mid
            if model_id is None:
                return
            dq = self._pending_by_model[model_id]

        # Build batch up to max_batch.
        batch: list[_Queued] = []
        while dq and len(batch) < self._max_batch:
            batch.append(dq.popleft())
        self._pending_total -= len(batch)
        if not dq:
            # Keep dict small.
            self._pending_by_model.pop(model_id, None)

        if _dbg_enabled():
            self._dbg_batch_ctr += 1
            if not sample:
                sample = (self._dbg_batch_ctr & 0x3F) == 0  # 1 / 64 flushes
        else:
            sample = False

        # region agent log
        if sample and _dbg_enabled():
            try:
                oldest_age_us = int((time.monotonic() - batch[0].t0) * 1_000_000) if batch else 0
                _dbg_emit(
                    {
                        "timestamp": int(time.time() * 1000),
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H1",
                        "location": "python/yatzy_az/server/batcher.py:Batcher._flush_one",
                        "message": "flushing per-model batch",
                        "data": {
                            "model_id": int(model_id),
                            "batch_len": int(len(batch)),
                            "oldest_age_us": int(oldest_age_us),
                            "queue_depth": int(self.queue_depth),
                            "max_batch": int(self._max_batch),
                            "max_wait_us": int(self._max_wait_s * 1_000_000),
                        },
                    }
                )
            except Exception:
                pass
        # endregion agent log

        oldest_wait_us = int((time.monotonic() - batch[0].t0) * 1_000_000) if batch else 0
        self._apply_model_batch(
            int(model_id),
            batch,
            flush_reason=str(reason),
            oldest_wait_us=oldest_wait_us,
        )

    def _apply_model_batch(
        self,
        model_id: int,
        items: list[_Queued],
        *,
        flush_reason: str,
        oldest_wait_us: int,
    ) -> None:
        if not items:
            return

        self._stats.batches_total += 1
        self._stats.requests_total += len(items)
        self._stats.max_batch_seen = max(self._stats.max_batch_seen, len(items))
        self._stats.batch_hist[len(items)] = self._stats.batch_hist.get(len(items), 0) + 1

        ms = self._stats.by_model.get(model_id)
        if ms is None:
            ms = ModelStats()
            self._stats.by_model[model_id] = ms
        ms.batches_total += 1
        ms.requests_total += len(items)
        ms.max_batch_seen = max(ms.max_batch_seen, len(items))
        ms.batch_hist[len(items)] = ms.batch_hist.get(len(items), 0) + 1
        ms.flush_reason_total[flush_reason] = ms.flush_reason_total.get(flush_reason, 0) + 1

        # Lazily create histograms on first real batch.
        if ms.batch_queue_wait_us is None:
            ms.batch_queue_wait_us = PromHistogram.with_buckets(_HIST_QUEUE_WAIT_US)
        if ms.batch_build_ms is None:
            ms.batch_build_ms = PromHistogram.with_buckets(_HIST_BUILD_MS)
        if ms.batch_forward_ms is None:
            ms.batch_forward_ms = PromHistogram.with_buckets(_HIST_FORWARD_MS)
        if ms.batch_post_ms is None:
            ms.batch_post_ms = PromHistogram.with_buckets(_HIST_POST_MS)
        ms.batch_queue_wait_us.observe(float(max(0, int(oldest_wait_us))))

        # Utilization proxies: EMA of underfill/full batch events.
        alpha = 0.02
        bsz = len(items)
        underfill_thr = max(1, int(self._max_batch * 0.25))
        ms.batch_underfill_frac = _ema_update(
            ms.batch_underfill_frac, 1.0 if bsz < underfill_thr else 0.0, alpha
        )
        ms.batch_full_frac = _ema_update(
            ms.batch_full_frac, 1.0 if bsz >= self._max_batch else 0.0, alpha
        )

        model = self._models.get(model_id)
        if model is None:
            for item in items:
                if not item.fut.cancelled():
                    item.fut.set_exception(ValueError(f"unknown model_id: {model_id}"))
            return

        bsz = len(items)
        t_build0 = time.monotonic()
        x_np = np.empty((bsz, FEATURE_LEN_V1), dtype=np.float32)
        for i, it in enumerate(items):
            row = np.frombuffer(it.req.features_f32, dtype=np.float32, count=FEATURE_LEN_V1)
            x_np[i, :] = row
        masks = [it.req.legal_mask for it in items]
        build_ms = (time.monotonic() - t_build0) * 1000.0
        ms.batch_build_ms.observe(float(build_ms))
        t0 = time.monotonic()
        logits_bytes, values_bytes, margin_bytes = model.infer_batch_packed(x_np, masks)
        dt_ms = (time.monotonic() - t0) * 1000.0
        ms.batch_forward_ms.observe(float(dt_ms))

        # region agent log
        if self._first_batch_logged is not None and model_id not in self._first_batch_logged:
            self._first_batch_logged.add(int(model_id))
            try:
                payload = {
                    "timestamp": int(time.time() * 1000),
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "H_warmup",
                    "location": "python/yatzy_az/server/batcher.py:_apply_model_batch",
                    "message": "first infer_batch for model_id",
                    "data": {
                        "model_id": int(model_id),
                        "items": int(len(items)),
                        "dt_ms": float(dt_ms),
                        "queue_depth": int(self.queue_depth),
                    },
                }
                if _dbg_enabled():
                    _dbg_emit(payload)
            except Exception:
                pass
        try:
            if _dbg_enabled():
                _dbg_emit(
                    {
                        "timestamp": int(time.time() * 1000),
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H_latency",
                        "location": "python/yatzy_az/server/batcher.py:_apply_model_batch",
                        "message": "infer_batch timing",
                        "data": {
                            "model_id": int(model_id),
                            "items": int(len(items)),
                            "dt_ms": float(dt_ms),
                            "dt_per_item_ms": float(dt_ms / max(1, len(items))),
                            "queue_depth": int(self.queue_depth),
                        },
                    }
                )
        except Exception:
            pass
        # endregion agent log

        t_post0 = time.monotonic()
        if len(logits_bytes) != bsz * ACTION_SPACE_A * 4:
            for item in items:
                if not item.fut.cancelled():
                    item.fut.set_exception(RuntimeError("model returned wrong logits byte size"))
            return
        if len(values_bytes) != bsz * 4:
            for item in items:
                if not item.fut.cancelled():
                    item.fut.set_exception(RuntimeError("model returned wrong value byte size"))
            return
        if margin_bytes is not None and len(margin_bytes) != bsz * 4:
            for item in items:
                if not item.fut.cancelled():
                    item.fut.set_exception(RuntimeError("model returned wrong margin byte size"))
            return

        logits_mv = memoryview(logits_bytes)
        values_mv = memoryview(values_bytes)
        margin_mv = memoryview(margin_bytes) if margin_bytes is not None else None

        for i, item in enumerate(items):
            if item.fut.cancelled():
                continue
            item.fut.set_result(
                InferResponsePacked(
                    protocol_version=int(item.req.protocol_version),
                    request_id=item.req.request_id,
                    policy_logits_f32=logits_mv[
                        i * ACTION_SPACE_A * 4 : (i + 1) * ACTION_SPACE_A * 4
                    ],
                    value_f32=values_mv[i * 4 : (i + 1) * 4],
                    margin_f32=None if margin_mv is None else margin_mv[i * 4 : (i + 1) * 4],
                )
            )
        post_ms = (time.monotonic() - t_post0) * 1000.0
        ms.batch_post_ms.observe(float(post_ms))