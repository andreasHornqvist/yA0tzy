"""Dynamic batching core (PRD §9.1)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Final

from .model import Model
from .protocol_v1 import InferRequestV1, InferResponseV1


@dataclass(slots=True)
class _Queued:
    req: InferRequestV1
    fut: asyncio.Future[InferResponseV1]
    t0: float


@dataclass(slots=True)
class BatcherStats:
    total_requests: int = 0
    total_batches: int = 0
    batch_hist: dict[int, int] | None = None
    max_batch_seen: int = 0

    def __post_init__(self) -> None:
        if self.batch_hist is None:
            self.batch_hist = {}


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
        self._stats = BatcherStats()
        self._stop = asyncio.Event()

    @property
    def stats(self) -> BatcherStats:
        return self._stats

    async def enqueue(self, req: InferRequestV1) -> InferResponseV1:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[InferResponseV1] = loop.create_future()
        await self._q.put(_Queued(req=req, fut=fut, t0=time.monotonic()))
        return await fut

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                first = await asyncio.wait_for(self._q.get(), timeout=0.1)
            except TimeoutError:
                continue

            batch: list[_Queued] = [first]
            deadline = first.t0 + self._max_wait_s

            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._q.get(), timeout=remaining))
                except TimeoutError:
                    break

            self._apply_batch(batch)

    def stop(self) -> None:
        self._stop.set()

    def _apply_batch(self, batch: list[_Queued]) -> None:
        self._stats.total_batches += 1
        self._stats.total_requests += len(batch)
        self._stats.max_batch_seen = max(self._stats.max_batch_seen, len(batch))
        self._stats.batch_hist[len(batch)] = self._stats.batch_hist.get(len(batch), 0) + 1

        # For v1 dummy, we process per-request but keep model_id routing.
        # In the real server, we'd group by model_id and stack tensors per group.
        for item in batch:
            if item.fut.cancelled():
                continue
            model = self._models.get(item.req.model_id)
            if model is None:
                item.fut.set_exception(ValueError(f"unknown model_id: {item.req.model_id}"))
                continue

            out = model.infer_batch([item.req.features], [item.req.legal_mask])[0]
            item.fut.set_result(
                InferResponseV1(
                    request_id=item.req.request_id,
                    policy_logits=out.policy_logits,
                    value=out.value,
                    margin=out.margin,
                )
            )
