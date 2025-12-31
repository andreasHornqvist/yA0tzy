"""Dynamic batching core (PRD §9.1)."""

from __future__ import annotations

import asyncio
import time
import json as _json
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
class ModelStats:
    requests_total: int = 0
    batches_total: int = 0
    batch_hist: dict[int, int] | None = None
    max_batch_seen: int = 0

    def __post_init__(self) -> None:
        if self.batch_hist is None:
            self.batch_hist = {}


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
        self._reloads_total: int = 0

    @property
    def stats(self) -> BatcherStats:
        return self._stats

    @property
    def queue_depth(self) -> int:
        return self._q.qsize()

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
            # IMPORTANT: the batching window is relative to *now*, not relative to when the first
            # request was enqueued. If the queue backs up (or inference is slow), using first.t0
            # would make `remaining` <= 0 immediately, forcing batch size 1 forever.
            deadline = time.monotonic() + self._max_wait_s

            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._q.get(), timeout=remaining))
                except TimeoutError:
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
                                "location": "python/yatzy_az/server/batcher.py:Batcher.run",
                                "message": "formed batch",
                                "data": {
                                    "batch_len": len(batch),
                                    "queue_depth_after_form": int(self._q.qsize()),
                                    "max_batch": int(self._max_batch),
                                    "max_wait_us": int(self._max_wait_s * 1_000_000),
                                },
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # endregion agent log

            self._apply_batch(batch)

    def stop(self) -> None:
        self._stop.set()

    def _apply_batch(self, batch: list[_Queued]) -> None:
        self._stats.batches_total += 1
        self._stats.requests_total += len(batch)
        self._stats.max_batch_seen = max(self._stats.max_batch_seen, len(batch))
        self._stats.batch_hist[len(batch)] = self._stats.batch_hist.get(len(batch), 0) + 1

        # Route by model_id. We process each model_id group in one call to `infer_batch`.
        groups: dict[int, list[_Queued]] = {}
        for item in batch:
            groups.setdefault(item.req.model_id, []).append(item)

        for model_id, items in groups.items():
            ms = self._stats.by_model.get(model_id)
            if ms is None:
                ms = ModelStats()
                self._stats.by_model[model_id] = ms
            ms.batches_total += 1
            ms.requests_total += len(items)
            ms.max_batch_seen = max(ms.max_batch_seen, len(items))
            ms.batch_hist[len(items)] = ms.batch_hist.get(len(items), 0) + 1

            model = self._models.get(model_id)
            if model is None:
                for item in items:
                    if not item.fut.cancelled():
                        item.fut.set_exception(ValueError(f"unknown model_id: {model_id}"))
                continue

            feats = [it.req.features for it in items]
            masks = [it.req.legal_mask for it in items]
            outs = model.infer_batch(feats, masks)
            if len(outs) != len(items):
                for item in items:
                    if not item.fut.cancelled():
                        item.fut.set_exception(RuntimeError("model returned wrong batch size"))
                continue

            for item, out in zip(items, outs, strict=True):
                if item.fut.cancelled():
                    continue
                item.fut.set_result(
                    InferResponseV1(
                        request_id=item.req.request_id,
                        policy_logits=out.policy_logits,
                        value=out.value,
                        margin=out.margin,
                    )
                )
