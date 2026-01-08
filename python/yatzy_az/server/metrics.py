"""Prometheus metrics snapshot + formatting for the infer-server (PRD E6S3)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .batcher import BatcherStats, ModelStats


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    now_s: float
    start_s: float
    queue_depth: int
    batcher: BatcherStats
    reloads_total: int = 0


def format_prometheus(snapshot: MetricsSnapshot) -> str:
    """Format metrics in Prometheus text exposition format."""
    lines: list[str] = []

    def gauge(name: str, value: float, labels: dict[str, str] | None = None) -> None:
        lines.append(_fmt_sample(name, value, labels))

    def counter(name: str, value: float, labels: dict[str, str] | None = None) -> None:
        lines.append(_fmt_sample(name, value, labels))

    # HELP/TYPE
    lines.append("# HELP yatzy_infer_uptime_seconds Process uptime (seconds).")
    lines.append("# TYPE yatzy_infer_uptime_seconds gauge")
    lines.append("# HELP yatzy_infer_queue_depth Number of queued inference requests.")
    lines.append("# TYPE yatzy_infer_queue_depth gauge")
    lines.append("# HELP yatzy_infer_requests_total Total requests handled by the batcher.")
    lines.append("# TYPE yatzy_infer_requests_total counter")
    lines.append("# HELP yatzy_infer_batches_total Total batches formed by the batcher.")
    lines.append("# TYPE yatzy_infer_batches_total counter")
    lines.append("# HELP yatzy_infer_batch_size_bucket Histogram of batch sizes (by model_id).")
    lines.append("# TYPE yatzy_infer_batch_size_bucket histogram")
    lines.append("# HELP yatzy_infer_batch_size_count Total number of batches (by model_id).")
    lines.append("# TYPE yatzy_infer_batch_size_count counter")
    lines.append("# HELP yatzy_infer_flush_reason_total Total batch flushes by reason (by model_id).")
    lines.append("# TYPE yatzy_infer_flush_reason_total counter")
    lines.append("# HELP yatzy_infer_batch_queue_wait_us Queue-wait (oldest request age) at batch flush (by model_id).")
    lines.append("# TYPE yatzy_infer_batch_queue_wait_us histogram")
    lines.append("# HELP yatzy_infer_batch_build_ms Time to build a batch input (by model_id).")
    lines.append("# TYPE yatzy_infer_batch_build_ms histogram")
    lines.append("# HELP yatzy_infer_batch_forward_ms Model forward time per batch (by model_id).")
    lines.append("# TYPE yatzy_infer_batch_forward_ms histogram")
    lines.append("# HELP yatzy_infer_batch_post_ms Postprocess/pack time per batch (by model_id).")
    lines.append("# TYPE yatzy_infer_batch_post_ms histogram")
    lines.append("# HELP yatzy_infer_batch_underfill_frac EMA of underfilled batches (<25% of max_batch) (by model_id).")
    lines.append("# TYPE yatzy_infer_batch_underfill_frac gauge")
    lines.append("# HELP yatzy_infer_batch_full_frac EMA of full batches (==max_batch) (by model_id).")
    lines.append("# TYPE yatzy_infer_batch_full_frac gauge")
    lines.append("# HELP yatzy_infer_model_reloads_total Total model hot-reloads (E13.2S4).")
    lines.append("# TYPE yatzy_infer_model_reloads_total counter")

    uptime = max(0.0, snapshot.now_s - snapshot.start_s)
    gauge("yatzy_infer_uptime_seconds", uptime)
    gauge("yatzy_infer_queue_depth", float(snapshot.queue_depth))

    # Global counters.
    counter("yatzy_infer_requests_total", float(snapshot.batcher.requests_total))
    counter("yatzy_infer_batches_total", float(snapshot.batcher.batches_total))
    counter("yatzy_infer_model_reloads_total", float(snapshot.reloads_total))

    # Per-model histogram. We expose histogram buckets over batch sizes.
    for model_id, ms in sorted(snapshot.batcher.by_model.items()):
        _emit_batch_hist(lines, model_id, ms)
        _emit_flush_reason(lines, model_id, ms)
        _emit_promhist(lines, "yatzy_infer_batch_queue_wait_us", model_id, getattr(ms, "batch_queue_wait_us", None))
        _emit_promhist(lines, "yatzy_infer_batch_build_ms", model_id, getattr(ms, "batch_build_ms", None))
        _emit_promhist(lines, "yatzy_infer_batch_forward_ms", model_id, getattr(ms, "batch_forward_ms", None))
        _emit_promhist(lines, "yatzy_infer_batch_post_ms", model_id, getattr(ms, "batch_post_ms", None))
        # Utilization proxies.
        gauge(
            "yatzy_infer_batch_underfill_frac",
            float(getattr(ms, "batch_underfill_frac", 0.0) or 0.0),
            {"model_id": str(model_id)},
        )
        gauge(
            "yatzy_infer_batch_full_frac",
            float(getattr(ms, "batch_full_frac", 0.0) or 0.0),
            {"model_id": str(model_id)},
        )

    return "\n".join(lines) + "\n"


def _emit_batch_hist(lines: list[str], model_id: int, ms: ModelStats) -> None:
    # Common bucket boundaries; +Inf handled separately.
    buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # batch_hist maps exact batch_size -> count.
    exact = ms.batch_hist or {}

    def count_le(le: int) -> int:
        return sum(c for sz, c in exact.items() if sz <= le)

    cum = 0
    for le in buckets:
        cum = count_le(le)
        lines.append(
            _fmt_sample(
                "yatzy_infer_batch_size_bucket",
                float(cum),
                {"model_id": str(model_id), "le": str(le)},
            )
        )
    # +Inf bucket equals total batches for that model.
    lines.append(
        _fmt_sample(
            "yatzy_infer_batch_size_bucket",
            float(ms.batches_total),
            {"model_id": str(model_id), "le": "+Inf"},
        )
    )
    lines.append(
        _fmt_sample(
            "yatzy_infer_batch_size_count",
            float(ms.batches_total),
            {"model_id": str(model_id)},
        )
    )


def _emit_flush_reason(lines: list[str], model_id: int, ms: ModelStats) -> None:
    fr = getattr(ms, "flush_reason_total", None) or {}
    for reason in ("full", "deadline", "forced"):
        lines.append(
            _fmt_sample(
                "yatzy_infer_flush_reason_total",
                float(fr.get(reason, 0)),
                {"model_id": str(model_id), "reason": str(reason)},
            )
        )


def _emit_promhist(lines: list[str], base: str, model_id: int, h: object | None) -> None:
    """Emit a Prometheus histogram from a PromHistogram-like object.

    Expected attributes:
      - buckets: iterable[float]
      - counts: iterable[int] (non-cumulative, aligned with buckets)
      - count: int (total count, incl overflow)
      - sum: float (sum of observed values)
    """
    if h is None:
        return
    buckets = getattr(h, "buckets", None)
    counts = getattr(h, "counts", None)
    tot = getattr(h, "count", 0) or 0
    s = getattr(h, "sum", 0.0) or 0.0
    if not buckets or counts is None:
        return

    cum = 0
    for le, c in zip(buckets, counts):
        cum += int(c)
        lines.append(
            _fmt_sample(
                f"{base}_bucket",
                float(cum),
                {"model_id": str(model_id), "le": str(le)},
            )
        )
    # +Inf bucket.
    lines.append(
        _fmt_sample(
            f"{base}_bucket",
            float(tot),
            {"model_id": str(model_id), "le": "+Inf"},
        )
    )
    lines.append(_fmt_sample(f"{base}_count", float(tot), {"model_id": str(model_id)}))
    lines.append(_fmt_sample(f"{base}_sum", float(s), {"model_id": str(model_id)}))


def _fmt_sample(name: str, value: float, labels: dict[str, str] | None) -> str:
    if labels:
        items = ",".join(f'{k}="{_escape_label(v)}"' for k, v in labels.items())
        return f"{name}{{{items}}} {value}"
    return f"{name} {value}"


def _escape_label(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def now_s() -> float:
    return time.time()
