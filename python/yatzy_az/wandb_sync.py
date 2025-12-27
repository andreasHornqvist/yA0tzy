"""W&B-compatible metrics stream consumer (Epic E10.5S2).

This is intentionally JSON-only for now (no wandb dependency):
- reads runs/<id>/logs/metrics.ndjson
- emits one JSON object per input event to stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True, help="Run directory (runs/<id>/)")


def _iter_metrics(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def normalize_event(ev: dict[str, Any]) -> dict[str, Any]:
    """Return a W&B-friendly JSON shape.

    Output contract:
    - step: best-effort numeric step (train_step, global_ply, tick, etc.)
    - metrics: flattened scalar dict (event-prefixed namespaces)
    """
    event = str(ev.get("event", ""))
    step = (
        ev.get("train_step")
        if "train_step" in ev
        else ev.get("global_ply", ev.get("tick", ev.get("gate_game_idx")))
    )

    metrics: dict[str, float] = {}

    def put(k: str, v: Any) -> None:
        if isinstance(v, (int, float)) and v is not None:
            metrics[k] = float(v)

    # Common numeric fields
    for k in [
        "loss_total",
        "loss_policy",
        "loss_value",
        "entropy",
        "lr",
        "throughput_steps_s",
        "win_rate",
        "mean_score_diff",
        "score_diff_se",
        "score_diff_ci95_low",
        "score_diff_ci95_high",
        "oracle_match_rate_overall",
        "oracle_match_rate_mark",
        "oracle_match_rate_reroll",
        "oracle_keepall_ignored",
        "completed_games",
        "steps",
        "would_block",
        "terminal",
    ]:
        if k in ev:
            put(f"{event}/{k}", ev.get(k))

    # Nested infer + pi summaries (best-effort)
    infer = ev.get("infer")
    if isinstance(infer, dict):
        for k in ["inflight", "sent", "received", "errors", "latency_p50_us", "latency_p95_us"]:
            if k in infer:
                put(f"{event}/infer_{k}", infer.get(k))
        if "latency_mean_us" in infer:
            put(f"{event}/infer_latency_mean_us", infer.get("latency_mean_us"))

    pi = ev.get("pi")
    if isinstance(pi, dict):
        for k in ["entropy", "max_p", "argmax_a"]:
            if k in pi:
                put(f"{event}/pi_{k}", pi.get(k))

    out: dict[str, Any] = {
        "event": event,
        "ts_ms": ev.get("ts_ms"),
        "run_id": ev.get("run_id"),
        "step": step,
        "metrics": metrics,
    }
    return out


def run_from_args(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    metrics_path = run_dir / "logs" / "metrics.ndjson"
    if not metrics_path.exists():
        raise RuntimeError(f"metrics stream not found: {metrics_path}")

    for ev in _iter_metrics(metrics_path):
        out = normalize_event(ev)
        sys.stdout.write(json.dumps(out) + "\n")

    return 0


