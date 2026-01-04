"""Tiny NDJSON debug logger used in Cursor debug mode.

Goal: avoid blocking the asyncio event loop on file I/O (open/write/close) in hot paths.
This module keeps a background thread that appends NDJSON lines to the configured log file.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any


_DEBUG_LOG_PATH = "/Users/andreashornqvist/code/yA0tzy/.cursor/debug.log"
_ENABLED = os.getenv("YZ_DEBUG_LOG") in ("1", "true", "yes")

_q: "queue.SimpleQueue[str]" = queue.SimpleQueue()
_started = False


def _writer_loop(fd: int) -> None:
    while True:
        line = _q.get()
        try:
            # Batch up as many queued lines as we can to reduce syscall + file contention.
            # (Important: multiple processes append to the same file; fewer writes helps a lot.)
            parts = [line]
            for _ in range(1023):
                try:
                    parts.append(_q.get_nowait())
                except Exception:
                    break

            payload_bytes = "".join(parts).encode("utf-8")
            t0 = time.perf_counter()
            os.write(fd, payload_bytes)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            # If debug log I/O itself is slow, record it (rare) from the writer thread
            # without involving the asyncio event loop.
            if dt_ms >= 5.0:
                try:
                    payload = {
                        "timestamp": int(time.time() * 1000),
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H_py_logio",
                        "location": "python/yatzy_az/server/debug_log.py:_writer_loop",
                        "message": "slow debug.log os.write",
                        "data": {"dt_ms": dt_ms, "bytes": len(payload_bytes)},
                    }
                    # Enqueue (do NOT immediately write again; that can amplify stalls).
                    _q.put(json.dumps(payload) + "\n")
                except Exception:
                    pass
        except Exception:
            # Best-effort only; never crash the server due to debug logging.
            pass


def _ensure_started() -> None:
    global _started
    if _started:
        return
    _started = True
    try:
        fd = os.open(_DEBUG_LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    except Exception:
        # If we can't open the log, we just drop logs.
        return
    t = threading.Thread(target=_writer_loop, args=(fd,), daemon=True)
    t.start()


def emit(payload: dict[str, Any]) -> None:
    """Enqueue a payload as one NDJSON line (best-effort)."""
    if not _ENABLED:
        return
    _ensure_started()
    try:
        if "timestamp" not in payload:
            payload["timestamp"] = int(time.time() * 1000)
        _q.put(json.dumps(payload) + "\n")
    except Exception:
        pass


def enabled() -> bool:
    return bool(_ENABLED)



