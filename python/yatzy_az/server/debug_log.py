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

_q: "queue.SimpleQueue[str]" = queue.SimpleQueue()
_started = False


def _writer_loop(fd: int) -> None:
    while True:
        line = _q.get()
        try:
            os.write(fd, line.encode("utf-8"))
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
    _ensure_started()
    try:
        if "timestamp" not in payload:
            payload["timestamp"] = int(time.time() * 1000)
        _q.put(json.dumps(payload) + "\n")
    except Exception:
        pass


