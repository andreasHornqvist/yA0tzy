"""Run manifest helpers (Epic E8.5.1).

The run manifest lives at `runs/<run_id>/run.json` and is updated by:
- Rust `yz selfplay` (initialization + selfplay counters)
- Python training (train_step + checkpoint pointers)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunManifestError(RuntimeError):
    pass


def load_manifest(run_root: Path) -> dict[str, Any]:
    p = Path(run_root) / "run.json"
    if not p.exists():
        raise RunManifestError(f"run.json not found: {p}")
    try:
        return json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        raise RunManifestError(f"failed to parse run.json: {p}") from e


def save_manifest_atomic(run_root: Path, manifest: dict[str, Any]) -> None:
    run_root = Path(run_root)
    p = run_root / "run.json"
    tmp = run_root / "run.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(p)
