"""Replay snapshot semantics (Epic E8.5.4).

Snapshot mode freezes the training dataset for an iteration:
- At train start, create `runs/<id>/replay_snapshot.json` listing shard filenames.
- Training uses only that list on resume, even if new shards appear in replay/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .replay_dataset import ReplayDatasetError, ReplayShard, discover_replay_shards

SNAPSHOT_VERSION: int = 1


class ReplaySnapshotError(RuntimeError):
    pass


def create_snapshot(*, replay_dir: Path, out_path: Path) -> dict[str, Any]:
    replay_dir = Path(replay_dir)
    out_path = Path(out_path)
    shards = discover_replay_shards(replay_dir)

    entries: list[dict[str, Any]] = []
    total_samples = 0
    for s in shards:
        entries.append(
            {
                "safetensors": s.safetensors_path.name,
                "meta": s.meta_path.name,
                "num_samples": s.meta.num_samples,
            }
        )
        total_samples += int(s.meta.num_samples)

    snap: dict[str, Any] = {
        "snapshot_version": SNAPSHOT_VERSION,
        "replay_dir": str(replay_dir.name),
        "shards": entries,
        "total_samples": total_samples,
    }

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, indent=2) + "\n")
    tmp.replace(out_path)
    return snap


def load_snapshot(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ReplaySnapshotError(f"snapshot not found: {path}")
    try:
        d = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        raise ReplaySnapshotError(f"failed to parse snapshot: {path}") from e
    if int(d.get("snapshot_version", -1)) != SNAPSHOT_VERSION:
        raise ReplaySnapshotError(
            f"snapshot_version mismatch: {d.get('snapshot_version')} != {SNAPSHOT_VERSION}"
        )
    if "shards" not in d or not isinstance(d["shards"], list):
        raise ReplaySnapshotError("invalid snapshot: missing shards list")
    return d


def shard_filenames(snapshot: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for e in snapshot["shards"]:
        if not isinstance(e, dict) or "safetensors" not in e:
            raise ReplaySnapshotError("invalid snapshot entry")
        out.append(str(e["safetensors"]))
    return out


def shards_from_snapshot(*, replay_dir: Path, snapshot: dict[str, Any]) -> list[ReplayShard]:
    """Resolve snapshot entries to ReplayShard objects by reading/validating meta."""
    replay_dir = Path(replay_dir)
    out: list[ReplayShard] = []
    # Validate meta for all shards once, then filter in snapshot order.
    try:
        shards_all = discover_replay_shards(replay_dir)
    except ReplayDatasetError as ex:
        raise ReplaySnapshotError(str(ex)) from ex

    by_name = {s.safetensors_path.name: s for s in shards_all}
    for e in snapshot["shards"]:
        name = str(e["safetensors"])
        if name not in by_name:
            raise ReplaySnapshotError(f"snapshot shard missing on disk: {name}")
        out.append(by_name[name])
    return out
