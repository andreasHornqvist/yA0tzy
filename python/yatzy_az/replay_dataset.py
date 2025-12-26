"""Replay dataset loader (PRD E8S1).

Reads replay shards emitted by `yz selfplay`:
- `shard_*.safetensors` (tensors)
- `shard_*.meta.json` (schema/version ids)

Provides:
- shard discovery + meta validation
- sample streaming
- a torch DataLoader-ready IterableDataset wrapper (optional torch dependency)
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

# Optional torch base class: we still validate availability at runtime in `_require_torch`.
try:  # pragma: no cover
    from torch.utils.data import IterableDataset as _TorchIterableDataset
except Exception:  # noqa: BLE001
    _TorchIterableDataset = object  # type: ignore[misc,assignment]

# v1 ids (must match Rust yz-replay / PRD §10.3)
PROTOCOL_VERSION: int = 1
FEATURE_SCHEMA_ID: int = 1
FEATURE_LEN: int = 45
ACTION_SPACE_ID: str = "oracle_keepmask_v1"
ACTION_SPACE_A: int = 47
RULESET_ID: str = "swedish_scandinavian_v1"

T_FEATURES = "features"
T_LEGAL_MASK = "legal_mask"
T_PI = "pi"
T_Z = "z"
T_Z_MARGIN = "z_margin"


class ReplayDatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShardMeta:
    protocol_version: int
    feature_schema_id: int
    feature_len: int
    action_space_id: str
    action_space_a: int
    ruleset_id: str
    num_samples: int

    @staticmethod
    def from_json(d: dict[str, Any]) -> ShardMeta:
        return ShardMeta(
            protocol_version=int(d["protocol_version"]),
            feature_schema_id=int(d["feature_schema_id"]),
            feature_len=int(d["feature_len"]),
            action_space_id=str(d["action_space_id"]),
            action_space_a=int(d["action_space_a"]),
            ruleset_id=str(d["ruleset_id"]),
            num_samples=int(d["num_samples"]),
        )


@dataclass(frozen=True)
class ReplayShard:
    safetensors_path: Path
    meta_path: Path
    meta: ShardMeta


def _validate_meta(meta: ShardMeta) -> None:
    if meta.protocol_version != PROTOCOL_VERSION:
        raise ReplayDatasetError(
            f"protocol_version mismatch: {meta.protocol_version} != {PROTOCOL_VERSION}"
        )
    if meta.feature_schema_id != FEATURE_SCHEMA_ID:
        raise ReplayDatasetError(
            f"feature_schema_id mismatch: {meta.feature_schema_id} != {FEATURE_SCHEMA_ID}"
        )
    if meta.feature_len != FEATURE_LEN:
        raise ReplayDatasetError(f"feature_len mismatch: {meta.feature_len} != {FEATURE_LEN}")
    if meta.action_space_id != ACTION_SPACE_ID:
        raise ReplayDatasetError(
            f"action_space_id mismatch: {meta.action_space_id!r} != {ACTION_SPACE_ID!r}"
        )
    if meta.action_space_a != ACTION_SPACE_A:
        raise ReplayDatasetError(
            f"action_space_a mismatch: {meta.action_space_a} != {ACTION_SPACE_A}"
        )
    if meta.ruleset_id != RULESET_ID:
        raise ReplayDatasetError(f"ruleset_id mismatch: {meta.ruleset_id!r} != {RULESET_ID!r}")


def discover_replay_shards(replay_dir: Path) -> list[ReplayShard]:
    """Discover shards under `replay_dir`, validate their meta, and return them in sorted order."""
    replay_dir = Path(replay_dir)
    if not replay_dir.exists():
        raise ReplayDatasetError(f"replay dir does not exist: {replay_dir}")
    if not replay_dir.is_dir():
        raise ReplayDatasetError(f"replay path is not a directory: {replay_dir}")

    st_files = sorted(replay_dir.glob("shard_*.safetensors"))
    shards: list[ReplayShard] = []
    for st in st_files:
        meta = st.with_suffix(".meta.json")
        if not meta.exists():
            raise ReplayDatasetError(f"missing meta file for shard: {st.name}")
        try:
            d = json.loads(meta.read_text())
        except Exception as e:  # noqa: BLE001 - we rethrow as domain error
            raise ReplayDatasetError(f"failed to parse meta json: {meta}") from e
        m = ShardMeta.from_json(d)
        _validate_meta(m)
        shards.append(ReplayShard(safetensors_path=st, meta_path=meta, meta=m))
    return shards


def _load_shard_numpy(path: Path) -> dict[str, np.ndarray]:
    """Load tensors from safetensors as NumPy arrays."""
    out: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        for name in f.keys():
            out[name] = f.get_tensor(name)
    return out


def iter_samples_numpy(
    shards: Sequence[ReplayShard],
    *,
    shuffle_shards: bool = False,
    seed: int = 0,
    repeat: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
    """Yield one sample at a time as NumPy arrays."""
    rng = random.Random(seed)

    while True:
        order = list(range(len(shards)))
        if shuffle_shards:
            rng.shuffle(order)

        for si in order:
            t = _load_shard_numpy(shards[si].safetensors_path)

            feats = t[T_FEATURES]
            legal = t[T_LEGAL_MASK]
            pi = t[T_PI]
            z = t[T_Z]
            z_margin = t.get(T_Z_MARGIN)

            n = feats.shape[0]
            if feats.shape != (n, FEATURE_LEN):
                raise ReplayDatasetError(f"bad features shape: {feats.shape}")
            if legal.shape != (n, ACTION_SPACE_A):
                raise ReplayDatasetError(f"bad legal_mask shape: {legal.shape}")
            if pi.shape != (n, ACTION_SPACE_A):
                raise ReplayDatasetError(f"bad pi shape: {pi.shape}")
            if z.shape != (n,):
                raise ReplayDatasetError(f"bad z shape: {z.shape}")
            if z_margin is not None and z_margin.shape != (n,):
                raise ReplayDatasetError(f"bad z_margin shape: {z_margin.shape}")

            for i in range(n):
                yield (
                    feats[i],
                    legal[i],
                    pi[i],
                    z[i],
                    None if z_margin is None else z_margin[i],
                )

        if not repeat:
            break


def _split_shards_for_worker(shards: Sequence[ReplayShard]) -> list[ReplayShard]:
    """Deterministic shard split across DataLoader workers."""
    try:
        import torch  # noqa: F401
        from torch.utils.data import get_worker_info
    except Exception:
        return list(shards)

    wi = get_worker_info()
    if wi is None:
        return list(shards)
    wid = int(wi.id)
    n = int(wi.num_workers)
    return list(shards)[wid::n]


def _require_torch():
    try:
        import torch  # noqa: F401
        from torch.utils.data import IterableDataset  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise ReplayDatasetError(
            "torch is required for ReplayIterableDataset; install with `uv sync --all-extras` or add the "
            "`train` extra."
        ) from e


class ReplayIterableDataset(_TorchIterableDataset):  # type: ignore[misc]
    """Torch DataLoader-ready dataset that streams samples from replay shards."""

    def __init__(
        self,
        replay_dir: Path,
        *,
        shuffle_shards: bool = False,
        seed: int = 0,
        repeat: bool = True,
    ) -> None:
        _require_torch()
        self.replay_dir = Path(replay_dir)
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.repeat = repeat

    def __iter__(self):
        import torch

        shards = discover_replay_shards(self.replay_dir)
        shards = _split_shards_for_worker(shards)
        it = iter_samples_numpy(
            shards,
            shuffle_shards=self.shuffle_shards,
            seed=self.seed + int(os.getpid()),
            repeat=self.repeat,
        )
        for feats, legal, pi, z, z_margin in it:
            x = torch.from_numpy(feats.astype(np.float32, copy=False))
            legal_t = torch.from_numpy(legal.astype(np.uint8, copy=False))
            pi_t = torch.from_numpy(pi.astype(np.float32, copy=False))
            z_t = torch.tensor(float(z), dtype=torch.float32)
            if z_margin is None:
                yield x, legal_t, pi_t, z_t, None
            else:
                zm_t = torch.tensor(float(z_margin), dtype=torch.float32)
                yield x, legal_t, pi_t, z_t, zm_t
