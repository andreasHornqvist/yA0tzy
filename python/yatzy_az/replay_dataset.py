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
import bisect
import collections
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

# Optional torch base class: we still validate availability at runtime in `_require_torch`.
try:  # pragma: no cover
    from torch.utils.data import IterableDataset as _TorchIterableDataset
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # noqa: BLE001
    _TorchIterableDataset = object  # type: ignore[misc,assignment]
    _TorchDataset = object  # type: ignore[misc,assignment]

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
        shard_files: list[str] | None = None,
        shuffle_shards: bool = False,
        seed: int = 0,
        repeat: bool = True,
    ) -> None:
        _require_torch()
        self.replay_dir = Path(replay_dir)
        self.shard_files = shard_files
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.repeat = repeat

    def __iter__(self):
        import torch

        if self.shard_files is None:
            shards = discover_replay_shards(self.replay_dir)
        else:
            # Validate meta for only the snapshot-listed shards.
            wanted = set(self.shard_files)
            shards_all = discover_replay_shards(self.replay_dir)
            shards = [s for s in shards_all if s.safetensors_path.name in wanted]
            missing = wanted.difference({s.safetensors_path.name for s in shards})
            if missing:
                raise ReplayDatasetError(f"snapshot missing shards on disk: {sorted(missing)}")
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
            # Use 0.0 as placeholder when z_margin is not present (avoids None in batch collate)
            zm_t = torch.tensor(0.0 if z_margin is None else float(z_margin), dtype=torch.float32)
            yield x, legal_t, pi_t, z_t, zm_t


class ReplayRandomAccessDataset(_TorchDataset):  # type: ignore[misc]
    """Map-style dataset that supports true random sampling across all samples.

    This builds a global index over all samples in the selected shard set and implements
    `__len__` + `__getitem__` so DataLoader can use `shuffle=True`.

    Note: This is designed for training stability. It intentionally trades some IO locality
    for better mixing, but mitigates overhead with a small per-process shard cache.
    """

    def __init__(
        self,
        replay_dir: Path,
        *,
        shard_files: list[str] | None = None,
        seed: int = 0,
        cache_shards: int = 4,
    ) -> None:
        _require_torch()
        self.replay_dir = Path(replay_dir)
        self.shard_files = shard_files
        self.seed = int(seed)
        self.cache_shards = int(cache_shards)

        # Lazily initialized (important for DataLoader worker spawn semantics on macOS).
        self._shards: list[ReplayShard] | None = None
        self._prefix_ends: list[int] | None = None  # cumulative sample counts
        self._total: int | None = None
        self._cache: "collections.OrderedDict[str, dict[str, np.ndarray]]" | None = None

    def _ensure_index(self) -> None:
        if self._shards is not None and self._prefix_ends is not None and self._total is not None:
            return

        # Resolve shard list (snapshot-filtered if shard_files is provided).
        if self.shard_files is None:
            shards = discover_replay_shards(self.replay_dir)
        else:
            wanted = set(self.shard_files)
            shards_all = discover_replay_shards(self.replay_dir)
            shards = [s for s in shards_all if s.safetensors_path.name in wanted]
            missing = wanted.difference({s.safetensors_path.name for s in shards})
            if missing:
                raise ReplayDatasetError(f"snapshot missing shards on disk: {sorted(missing)}")

            # Preserve snapshot order for determinism.
            by_name = {s.safetensors_path.name: s for s in shards}
            shards = [by_name[name] for name in self.shard_files if name in by_name]

        if not shards:
            raise ReplayDatasetError(f"no replay shards found under: {self.replay_dir}")

        # Build prefix sums over shard sizes.
        prefix: list[int] = []
        total = 0
        for s in shards:
            n = int(s.meta.num_samples)
            if n <= 0:
                continue
            total += n
            prefix.append(total)
        if total <= 0:
            raise ReplayDatasetError("replay shards have zero total_samples")

        self._shards = shards
        self._prefix_ends = prefix
        self._total = total
        self._cache = collections.OrderedDict()

    def __len__(self) -> int:
        self._ensure_index()
        assert self._total is not None
        return int(self._total)

    def _get_shard_tensors(self, shard: ReplayShard) -> dict[str, np.ndarray]:
        assert self._cache is not None
        key = str(shard.safetensors_path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        t = _load_shard_numpy(shard.safetensors_path)
        self._cache[key] = t
        self._cache.move_to_end(key)
        # Evict LRU.
        while len(self._cache) > max(1, self.cache_shards):
            self._cache.popitem(last=False)
        return t

    def __getitem__(self, idx: int):
        import torch

        self._ensure_index()
        assert self._shards is not None
        assert self._prefix_ends is not None
        assert self._total is not None

        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        idx = int(idx)
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)

        # Locate shard by prefix sum.
        si = bisect.bisect_right(self._prefix_ends, idx)
        prev_end = 0 if si == 0 else int(self._prefix_ends[si - 1])
        local = idx - prev_end

        shard = self._shards[si]
        t = self._get_shard_tensors(shard)

        feats = t[T_FEATURES]
        legal = t[T_LEGAL_MASK]
        pi = t[T_PI]
        z = t[T_Z]
        z_margin = t.get(T_Z_MARGIN)

        n = feats.shape[0]
        if local < 0 or local >= n:
            raise ReplayDatasetError(f"index out of range within shard: {shard.safetensors_path.name}")

        # Validate shapes once (cheap) for safety.
        if feats.shape[1] != FEATURE_LEN:
            raise ReplayDatasetError(f"bad features shape: {feats.shape}")
        if legal.shape[1] != ACTION_SPACE_A:
            raise ReplayDatasetError(f"bad legal_mask shape: {legal.shape}")
        if pi.shape[1] != ACTION_SPACE_A:
            raise ReplayDatasetError(f"bad pi shape: {pi.shape}")
        if z.shape != (n,):
            raise ReplayDatasetError(f"bad z shape: {z.shape}")
        if z_margin is not None and z_margin.shape != (n,):
            raise ReplayDatasetError(f"bad z_margin shape: {z_margin.shape}")

        feats_i = feats[local]
        legal_i = legal[local]
        pi_i = pi[local]
        z_i = z[local]
        zm_i = 0.0 if z_margin is None else float(z_margin[local])

        x = torch.from_numpy(feats_i.astype(np.float32, copy=False))
        legal_t = torch.from_numpy(legal_i.astype(np.uint8, copy=False))
        pi_t = torch.from_numpy(pi_i.astype(np.float32, copy=False))
        z_t = torch.tensor(float(z_i), dtype=torch.float32)
        zm_t = torch.tensor(float(zm_i), dtype=torch.float32)
        return x, legal_t, pi_t, z_t, zm_t
