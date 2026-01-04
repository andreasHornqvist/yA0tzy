import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from yatzy_az import replay_dataset as rd


def _write_one_shard(replay_dir: Path, *, schema_ok: bool = True) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)

    n = 3
    feats = np.zeros((n, rd.FEATURE_LEN), dtype=np.float32)
    legal = np.ones((n, rd.ACTION_SPACE_A), dtype=np.uint8)
    pi = np.zeros((n, rd.ACTION_SPACE_A), dtype=np.float32)
    pi[:, 0] = 1.0
    z = np.zeros((n,), dtype=np.float32)

    st_path = replay_dir / "shard_000000.safetensors"
    save_file(
        {
            rd.T_FEATURES: feats,
            rd.T_LEGAL_MASK: legal,
            rd.T_PI: pi,
            rd.T_Z: z,
        },
        str(st_path),
    )

    meta = {
        "protocol_version": rd.PROTOCOL_VERSION,
        "feature_schema_id": rd.FEATURE_SCHEMA_ID if schema_ok else (rd.FEATURE_SCHEMA_ID + 1),
        "feature_len": rd.FEATURE_LEN,
        "action_space_id": rd.ACTION_SPACE_ID,
        "action_space_a": rd.ACTION_SPACE_A,
        "ruleset_id": rd.RULESET_ID,
        "num_samples": n,
        "git_hash": None,
        "config_hash": None,
    }
    (replay_dir / "shard_000000.meta.json").write_text(json.dumps(meta))


def test_discover_replay_shards_validates_schema_ids(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    _write_one_shard(replay_dir, schema_ok=True)
    shards = rd.discover_replay_shards(replay_dir)
    assert len(shards) == 1
    assert shards[0].meta.num_samples == 3


def test_discover_replay_shards_rejects_schema_mismatch(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    _write_one_shard(replay_dir, schema_ok=False)
    with pytest.raises(rd.ReplayDatasetError, match="feature_schema_id mismatch"):
        rd.discover_replay_shards(replay_dir)


def test_iter_samples_numpy_yields_expected_shapes(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    _write_one_shard(replay_dir, schema_ok=True)
    shards = rd.discover_replay_shards(replay_dir)
    it = rd.iter_samples_numpy(shards, shuffle_shards=False, repeat=False)
    x, legal, pi, z, z_margin = next(it)
    assert x.shape == (rd.FEATURE_LEN,)
    assert legal.shape == (rd.ACTION_SPACE_A,)
    assert pi.shape == (rd.ACTION_SPACE_A,)
    assert isinstance(z.item() if hasattr(z, "item") else z, (float, np.floating))
    assert z_margin is None


def test_random_indexed_dataset_is_map_style_and_dataloader_shuffle_works(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    replay_dir = tmp_path / "replay"
    _write_one_shard(replay_dir, schema_ok=True)

    ds = rd.ReplayRandomAccessDataset(replay_dir, shard_files=["shard_000000.safetensors"], seed=0, cache_shards=1)
    assert len(ds) == 3

    x, legal, pi, z, zm = ds[0]
    assert tuple(x.shape) == (rd.FEATURE_LEN,)
    assert tuple(legal.shape) == (rd.ACTION_SPACE_A,)
    assert tuple(pi.shape) == (rd.ACTION_SPACE_A,)
    assert tuple(z.shape) == ()
    assert tuple(zm.shape) == ()

    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, drop_last=False)
    xb, legalb, pib, zb, zmb = next(iter(dl))
    assert tuple(xb.shape) == (2, rd.FEATURE_LEN)
    assert tuple(legalb.shape) == (2, rd.ACTION_SPACE_A)
    assert tuple(pib.shape) == (2, rd.ACTION_SPACE_A)
    assert tuple(zb.shape) == (2,)
    assert tuple(zmb.shape) == (2,)
