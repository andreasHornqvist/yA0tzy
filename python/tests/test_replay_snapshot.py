import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from yatzy_az import replay_dataset as rd
from yatzy_az import replay_snapshot as rs


def _write_shard(replay_dir: Path, idx: int, n: int) -> None:
    st = replay_dir / f"shard_{idx:06}.safetensors"
    meta = replay_dir / f"shard_{idx:06}.meta.json"

    feats = np.zeros((n, rd.FEATURE_LEN), dtype=np.float32)
    legal = np.ones((n, rd.ACTION_SPACE_A), dtype=np.uint8)
    pi = np.zeros((n, rd.ACTION_SPACE_A), dtype=np.float32)
    pi[:, 0] = 1.0
    z = np.zeros((n,), dtype=np.float32)

    save_file(
        {
            rd.T_FEATURES: feats,
            rd.T_LEGAL_MASK: legal,
            rd.T_PI: pi,
            rd.T_Z: z,
        },
        str(st),
    )
    meta.write_text(
        json.dumps(
            {
                "protocol_version": rd.PROTOCOL_VERSION,
                "feature_schema_id": rd.FEATURE_SCHEMA_ID,
                "feature_len": rd.FEATURE_LEN,
                "action_space_id": rd.ACTION_SPACE_ID,
                "action_space_a": rd.ACTION_SPACE_A,
                "ruleset_id": rd.RULESET_ID,
                "num_samples": n,
                "git_hash": None,
                "config_hash": None,
            }
        )
    )


def test_create_snapshot_is_sorted_and_stable(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    _write_shard(replay_dir, 2, 3)
    _write_shard(replay_dir, 1, 5)

    snap_path = tmp_path / "replay_snapshot.json"
    snap = rs.create_snapshot(replay_dir=replay_dir, out_path=snap_path)
    assert snap_path.exists()

    names = rs.shard_filenames(snap)
    assert names == ["shard_000001.safetensors", "shard_000002.safetensors"]


def test_snapshot_does_not_silently_include_new_shards(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    _write_shard(replay_dir, 1, 2)
    snap_path = tmp_path / "replay_snapshot.json"
    snap = rs.create_snapshot(replay_dir=replay_dir, out_path=snap_path)
    assert rs.shard_filenames(snap) == ["shard_000001.safetensors"]

    # Add a new shard after snapshot creation.
    _write_shard(replay_dir, 2, 2)

    snap2 = rs.load_snapshot(snap_path)
    shards = rs.shards_from_snapshot(replay_dir=replay_dir, snapshot=snap2)
    assert [s.safetensors_path.name for s in shards] == ["shard_000001.safetensors"]


def test_snapshot_missing_shard_fails_loudly(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    _write_shard(replay_dir, 1, 2)
    snap_path = tmp_path / "replay_snapshot.json"
    rs.create_snapshot(replay_dir=replay_dir, out_path=snap_path)

    # Remove the shard after snapshot.
    (replay_dir / "shard_000001.safetensors").unlink()

    snap = rs.load_snapshot(snap_path)
    with pytest.raises(rs.ReplaySnapshotError, match="missing on disk"):
        rs.shards_from_snapshot(replay_dir=replay_dir, snapshot=snap)


def test_snapshot_write_is_atomic_wrt_tmp_file(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    _write_shard(replay_dir, 1, 2)

    snap_path = tmp_path / "replay_snapshot.json"
    rs.create_snapshot(replay_dir=replay_dir, out_path=snap_path)

    # Leave a corrupt tmp file around; final snapshot must still be readable.
    tmp = snap_path.with_suffix(snap_path.suffix + ".tmp")
    tmp.write_text("{not json")

    snap = rs.load_snapshot(snap_path)
    assert rs.shard_filenames(snap) == ["shard_000001.safetensors"]
