import json
from pathlib import Path

import pytest

from yatzy_az.run_manifest import RunManifestError, load_manifest, save_manifest_atomic


def test_save_manifest_atomic_roundtrip(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)

    m = {
        "run_manifest_version": 1,
        "run_id": "x",
        "train_step": 0,
        "selfplay_games_completed": 10,
        "candidate_checkpoint": None,
    }
    save_manifest_atomic(run_root, m)

    got = load_manifest(run_root)
    assert got["run_id"] == "x"
    assert got["train_step"] == 0

    got["train_step"] = 123
    got["candidate_checkpoint"] = "models/candidate.pt"
    save_manifest_atomic(run_root, got)

    got2 = json.loads((run_root / "run.json").read_text())
    assert got2["train_step"] == 123
    assert got2["candidate_checkpoint"] == "models/candidate.pt"


def test_load_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(RunManifestError):
        load_manifest(tmp_path)
