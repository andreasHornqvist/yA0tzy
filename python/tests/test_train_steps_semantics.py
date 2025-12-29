from __future__ import annotations

import json
from pathlib import Path

import pytest

from yatzy_az.trainer.train import (
    _best_effort_update_run_manifest_train_progress,
    derive_steps_from_epochs,
)


def test_derive_steps_from_epochs_uses_ceil() -> None:
    # total_samples=1001, batch_size=256 => ceil=4 => epochs=3 => 12
    assert derive_steps_from_epochs(epochs=3, total_samples=1001, batch_size=256) == 12


def test_best_effort_updates_current_iteration_steps_target(tmp_path: Path) -> None:
    run_root = tmp_path / "run_1"
    run_root.mkdir()
    (run_root / "logs").mkdir()

    run_json = run_root / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "run_id": "run_1",
                "controller_iteration_idx": 2,
                "iterations": [
                    {"idx": 0, "train": {}},
                    {"idx": 1, "train": {}},
                    {"idx": 2, "train": {}},
                ],
            },
            indent=2,
        )
        + "\n"
    )

    _best_effort_update_run_manifest_train_progress(
        run_root=run_root, steps_target=123, steps_completed=7
    )
    m = json.loads(run_json.read_text())
    it = next(x for x in m["iterations"] if x["idx"] == 2)
    assert it["train"]["steps_target"] == 123
    assert it["train"]["steps_completed"] == 7


def test_epochs_mode_requires_snapshot_when_steps_not_set(tmp_path: Path) -> None:
    # In epochs-mode (cfg exists, steps_per_iteration is None), disabling snapshot must error.
    # This is enforced before importing torch / constructing the dataset.
    from yatzy_az.trainer import train as train_mod

    run_root = tmp_path / "run_1"
    models_dir = run_root / "models"
    replay_dir = run_root / "replay"
    run_root.mkdir()
    models_dir.mkdir()
    replay_dir.mkdir()

    # Minimal run.json to allow best-effort manifest updates (not required for the error).
    (run_root / "run.json").write_text(
        json.dumps({"run_id": "run_1", "controller_iteration_idx": 0, "iterations": [{"idx": 0, "train": {}}]})
        + "\n"
    )

    # Config: epochs-mode (steps_per_iteration unset).
    (run_root / "config.yaml").write_text(
        """\
inference:
  bind: "unix:///tmp/yatzy_infer.sock"
  device: "cpu"
  max_batch: 32
  max_wait_us: 1000
mcts:
  c_puct: 1.5
  budget_reroll: 100
  budget_mark: 100
  max_inflight_per_game: 4
selfplay:
  games_per_iteration: 1
  workers: 1
  threads_per_worker: 1
training:
  batch_size: 256
  learning_rate: 0.001
  epochs: 2
  weight_decay: 0.0
  steps_per_iteration: null
gating:
  games: 1
  paired_seed_swap: true
  deterministic_chance: true
replay:
  capacity_shards: null
controller:
  total_iterations: 1
"""
    )

    args = train_mod.argparse.Namespace(
        replay=str(replay_dir),
        config=None,
        best=str(models_dir / "best.pt"),
        resume=None,
        out=str(models_dir),
        batch_size=None,
        num_workers=0,
        steps=None,
        shuffle_shards=False,
        no_repeat=True,
        device="cpu",
        seed=0,
        lr=None,
        weight_decay=None,
        value_lambda=1.0,
        hidden=8,
        blocks=1,
        log_every=25,
        snapshot=None,
        no_snapshot=True,
    )

    with pytest.raises(RuntimeError, match="epochs-mode requires replay snapshot semantics"):
        train_mod.run_from_args(args)


