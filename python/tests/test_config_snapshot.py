from __future__ import annotations

from pathlib import Path

from yatzy_az.run_manifest import save_manifest_atomic
from yatzy_az.trainer.train import _ensure_run_config_snapshot


def test_trainer_config_snapshot_written_when_missing(tmp_path: Path) -> None:
    # Simulate run layout: run.json exists but config.yaml does not.
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    save_manifest_atomic(run_root, {"run_id": "run"})

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
inference:
  bind: "unix:///tmp/test.sock"
  device: "cpu"
  max_batch: 16
  max_wait_us: 500
mcts:
  c_puct: 1.0
  budget_reroll: 10
  budget_mark: 10
  max_inflight_per_game: 2
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25
  temperature_schedule:
    kind: "constant"
    t0: 1.0
selfplay:
  games_per_iteration: 1
  workers: 1
  threads_per_worker: 1
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 1
gating:
  games: 2
  win_rate_threshold: 0.5
  paired_seed_swap: false
  deterministic_chance: true
"""
    )

    _ensure_run_config_snapshot(run_root=run_root, config_path=cfg_path)
    assert (run_root / "config.yaml").exists()

