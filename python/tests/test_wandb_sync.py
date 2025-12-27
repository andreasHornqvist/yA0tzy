import json
from pathlib import Path

import pytest

from yatzy_az import wandb_sync


def test_normalize_event_flattens_metrics() -> None:
    ev = {
        "event": "train_step",
        "ts_ms": 1,
        "run_id": "r",
        "train_step": 7,
        "loss_total": 1.0,
        "entropy": 0.5,
    }
    out = wandb_sync.normalize_event(ev)
    assert out["step"] == 7
    assert out["metrics"]["train_step/loss_total"] == 1.0
    assert out["metrics"]["train_step/entropy"] == 0.5


def test_wandb_sync_reads_metrics_ndjson(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "run1"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "metrics.ndjson").write_text(
        json.dumps({"event": "train_step", "ts_ms": 1, "run_id": "r", "train_step": 1})
        + "\n"
    )

    args = type("Args", (), {"run": str(run_dir)})()
    rc = wandb_sync.run_from_args(args)
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    v = json.loads(out[0])
    assert v["event"] == "train_step"
    assert v["step"] == 1


