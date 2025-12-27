import json
from pathlib import Path

from yatzy_az.trainer import train as train_mod


def test_train_appends_metrics_line_when_run_layout_present(tmp_path: Path) -> None:
    run_root = tmp_path / "runX"
    run_root.mkdir()
    (run_root / "run.json").write_text(json.dumps({"run_id": "runX"}) + "\n")

    train_mod._append_metrics_train_step(  # type: ignore[attr-defined]
        run_root=run_root,
        train_step=1,
        loss_total=1.0,
        loss_policy=0.5,
        loss_value=0.5,
        entropy=0.1,
        lr=0.001,
        throughput_steps_s=123.0,
    )

    p = run_root / "logs" / "metrics.ndjson"
    assert p.exists()
    line = p.read_text().strip()
    ev = json.loads(line)
    assert ev["event"] == "train_step"
    assert ev["train_step"] == 1
    assert "v" in ev


