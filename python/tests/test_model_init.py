from __future__ import annotations

from pathlib import Path


def test_model_init_writes_contract_checkpoint_and_torchmodel_loads(tmp_path: Path) -> None:
    from yatzy_az.model.init import init_model_checkpoint
    from yatzy_az.model.net import A, F
    from yatzy_az.server.checkpoint import load_checkpoint
    from yatzy_az.server.model import TorchModel

    out = tmp_path / "models" / "best.pt"
    init_model_checkpoint(out, hidden=8, blocks=1)

    ck = load_checkpoint(out)
    assert ck.checkpoint_version == 1

    tm = TorchModel(checkpoint_path=out, device="cpu")
    outs = tm.infer_batch([[0.0] * F], [bytes([1] * A)])
    assert len(outs) == 1
    assert len(outs[0].policy_logits) == A
    assert -1.0 <= outs[0].value <= 1.0


