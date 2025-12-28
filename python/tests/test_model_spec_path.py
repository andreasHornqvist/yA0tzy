from __future__ import annotations

from pathlib import Path


def test_build_model_supports_path_and_dummy_specs(tmp_path: Path) -> None:
    import torch

    from yatzy_az.model import YatzyNet, YatzyNetConfig
    from yatzy_az.model.net import A, F
    from yatzy_az.replay_dataset import FEATURE_SCHEMA_ID, PROTOCOL_VERSION, RULESET_ID
    from yatzy_az.server.model import DummyModel, TorchModel, build_model

    # Create a valid checkpoint payload.
    m = YatzyNet(YatzyNetConfig(hidden=8, blocks=1))
    payload = {
        "checkpoint_version": 1,
        "model": m.state_dict(),
        "config": {"hidden": 8, "blocks": 1, "feature_len": F, "action_space_a": A},
        "meta": {
            "protocol_version": PROTOCOL_VERSION,
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "ruleset_id": RULESET_ID,
            "action_space_a": A,
        },
    }
    ckpt = tmp_path / "best.pt"
    torch.save(payload, ckpt)

    tm = build_model(f"path:{ckpt}", device="cpu")
    assert isinstance(tm, TorchModel)
    outs = tm.infer_batch([[0.0] * F], [bytes([1] * A)])
    assert len(outs) == 1
    assert len(outs[0].policy_logits) == A

    dm = build_model("dummy:0.2", device="cpu")
    assert isinstance(dm, DummyModel)


