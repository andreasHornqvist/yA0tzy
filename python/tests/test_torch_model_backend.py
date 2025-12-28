from __future__ import annotations

from pathlib import Path

import pytest


def test_torch_model_backend_shapes_and_determinism(tmp_path: Path) -> None:
    import torch

    from yatzy_az.model import YatzyNet, YatzyNetConfig
    from yatzy_az.model.net import A, F
    from yatzy_az.replay_dataset import FEATURE_SCHEMA_ID, PROTOCOL_VERSION, RULESET_ID
    from yatzy_az.server.model import TorchModel

    # Build a tiny model and write a v1 checkpoint matching the contract.
    model = YatzyNet(YatzyNetConfig(hidden=8, blocks=1))
    model.eval()

    payload = {
        "checkpoint_version": 1,
        "model": model.state_dict(),
        "config": {"hidden": 8, "blocks": 1, "feature_len": F, "action_space_a": A},
        "meta": {
            "protocol_version": PROTOCOL_VERSION,
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "ruleset_id": RULESET_ID,
            "action_space_a": A,
            "train_step": 0,
        },
    }

    ckpt_path = tmp_path / "best.pt"
    torch.save(payload, ckpt_path)

    tm = TorchModel(checkpoint_path=ckpt_path, device="cpu")

    feats = [[0.0] * F, [1.0] * F, [0.1 * i for i in range(F)]]
    masks = [bytes([1] * A) for _ in feats]  # ignored by TorchModel (masking is Rust-side)

    out1 = tm.infer_batch(feats, masks)
    out2 = tm.infer_batch(feats, masks)

    assert len(out1) == len(feats)
    assert len(out2) == len(feats)

    for a, b in zip(out1, out2, strict=True):
        assert len(a.policy_logits) == A
        assert all(isinstance(x, float) for x in a.policy_logits)
        assert pytest.approx(a.policy_logits, abs=0.0) == b.policy_logits
        assert -1.0 <= a.value <= 1.0
        assert pytest.approx(a.value, abs=0.0) == b.value


