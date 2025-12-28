from __future__ import annotations

from pathlib import Path

import pytest


def _save_ckpt(tmp_path: Path, payload: dict) -> Path:
    import torch

    p = tmp_path / "ckpt.pt"
    torch.save(payload, p)
    return p


def test_checkpoint_contract_v1_loads_valid_checkpoint(tmp_path: Path) -> None:
    import torch

    from yatzy_az.model import YatzyNet, YatzyNetConfig
    from yatzy_az.model.net import A, F
    from yatzy_az.replay_dataset import FEATURE_SCHEMA_ID, PROTOCOL_VERSION, RULESET_ID
    from yatzy_az.server.checkpoint import load_checkpoint

    model = YatzyNet(YatzyNetConfig(hidden=8, blocks=1))
    _ = model(torch.zeros((2, F), dtype=torch.float32))

    payload = {
        "checkpoint_version": 1,
        "model": model.state_dict(),
        "config": {"hidden": 8, "blocks": 1, "feature_len": F, "action_space_a": A},
        "meta": {
            "protocol_version": PROTOCOL_VERSION,
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "ruleset_id": RULESET_ID,
            "action_space_a": A,
            "train_step": 123,
        },
    }
    p = _save_ckpt(tmp_path, payload)
    ck = load_checkpoint(p)
    assert ck.checkpoint_version == 1


def test_checkpoint_contract_rejects_missing_version(tmp_path: Path) -> None:
    from yatzy_az.server.checkpoint import CheckpointError, load_checkpoint

    p = _save_ckpt(tmp_path, {"model": {}, "config": {}, "meta": {}})
    with pytest.raises(CheckpointError, match="checkpoint_version"):
        load_checkpoint(p)


def test_checkpoint_contract_rejects_wrong_feature_len(tmp_path: Path) -> None:
    import torch

    from yatzy_az.model import YatzyNet, YatzyNetConfig
    from yatzy_az.model.net import A, F
    from yatzy_az.replay_dataset import FEATURE_SCHEMA_ID, PROTOCOL_VERSION, RULESET_ID
    from yatzy_az.server.checkpoint import CheckpointError, load_checkpoint

    model = YatzyNet(YatzyNetConfig(hidden=8, blocks=1))
    _ = model(torch.zeros((1, F), dtype=torch.float32))

    payload = {
        "checkpoint_version": 1,
        "model": model.state_dict(),
        "config": {"hidden": 8, "blocks": 1, "feature_len": F + 1, "action_space_a": A},
        "meta": {
            "protocol_version": PROTOCOL_VERSION,
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "ruleset_id": RULESET_ID,
            "action_space_a": A,
        },
    }
    p = _save_ckpt(tmp_path, payload)
    with pytest.raises(CheckpointError, match="feature_len"):
        load_checkpoint(p)


def test_checkpoint_contract_rejects_wrong_schema_id(tmp_path: Path) -> None:
    import torch

    from yatzy_az.model import YatzyNet, YatzyNetConfig
    from yatzy_az.model.net import A, F
    from yatzy_az.replay_dataset import FEATURE_SCHEMA_ID, PROTOCOL_VERSION, RULESET_ID
    from yatzy_az.server.checkpoint import CheckpointError, load_checkpoint

    model = YatzyNet(YatzyNetConfig(hidden=8, blocks=1))
    _ = model(torch.zeros((1, F), dtype=torch.float32))

    payload = {
        "checkpoint_version": 1,
        "model": model.state_dict(),
        "config": {"hidden": 8, "blocks": 1, "feature_len": F, "action_space_a": A},
        "meta": {
            "protocol_version": PROTOCOL_VERSION,
            "feature_schema_id": FEATURE_SCHEMA_ID + 999,
            "ruleset_id": RULESET_ID,
            "action_space_a": A,
        },
    }
    p = _save_ckpt(tmp_path, payload)
    with pytest.raises(CheckpointError, match="feature_schema_id"):
        load_checkpoint(p)


