"""Model initialization utilities (PRD E6.5S4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def init_model_checkpoint(out: Path, *, hidden: int, blocks: int) -> None:
    """Create a fresh, contract-compliant checkpoint for inference/training bootstrap."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import torch

    from ..model import A, F, YatzyNet, YatzyNetConfig
    from ..replay_dataset import FEATURE_SCHEMA_ID, PROTOCOL_VERSION, RULESET_ID

    model = YatzyNet(YatzyNetConfig(hidden=int(hidden), blocks=int(blocks)))
    model.eval()

    payload: dict[str, Any] = {
        "checkpoint_version": 1,
        "model": model.state_dict(),
        "config": {
            "hidden": int(hidden),
            "blocks": int(blocks),
            "feature_len": int(F),
            "action_space_a": int(A),
        },
        "meta": {
            "protocol_version": int(PROTOCOL_VERSION),
            "feature_schema_id": int(FEATURE_SCHEMA_ID),
            "ruleset_id": str(RULESET_ID),
            "action_space_a": int(A),
            "train_step": 0,
        },
    }

    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)


