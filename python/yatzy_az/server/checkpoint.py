"""Inference checkpoint contract (PRD Epic E6.5S1).

This module defines the expected payload format for checkpoints that the inference server can load.

Contract v1:
- torch.load() returns a dict
- required keys: checkpoint_version, model, config, meta
- strict compatibility checks against feature/action/schema IDs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CheckpointV1:
    checkpoint_version: int
    model: dict[str, Any]
    config: dict[str, Any]
    meta: dict[str, Any]


def _require_dict(x: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(x, dict):
        raise CheckpointError(f"{where}: expected dict, got {type(x).__name__}")
    return x


def _require_int(d: dict[str, Any], key: str, *, where: str) -> int:
    if key not in d:
        raise CheckpointError(f"{where}: missing {key!r}")
    v = d[key]
    if not isinstance(v, int):
        raise CheckpointError(f"{where}: {key!r} expected int, got {type(v).__name__}")
    return int(v)


def _require_str(d: dict[str, Any], key: str, *, where: str) -> str:
    if key not in d:
        raise CheckpointError(f"{where}: missing {key!r}")
    v = d[key]
    if not isinstance(v, str):
        raise CheckpointError(f"{where}: {key!r} expected str, got {type(v).__name__}")
    return v


def load_checkpoint(path: Path, *, map_location: str = "cpu") -> CheckpointV1:
    """Load and validate a checkpoint for inference use."""
    path = Path(path)
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise CheckpointError("torch is required to load checkpoints") from e

    raw = torch.load(path, map_location=map_location)
    raw = _require_dict(raw, where=str(path))

    ver = raw.get("checkpoint_version", None)
    if ver is None:
        raise CheckpointError(f"{path}: missing 'checkpoint_version'")
    if not isinstance(ver, int):
        raise CheckpointError(
            f"{path}: 'checkpoint_version' expected int, got {type(ver).__name__}"
        )
    if int(ver) != 1:
        raise CheckpointError(f"{path}: unsupported checkpoint_version={ver} (expected 1)")

    model = raw.get("model", None)
    if model is None:
        raise CheckpointError(f"{path}: missing 'model'")
    model = _require_dict(model, where=f"{path}:model")

    cfg = raw.get("config", None)
    if cfg is None:
        raise CheckpointError(f"{path}: missing 'config'")
    cfg = _require_dict(cfg, where=f"{path}:config")

    meta = raw.get("meta", None)
    if meta is None:
        raise CheckpointError(f"{path}: missing 'meta'")
    meta = _require_dict(meta, where=f"{path}:meta")

    # Compatibility checks.
    from ..model.net import F as FEATURE_LEN, A as ACTION_SPACE_A_MODEL
    from ..replay_dataset import FEATURE_SCHEMA_ID, PROTOCOL_VERSION, RULESET_ID
    from .protocol_v1 import ACTION_SPACE_A as ACTION_SPACE_A_PROTOCOL

    feature_len = _require_int(cfg, "feature_len", where=f"{path}:config")
    if feature_len != int(FEATURE_LEN):
        raise CheckpointError(
            f"{path}:config['feature_len']={feature_len} mismatches expected {FEATURE_LEN}"
        )

    action_space_a = _require_int(cfg, "action_space_a", where=f"{path}:config")
    if action_space_a != int(ACTION_SPACE_A_PROTOCOL) or action_space_a != int(
        ACTION_SPACE_A_MODEL
    ):
        raise CheckpointError(
            f"{path}:config['action_space_a']={action_space_a} mismatches expected {ACTION_SPACE_A_PROTOCOL}"
        )

    _require_int(cfg, "hidden", where=f"{path}:config")
    _require_int(cfg, "blocks", where=f"{path}:config")

    pv = _require_int(meta, "protocol_version", where=f"{path}:meta")
    if pv != int(PROTOCOL_VERSION):
        raise CheckpointError(
            f"{path}:meta['protocol_version']={pv} mismatches expected {PROTOCOL_VERSION}"
        )

    fsid = _require_int(meta, "feature_schema_id", where=f"{path}:meta")
    if fsid != int(FEATURE_SCHEMA_ID):
        raise CheckpointError(
            f"{path}:meta['feature_schema_id']={fsid} mismatches expected {FEATURE_SCHEMA_ID}"
        )

    rid = _require_str(meta, "ruleset_id", where=f"{path}:meta")
    if rid != str(RULESET_ID):
        raise CheckpointError(
            f"{path}:meta['ruleset_id']={rid!r} mismatches expected {RULESET_ID!r}"
        )

    # Redundant sanity (meta.action_space_a should match too, if present).
    if "action_space_a" in meta:
        asa = _require_int(meta, "action_space_a", where=f"{path}:meta")
        if asa != int(ACTION_SPACE_A_PROTOCOL):
            raise CheckpointError(
                f"{path}:meta['action_space_a']={asa} mismatches expected {ACTION_SPACE_A_PROTOCOL}"
            )

    return CheckpointV1(checkpoint_version=1, model=model, config=cfg, meta=meta)


