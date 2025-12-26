"""Inference model interface + dummy implementation."""

from __future__ import annotations

from dataclasses import dataclass

from .protocol_v1 import ACTION_SPACE_A


@dataclass(frozen=True, slots=True)
class InferOutput:
    policy_logits: list[float]  # len=A
    value: float
    margin: float | None = None


class Model:
    def infer_batch(
        self, features_batch: list[list[float]], legal_mask_batch: list[bytes]
    ) -> list[InferOutput]:
        raise NotImplementedError


def parse_model_spec(spec: str) -> tuple[str, float | None]:
    """Parse a model spec string.

    Supported:
    - 'dummy'
    - 'dummy:<value>' where value is a float (e.g. dummy:0.1, dummy:-0.1)
    """
    head, sep, tail = spec.partition(":")
    head = head.strip().lower()
    if sep == "":
        return (head, None)
    tail = tail.strip()
    try:
        v = float(tail)
    except ValueError as e:
        raise ValueError(f"invalid model spec value: {spec!r}") from e
    return (head, v)


class DummyModel(Model):
    """Uniform logits over legal actions, value configurable."""

    def __init__(self, *, value: float = 0.0) -> None:
        self._value = float(value)

    def infer_batch(
        self, features_batch: list[list[float]], legal_mask_batch: list[bytes]
    ) -> list[InferOutput]:
        out: list[InferOutput] = []
        for _features, legal in zip(features_batch, legal_mask_batch, strict=True):
            logits = [0.0] * ACTION_SPACE_A
            for i, b in enumerate(legal):
                if b == 0:
                    logits[i] = -1.0e9
            out.append(InferOutput(policy_logits=logits, value=self._value, margin=None))
        return out


def build_model(spec: str) -> Model:
    kind, value = parse_model_spec(spec)
    if kind == "dummy":
        return DummyModel(value=0.0 if value is None else value)
    raise ValueError(f"unknown model spec kind: {kind!r}")
