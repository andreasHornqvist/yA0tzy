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


class DummyModel(Model):
    """Uniform logits over legal actions, value=0."""

    def infer_batch(
        self, features_batch: list[list[float]], legal_mask_batch: list[bytes]
    ) -> list[InferOutput]:
        out: list[InferOutput] = []
        for _features, legal in zip(features_batch, legal_mask_batch, strict=True):
            logits = [0.0] * ACTION_SPACE_A
            for i, b in enumerate(legal):
                if b == 0:
                    logits[i] = -1.0e9
            out.append(InferOutput(policy_logits=logits, value=0.0, margin=None))
        return out
