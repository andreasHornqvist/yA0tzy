"""Inference model interface + dummy implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .checkpoint import load_checkpoint
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


class TorchModel(Model):
    """Real PyTorch-backed model (YatzyNet).

    Returns *raw* logits; masking/softmax is handled by the Rust client.
    """

    def __init__(self, *, checkpoint_path: Path, device: str) -> None:
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("TorchModel requires torch") from e

        ckpt = load_checkpoint(Path(checkpoint_path), map_location="cpu")

        cfg = ckpt.config
        hidden = int(cfg["hidden"])
        blocks = int(cfg["blocks"])

        from ..model import YatzyNet, YatzyNetConfig

        self._device = torch.device(str(device))
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but torch.cuda.is_available() is false")

        m = YatzyNet(YatzyNetConfig(hidden=hidden, blocks=blocks))
        m.load_state_dict(ckpt.model, strict=True)
        m.eval()
        m.to(self._device)

        self._model = m

    def infer_batch(
        self, features_batch: list[list[float]], legal_mask_batch: list[bytes]
    ) -> list[InferOutput]:
        # NOTE: we keep legal_mask_batch in the interface for protocol compatibility, but do not
        # apply masking here (Rust is responsible for masking + softmax).
        _ = legal_mask_batch
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("TorchModel requires torch") from e

        if len(features_batch) == 0:
            return []

        # One conversion per batch. For GPU, do a single host->device transfer per batch.
        x = torch.as_tensor(features_batch, dtype=torch.float32)
        if x.ndim != 2:
            raise RuntimeError(f"bad features tensor rank: {x.ndim} (expected 2)")
        if self._device.type != "cpu":
            x = x.to(self._device)
        x = x.contiguous()

        with torch.inference_mode():
            logits, v = self._model(x)

        if logits.ndim != 2 or logits.shape[0] != x.shape[0] or logits.shape[1] != ACTION_SPACE_A:
            raise RuntimeError(f"bad logits shape: {tuple(logits.shape)} (expected [B,{ACTION_SPACE_A}])")
        if v.ndim != 1 or v.shape[0] != x.shape[0]:
            raise RuntimeError(f"bad value shape: {tuple(v.shape)} (expected [B])")

        logits_cpu = logits.detach().to("cpu")
        v_cpu = v.detach().to("cpu")
        logits_list: list[list[float]] = logits_cpu.tolist()
        v_list: list[float] = v_cpu.tolist()

        out: list[InferOutput] = []
        for ls, vv in zip(logits_list, v_list, strict=True):
            out.append(
                InferOutput(policy_logits=[float(z) for z in ls], value=float(vv), margin=None)
            )
        return out


def parse_model_spec(spec: str) -> tuple[str, float | None]:
    """Parse a model spec string.

    Supported:
    - 'dummy'
    - 'dummy:<value>' where value is a float (e.g. dummy:0.1, dummy:-0.1)
    - 'path:<checkpoint_path>' (e.g. path:runs/smoke/models/best.pt)
    """
    head, sep, tail = spec.partition(":")
    head = head.strip().lower()
    if sep == "":
        return (head, None)
    tail = tail.strip()
    if head == "path":
        if tail == "":
            raise ValueError(f"empty path model spec: {spec!r}")
        # For 'path:' specs, we return a placeholder value and let build_model parse the path.
        # (We keep this function minimal and backward-compatible for dummy parsing.)
        return ("path", None)
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


def build_model(spec: str, *, device: str) -> Model:
    head, _sep, tail = spec.partition(":")
    kind, value = parse_model_spec(spec)
    if kind == "dummy":
        return DummyModel(value=0.0 if value is None else value)
    if kind == "path":
        # Preserve original tail as the path string (can contain ':' on Windows paths, but we
        # don't support Windows in v1 and this keeps the syntax simple).
        path_s = tail.strip()
        if path_s == "":
            raise ValueError(f"empty path model spec: {spec!r}")
        return TorchModel(checkpoint_path=Path(path_s), device=device)
    raise ValueError(f"unknown model spec kind: {kind!r}")


def build_torch_model(*, checkpoint_path: Path, device: str) -> TorchModel:
    """Internal helper (CLI wiring is handled in E6.5S3)."""
    return TorchModel(checkpoint_path=checkpoint_path, device=device)
