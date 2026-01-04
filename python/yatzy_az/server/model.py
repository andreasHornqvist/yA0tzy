"""Inference model interface + dummy implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time as _time
import json as _json

import numpy as np

from .checkpoint import load_checkpoint
from .debug_log import emit as _dbg_emit, enabled as _dbg_enabled
from .protocol_v1 import ACTION_SPACE_A, LEGAL_MASK_BITSET_BYTES


_BITS_LUT_U8: np.ndarray | None = None


def _bits_lut_u8() -> np.ndarray:
    """Return a [256,8] uint8 LUT mapping byte->LSB-first bits."""
    global _BITS_LUT_U8
    if _BITS_LUT_U8 is not None:
        return _BITS_LUT_U8
    lut = np.zeros((256, 8), dtype=np.uint8)
    for b in range(256):
        for bit in range(8):
            lut[b, bit] = (b >> bit) & 1
    _BITS_LUT_U8 = lut
    return lut


def _unpack_legal_mask_bitset_lsb(legal6: bytes) -> np.ndarray:
    """Unpack 6-byte bitset into a uint8 mask of length ACTION_SPACE_A (0/1)."""
    if len(legal6) != LEGAL_MASK_BITSET_BYTES:
        raise ValueError(f"expected {LEGAL_MASK_BITSET_BYTES} bytes, got {len(legal6)}")
    arr = np.frombuffer(legal6, dtype=np.uint8, count=LEGAL_MASK_BITSET_BYTES)
    bits = _bits_lut_u8()[arr].reshape(-1)  # 48 bits
    return bits[:ACTION_SPACE_A]


class Model:
    def infer_batch_packed(
        self, x_np: np.ndarray, legal_mask_batch: list[bytes]
    ) -> tuple[bytes, bytes, bytes | None]:
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

    def infer_batch_packed(
        self, x_np: np.ndarray, legal_mask_batch: list[bytes]
    ) -> tuple[bytes, bytes, bytes | None]:
        # NOTE: we keep legal_mask_batch in the interface for protocol compatibility, but do not
        # apply masking here (Rust is responsible for masking + softmax).
        _ = legal_mask_batch
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("TorchModel requires torch") from e

        if x_np.size == 0:
            return (b"", b"", None)

        t0 = _time.monotonic()
        # One conversion per batch. For GPU, do a single host->device transfer per batch.
        if x_np.dtype != np.float32:
            x_np = x_np.astype(np.float32, copy=False)
        if not x_np.flags["C_CONTIGUOUS"]:
            x_np = np.ascontiguousarray(x_np, dtype=np.float32)
        x = torch.from_numpy(x_np)
        t_as_tensor = _time.monotonic()
        if x.ndim != 2:
            raise RuntimeError(f"bad features tensor rank: {x.ndim} (expected 2)")
        if self._device.type != "cpu":
            x = x.to(self._device)
        t_to_dev = _time.monotonic()
        x = x.contiguous()

        with torch.inference_mode():
            logits, v = self._model(x)
        t_fwd = _time.monotonic()

        if logits.ndim != 2 or logits.shape[0] != x.shape[0] or logits.shape[1] != ACTION_SPACE_A:
            raise RuntimeError(f"bad logits shape: {tuple(logits.shape)} (expected [B,{ACTION_SPACE_A}])")
        if v.ndim != 1 or v.shape[0] != x.shape[0]:
            raise RuntimeError(f"bad value shape: {tuple(v.shape)} (expected [B])")

        logits_cpu = logits.detach().to("cpu")
        v_cpu = v.detach().to("cpu")
        t_to_cpu = _time.monotonic()
        logits_bytes = (
            logits_cpu.contiguous().numpy().astype(np.float32, copy=False).tobytes()
        )
        values_bytes = v_cpu.contiguous().numpy().astype(np.float32, copy=False).tobytes()
        t_pack = _time.monotonic()

        # region agent log
        if _dbg_enabled():
            try:
                total_ms = (t_pack - t0) * 1000.0
                sampled = (int(t_pack * 1000) & 0x3F) == 0  # ~1/64
                if total_ms >= 5.0 or sampled:
                    _dbg_emit(
                        {
                            "timestamp": int(_time.time() * 1000),
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H_latency",
                            "location": "python/yatzy_az/server/model.py:TorchModel.infer_batch",
                            "message": "torch timings",
                            "data": {
                                "device": str(self._device),
                                "batch": int(x_np.shape[0]),
                                "sampled": bool(sampled),
                                "as_tensor_ms": (t_as_tensor - t0) * 1000.0,
                                "to_device_ms": (t_to_dev - t_as_tensor) * 1000.0,
                                "forward_ms": (t_fwd - t_to_dev) * 1000.0,
                                "to_cpu_ms": (t_to_cpu - t_fwd) * 1000.0,
                                "pack_ms": (t_pack - t_to_cpu) * 1000.0,
                                "total_ms": total_ms,
                            },
                        }
                    )
            except Exception:
                pass
        # endregion agent log
        return (logits_bytes, values_bytes, None)


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

    def infer_batch_packed(
        self, x_np: np.ndarray, legal_mask_batch: list[bytes]
    ) -> tuple[bytes, bytes, bytes | None]:
        # For dummy we ignore features (but keep shape checks minimal).
        bsz = int(x_np.shape[0])
        logits = np.zeros((bsz, ACTION_SPACE_A), dtype=np.float32)
        for i, legal in enumerate(legal_mask_batch):
            if len(legal) == LEGAL_MASK_BITSET_BYTES:
                m = _unpack_legal_mask_bitset_lsb(legal)
            else:
                m = np.frombuffer(legal, dtype=np.uint8, count=ACTION_SPACE_A)
            logits[i, m == 0] = -1.0e9
        values = np.full((bsz,), float(self._value), dtype=np.float32)
        return (logits.tobytes(), values.tobytes(), None)


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
