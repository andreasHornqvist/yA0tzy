"""Microbench for Protocol v1 codec + tensorization (no pytest required).

Usage:
  python -m yatzy_az.server.bench_codec_v1 --n 200000
"""

from __future__ import annotations

import argparse
import struct
import time

import numpy as np

from .protocol_v1 import (
    ACTION_SPACE_A,
    FEATURE_LEN_V1,
    FEATURE_SCHEMA_ID_V1,
    InferRequestV1,
    InferResponseV1,
    InferResponseV1Packed,
    decode_request_v1,
    decode_request_v1_packed,
    encode_request_v1,
    encode_response_v1,
    encode_response_v1_packed,
)


def _bench(name: str, n: int, fn) -> None:
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    it_s = n / max(1e-12, dt)
    us_it = (dt / max(1, n)) * 1e6
    print(f"{name:28s} {it_s:12.0f} it/s  {us_it:9.3f} us/it")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="Iterations")
    args = ap.parse_args()
    n = int(args.n)

    # Pre-build one request payload to avoid measuring encode_request_v1 itself.
    req = InferRequestV1(
        request_id=123,
        model_id=0,
        feature_schema_id=FEATURE_SCHEMA_ID_V1,
        features=[0.0] * FEATURE_LEN_V1,
        legal_mask=bytes([1] * ACTION_SPACE_A),
    )
    payload = encode_request_v1(req)

    def run_decode_v1() -> None:
        for _ in range(n):
            _ = decode_request_v1(payload)

    def run_decode_packed() -> None:
        for _ in range(n):
            _ = decode_request_v1_packed(payload)

    # Response encode: build representative outputs.
    logits_list = [0.1] * ACTION_SPACE_A
    resp = InferResponseV1(request_id=123, policy_logits=logits_list, value=0.25, margin=None)

    logits_np = (np.arange(ACTION_SPACE_A, dtype=np.float32) * 0.001).astype(np.float32, copy=False)
    logits_bytes = logits_np.tobytes()
    value_bytes = struct.pack("<f", 0.25)
    resp_p = InferResponseV1Packed(
        request_id=123,
        policy_logits_f32=memoryview(logits_bytes),
        value_f32=memoryview(value_bytes),
        margin_f32=None,
    )

    def run_encode_v1() -> None:
        for _ in range(n):
            _ = encode_response_v1(resp)

    def run_encode_packed() -> None:
        for _ in range(n):
            _ = encode_response_v1_packed(resp_p)

    # Tensorization: bytes -> np row view -> contiguous batch
    req_p = decode_request_v1_packed(payload)
    row = np.frombuffer(req_p.features_f32, dtype=np.float32, count=FEATURE_LEN_V1)

    def run_tensorize_copy() -> None:
        x = np.empty((64, FEATURE_LEN_V1), dtype=np.float32)
        for _ in range(n):
            # Model the batcher path: copy one row into a contiguous batch buffer.
            x[0, :] = row

    print(f"n={n}")
    _bench("decode_request_v1", n, run_decode_v1)
    _bench("decode_request_v1_packed", n, run_decode_packed)
    _bench("encode_response_v1", n, run_encode_v1)
    _bench("encode_response_v1_packed", n, run_encode_packed)
    _bench("tensorize(copy one row)", n, run_tensorize_copy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


