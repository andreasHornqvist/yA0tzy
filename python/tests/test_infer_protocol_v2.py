import asyncio
import pathlib
import struct
import time

import pytest

from yatzy_az.server.protocol_v1 import (
    ACTION_SPACE_A,
    FLAG_LEGAL_MASK_BITSET,
    FEATURE_LEN_V1,
    FEATURE_SCHEMA_ID_V1,
    LEGAL_MASK_BITSET_BYTES,
    PROTOCOL_VERSION_V2,
    encode_frame,
)
from yatzy_az.server.protocol_v1 import decode_response_v1  # reuse float decoder for sanity
from yatzy_az.server.server import ServerConfig, serve


def _encode_request_v2(*, request_id: int, model_id: int, features_f32: bytes, legal_mask: bytes) -> bytes:
    assert len(features_f32) == FEATURE_LEN_V1 * 4
    assert len(legal_mask) == ACTION_SPACE_A
    out = bytearray()
    out += struct.pack("<I", PROTOCOL_VERSION_V2)
    out += struct.pack("<B", 1)  # request
    out += struct.pack("<B", 0)  # flags
    out += b"\x00\x00"
    out += struct.pack("<Q", int(request_id))
    out += struct.pack("<I", int(model_id))
    out += struct.pack("<I", int(FEATURE_SCHEMA_ID_V1))
    out += struct.pack("<I", int(len(features_f32)))
    out += features_f32
    out += struct.pack("<I", int(len(legal_mask)))
    out += legal_mask
    return bytes(out)


def _encode_request_v2_bitset(*, request_id: int, model_id: int, features_f32: bytes, legal_bitset: bytes) -> bytes:
    assert len(features_f32) == FEATURE_LEN_V1 * 4
    assert len(legal_bitset) == LEGAL_MASK_BITSET_BYTES
    out = bytearray()
    out += struct.pack("<I", PROTOCOL_VERSION_V2)
    out += struct.pack("<B", 1)  # request
    out += struct.pack("<B", int(FLAG_LEGAL_MASK_BITSET))  # flags
    out += b"\x00\x00"
    out += struct.pack("<Q", int(request_id))
    out += struct.pack("<I", int(model_id))
    out += struct.pack("<I", int(FEATURE_SCHEMA_ID_V1))
    out += struct.pack("<I", int(len(features_f32)))
    out += features_f32
    out += struct.pack("<I", int(len(legal_bitset)))
    out += legal_bitset
    return bytes(out)


def _decode_response_v2(payload: bytes) -> tuple[int, bytes, bytes]:
    off = 0

    def take(n: int) -> bytes:
        nonlocal off
        b = payload[off : off + n]
        if len(b) != n:
            raise RuntimeError("short payload")
        off += n
        return b

    ver = struct.unpack("<I", take(4))[0]
    assert ver == PROTOCOL_VERSION_V2
    kind = struct.unpack("<B", take(1))[0]
    assert kind == 2
    take(1)  # flags
    take(2)  # reserved
    rid = struct.unpack("<Q", take(8))[0]
    pol_b = struct.unpack("<I", take(4))[0]
    assert pol_b == ACTION_SPACE_A * 4
    logits = take(pol_b)
    value = take(4)
    has_margin = struct.unpack("<B", take(1))[0]
    assert has_margin in (0, 1)
    if has_margin == 1:
        take(4)
    return (int(rid), logits, value)


def test_uds_server_accepts_v2_and_replies_v2() -> None:
    asyncio.run(_uds_server_accepts_v2_and_replies_v2())


async def _uds_server_accepts_v2_and_replies_v2() -> None:
    sock = pathlib.Path("/tmp") / f"yatzy_infer_v2_test_{time.time_ns()}.sock"
    cfg = ServerConfig(
        bind=f"unix://{sock}",
        device="cpu",
        max_batch=128,
        max_wait_us=50_000,
        print_stats_every_s=0,
        best_id=0,
        cand_id=1,
        best_spec="dummy:0.1",
        cand_spec="dummy:-0.1",
        metrics_bind="127.0.0.1:0",
        metrics_disable=True,
        torch_threads=None,
        torch_interop_threads=None,
    )
    task = asyncio.create_task(serve(cfg))
    # Wait for socket.
    for _ in range(200):
        if sock.exists():
            break
        await asyncio.sleep(0.01)
    assert sock.exists()

    # v2 request payload: features are packed float32 bytes.
    feats = (b"\x00\x00\x00\x00") * FEATURE_LEN_V1
    legal = bytes([1] * ACTION_SPACE_A)
    payload = _encode_request_v2(request_id=123, model_id=0, features_f32=feats, legal_mask=legal)

    r, w = await asyncio.open_unix_connection(str(sock))
    w.write(encode_frame(payload))
    await w.drain()
    frame = await r.readexactly(4)
    (n,) = struct.unpack("<I", frame)
    body = await r.readexactly(n)
    w.close()
    await w.wait_closed()

    # Verify it is a v2 response.
    rid, logits_b, value_b = _decode_response_v2(body)
    assert rid == 123
    assert len(logits_b) == ACTION_SPACE_A * 4
    assert len(value_b) == 4

    # Sanity: v1 decoder should reject v2 response.
    with pytest.raises(Exception):
        _ = decode_response_v1(body)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_uds_server_accepts_v2_bitset_legal_mask() -> None:
    asyncio.run(_uds_server_accepts_v2_bitset_legal_mask())


async def _uds_server_accepts_v2_bitset_legal_mask() -> None:
    sock = pathlib.Path("/tmp") / f"yatzy_infer_v2_bitset_{time.time_ns()}.sock"
    cfg = ServerConfig(
        bind=f"unix://{sock}",
        device="cpu",
        max_batch=128,
        max_wait_us=50_000,
        print_stats_every_s=0,
        best_id=0,
        cand_id=1,
        best_spec="dummy:0.1",
        cand_spec="dummy:-0.1",
        metrics_bind="127.0.0.1:0",
        metrics_disable=True,
        torch_threads=None,
        torch_interop_threads=None,
    )
    task = asyncio.create_task(serve(cfg))
    for _ in range(200):
        if sock.exists():
            break
        await asyncio.sleep(0.01)
    assert sock.exists()

    feats = (b"\x00\x00\x00\x00") * FEATURE_LEN_V1
    # LSB-first bitset: actions 0,1,8 legal.
    legal = bytes([0b0000_0011, 0b0000_0001, 0, 0, 0, 0])
    payload = _encode_request_v2_bitset(
        request_id=777, model_id=0, features_f32=feats, legal_bitset=legal
    )

    r, w = await asyncio.open_unix_connection(str(sock))
    w.write(encode_frame(payload))
    await w.drain()
    resp = await read_frame(r)
    w.close()
    await w.wait_closed()

    rid, _logits_b, _value_b = _decode_response_v2(resp)
    assert rid == 777

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_mixed_v1_v2_requests_concurrently() -> None:
    asyncio.run(_mixed_v1_v2_requests_concurrently())


async def _mixed_v1_v2_requests_concurrently() -> None:
    sock = pathlib.Path("/tmp") / f"yatzy_infer_v2_mix_{time.time_ns()}.sock"
    cfg = ServerConfig(
        bind=f"unix://{sock}",
        device="cpu",
        max_batch=128,
        max_wait_us=50_000,
        print_stats_every_s=0,
        best_id=0,
        cand_id=1,
        best_spec="dummy",
        cand_spec="dummy",
        metrics_bind="127.0.0.1:0",
        metrics_disable=True,
        torch_threads=None,
        torch_interop_threads=None,
    )
    task = asyncio.create_task(serve(cfg))
    for _ in range(200):
        if sock.exists():
            break
        await asyncio.sleep(0.01)
    assert sock.exists()

    from yatzy_az.server.protocol_v1 import InferRequestV1, encode_request_v1, read_frame

    feats_b = (b"\x00\x00\x00\x00") * FEATURE_LEN_V1
    legal = bytes([1] * ACTION_SPACE_A)

    async def req_v1(rid: int):
        r, w = await asyncio.open_unix_connection(str(sock))
        req = InferRequestV1(
            request_id=rid,
            model_id=0,
            feature_schema_id=FEATURE_SCHEMA_ID_V1,
            features=[0.0] * FEATURE_LEN_V1,
            legal_mask=legal,
        )
        w.write(encode_frame(encode_request_v1(req)))
        await w.drain()
        resp = await read_frame(r)
        w.close()
        await w.wait_closed()
        # v1 responses still decodable.
        return decode_response_v1(resp).request_id

    async def req_v2(rid: int):
        r, w = await asyncio.open_unix_connection(str(sock))
        w.write(encode_frame(_encode_request_v2(request_id=rid, model_id=0, features_f32=feats_b, legal_mask=legal)))
        await w.drain()
        resp = await read_frame(r)
        w.close()
        await w.wait_closed()
        rid2, _, _ = _decode_response_v2(resp)
        return rid2

    jobs = []
    for i in range(64):
        jobs.append(req_v1(i))
        jobs.append(req_v2(1000 + i))
    got = await asyncio.gather(*jobs)
    assert set(got) == set(range(64)).union({1000 + i for i in range(64)})

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


