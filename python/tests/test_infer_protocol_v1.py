import asyncio
import pathlib
import time

import pytest

from yatzy_az.infer_server.protocol_v1 import (
    ACTION_SPACE_A,
    FEATURE_LEN_V1,
    FEATURE_SCHEMA_ID_V1,
    InferRequestV1,
    InferResponseV1,
    decode_request_v1,
    decode_response_v1,
    encode_request_v1,
    encode_response_v1,
)
from yatzy_az.infer_server.server import ServerConfig, serve


def test_request_roundtrip_codec() -> None:
    req = InferRequestV1(
        request_id=123,
        model_id=7,
        feature_schema_id=FEATURE_SCHEMA_ID_V1,
        features=[0.0] * FEATURE_LEN_V1,
        legal_mask=bytes([1] * ACTION_SPACE_A),
    )
    payload = encode_request_v1(req)
    req2 = decode_request_v1(payload)
    assert req2 == req


def test_response_roundtrip_codec() -> None:
    resp = InferResponseV1(
        request_id=123,
        policy_logits=[0.0] * ACTION_SPACE_A,
        value=0.0,
        margin=None,
    )
    payload = encode_response_v1(resp)
    resp2 = decode_response_v1(payload)
    assert resp2 == resp


def test_uds_server_smoke_batches() -> None:
    asyncio.run(_uds_server_smoke_batches())


async def _uds_server_smoke_batches() -> None:
    # Start server in background on a temp socket.
    sock = pathlib.Path("/tmp") / f"yatzy_infer_test_{time.time_ns()}.sock"
    cfg = ServerConfig(
        bind=f"unix://{sock}",
        max_batch=128,
        max_wait_us=50_000,
        print_stats_every_s=0,
        best_id=0,
        cand_id=1,
        best_spec="dummy",
        cand_spec="dummy",
    )
    task = asyncio.create_task(serve(cfg))
    await asyncio.sleep(0.05)

    async def client_req(request_id: int) -> InferResponseV1:
        r, w = await asyncio.open_unix_connection(str(sock))
        # Construct request payload directly.
        req = InferRequestV1(
            request_id=request_id,
            model_id=0,
            feature_schema_id=FEATURE_SCHEMA_ID_V1,
            features=[0.0] * FEATURE_LEN_V1,
            legal_mask=bytes([1] * ACTION_SPACE_A),
        )
        from yatzy_az.infer_server.protocol_v1 import encode_frame, read_frame

        w.write(encode_frame(encode_request_v1(req)))
        await w.drain()
        payload = await read_frame(r)
        w.close()
        await w.wait_closed()
        return decode_response_v1(payload)

    # Fire multiple requests concurrently to encourage batching.
    resps = await asyncio.gather(*(client_req(i) for i in range(64)))
    assert {r.request_id for r in resps} == set(range(64))
    assert all(len(r.policy_logits) == ACTION_SPACE_A for r in resps)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_uds_server_routes_by_model_id() -> None:
    asyncio.run(_uds_server_routes_by_model_id())


async def _uds_server_routes_by_model_id() -> None:
    sock = pathlib.Path("/tmp") / f"yatzy_infer_route_test_{time.time_ns()}.sock"
    cfg = ServerConfig(
        bind=f"unix://{sock}",
        max_batch=128,
        max_wait_us=50_000,
        print_stats_every_s=0,
        best_id=0,
        cand_id=1,
        best_spec="dummy:0.1",
        cand_spec="dummy:-0.1",
    )
    task = asyncio.create_task(serve(cfg))
    await asyncio.sleep(0.05)

    async def client_req(request_id: int, model_id: int) -> InferResponseV1:
        r, w = await asyncio.open_unix_connection(str(sock))
        req = InferRequestV1(
            request_id=request_id,
            model_id=model_id,
            feature_schema_id=FEATURE_SCHEMA_ID_V1,
            features=[0.0] * FEATURE_LEN_V1,
            legal_mask=bytes([1] * ACTION_SPACE_A),
        )
        from yatzy_az.infer_server.protocol_v1 import encode_frame, read_frame

        w.write(encode_frame(encode_request_v1(req)))
        await w.drain()
        payload = await read_frame(r)
        w.close()
        await w.wait_closed()
        return decode_response_v1(payload)

    # Mix best/candidate model ids in flight.
    jobs = []
    expected = {}
    for i in range(64):
        mid = 0 if (i % 2 == 0) else 1
        expected[i] = 0.1 if mid == 0 else -0.1
        jobs.append(client_req(i, mid))
    resps = await asyncio.gather(*jobs)

    for r in resps:
        assert r.value == pytest.approx(expected[r.request_id])

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
