"""Protocol v1 framing + codec compatible with `rust/yz-infer`."""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from typing import Final

PROTOCOL_VERSION_V1: Final[int] = 1
PROTOCOL_VERSION_V2: Final[int] = 2

# Backwards-compatible default.
PROTOCOL_VERSION: Final[int] = PROTOCOL_VERSION_V1
ACTION_SPACE_A: Final[int] = 47
FLAG_LEGAL_MASK_BITSET: Final[int] = 0x01
LEGAL_MASK_BITSET_BYTES: Final[int] = 6
FEATURE_SCHEMA_ID_V1: Final[int] = 1
FEATURE_LEN_V1: Final[int] = 45
MAX_FRAME_LEN: Final[int] = 64 * 1024 * 1024  # 64 MiB (matches rust/yz-infer)


class ProtocolError(Exception):
    pass


class FrameTooLargeError(ProtocolError):
    pass


class UnexpectedEofError(ProtocolError):
    pass


class DecodeError(ProtocolError):
    pass


class EncodeError(ProtocolError):
    pass


@dataclass(frozen=True, slots=True)
class InferRequestV1:
    request_id: int
    model_id: int
    feature_schema_id: int
    features: list[float]  # len=F
    legal_mask: bytes  # len=A, each byte 0/1


@dataclass(frozen=True, slots=True)
class InferRequestV1Packed:
    """Protocol-v1 compatible request representation that avoids Python float lists.

    `features_f32` is a bytes-like view of exactly FEATURE_LEN_V1 * 4 bytes (little-endian f32).
    """

    request_id: int
    model_id: int
    feature_schema_id: int
    features_f32: memoryview  # len=F*4
    legal_mask: bytes  # len=A, each byte 0/1


@dataclass(frozen=True, slots=True)
class InferResponseV1:
    request_id: int
    policy_logits: list[float]  # len=A
    value: float
    margin: float | None


@dataclass(frozen=True, slots=True)
class InferResponseV1Packed:
    """Protocol-v1 compatible response representation that avoids Python float lists.

    `policy_logits_f32` is a bytes-like view of exactly ACTION_SPACE_A * 4 bytes (little-endian f32).
    `value_f32` is a bytes-like view of exactly 4 bytes (little-endian f32).
    """

    request_id: int
    policy_logits_f32: memoryview  # len=A*4
    value_f32: memoryview  # len=4
    margin_f32: memoryview | None


@dataclass(frozen=True, slots=True)
class InferRequestPacked:
    """Internal request representation that supports multiple protocol versions."""

    protocol_version: int
    request_id: int
    model_id: int
    feature_schema_id: int
    features_f32: memoryview  # len=F*4
    legal_mask: bytes  # len=47 (byte-mask) or len=6 (bitset)
    legal_is_bitset: bool


@dataclass(frozen=True, slots=True)
class InferResponsePacked:
    """Internal response representation that supports multiple protocol versions."""

    protocol_version: int
    request_id: int
    policy_logits_f32: memoryview  # len=A*4
    value_f32: memoryview  # len=4
    margin_f32: memoryview | None


def encode_frame(payload: bytes) -> bytes:
    n = len(payload)
    if n > MAX_FRAME_LEN:
        raise FrameTooLargeError(f"frame too large: {n} > {MAX_FRAME_LEN}")
    return struct.pack("<I", n) + payload


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    """Read one length-delimited frame payload from an asyncio StreamReader."""
    try:
        hdr = await reader.readexactly(4)
    except asyncio.IncompleteReadError as e:
        raise UnexpectedEofError("unexpected EOF while reading frame length") from e
    (n,) = struct.unpack("<I", hdr)
    if n > MAX_FRAME_LEN:
        raise FrameTooLargeError(f"frame too large: {n} > {MAX_FRAME_LEN}")
    try:
        return await reader.readexactly(n)
    except asyncio.IncompleteReadError as e:
        raise UnexpectedEofError("unexpected EOF while reading frame payload") from e


async def write_frame(writer: asyncio.StreamWriter, payload: bytes) -> None:
    """Write one length-delimited frame to an asyncio StreamWriter."""
    writer.write(encode_frame(payload))
    await writer.drain()


def decode_frame_from_buffer(buf: bytes) -> bytes:
    if len(buf) < 4:
        raise UnexpectedEofError("need 4 bytes for frame length")
    (n,) = struct.unpack_from("<I", buf, 0)
    if n > MAX_FRAME_LEN:
        raise FrameTooLargeError(f"frame too large: {n} > {MAX_FRAME_LEN}")
    if len(buf) < 4 + n:
        raise UnexpectedEofError("buffer ended early while reading frame")
    return buf[4 : 4 + n]


def encode_request_v1(req: InferRequestV1) -> bytes:
    # Header: u32 version, u8 kind, u8 flags, u16 reserved
    out = bytearray()
    out += struct.pack("<I", PROTOCOL_VERSION_V1)
    out += struct.pack("<B", 1)  # MsgKind::Request
    out += struct.pack("<B", 0)  # flags
    out += b"\x00\x00"  # reserved

    out += struct.pack("<Q", req.request_id)
    out += struct.pack("<I", req.model_id)
    out += struct.pack("<I", req.feature_schema_id)

    out += struct.pack("<I", len(req.features))
    for f in req.features:
        out += struct.pack("<f", float(f))

    out += struct.pack("<I", len(req.legal_mask))
    out += req.legal_mask
    return bytes(out)


def decode_request_v1(payload: bytes) -> InferRequestV1:
    off = 0

    def take(n: int) -> bytes:
        nonlocal off
        if off + n > len(payload):
            raise DecodeError("payload too short")
        b = payload[off : off + n]
        off += n
        return b

    version = struct.unpack("<I", take(4))[0]
    if version != PROTOCOL_VERSION_V1:
        raise DecodeError(f"bad version: {version}")
    kind = struct.unpack("<B", take(1))[0]
    if kind != 1:
        raise DecodeError(f"bad kind: {kind}")
    _flags = struct.unpack("<B", take(1))[0]
    take(2)  # reserved

    request_id = struct.unpack("<Q", take(8))[0]
    model_id = struct.unpack("<I", take(4))[0]
    feature_schema_id = struct.unpack("<I", take(4))[0]
    if feature_schema_id != FEATURE_SCHEMA_ID_V1:
        raise DecodeError(f"bad schema: {feature_schema_id}")

    features_len = struct.unpack("<I", take(4))[0]
    if features_len != FEATURE_LEN_V1:
        raise DecodeError(f"bad features len: got {features_len}, expected {FEATURE_LEN_V1}")
    features: list[float] = []
    for _ in range(features_len):
        features.append(struct.unpack("<f", take(4))[0])

    legal_len = struct.unpack("<I", take(4))[0]
    if legal_len != ACTION_SPACE_A:
        raise DecodeError(f"bad legal len: got {legal_len}, expected {ACTION_SPACE_A}")
    legal_mask = take(legal_len)
    for b in legal_mask:
        if b not in (0, 1):
            raise DecodeError(f"bad legal byte: {b}")

    return InferRequestV1(
        request_id=request_id,
        model_id=model_id,
        feature_schema_id=feature_schema_id,
        features=features,
        legal_mask=bytes(legal_mask),
    )


def decode_request_v1_packed(payload: bytes) -> InferRequestV1Packed:
    """Decode a v1 request without constructing Python float lists (hot path)."""
    off = 0

    def take(n: int) -> memoryview:
        nonlocal off
        if off + n > len(payload):
            raise DecodeError("payload too short")
        mv = memoryview(payload)[off : off + n]
        off += n
        return mv

    version = struct.unpack("<I", take(4))[0]
    if version != PROTOCOL_VERSION_V1:
        raise DecodeError(f"bad version: {version}")
    kind = struct.unpack("<B", take(1))[0]
    if kind != 1:
        raise DecodeError(f"bad kind: {kind}")
    _flags = struct.unpack("<B", take(1))[0]
    take(2)  # reserved

    request_id = struct.unpack("<Q", take(8))[0]
    model_id = struct.unpack("<I", take(4))[0]
    feature_schema_id = struct.unpack("<I", take(4))[0]
    if feature_schema_id != FEATURE_SCHEMA_ID_V1:
        raise DecodeError(f"bad schema: {feature_schema_id}")

    features_len = struct.unpack("<I", take(4))[0]
    if features_len != FEATURE_LEN_V1:
        raise DecodeError(f"bad features len: got {features_len}, expected {FEATURE_LEN_V1}")
    features_f32 = take(int(features_len) * 4)

    legal_len = struct.unpack("<I", take(4))[0]
    if legal_len != ACTION_SPACE_A:
        raise DecodeError(f"bad legal len: got {legal_len}, expected {ACTION_SPACE_A}")
    legal_mv = take(int(legal_len))
    legal_mask = bytes(legal_mv)
    for b in legal_mask:
        if b not in (0, 1):
            raise DecodeError(f"bad legal byte: {b}")

    return InferRequestV1Packed(
        request_id=request_id,
        model_id=model_id,
        feature_schema_id=feature_schema_id,
        features_f32=features_f32,
        legal_mask=legal_mask,
    )


def decode_request_v2_packed(payload: bytes) -> InferRequestPacked:
    """Decode a v2 request (packed f32 bytes) into the unified internal representation."""
    off = 0

    def take(n: int) -> memoryview:
        nonlocal off
        if off + n > len(payload):
            raise DecodeError("payload too short")
        mv = memoryview(payload)[off : off + n]
        off += n
        return mv

    version = struct.unpack("<I", take(4))[0]
    if version != PROTOCOL_VERSION_V2:
        raise DecodeError(f"bad version: {version}")
    kind = struct.unpack("<B", take(1))[0]
    if kind != 1:
        raise DecodeError(f"bad kind: {kind}")
    flags = struct.unpack("<B", take(1))[0]
    take(2)  # reserved

    request_id = struct.unpack("<Q", take(8))[0]
    model_id = struct.unpack("<I", take(4))[0]
    feature_schema_id = struct.unpack("<I", take(4))[0]
    if feature_schema_id != FEATURE_SCHEMA_ID_V1:
        raise DecodeError(f"bad schema: {feature_schema_id}")

    features_byte_len = struct.unpack("<I", take(4))[0]
    expected = FEATURE_LEN_V1 * 4
    if features_byte_len != expected:
        raise DecodeError(f"bad features byte len: got {features_byte_len}, expected {expected}")
    features_f32 = take(int(features_byte_len))

    legal_is_bitset = (int(flags) & int(FLAG_LEGAL_MASK_BITSET)) != 0
    legal_len = struct.unpack("<I", take(4))[0]
    if legal_is_bitset:
        if legal_len != LEGAL_MASK_BITSET_BYTES:
            raise DecodeError(
                f"bad legal len (bitset): got {legal_len}, expected {LEGAL_MASK_BITSET_BYTES}"
            )
        legal_mv = take(int(legal_len))
        legal_mask = bytes(legal_mv)
    else:
        if legal_len != ACTION_SPACE_A:
            raise DecodeError(f"bad legal len: got {legal_len}, expected {ACTION_SPACE_A}")
        legal_mv = take(int(legal_len))
        legal_mask = bytes(legal_mv)
        for b in legal_mask:
            if b not in (0, 1):
                raise DecodeError(f"bad legal byte: {b}")

    return InferRequestPacked(
        protocol_version=PROTOCOL_VERSION_V2,
        request_id=request_id,
        model_id=model_id,
        feature_schema_id=feature_schema_id,
        features_f32=features_f32,
        legal_mask=legal_mask,
        legal_is_bitset=legal_is_bitset,
    )


def decode_request_packed(payload: bytes) -> InferRequestPacked:
    """Decode either v1 or v2 into a single internal request type."""
    if len(payload) < 4:
        raise DecodeError("payload too short")
    (version,) = struct.unpack_from("<I", payload, 0)
    if version == PROTOCOL_VERSION_V1:
        r1 = decode_request_v1_packed(payload)
        return InferRequestPacked(
            protocol_version=PROTOCOL_VERSION_V1,
            request_id=r1.request_id,
            model_id=r1.model_id,
            feature_schema_id=r1.feature_schema_id,
            features_f32=r1.features_f32,
            legal_mask=r1.legal_mask,
            legal_is_bitset=False,
        )
    if version == PROTOCOL_VERSION_V2:
        return decode_request_v2_packed(payload)
    raise DecodeError(f"bad version: {version}")


def encode_response_v1(resp: InferResponseV1) -> bytes:
    out = bytearray()
    out += struct.pack("<I", PROTOCOL_VERSION_V1)
    out += struct.pack("<B", 2)  # MsgKind::Response
    out += struct.pack("<B", 0)  # flags
    out += b"\x00\x00"  # reserved

    out += struct.pack("<Q", resp.request_id)

    out += struct.pack("<I", len(resp.policy_logits))
    for f in resp.policy_logits:
        out += struct.pack("<f", float(f))

    out += struct.pack("<f", float(resp.value))
    if resp.margin is None:
        out += struct.pack("<B", 0)
    else:
        out += struct.pack("<B", 1)
        out += struct.pack("<f", float(resp.margin))

    return bytes(out)


def encode_response_v1_packed(resp: InferResponseV1Packed) -> bytes:
    """Encode a v1 response from packed float32 bytes (hot path)."""
    if len(resp.policy_logits_f32) != ACTION_SPACE_A * 4:
        raise EncodeError(
            f"bad policy_logits_f32 len: got {len(resp.policy_logits_f32)}, expected {ACTION_SPACE_A * 4}"
        )
    if len(resp.value_f32) != 4:
        raise EncodeError(f"bad value_f32 len: got {len(resp.value_f32)}, expected 4")
    if resp.margin_f32 is not None and len(resp.margin_f32) != 4:
        raise EncodeError(f"bad margin_f32 len: got {len(resp.margin_f32)}, expected 4")

    out = bytearray()
    out += struct.pack("<I", PROTOCOL_VERSION_V1)
    out += struct.pack("<B", 2)  # MsgKind::Response
    out += struct.pack("<B", 0)  # flags
    out += b"\x00\x00"  # reserved
    out += struct.pack("<Q", int(resp.request_id))

    out += struct.pack("<I", ACTION_SPACE_A)
    out += resp.policy_logits_f32

    out += resp.value_f32

    if resp.margin_f32 is None:
        out += struct.pack("<B", 0)
    else:
        out += struct.pack("<B", 1)
        out += resp.margin_f32
    return bytes(out)


def encode_response_v2_packed(resp: InferResponsePacked) -> bytes:
    """Encode a v2 response from packed float32 bytes (hot path)."""
    if len(resp.policy_logits_f32) != ACTION_SPACE_A * 4:
        raise EncodeError(
            f"bad policy_logits_f32 len: got {len(resp.policy_logits_f32)}, expected {ACTION_SPACE_A * 4}"
        )
    if len(resp.value_f32) != 4:
        raise EncodeError(f"bad value_f32 len: got {len(resp.value_f32)}, expected 4")
    if resp.margin_f32 is not None and len(resp.margin_f32) != 4:
        raise EncodeError(f"bad margin_f32 len: got {len(resp.margin_f32)}, expected 4")

    out = bytearray()
    out += struct.pack("<I", PROTOCOL_VERSION_V2)
    out += struct.pack("<B", 2)  # MsgKind::Response
    out += struct.pack("<B", 0)  # flags
    out += b"\x00\x00"  # reserved
    out += struct.pack("<Q", int(resp.request_id))

    out += struct.pack("<I", ACTION_SPACE_A * 4)
    out += resp.policy_logits_f32
    out += resp.value_f32

    if resp.margin_f32 is None:
        out += struct.pack("<B", 0)
    else:
        out += struct.pack("<B", 1)
        out += resp.margin_f32
    return bytes(out)


def encode_response_packed(resp: InferResponsePacked) -> bytes:
    """Encode response according to request protocol version (v1 or v2)."""
    if resp.protocol_version == PROTOCOL_VERSION_V1:
        r1 = InferResponseV1Packed(
            request_id=resp.request_id,
            policy_logits_f32=resp.policy_logits_f32,
            value_f32=resp.value_f32,
            margin_f32=resp.margin_f32,
        )
        return encode_response_v1_packed(r1)
    if resp.protocol_version == PROTOCOL_VERSION_V2:
        return encode_response_v2_packed(resp)
    raise EncodeError(f"bad protocol_version: {resp.protocol_version}")


def decode_response_v1(payload: bytes) -> InferResponseV1:
    off = 0

    def take(n: int) -> bytes:
        nonlocal off
        if off + n > len(payload):
            raise DecodeError("payload too short")
        b = payload[off : off + n]
        off += n
        return b

    version = struct.unpack("<I", take(4))[0]
    if version != PROTOCOL_VERSION_V1:
        raise DecodeError(f"bad version: {version}")
    kind = struct.unpack("<B", take(1))[0]
    if kind != 2:
        raise DecodeError(f"bad kind: {kind}")
    _flags = struct.unpack("<B", take(1))[0]
    take(2)  # reserved

    request_id = struct.unpack("<Q", take(8))[0]

    policy_len = struct.unpack("<I", take(4))[0]
    if policy_len != ACTION_SPACE_A:
        raise DecodeError(f"bad policy len: got {policy_len}, expected {ACTION_SPACE_A}")
    policy_logits: list[float] = []
    for _ in range(policy_len):
        policy_logits.append(struct.unpack("<f", take(4))[0])

    value = struct.unpack("<f", take(4))[0]
    has_margin = struct.unpack("<B", take(1))[0]
    margin = struct.unpack("<f", take(4))[0] if has_margin == 1 else None

    return InferResponseV1(
        request_id=request_id, policy_logits=policy_logits, value=value, margin=margin
    )
