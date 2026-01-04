//! Binary codec for protocol v1 (payload inside a length-delimited frame).

use thiserror::Error;

use crate::protocol::{FLAG_LEGAL_MASK_BITSET, LEGAL_MASK_BITSET_BYTES, PROTOCOL_VERSION_V2};
use crate::protocol::{
    InferRequestV1, InferResponseV1, MsgKind, ACTION_SPACE_A, FEATURE_LEN_V1, FEATURE_SCHEMA_ID_V1,
    PROTOCOL_VERSION,
};

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("payload too short")]
    TooShort,
    #[error("unsupported protocol version: {0}")]
    BadVersion(u32),
    #[error("unexpected message kind: {0}")]
    BadKind(u8),
    #[error("unsupported feature_schema_id: {0}")]
    BadSchema(u32),
    #[error("invalid vector length for features: got {got}, expected {expected}")]
    BadFeaturesLen { got: u32, expected: u32 },
    #[error("invalid vector length for legal_mask: got {got}, expected {expected}")]
    BadLegalLen { got: u32, expected: u32 },
    #[error("invalid vector length for policy_logits: got {got}, expected {expected}")]
    BadPolicyLen { got: u32, expected: u32 },
    #[error("invalid byte length for features: got {got}, expected {expected}")]
    BadFeaturesByteLen { got: u32, expected: u32 },
    #[error("invalid byte length for policy_logits: got {got}, expected {expected}")]
    BadPolicyByteLen { got: u32, expected: u32 },
    #[error("invalid boolean byte in legal_mask: {0}")]
    BadLegalByte(u8),
}

pub fn encode_request_v1(req: &InferRequestV1) -> Vec<u8> {
    let mut out = Vec::with_capacity(encode_request_v1_len(req));
    encode_request_v1_into(&mut out, req);
    out
}

pub fn encode_request_v1_len(req: &InferRequestV1) -> usize {
    // header: u32 version + u8 kind + u8 flags + u16 reserved = 8 bytes
    // ids: u64 request_id + u32 model_id + u32 schema_id = 16 bytes
    // vectors: u32 features_len + f32[...] + u32 legal_len + u8[...] = 8 + features*4 + legal
    32 + req.features.len() * 4 + req.legal_mask.len()
}

pub fn encode_request_v1_into(out: &mut Vec<u8>, req: &InferRequestV1) {
    out.clear();
    out.reserve(encode_request_v1_len(req));

    out.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    out.push(MsgKind::Request as u8);
    out.push(0); // flags
    out.extend_from_slice(&[0, 0]); // reserved

    out.extend_from_slice(&req.request_id.to_le_bytes());
    out.extend_from_slice(&req.model_id.to_le_bytes());
    out.extend_from_slice(&req.feature_schema_id.to_le_bytes());

    let features_len: u32 = req.features.len() as u32;
    out.extend_from_slice(&features_len.to_le_bytes());
    for &f in &req.features {
        out.extend_from_slice(&f.to_le_bytes());
    }

    let legal_len: u32 = req.legal_mask.len() as u32;
    out.extend_from_slice(&legal_len.to_le_bytes());
    out.extend_from_slice(&req.legal_mask);
}

pub fn encode_request_into(
    out: &mut Vec<u8>,
    req: &InferRequestV1,
    protocol_version: u32,
    legal_mask_bitset: bool,
) {
    match protocol_version {
        PROTOCOL_VERSION => encode_request_v1_into(out, req),
        PROTOCOL_VERSION_V2 => encode_request_v2_into(out, req, legal_mask_bitset),
        _ => encode_request_v1_into(out, req), // fallback
    }
}

pub fn encode_request_v2_into(out: &mut Vec<u8>, req: &InferRequestV1, legal_mask_bitset: bool) {
    out.clear();
    // header (8) + ids (16) + lens (8) + features bytes + legal bytes
    let features_bytes_len = (req.features.len() * 4) as usize;
    let legal_bytes_len = if legal_mask_bitset {
        LEGAL_MASK_BITSET_BYTES
    } else {
        req.legal_mask.len()
    };
    out.reserve(32 + features_bytes_len + legal_bytes_len);

    out.extend_from_slice(&PROTOCOL_VERSION_V2.to_le_bytes());
    out.push(MsgKind::Request as u8);
    let mut flags: u8 = 0;
    if legal_mask_bitset {
        flags |= FLAG_LEGAL_MASK_BITSET;
    }
    out.push(flags);
    out.extend_from_slice(&[0, 0]); // reserved

    out.extend_from_slice(&req.request_id.to_le_bytes());
    out.extend_from_slice(&req.model_id.to_le_bytes());
    out.extend_from_slice(&req.feature_schema_id.to_le_bytes());

    let features_byte_len: u32 = (req.features.len() as u32) * 4;
    out.extend_from_slice(&features_byte_len.to_le_bytes());
    #[cfg(target_endian = "little")]
    {
        out.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&req.features));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &f in &req.features {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }

    if legal_mask_bitset {
        // Pack 47 bytes (0/1) into 6 bytes (LSB-first).
        debug_assert_eq!(req.legal_mask.len(), ACTION_SPACE_A as usize);
        let packed = pack_legal_mask_bitset(&req.legal_mask);
        let legal_len: u32 = LEGAL_MASK_BITSET_BYTES as u32;
        out.extend_from_slice(&legal_len.to_le_bytes());
        out.extend_from_slice(&packed);
    } else {
        let legal_len: u32 = req.legal_mask.len() as u32;
        out.extend_from_slice(&legal_len.to_le_bytes());
        out.extend_from_slice(&req.legal_mask);
    }
}

fn pack_legal_mask_bitset(legal: &[u8]) -> [u8; LEGAL_MASK_BITSET_BYTES] {
    // Hot path: keep this cheap. Avoid div/mod in the per-bit loop.
    debug_assert_eq!(legal.len(), ACTION_SPACE_A as usize);
    let mut out = [0u8; LEGAL_MASK_BITSET_BYTES];
    let mut i = 0usize;
    for byte_i in 0..LEGAL_MASK_BITSET_BYTES {
        let mut b = 0u8;
        // LSB-first within each byte.
        for bit in 0..8usize {
            if i >= legal.len() {
                break;
            }
            // Legal bytes are 0/1; mask just in case.
            b |= (legal[i] & 1) << bit;
            i += 1;
        }
        out[byte_i] = b;
    }
    out
}

pub fn decode_request_v1(bytes: &[u8]) -> Result<InferRequestV1, DecodeError> {
    let mut c = Cursor::new(bytes);

    let version = c.read_u32()?;
    if version != PROTOCOL_VERSION {
        return Err(DecodeError::BadVersion(version));
    }
    let kind = c.read_u8()?;
    if kind != (MsgKind::Request as u8) {
        return Err(DecodeError::BadKind(kind));
    }
    let _flags = c.read_u8()?;
    c.skip(2)?;

    let request_id = c.read_u64()?;
    let model_id = c.read_u32()?;
    let feature_schema_id = c.read_u32()?;
    if feature_schema_id != FEATURE_SCHEMA_ID_V1 {
        return Err(DecodeError::BadSchema(feature_schema_id));
    }

    let features_len = c.read_u32()?;
    if features_len != FEATURE_LEN_V1 {
        return Err(DecodeError::BadFeaturesLen {
            got: features_len,
            expected: FEATURE_LEN_V1,
        });
    }
    let mut features = Vec::with_capacity(features_len as usize);
    for _ in 0..features_len {
        features.push(c.read_f32()?);
    }

    let legal_len = c.read_u32()?;
    if legal_len != ACTION_SPACE_A {
        return Err(DecodeError::BadLegalLen {
            got: legal_len,
            expected: ACTION_SPACE_A,
        });
    }
    let mut legal_mask = vec![0u8; legal_len as usize];
    c.read_bytes_into(&mut legal_mask)?;
    for &b in &legal_mask {
        if b != 0 && b != 1 {
            return Err(DecodeError::BadLegalByte(b));
        }
    }

    Ok(InferRequestV1 {
        request_id,
        model_id,
        feature_schema_id,
        features,
        legal_mask,
    })
}

pub fn encode_response_v1(resp: &InferResponseV1) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + resp.policy_logits.len() * 4);

    out.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    out.push(MsgKind::Response as u8);
    out.push(0); // flags
    out.extend_from_slice(&[0, 0]); // reserved

    out.extend_from_slice(&resp.request_id.to_le_bytes());

    let policy_len: u32 = resp.policy_logits.len() as u32;
    out.extend_from_slice(&policy_len.to_le_bytes());
    for &f in &resp.policy_logits {
        out.extend_from_slice(&f.to_le_bytes());
    }

    out.extend_from_slice(&resp.value.to_le_bytes());
    match resp.margin {
        Some(m) => {
            out.push(1);
            out.extend_from_slice(&m.to_le_bytes());
        }
        None => {
            out.push(0);
        }
    }

    out
}

pub fn encode_response_v2(resp: &InferResponseV1) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + (resp.policy_logits.len() * 4));
    out.extend_from_slice(&PROTOCOL_VERSION_V2.to_le_bytes());
    out.push(MsgKind::Response as u8);
    out.push(0); // flags
    out.extend_from_slice(&[0, 0]); // reserved

    out.extend_from_slice(&resp.request_id.to_le_bytes());

    let policy_byte_len: u32 = (resp.policy_logits.len() as u32) * 4;
    out.extend_from_slice(&policy_byte_len.to_le_bytes());
    #[cfg(target_endian = "little")]
    {
        out.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&resp.policy_logits));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &f in &resp.policy_logits {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }

    out.extend_from_slice(&resp.value.to_le_bytes());
    match resp.margin {
        Some(m) => {
            out.push(1);
            out.extend_from_slice(&m.to_le_bytes());
        }
        None => out.push(0),
    }
    out
}

pub fn decode_response_v1(bytes: &[u8]) -> Result<InferResponseV1, DecodeError> {
    let mut c = Cursor::new(bytes);

    let version = c.read_u32()?;
    if version != PROTOCOL_VERSION {
        return Err(DecodeError::BadVersion(version));
    }
    let kind = c.read_u8()?;
    if kind != (MsgKind::Response as u8) {
        return Err(DecodeError::BadKind(kind));
    }
    let _flags = c.read_u8()?;
    c.skip(2)?;

    let request_id = c.read_u64()?;

    let policy_len = c.read_u32()?;
    if policy_len != ACTION_SPACE_A {
        return Err(DecodeError::BadPolicyLen {
            got: policy_len,
            expected: ACTION_SPACE_A,
        });
    }
    let mut policy_logits = Vec::with_capacity(policy_len as usize);
    for _ in 0..policy_len {
        policy_logits.push(c.read_f32()?);
    }

    let value = c.read_f32()?;
    let has_margin = c.read_u8()?;
    let margin = if has_margin == 1 {
        Some(c.read_f32()?)
    } else {
        None
    };

    Ok(InferResponseV1 {
        request_id,
        policy_logits,
        value,
        margin,
    })
}

pub fn decode_response_any(bytes: &[u8]) -> Result<InferResponseV1, DecodeError> {
    if bytes.len() < 4 {
        return Err(DecodeError::TooShort);
    }
    let version = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if version == PROTOCOL_VERSION {
        return decode_response_v1(bytes);
    }
    if version == PROTOCOL_VERSION_V2 {
        return decode_response_v2(bytes);
    }
    Err(DecodeError::BadVersion(version))
}

pub fn decode_response_v2(bytes: &[u8]) -> Result<InferResponseV1, DecodeError> {
    let mut c = Cursor::new(bytes);
    let version = c.read_u32()?;
    if version != PROTOCOL_VERSION_V2 {
        return Err(DecodeError::BadVersion(version));
    }
    let kind = c.read_u8()?;
    if kind != (MsgKind::Response as u8) {
        return Err(DecodeError::BadKind(kind));
    }
    let _flags = c.read_u8()?;
    c.skip(2)?;

    let request_id = c.read_u64()?;

    let policy_byte_len = c.read_u32()?;
    let expected = ACTION_SPACE_A * 4;
    if policy_byte_len != expected {
        return Err(DecodeError::BadPolicyByteLen {
            got: policy_byte_len,
            expected,
        });
    }
    let logits_bytes = c.take(policy_byte_len as usize)?;
    let policy_logits = decode_f32_le_slice(logits_bytes)?;

    let value = c.read_f32()?;
    let has_margin = c.read_u8()?;
    let margin = if has_margin == 1 { Some(c.read_f32()?) } else { None };

    Ok(InferResponseV1 {
        request_id,
        policy_logits,
        value,
        margin,
    })
}

fn decode_f32_le_slice(bytes: &[u8]) -> Result<Vec<f32>, DecodeError> {
    if bytes.len() % 4 != 0 {
        return Err(DecodeError::TooShort);
    }
    #[cfg(target_endian = "little")]
    {
        let s: &[f32] = bytemuck::cast_slice(bytes);
        return Ok(s.to_vec());
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for i in 0..(bytes.len() / 4) {
            let j = i * 4;
            out.push(f32::from_le_bytes([bytes[j], bytes[j + 1], bytes[j + 2], bytes[j + 3]]));
        }
        Ok(out)
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.off + n > self.bytes.len() {
            return Err(DecodeError::TooShort);
        }
        let s = &self.bytes[self.off..self.off + n];
        self.off += n;
        Ok(s)
    }

    fn skip(&mut self, n: usize) -> Result<(), DecodeError> {
        self.take(n).map(|_| ())
    }

    fn read_u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, DecodeError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, DecodeError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32, DecodeError> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_bytes_into(&mut self, out: &mut [u8]) -> Result<(), DecodeError> {
        let b = self.take(out.len())?;
        out.copy_from_slice(b);
        Ok(())
    }
}

#[cfg(test)]
mod v2_tests {
    use super::*;
    use crate::protocol::{InferRequestV1, InferResponseV1, ACTION_SPACE_A, FEATURE_LEN_V1, FEATURE_SCHEMA_ID_V1};

    #[test]
    fn encode_request_v2_has_expected_lengths() {
        let req = InferRequestV1 {
            request_id: 1,
            model_id: 7,
            feature_schema_id: FEATURE_SCHEMA_ID_V1,
            features: vec![0.0; FEATURE_LEN_V1 as usize],
            legal_mask: vec![1u8; ACTION_SPACE_A as usize],
        };
        let mut out = Vec::new();
        encode_request_v2_into(&mut out, &req, false);
        // version
        assert_eq!(u32::from_le_bytes(out[0..4].try_into().unwrap()), PROTOCOL_VERSION_V2);
        // features_byte_len field is at offset:
        // header 8 + ids 16 = 24
        // request_id u64 at 8..16; model_id 16..20; schema 20..24; then features_byte_len 24..28
        let features_byte_len = u32::from_le_bytes(out[24..28].try_into().unwrap());
        assert_eq!(features_byte_len, FEATURE_LEN_V1 * 4);
    }

    #[test]
    fn encode_request_v2_bitset_sets_flag_and_uses_6_bytes() {
        let mut legal = vec![0u8; ACTION_SPACE_A as usize];
        legal[0] = 1;
        legal[1] = 1;
        legal[8] = 1;
        let req = InferRequestV1 {
            request_id: 1,
            model_id: 7,
            feature_schema_id: FEATURE_SCHEMA_ID_V1,
            features: vec![0.0; FEATURE_LEN_V1 as usize],
            legal_mask: legal,
        };
        let mut out = Vec::new();
        encode_request_v2_into(&mut out, &req, true);
        assert_eq!(u32::from_le_bytes(out[0..4].try_into().unwrap()), PROTOCOL_VERSION_V2);
        assert_eq!(out[4], MsgKind::Request as u8);
        assert_eq!(out[5] & FLAG_LEGAL_MASK_BITSET, FLAG_LEGAL_MASK_BITSET);
        // legal_len is after: header(8)+ids(16)+features_len_u32(4)+features_bytes(180) = 208
        let legal_len_off = 8 + 16 + 4 + (FEATURE_LEN_V1 as usize) * 4;
        let legal_len = u32::from_le_bytes(out[legal_len_off..legal_len_off + 4].try_into().unwrap());
        assert_eq!(legal_len as usize, LEGAL_MASK_BITSET_BYTES);
        let mask = &out[legal_len_off + 4..legal_len_off + 4 + LEGAL_MASK_BITSET_BYTES];
        // LSB-first: actions 0,1 set => 0b00000011 in byte0; action 8 set => bit0 in byte1.
        assert_eq!(mask[0], 0b0000_0011);
        assert_eq!(mask[1], 0b0000_0001);
    }

    #[test]
    fn roundtrip_decode_response_any_accepts_v1_and_v2() {
        let resp = InferResponseV1 {
            request_id: 42,
            policy_logits: vec![0.5; ACTION_SPACE_A as usize],
            value: -0.25,
            margin: None,
        };
        let p1 = encode_response_v1(&resp);
        let p2 = encode_response_v2(&resp);
        let r1 = decode_response_any(&p1).unwrap();
        let r2 = decode_response_any(&p2).unwrap();
        assert_eq!(r1.request_id, 42);
        assert_eq!(r2.request_id, 42);
        assert_eq!(r2.policy_logits.len(), ACTION_SPACE_A as usize);
        assert!((r2.value - (-0.25)).abs() < 1e-6);
    }
}
