//! Binary codec for protocol v1 (payload inside a length-delimited frame).

use thiserror::Error;

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
