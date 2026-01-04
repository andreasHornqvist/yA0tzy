//! Protocol v1 definitions for Rust ↔ inference server.
//!
//! This module defines the on-wire contract and sizes to validate against.

/// Protocol version constants (PRD §9 / E6.6).
pub const PROTOCOL_VERSION_V1: u32 = 1;
pub const PROTOCOL_VERSION_V2: u32 = 2;

/// Default protocol version (backwards compatible).
pub const PROTOCOL_VERSION: u32 = PROTOCOL_VERSION_V1;

/// Fixed action space size (PRD §5.2).
pub const ACTION_SPACE_A: u32 = 47;

/// v2 request header flags.
pub const FLAG_LEGAL_MASK_BITSET: u8 = 0x01;

/// Packed legal mask size in bytes for A=47 (48 bits).
pub const LEGAL_MASK_BITSET_BYTES: usize = 6;

/// Internal legal mask representation: 47 bits in a u64 (bit i => action i).
pub type LegalMask = u64;

/// Feature schema id currently supported (PRD Epic E3/E3.5).
pub const FEATURE_SCHEMA_ID_V1: u32 = 1;

/// Feature vector length for schema v1.
///
/// NOTE: keep in sync with `yz-features` schema constants.
pub const FEATURE_LEN_V1: u32 = 45;

#[derive(Debug, Clone, PartialEq)]
pub struct InferRequestV1 {
    pub request_id: u64,
    pub model_id: u32,
    pub feature_schema_id: u32,
    pub features: Vec<f32>,  // length = F (for schema)
    pub legal_mask: LegalMask,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferResponseV1 {
    pub request_id: u64,
    pub policy_logits: Vec<f32>, // length = A
    pub value: f32,
    pub margin: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgKind {
    Request = 1,
    Response = 2,
}
