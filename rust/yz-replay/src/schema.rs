//! Replay tensor schema + ids (PRD §10.1–§10.3).

/// Protocol version for replay shards.
pub const PROTOCOL_VERSION: u32 = 1;

/// Action space id (PRD §10.3).
pub const ACTION_SPACE_ID: &str = "oracle_keepmask_v1";

/// Ruleset id (PRD §10.3).
pub const RULESET_ID: &str = "swedish_scandinavian_v1";

/// Tensor names inside safetensors.
pub const T_FEATURES: &str = "features";
pub const T_LEGAL_MASK: &str = "legal_mask";
pub const T_PI: &str = "pi";
pub const T_Z: &str = "z";
pub const T_Z_MARGIN: &str = "z_margin";
