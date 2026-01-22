//! Replay tensor schema + ids (PRD §10.1–§10.3).

/// Protocol version for replay shards.
pub const PROTOCOL_VERSION: u32 = 1;

/// Action space id (PRD §10.3).
/// v2: mark-only-at-roll-3 rules (KeepMask(31) legal, Mark illegal at rerolls>0).
pub const ACTION_SPACE_ID: &str = "oracle_keepmask_v2";

/// Ruleset id (PRD §10.3).
/// mark_at_r3_v1: marking only allowed at roll 3 (rerolls_left=0).
pub const RULESET_ID: &str = "swedish_scandinavian_mark_at_r3_v1";

/// Tensor names inside safetensors.
pub const T_FEATURES: &str = "features";
pub const T_LEGAL_MASK: &str = "legal_mask";
pub const T_PI: &str = "pi";
pub const T_Z: &str = "z";
pub const T_Z_MARGIN: &str = "z_margin";
