//! Feature schema v1 (PRD Epic E3, Story 1).
//!
//! This schema defines a stable, versioned feature vector used by the NN.
//! Encoding is from POV of `player_to_move` (PRD Epic E3, Story 2).
//!
//! ### Layout (v1)
//! Let `my` mean the encoded current player (player_to_move), and `opp` the other.
//!
//! - **my_avail_mask_bits**: 15 floats, for cats 0..14 in PRD order, 1.0 if available else 0.0
//! - **my_upper_total_cap_norm**: 1 float, upper_total_cap / 63
//! - **my_total_score_norm**: 1 float, total_score / SCORE_NORM
//! - **my_filled_count_norm**: 1 float, filled_count / 15
//! - **opp_avail_mask_bits**: 15 floats
//! - **opp_upper_total_cap_norm**: 1 float
//! - **opp_total_score_norm**: 1 float
//! - **opp_filled_count_norm**: 1 float
//! - **dice_counts**: 6 floats, counts of faces 1..6 divided by 5
//! - **rerolls_left_onehot**: 3 floats for rerolls_left in {0,1,2}
//!
//! Total: F = 45.

/// Increment this whenever the feature layout changes.
pub const FEATURE_SCHEMA_ID: u32 = 1;

/// Feature vector length for schema v1.
pub const F: usize = 45;

/// Normalization scale for total_score.
///
/// Yatzy solitaire scores are typically < 400. For 1v1, totals are in the same ballpark.
pub const SCORE_NORM: f32 = 400.0;
