//! Feature encoding implementation for FeatureSchema v1.

use crate::schema::{F, SCORE_NORM};

/// Minimal game-state view needed for encoding.
///
/// This is intentionally small and self-contained (no dependency on a full engine struct yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerView {
    /// Category availability mask (oracle convention: bit (14-cat) is 1 if available).
    pub avail_mask: u16,
    /// Upper total (clamped to 63).
    pub upper_total_cap: u8,
    /// Total score so far.
    pub total_score: i16,
}

/// Minimal game-state view for encoding, including player-to-move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameStateView {
    pub players: [PlayerView; 2],
    /// Dice (sorted ascending).
    pub dice_sorted: [u8; 5],
    pub rerolls_left: u8,
    pub player_to_move: u8, // 0 or 1
}

impl GameStateView {
    pub fn swap_players(self) -> Self {
        Self {
            players: [self.players[1], self.players[0]],
            dice_sorted: self.dice_sorted,
            rerolls_left: self.rerolls_left,
            player_to_move: 1u8.saturating_sub(self.player_to_move),
        }
    }
}

fn filled_count_from_avail(avail_mask: u16) -> u8 {
    // 15 categories total. filled_count = 15 - popcount(avail_mask).
    let avail = (avail_mask & 0x7FFF).count_ones() as u8;
    15u8 - avail
}

fn avail_bit_for_cat(cat: u8) -> u16 {
    1u16 << (14 - cat as u16)
}

fn push_avail_bits(out: &mut [f32], offset: &mut usize, avail_mask: u16) {
    for cat in 0u8..15 {
        let bit = avail_bit_for_cat(cat);
        out[*offset] = if (avail_mask & bit) != 0 { 1.0 } else { 0.0 };
        *offset += 1;
    }
}

fn push_scalar(out: &mut [f32], offset: &mut usize, v: f32) {
    out[*offset] = v;
    *offset += 1;
}

/// Encode state into feature vector v1, from POV of `player_to_move`.
pub fn encode_state_v1(s: &GameStateView) -> [f32; F] {
    assert!(s.player_to_move <= 1, "player_to_move must be 0 or 1");
    let me = s.players[s.player_to_move as usize];
    let opp = s.players[(1 - s.player_to_move) as usize];

    let mut out = [0.0f32; F];
    let mut off = 0usize;

    // my
    push_avail_bits(&mut out, &mut off, me.avail_mask);
    push_scalar(&mut out, &mut off, (me.upper_total_cap as f32) / 63.0);
    push_scalar(&mut out, &mut off, (me.total_score as f32) / SCORE_NORM);
    push_scalar(
        &mut out,
        &mut off,
        (filled_count_from_avail(me.avail_mask) as f32) / 15.0,
    );

    // opp
    push_avail_bits(&mut out, &mut off, opp.avail_mask);
    push_scalar(&mut out, &mut off, (opp.upper_total_cap as f32) / 63.0);
    push_scalar(&mut out, &mut off, (opp.total_score as f32) / SCORE_NORM);
    push_scalar(
        &mut out,
        &mut off,
        (filled_count_from_avail(opp.avail_mask) as f32) / 15.0,
    );

    // dice counts (faces 1..6)
    let mut counts = [0u8; 6];
    for &d in &s.dice_sorted {
        debug_assert!((1..=6).contains(&d));
        counts[(d - 1) as usize] += 1;
    }
    for &c in &counts {
        push_scalar(&mut out, &mut off, (c as f32) / 5.0);
    }

    // rerolls_left one-hot
    let r = s.rerolls_left.min(2) as usize;
    for i in 0..3 {
        push_scalar(&mut out, &mut off, if i == r { 1.0 } else { 0.0 });
    }

    debug_assert_eq!(off, F);
    out
}
