//! Stable state key for mapping realized stochastic children.

use yz_core::GameState;

/// A compact, stable key for a `GameState`.
///
/// This is intentionally independent of Rust's `Hash` randomness so we can use it for deterministic
/// behavior in eval mode.
pub type StateKey = u128;

pub fn state_key(s: &GameState) -> StateKey {
    // Layout (low -> high bits), little-endian conceptual:
    // - players[0].avail_mask: 15 bits (stored in u16)
    // - players[1].avail_mask: 15 bits
    // - players[0].upper_total_cap: 6 bits (0..63)
    // - players[1].upper_total_cap: 6 bits
    // - players[0].total_score: i16 bits (16)
    // - players[1].total_score: i16 bits (16)
    // - dice: 5 * 3 bits (values 1..6)
    // - rerolls_left: 2 bits
    // - player_to_move: 1 bit
    //
    // Total: 15+15+6+6+16+16+15+2+1 = 92 bits.

    let mut x: u128 = 0;
    let mut shift: u32 = 0;

    let p0 = s.players[0];
    let p1 = s.players[1];

    x |= (p0.avail_mask as u128) << shift;
    shift += 16;
    x |= (p1.avail_mask as u128) << shift;
    shift += 16;

    x |= ((p0.upper_total_cap as u128) & 0x3F) << shift;
    shift += 6;
    x |= ((p1.upper_total_cap as u128) & 0x3F) << shift;
    shift += 6;

    x |= ((p0.total_score as u16) as u128) << shift;
    shift += 16;
    x |= ((p1.total_score as u16) as u128) << shift;
    shift += 16;

    for &d in &s.dice_sorted {
        debug_assert!((1..=6).contains(&d));
        x |= ((d as u128) & 0x7) << shift;
        shift += 3;
    }

    x |= ((s.rerolls_left as u128) & 0x3) << shift;
    shift += 2;
    x |= ((s.player_to_move as u128) & 0x1) << shift;

    x
}
