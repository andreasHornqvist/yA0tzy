//! Afterstate + chance-outcome utilities for explicit chance nodes.
//!
//! This module is intentionally kept inside `yz-mcts` so we can refactor the search tree without
//! changing `yz-core` semantics.

use rand::Rng;
use yz_core::{canonicalize_keepmask, GameState, PlayerState};

/// Key for a chance outcome histogram `[u8; 6]` (counts of faces 1..6).
///
/// Encoding: pack 6 counts (each 0..=5) into 3 bits each (18 bits total).
pub type OutcomeHistKey = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AfterState {
    pub players: [PlayerState; 2],
    pub player_to_act: u8,
    /// Rerolls left *after* taking the KeepMask decision (already decremented).
    pub rerolls_left: u8,
    pub kept_hist: [u8; 6],
    pub k_to_roll: u8,
}

pub fn kept_hist_from_keepmask(dice_sorted: [u8; 5], mask: u8) -> ([u8; 6], u8) {
    debug_assert!(dice_sorted.windows(2).all(|w| w[0] <= w[1]));
    debug_assert!(mask < 32);

    let mut kept = [0u8; 6];
    let mut kept_n: u8 = 0;
    for i in 0..5usize {
        let bit = 1u8 << (4 - i);
        if (mask & bit) != 0 {
            let d = dice_sorted[i];
            debug_assert!((1..=6).contains(&d));
            kept[(d - 1) as usize] = kept[(d - 1) as usize].saturating_add(1);
            kept_n = kept_n.saturating_add(1);
        }
    }
    let k_to_roll = 5u8.saturating_sub(kept_n);
    (kept, k_to_roll)
}

pub fn dice_sorted_from_hist(hist: [u8; 6]) -> [u8; 5] {
    let total: u8 = hist.iter().sum();
    assert_eq!(total, 5, "dice histogram must sum to 5, got {total}");

    let mut out = [0u8; 5];
    let mut idx = 0usize;
    for (face0, &c) in hist.iter().enumerate() {
        let face = (face0 as u8) + 1;
        for _ in 0..c {
            out[idx] = face;
            idx += 1;
        }
    }
    debug_assert_eq!(idx, 5);
    out
}

pub fn sample_roll_hist<R: Rng>(k_to_roll: u8, rng: &mut R) -> [u8; 6] {
    assert!(k_to_roll <= 5, "k_to_roll must be 0..=5, got {k_to_roll}");
    let mut h = [0u8; 6];
    for _ in 0..k_to_roll {
        let face: u8 = rng.gen_range(1..=6);
        h[(face - 1) as usize] += 1;
    }
    h
}

pub fn pack_outcome_key(hist: [u8; 6]) -> OutcomeHistKey {
    let mut k: u32 = 0;
    for (i, &c) in hist.iter().enumerate() {
        assert!(c <= 5, "hist count out of range: hist[{i}]={c}");
        k |= (c as u32) << (i * 3);
    }
    k
}

pub fn unpack_outcome_key(key: OutcomeHistKey) -> [u8; 6] {
    let mut h = [0u8; 6];
    for i in 0..6usize {
        h[i] = ((key >> (i * 3)) & 0x7) as u8;
    }
    h
}

pub fn afterstate_from_keepmask(state: &GameState, keepmask_raw: u8) -> AfterState {
    assert!(keepmask_raw < 32);
    assert!(
        state.rerolls_left > 0,
        "KeepMask afterstate only valid at rerolls_left>0"
    );

    let canonical = canonicalize_keepmask(state.dice_sorted, keepmask_raw);
    let (kept_hist, k_to_roll) = kept_hist_from_keepmask(state.dice_sorted, canonical);

    AfterState {
        players: state.players,
        player_to_act: state.player_to_move,
        rerolls_left: state
            .rerolls_left
            .checked_sub(1)
            .expect("rerolls_left checked above"),
        kept_hist,
        k_to_roll,
    }
}

pub fn apply_roll_hist_to_afterstate(as_: &AfterState, roll_hist: [u8; 6]) -> GameState {
    let roll_total: u8 = roll_hist.iter().sum();
    assert_eq!(
        roll_total, as_.k_to_roll,
        "roll_hist sum must equal k_to_roll"
    );

    let mut next = [0u8; 6];
    for i in 0..6usize {
        next[i] = as_.kept_hist[i] + roll_hist[i];
    }
    let dice_sorted = dice_sorted_from_hist(next);

    GameState {
        players: as_.players,
        dice_sorted,
        rerolls_left: as_.rerolls_left,
        player_to_move: as_.player_to_act,
    }
}

