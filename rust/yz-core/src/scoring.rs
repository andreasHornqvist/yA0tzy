//! Scoring utilities (oracle-compatible).
//!
//! This module intentionally delegates dice→category scoring to the vendored
//! oracle implementation to avoid drift.

use crate::action::NUM_CATS;

/// Compute raw category scores for a 5-dice hand.
///
/// - Input dice must be in 1..=6. Order does not matter.
/// - Returned scores are **raw** category scores; upper bonus is not included.
pub fn scores_for_dice(dice: [u8; 5]) -> [i32; NUM_CATS] {
    swedish_yatzy_dp::game::scores_for_dice(dice)
}

/// Apply the realized score delta for marking category `cat` given the current dice.
///
/// Returns `(delta_score, new_upper_total_cap)` where:
/// - `delta_score` includes the one-time +50 upper bonus if crossing 63 from below
/// - `new_upper_total_cap` is clamped to 63 (per PRD + oracle convention)
///
/// # Panics
/// Panics if `cat` is out of range 0..14.
pub fn apply_mark_score(dice: [u8; 5], cat: u8, upper_total_cap: u8) -> (i32, u8) {
    assert!((cat as usize) < NUM_CATS, "cat out of range: {}", cat);

    let raw = scores_for_dice(dice)[cat as usize];

    // Only upper categories (0..=5) contribute to upper_total, with cap at 63.
    if cat < 6 {
        let prev_upper = upper_total_cap as i32;
        let unclamped_new_upper = prev_upper + raw;
        let new_upper_cap = unclamped_new_upper.clamp(0, 63) as u8;

        // Bonus triggers once when crossing from <63 to >=63.
        let bonus = if prev_upper < 63 && unclamped_new_upper >= 63 {
            50
        } else {
            0
        };

        (raw + bonus, new_upper_cap)
    } else {
        // Non-upper categories don't affect upper total.
        (raw, upper_total_cap)
    }
}
