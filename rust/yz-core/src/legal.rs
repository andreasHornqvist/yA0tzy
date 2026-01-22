//! Legal action mask generation (PRD §5.2).

use crate::action::{avail_bit_for_cat, Action, A, NUM_CATS};

/// Bitset of legal actions for the fixed action space A=47.
///
/// Bit `i` corresponds to action index `i`.
pub type LegalMask = u64;

/// Return whether action index `idx` is legal under this mask.
#[inline]
pub fn is_legal(mask: LegalMask, idx: usize) -> bool {
    debug_assert!(idx < A);
    ((mask >> idx) & 1) != 0
}

/// Convert a legal bitset into a `[u8; A]` array (0/1 bytes).
pub fn to_u8_array(mask: LegalMask) -> [u8; A] {
    let mut out = [0u8; A];
    for i in 0..A {
        out[i] = if is_legal(mask, i) { 1 } else { 0 };
    }
    out
}

/// Return legality of each action index in the fixed action space A=47.
///
/// Mark-only-at-roll-3 rules:
/// - If rerolls_left == 0 (roll 3): only Mark(cat) is legal (if category available)
/// - If rerolls_left > 0 (rolls 1-2): only KeepMask(0..31) is legal, Mark is illegal
///   - KeepMask(31) = "keep all" advances to next roll without changing dice
pub fn legal_action_mask(avail_mask: u16, rerolls_left: u8) -> LegalMask {
    let mut legal: LegalMask = 0;

    if rerolls_left == 0 {
        // Roll 3: only Mark(cat) is legal
        for cat in 0..(NUM_CATS as u8) {
            let bit = avail_bit_for_cat(cat);
            let is_avail = (avail_mask & bit) != 0;
            let idx = crate::action::action_to_index(Action::Mark(cat)) as usize;
            if is_avail {
                legal |= 1u64 << idx;
            }
        }
    } else {
        // Rolls 1-2: only KeepMask(0..31) is legal, Mark is illegal
        for mask in 0u8..=31 {
            let idx = crate::action::action_to_index(Action::KeepMask(mask)) as usize;
            legal |= 1u64 << idx;
        }
    }

    legal
}
