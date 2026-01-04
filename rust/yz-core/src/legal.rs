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
/// PRD §5.2 legality rules:
/// - If rerolls_left == 0: only Mark(cat) is legal (if category available)
/// - If rerolls_left > 0:
///   - Mark(cat) legal iff available
///   - KeepMask(mask) legal for mask 0..=30
///   - KeepMask(31) illegal (keep-all is dominated)
pub fn legal_action_mask(avail_mask: u16, rerolls_left: u8) -> LegalMask {
    let mut legal: LegalMask = 0;

    // Mark(cat) indices: 32..=46
    for cat in 0..(NUM_CATS as u8) {
        let bit = avail_bit_for_cat(cat);
        let is_avail = (avail_mask & bit) != 0;
        let idx = crate::action::action_to_index(Action::Mark(cat)) as usize;
        if is_avail {
            legal |= 1u64 << idx;
        }
    }

    if rerolls_left == 0 {
        // KeepMask actions remain illegal
        return legal;
    }

    // KeepMask indices 0..=30 are legal; 31 illegal.
    for mask in 0u8..=30 {
        let idx = crate::action::action_to_index(Action::KeepMask(mask)) as usize;
        legal |= 1u64 << idx;
    }
    // mask 31 left as 0

    legal
}
