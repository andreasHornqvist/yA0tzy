//! Legal action mask generation (PRD §5.2).

use crate::action::{avail_bit_for_cat, Action, A, NUM_CATS};

/// Return legality of each action index in the fixed action space A=47.
///
/// PRD §5.2 legality rules:
/// - If rerolls_left == 0: only Mark(cat) is legal (if category available)
/// - If rerolls_left > 0:
///   - Mark(cat) legal iff available
///   - KeepMask(mask) legal for mask 0..=30
///   - KeepMask(31) illegal (keep-all is dominated)
pub fn legal_action_mask(avail_mask: u16, rerolls_left: u8) -> [bool; A] {
    let mut legal = [false; A];

    // Mark(cat) indices: 32..=46
    for cat in 0..(NUM_CATS as u8) {
        let bit = avail_bit_for_cat(cat);
        let is_avail = (avail_mask & bit) != 0;
        let idx = crate::action::action_to_index(Action::Mark(cat)) as usize;
        legal[idx] = is_avail;
    }

    if rerolls_left == 0 {
        // KeepMask actions remain illegal
        return legal;
    }

    // KeepMask indices 0..=30 are legal; 31 illegal.
    for mask in 0u8..=30 {
        let idx = crate::action::action_to_index(Action::KeepMask(mask)) as usize;
        legal[idx] = true;
    }
    // mask 31 left as false

    legal
}
