//! Action space definition and index mapping (PRD §5.2).
//!
//! Action space size: A = 47
//! - idx 0..=31  : KeepMask(mask)
//! - idx 32..=46 : Mark(cat) where cat = idx - 32

pub const NUM_CATS: usize = 15;
pub const A: usize = 32 + NUM_CATS; // 47

/// Canonicalize a KeepMask action for a given sorted dice configuration.
///
/// When dice contain duplicates, multiple different masks can represent the same kept multiset.
/// This helper maps any keepmask to a canonical representative by keeping the rightmost
/// occurrences for each face.
///
/// # Preconditions
/// - `dice_sorted` must be sorted non-decreasing.
/// - `mask` must be in `0..=31`.
pub fn canonicalize_keepmask(dice_sorted: [u8; 5], mask: u8) -> u8 {
    debug_assert!(dice_sorted.windows(2).all(|w| w[0] <= w[1]));
    debug_assert!(mask < 32);

    // Count kept faces.
    let mut need = [0u8; 6];
    for i in 0..5usize {
        let bit = 1u8 << (4 - i);
        if (mask & bit) != 0 {
            let face = dice_sorted[i] as usize;
            debug_assert!((1..=6).contains(&face));
            need[face - 1] = need[face - 1].saturating_add(1);
        }
    }

    // Reconstruct canonical mask by keeping the rightmost occurrences for each face.
    let mut out: u8 = 0;
    for i in (0..5usize).rev() {
        let bit = 1u8 << (4 - i);
        let face = dice_sorted[i] as usize;
        let slot = face - 1;
        if need[slot] > 0 {
            need[slot] -= 1;
            out |= bit;
        }
    }
    out
}

/// Oracle-compatible action representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// KeepMask over sorted dice; mask is 0..=31.
    KeepMask(u8),
    /// Mark category index 0..=14.
    Mark(u8),
}

/// Convert an `Action` to its policy index (0..=46).
///
/// # Panics
/// Panics if the action is out of range (mask > 31 or cat > 14).
pub fn action_to_index(a: Action) -> u8 {
    match a {
        Action::KeepMask(mask) => {
            assert!(mask < 32, "KeepMask out of range: {}", mask);
            mask
        }
        Action::Mark(cat) => {
            assert!((cat as usize) < NUM_CATS, "Mark cat out of range: {}", cat);
            32 + cat
        }
    }
}

/// Convert a policy index (0..=46) to an `Action`.
///
/// # Panics
/// Panics if idx > 46.
pub fn index_to_action(idx: u8) -> Action {
    assert!((idx as usize) < A, "Action index out of range: {}", idx);
    if idx < 32 {
        Action::KeepMask(idx)
    } else {
        Action::Mark(idx - 32)
    }
}

/// True if `idx` is a Mark action (32..=46).
pub fn is_mark_index(idx: u8) -> bool {
    idx >= 32
}

/// If `idx` is a Mark action, return its category (0..=14).
///
/// # Panics
/// Panics if `idx` is not a Mark index.
pub fn mark_cat_from_index(idx: u8) -> u8 {
    assert!(is_mark_index(idx), "Not a Mark index: {}", idx);
    idx - 32
}

/// Oracle bit convention: bit (14 - cat) is 1 if category is available.
pub fn avail_bit_for_cat(cat: u8) -> u16 {
    assert!((cat as usize) < NUM_CATS, "cat out of range: {}", cat);
    1u16 << (14 - (cat as u16))
}
