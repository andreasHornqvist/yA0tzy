#[cfg(test)]
mod tests {
    use crate::{
        action_to_index, avail_bit_for_cat, index_to_action, is_mark_index, legal_action_mask,
        mark_cat_from_index, Action, A, NUM_CATS,
    };

    #[test]
    fn action_index_roundtrip_keepmask() {
        for mask in 0u8..=31 {
            let a = Action::KeepMask(mask);
            let idx = action_to_index(a);
            assert_eq!(idx, mask);
            assert_eq!(index_to_action(idx), a);
            assert!(!is_mark_index(idx));
        }
    }

    #[test]
    fn action_index_roundtrip_mark() {
        for cat in 0u8..(NUM_CATS as u8) {
            let a = Action::Mark(cat);
            let idx = action_to_index(a);
            assert_eq!(idx, 32 + cat);
            assert_eq!(index_to_action(idx), a);
            assert!(is_mark_index(idx));
            assert_eq!(mark_cat_from_index(idx), cat);
        }
    }

    #[test]
    #[should_panic]
    fn index_to_action_out_of_range_panics() {
        let _ = index_to_action(47);
    }

    #[test]
    fn avail_bit_convention_spot_checks() {
        // cat=0 -> bit 14
        assert_eq!(avail_bit_for_cat(0), 1u16 << 14);
        // cat=14 -> bit 0
        assert_eq!(avail_bit_for_cat(14), 1u16 << 0);
    }

    #[test]
    fn legal_mask_rerolls_zero_only_marks() {
        // Only cats 0 and 14 available.
        let avail_mask = avail_bit_for_cat(0) | avail_bit_for_cat(14);
        let legal = legal_action_mask(avail_mask, 0);

        // KeepMasks all illegal
        for idx in 0..=31usize {
            assert!(
                !legal[idx],
                "KeepMask idx {} should be illegal at rerolls=0",
                idx
            );
        }

        // Marks follow avail_mask
        for cat in 0u8..(NUM_CATS as u8) {
            let idx = (32 + cat) as usize;
            let should_be_legal = cat == 0 || cat == 14;
            assert_eq!(
                legal[idx], should_be_legal,
                "Mark cat {} (idx {}) mismatch",
                cat, idx
            );
        }
    }

    #[test]
    fn legal_mask_rerolls_positive_keepmask_0_30_legal_31_illegal() {
        // All categories available.
        let avail_mask = (1u16 << 15) - 1;
        let legal = legal_action_mask(avail_mask, 2);

        // KeepMask 0..=30 legal
        for idx in 0..=30usize {
            assert!(
                legal[idx],
                "KeepMask idx {} should be legal at rerolls>0",
                idx
            );
        }
        // KeepMask 31 illegal
        assert!(!legal[31], "KeepMask(31) should be illegal when rerolls>0");

        // Marks all legal
        for idx in 32..A {
            assert!(
                legal[idx],
                "Mark idx {} should be legal when category available",
                idx
            );
        }
    }
}
