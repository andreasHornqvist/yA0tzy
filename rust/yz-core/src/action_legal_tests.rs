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
                ((legal >> idx) & 1) == 0,
                "KeepMask idx {} should be illegal at rerolls=0",
                idx
            );
        }

        // Marks follow avail_mask
        for cat in 0u8..(NUM_CATS as u8) {
            let idx = (32 + cat) as usize;
            let should_be_legal = cat == 0 || cat == 14;
            assert_eq!(
                (((legal >> idx) & 1) != 0),
                should_be_legal,
                "Mark cat {} (idx {}) mismatch",
                cat, idx
            );
        }
    }

    #[test]
    fn legal_mask_rerolls_positive_only_keepmask_legal() {
        // Mark-only-at-roll-3 rules: at rerolls > 0, only KeepMask(0..31) legal, Mark illegal.
        let avail_mask = (1u16 << 15) - 1;
        let legal = legal_action_mask(avail_mask, 2);

        // KeepMask 0..=31 all legal
        for idx in 0..=31usize {
            assert!(
                ((legal >> idx) & 1) != 0,
                "KeepMask idx {} should be legal at rerolls>0",
                idx
            );
        }

        // Marks all illegal at rerolls > 0
        for idx in 32..A {
            assert!(
                ((legal >> idx) & 1) == 0,
                "Mark idx {} should be illegal at rerolls>0 (mark-only-at-roll-3)",
                idx
            );
        }
    }

    #[test]
    fn legal_mask_rerolls_one_still_keepmask_only() {
        // Verify rule applies at rerolls_left=1 as well.
        let avail_mask = (1u16 << 15) - 1;
        let legal = legal_action_mask(avail_mask, 1);

        // KeepMask 0..=31 all legal
        for idx in 0..=31usize {
            assert!(
                ((legal >> idx) & 1) != 0,
                "KeepMask idx {} should be legal at rerolls=1",
                idx
            );
        }

        // Marks all illegal
        for idx in 32..A {
            assert!(
                ((legal >> idx) & 1) == 0,
                "Mark idx {} should be illegal at rerolls=1",
                idx
            );
        }
    }
}
