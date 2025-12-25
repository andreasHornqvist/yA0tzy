#[cfg(test)]
mod tests {
    use crate::chance::{apply_keepmask, roll5, EventKey};

    #[test]
    fn roll5_is_deterministic() {
        let key = EventKey {
            episode_seed: 123,
            player: 0,
            round_idx: 7,
            roll_idx: 1,
        };
        assert_eq!(roll5(key), roll5(key));
    }

    #[test]
    fn roll5_values_in_range() {
        let key = EventKey {
            episode_seed: 999,
            player: 1,
            round_idx: 0,
            roll_idx: 0,
        };
        let d = roll5(key);
        for x in d {
            assert!((1..=6).contains(&x), "die out of range: {}", x);
        }
    }

    #[test]
    fn roll_idx_changes_stream() {
        let k0 = EventKey {
            episode_seed: 42,
            player: 0,
            round_idx: 3,
            roll_idx: 0,
        };
        let k1 = EventKey { roll_idx: 1, ..k0 };
        assert_ne!(roll5(k0), roll5(k1));
    }

    #[test]
    fn keepmask_bit_mapping_sanity() {
        // prev_sorted indices 0..4 map to bits 4..0
        let prev = [1, 2, 3, 4, 6];
        let key = EventKey {
            episode_seed: 1,
            player: 0,
            round_idx: 0,
            roll_idx: 1,
        };

        // Keep only index 0 (bit4)
        let out = apply_keepmask(prev, 0b1_0000, key);
        assert_eq!(out.len(), 5);
        // Kept value 1 should still exist in the resulting multiset.
        assert!(out.contains(&1));
    }

    #[test]
    fn duplicate_die_reroll_exploit_prevented() {
        // Two identical dice (1,1). Rerolling the first vs second 1 should not
        // change the outcome under event-keyed generation (only k matters).
        let prev = [1, 1, 3, 4, 6];
        let key = EventKey {
            episode_seed: 777,
            player: 0,
            round_idx: 5,
            roll_idx: 1,
        };

        // Reroll only index 0 (a 1): keep indices 1..4 => 0b01111 = 15
        let mask_reroll_i0 = 0b0_1111u8;
        // Reroll only index 1 (the other 1): keep indices 0,2,3,4 => 0b10111 = 23
        let mask_reroll_i1 = 0b1_0111u8;

        let out0 = apply_keepmask(prev, mask_reroll_i0, key);
        let out1 = apply_keepmask(prev, mask_reroll_i1, key);
        assert_eq!(out0, out1);
    }
}
