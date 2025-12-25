#[cfg(test)]
mod tests {
    use crate::{apply_mark_score, scores_for_dice};

    #[test]
    fn scoring_parity_exhaustive_5dice() {
        // Exhaustive parity against oracle scorer (6^5 = 7776 hands).
        for a in 1u8..=6 {
            for b in 1u8..=6 {
                for c in 1u8..=6 {
                    for d in 1u8..=6 {
                        for e in 1u8..=6 {
                            let dice = [a, b, c, d, e];
                            let ours = scores_for_dice(dice);
                            let oracle = swedish_yatzy_dp::game::scores_for_dice(dice);
                            assert_eq!(ours, oracle, "Mismatch for dice {:?}", dice);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn upper_bonus_triggers_on_crossing_63() {
        // cat=0 (ones), dice has 5 ones -> raw=5
        let dice = [1, 1, 1, 1, 1];
        let cat = 0u8;

        // 60 + 5 crosses 63 -> +50 bonus
        let (delta, new_upper) = apply_mark_score(dice, cat, 60);
        assert_eq!(delta, 5 + 50);
        assert_eq!(new_upper, 63);
    }

    #[test]
    fn upper_bonus_does_not_trigger_if_already_at_cap() {
        // Already capped at 63 means bonus already achieved; no further bonus.
        let dice = [6, 6, 6, 6, 6];
        let cat = 5u8; // sixes, raw=30
        let (delta, new_upper) = apply_mark_score(dice, cat, 63);
        assert_eq!(delta, 30);
        assert_eq!(new_upper, 63);
    }

    #[test]
    fn non_upper_categories_do_not_change_upper_total() {
        let dice = [2, 2, 3, 3, 3];
        let cat = 12u8; // house
        let (delta, new_upper) = apply_mark_score(dice, cat, 10);
        // house raw score is sum=13
        assert_eq!(delta, 13);
        assert_eq!(new_upper, 10);
    }
}
