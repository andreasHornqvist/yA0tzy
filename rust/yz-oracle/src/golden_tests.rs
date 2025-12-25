//! Golden tests for oracle conventions.
//!
//! These tests verify that our understanding of the oracle's conventions
//! matches the actual implementation, as documented in PRD Section 3.

use crate::{oracle, Action, CAT_NAMES, FULL_MASK, NUM_CATS};
use swedish_yatzy_dp::game::scores_for_dice;

// =============================================================================
// Category Order Tests (PRD 3.1)
// =============================================================================

#[test]
fn category_order_matches_prd() {
    // PRD 3.1: Categories indexed 0..14 in this exact order
    let expected = [
        "ones",
        "twos",
        "threes",
        "fours",
        "fives",
        "sixes",
        "pair",
        "two_pairs",
        "three_kind",
        "four_kind",
        "small_straight",
        "large_straight",
        "house",
        "chance",
        "yatzy",
    ];

    assert_eq!(CAT_NAMES.len(), 15);
    assert_eq!(NUM_CATS, 15);

    for (i, &name) in expected.iter().enumerate() {
        assert_eq!(
            CAT_NAMES[i], name,
            "Category {} should be '{}' but got '{}'",
            i, name, CAT_NAMES[i]
        );
    }
}

#[test]
fn category_indices_are_correct() {
    // Verify specific category indices
    assert_eq!(CAT_NAMES[0], "ones");
    assert_eq!(CAT_NAMES[5], "sixes");
    assert_eq!(CAT_NAMES[6], "pair");
    assert_eq!(CAT_NAMES[10], "small_straight");
    assert_eq!(CAT_NAMES[11], "large_straight");
    assert_eq!(CAT_NAMES[14], "yatzy");
}

// =============================================================================
// Availability Mask Tests (PRD 3.2)
// =============================================================================

#[test]
fn full_mask_value() {
    // FULL_MASK should be 0x7FFF = 32767 = (1 << 15) - 1
    assert_eq!(FULL_MASK, 0x7FFF);
    assert_eq!(FULL_MASK, (1u16 << 15) - 1);
    assert_eq!(FULL_MASK, 32767);
}

#[test]
fn avail_mask_bit_convention() {
    // PRD 3.2: bit (14 - cat) is 1 if category cat is available

    // For FULL_MASK, all bits 0..14 should be set
    for cat in 0..NUM_CATS {
        let bit_position = 14 - cat;
        let bit = 1u16 << bit_position;
        assert!(
            (FULL_MASK & bit) != 0,
            "Category {} (bit {}) should be set in FULL_MASK",
            cat,
            bit_position
        );
    }
}

#[test]
fn avail_mask_single_category_available() {
    // Test mask with only one category available
    for cat in 0..NUM_CATS {
        let bit_position = 14 - cat;
        let single_cat_mask = 1u16 << bit_position;

        // Verify the mask has only this category's bit set
        for other_cat in 0..NUM_CATS {
            let other_bit = 1u16 << (14 - other_cat);
            if other_cat == cat {
                assert!(
                    (single_cat_mask & other_bit) != 0,
                    "Category {} should be available",
                    cat
                );
            } else {
                assert!(
                    (single_cat_mask & other_bit) == 0,
                    "Category {} should not be available",
                    other_cat
                );
            }
        }
    }
}

#[test]
fn avail_mask_remove_category() {
    // Test removing a category from FULL_MASK
    for cat in 0..NUM_CATS {
        let bit_position = 14 - cat;
        let removed_mask = FULL_MASK & !(1u16 << bit_position);

        // The removed category should now be unavailable
        assert!(
            (removed_mask & (1u16 << bit_position)) == 0,
            "Category {} should be unavailable after removal",
            cat
        );

        // All other categories should still be available
        for other_cat in 0..NUM_CATS {
            if other_cat != cat {
                let other_bit = 1u16 << (14 - other_cat);
                assert!(
                    (removed_mask & other_bit) != 0,
                    "Category {} should still be available",
                    other_cat
                );
            }
        }
    }
}

// =============================================================================
// KeepMask Tests (PRD 3.5)
// =============================================================================

#[test]
fn keepmask_keep_all() {
    // mask = 31 (0b11111) keeps all 5 dice
    let mask: u8 = 0b11111;
    assert_eq!(mask, 31);

    // All bits 0..4 should be set
    for i in 0..5 {
        let bit = 1u8 << (4 - i);
        assert!(
            (mask & bit) != 0,
            "Bit for dice[{}] should be set in keep-all mask",
            i
        );
    }
}

#[test]
fn keepmask_keep_none() {
    // mask = 0 keeps no dice (reroll all)
    let mask: u8 = 0b00000;
    assert_eq!(mask, 0);

    // No bits should be set
    for i in 0..5 {
        let bit = 1u8 << (4 - i);
        assert!(
            (mask & bit) == 0,
            "Bit for dice[{}] should not be set in keep-none mask",
            i
        );
    }
}

#[test]
fn keepmask_bit_position_convention() {
    // PRD 3.5: bit (4 - i) corresponds to dice[i] (sorted)

    // Test each single-die keep mask
    for i in 0..5 {
        let bit_position = 4 - i;
        let single_die_mask = 1u8 << bit_position;

        // Verify only this die's bit is set
        for j in 0..5 {
            let other_bit = 1u8 << (4 - j);
            if j == i {
                assert!(
                    (single_die_mask & other_bit) != 0,
                    "Mask for keeping dice[{}] should have bit {} set",
                    i,
                    bit_position
                );
            } else {
                assert!(
                    (single_die_mask & other_bit) == 0,
                    "Mask for keeping dice[{}] should not have bit for dice[{}] set",
                    i,
                    j
                );
            }
        }
    }
}

#[test]
fn keepmask_examples() {
    // Some concrete examples

    // Keep first two dice (dice[0] and dice[1])
    // bit(4-0)=4 and bit(4-1)=3 -> 0b11000 = 24
    let keep_first_two: u8 = (1 << 4) | (1 << 3);
    assert_eq!(keep_first_two, 0b11000);
    assert_eq!(keep_first_two, 24);

    // Keep last two dice (dice[3] and dice[4])
    // bit(4-3)=1 and bit(4-4)=0 -> 0b00011 = 3
    let keep_last_two: u8 = (1 << 1) | (1 << 0);
    assert_eq!(keep_last_two, 0b00011);
    assert_eq!(keep_last_two, 3);

    // Keep middle die (dice[2])
    // bit(4-2)=2 -> 0b00100 = 4
    let keep_middle: u8 = 1 << 2;
    assert_eq!(keep_middle, 0b00100);
    assert_eq!(keep_middle, 4);
}

// =============================================================================
// Scoring Tests (PRD 5.4)
// =============================================================================

#[test]
fn scoring_upper_section() {
    // Test upper section scoring (ones through sixes)
    let dice = [1, 2, 3, 4, 5];
    let scores = scores_for_dice(dice);

    assert_eq!(scores[0], 1); // ones: 1×1 = 1
    assert_eq!(scores[1], 2); // twos: 1×2 = 2
    assert_eq!(scores[2], 3); // threes: 1×3 = 3
    assert_eq!(scores[3], 4); // fours: 1×4 = 4
    assert_eq!(scores[4], 5); // fives: 1×5 = 5
    assert_eq!(scores[5], 0); // sixes: 0×6 = 0
}

#[test]
fn scoring_upper_section_multiples() {
    // Test with multiple of same face
    let dice = [3, 3, 3, 4, 5];
    let scores = scores_for_dice(dice);

    assert_eq!(scores[2], 9); // threes: 3×3 = 9
    assert_eq!(scores[3], 4); // fours: 1×4 = 4
}

#[test]
fn scoring_small_straight() {
    // PRD: small straight = 15
    let dice = [1, 2, 3, 4, 5];
    let scores = scores_for_dice(dice);

    assert_eq!(scores[10], 15, "Small straight (1-2-3-4-5) should score 15");
    assert_eq!(scores[11], 0, "Large straight should score 0 for 1-2-3-4-5");
}

#[test]
fn scoring_large_straight() {
    // PRD: large straight = 20
    let dice = [2, 3, 4, 5, 6];
    let scores = scores_for_dice(dice);

    assert_eq!(scores[10], 0, "Small straight should score 0 for 2-3-4-5-6");
    assert_eq!(scores[11], 20, "Large straight (2-3-4-5-6) should score 20");
}

#[test]
fn scoring_yatzy() {
    // PRD: yatzy = 50
    let dice = [6, 6, 6, 6, 6];
    let scores = scores_for_dice(dice);

    assert_eq!(scores[14], 50, "Yatzy should score 50");
    assert_eq!(scores[5], 30, "Sixes should score 30 for five 6s");
}

#[test]
fn scoring_house() {
    // House = full house (3 of a kind + pair)
    let dice = [2, 2, 3, 3, 3];
    let scores = scores_for_dice(dice);

    // House scores sum of dice
    let sum: i32 = dice.iter().map(|&d| d as i32).sum();
    assert_eq!(scores[12], sum, "House should score sum of dice");
    assert_eq!(scores[12], 13); // 2+2+3+3+3 = 13
}

#[test]
fn scoring_pair() {
    // Pair = highest pair × 2
    let dice = [1, 2, 2, 5, 5];
    let scores = scores_for_dice(dice);

    assert_eq!(scores[6], 10, "Pair should score highest pair (5+5=10)");
}

#[test]
fn scoring_two_pairs() {
    // Two pairs = sum of both pairs
    let dice = [1, 2, 2, 5, 5];
    let scores = scores_for_dice(dice);

    assert_eq!(scores[7], 14, "Two pairs should score 2+2+5+5=14");
}

#[test]
fn scoring_three_of_a_kind() {
    let dice = [3, 3, 3, 4, 5];
    let scores = scores_for_dice(dice);

    assert_eq!(scores[8], 9, "Three of a kind should score 3×3=9");
}

#[test]
fn scoring_four_of_a_kind() {
    let dice = [4, 4, 4, 4, 5];
    let scores = scores_for_dice(dice);

    assert_eq!(scores[9], 16, "Four of a kind should score 4×4=16");
}

#[test]
fn scoring_chance() {
    // Chance = sum of all dice
    let dice = [1, 2, 3, 4, 6];
    let scores = scores_for_dice(dice);

    assert_eq!(scores[13], 16, "Chance should score sum 1+2+3+4+6=16");
}

// =============================================================================
// Oracle Action Tests
// =============================================================================

#[test]
fn oracle_action_mark_uses_correct_indices() {
    let oracle = oracle();

    // With no rerolls left, oracle must return Mark action
    let dice = [1, 2, 3, 4, 5];
    let (action, _ev) = oracle.best_action(FULL_MASK, 0, dice, 0);

    match action {
        Action::Mark { cat } => {
            assert!(
                (cat as usize) < NUM_CATS,
                "Mark category {} should be < {}",
                cat,
                NUM_CATS
            );
        }
        Action::KeepMask { .. } => {
            panic!("Oracle should return Mark when rerolls_left=0");
        }
    }
}

#[test]
fn oracle_action_keepmask_uses_correct_range() {
    let oracle = oracle();

    // With rerolls left, oracle might return KeepMask
    // Use dice that oracle would want to reroll
    let dice = [1, 1, 2, 3, 4];
    let (action, _ev) = oracle.best_action(FULL_MASK, 0, dice, 2);

    match action {
        Action::KeepMask { mask } => {
            assert!(mask < 32, "KeepMask {} should be < 32 (5 bits)", mask);
            // mask=31 (keep all) would trigger a Mark instead
            assert!(mask < 31, "KeepMask should not be 31 (keep all)");
        }
        Action::Mark { cat } => {
            // Oracle chose to mark early - this is valid
            assert!(
                (cat as usize) < NUM_CATS,
                "Mark category {} should be < {}",
                cat,
                NUM_CATS
            );
        }
    }
}

#[test]
fn oracle_respects_avail_mask() {
    let oracle = oracle();

    // Only yatzy (cat 14) available
    let only_yatzy_avail = 1u16 << (14 - 14); // bit 0
    let dice = [6, 6, 6, 6, 6]; // Perfect yatzy

    let (action, _ev) = oracle.best_action(only_yatzy_avail, 0, dice, 0);

    match action {
        Action::Mark { cat } => {
            assert_eq!(cat, 14, "Should mark yatzy when it's the only option");
        }
        Action::KeepMask { .. } => {
            panic!("Should Mark when rerolls=0");
        }
    }
}
