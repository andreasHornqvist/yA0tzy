//! Game primitives for Swedish/Scandinavian Yatzy.
//!
//! Category order matches the Java reference implementation (`ansjob/optimalt-yatzy`).

pub const NUM_CATS: usize = 15;
pub const FULL_MASK: u16 = (1 << NUM_CATS) - 1;

/// Category names in index order (0..14).
pub const CAT_NAMES: [&str; NUM_CATS] = [
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

/// Compute all category scores for a 5-dice hand.
///
/// - Input dice must be in 1..=6. Order does not matter (function sorts internally).
/// - Returned scores are **raw** category scores (upper bonus is *not* applied here).
pub fn scores_for_dice(mut dice: [u8; 5]) -> [i32; NUM_CATS] {
    dice.sort();

    let mut counts = [0u8; 6];
    for &d in &dice {
        counts[(d - 1) as usize] += 1;
    }

    let mut s = [0i32; NUM_CATS];

    // Upper section
    for i in 0..6 {
        s[i] = counts[i] as i32 * (i as i32 + 1);
    }

    // Pair (highest)
    for i in (0..6).rev() {
        if counts[i] >= 2 {
            s[6] = 2 * (i as i32 + 1);
            break;
        }
    }

    // Two pairs
    let pairs: Vec<i32> = (0..6)
        .filter(|&i| counts[i] >= 2)
        .map(|i| i as i32 + 1)
        .collect();
    if pairs.len() >= 2 {
        s[7] = 2 * (pairs[pairs.len() - 1] + pairs[pairs.len() - 2]);
    }

    // Three of a kind
    for i in (0..6).rev() {
        if counts[i] >= 3 {
            s[8] = 3 * (i as i32 + 1);
            break;
        }
    }

    // Four of a kind
    for i in (0..6).rev() {
        if counts[i] >= 4 {
            s[9] = 4 * (i as i32 + 1);
            break;
        }
    }

    // Small straight (1-2-3-4-5)
    if dice == [1, 2, 3, 4, 5] {
        s[10] = 15;
    }

    // Large straight (2-3-4-5-6)
    if dice == [2, 3, 4, 5, 6] {
        s[11] = 20;
    }

    // House (3+2)
    let has3 = counts.iter().any(|&x| x == 3);
    let has2 = counts.iter().any(|&x| x == 2);
    if has3 && has2 {
        s[12] = dice.iter().map(|&x| x as i32).sum();
    }

    // Chance
    s[13] = dice.iter().map(|&x| x as i32).sum();

    // Yatzy
    if counts.iter().any(|&x| x == 5) {
        s[14] = 50;
    }

    s
}
