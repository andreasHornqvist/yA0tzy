//! Dynamic-programming oracle for optimal Swedish/Scandinavian Yatzy play.
//!
//! ## Low-level state specification (for integrating with your own Yatzy engine)
//! - **Categories**: indices `0..14` in [`crate::game::CAT_NAMES`] order.
//! - **avail_mask**: `u16` where bit `(14 - cat)` is **1 if category `cat` is available**.
//! - **upper_total**: `0..=63` (values above 63 are clamped to 63).
//! - **dice**: 5 values in `1..=6` (sorted or unsorted; the oracle sorts internally).
//! - **rerolls_left**: `0..=2`.
//!
//! ## Action encoding
//! - [`Action::Mark`]: fill category `cat` (0..14).
//! - [`Action::KeepMask`]: keep mask over the **sorted** dice. Bit `(4-i)` refers to `dice[i]`
//!   after sorting. Dice with a 0-bit are rerolled.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::game::{FULL_MASK, NUM_CATS};

const MAX_HAND: usize = 252; // All 5-dice multisets (1..=252 used, 0 unused)
const MAX_KEEPER: usize = 462; // All 0-5 dice multisets (0..=461)

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Mark category index 0..14
    Mark { cat: u8 },
    /// Keep mask over sorted dice; bit (4-i) corresponds to dice[i]
    KeepMask { mask: u8 },
}

// ============================================================================
// Hand: 5-dice multiset (sorted)
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Hand {
    dice: [u8; 5], // sorted
}

impl Hand {
    fn new(a: u8, b: u8, c: u8, d: u8, e: u8) -> Self {
        let mut dice = [a, b, c, d, e];
        dice.sort();
        Hand { dice }
    }

    fn to_counts(&self) -> [u8; 6] {
        let mut counts = [0u8; 6];
        for &d in &self.dice {
            counts[(d - 1) as usize] += 1;
        }
        counts
    }
}

// ============================================================================
// Keeper: 0-5 dice multiset (the "keeper trick" optimization)
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Keeper {
    counts: [u8; 6], // counts[i] = count of face (i+1)
    total: u8,
}

impl Keeper {
    fn new(counts: [u8; 6]) -> Self {
        let total = counts.iter().sum();
        Keeper { counts, total }
    }

    fn add(&self, face: u8) -> Keeper {
        let mut new_counts = self.counts;
        new_counts[(face - 1) as usize] += 1;
        Keeper {
            counts: new_counts,
            total: self.total + 1,
        }
    }
}

// ============================================================================
// Precomputed tables
// ============================================================================

#[allow(dead_code)]
struct Tables {
    // All hands indexed 1..252 (0 unused, matching Java)
    hands: Vec<Hand>,
    hand_to_idx: FxHashMap<Hand, usize>,

    // All keepers indexed 0..461
    keepers: Vec<Keeper>,
    keeper_to_idx: FxHashMap<Keeper, usize>,

    // Keepers grouped by count (0..5)
    keepers_by_count: Vec<Vec<usize>>,

    // Score table: scores[hand_idx][cat] (hand_idx 1-based)
    scores: Vec<[i32; NUM_CATS]>,

    // OPTIMIZATION: Precomputed lookups to avoid hash operations
    // hand_keeper_map[hand_idx][mask] = keeper_idx for that keep mask
    hand_keeper_map: Vec<[usize; 32]>,

    // keeper_add[keeper_idx][face-1] = new keeper_idx after adding that face
    keeper_add: Vec<[usize; 6]>,

    // hand_to_keeper[hand_idx] = keeper_idx for the 5-dice keeper
    hand_to_keeper: Vec<usize>,
}

impl Tables {
    fn new() -> Self {
        // Generate all hands (252 total)
        let mut hands = vec![Hand::new(1, 1, 1, 1, 1); MAX_HAND + 1]; // 0 placeholder
        let mut hand_to_idx = FxHashMap::default();
        let mut idx = 1;
        for a in 1u8..=6 {
            for b in a..=6 {
                for c in b..=6 {
                    for d in c..=6 {
                        for e in d..=6 {
                            let h = Hand::new(a, b, c, d, e);
                            hands[idx] = h;
                            hand_to_idx.insert(h, idx);
                            idx += 1;
                        }
                    }
                }
            }
        }

        // Generate all keepers (462 total, including empty)
        let mut keepers = Vec::with_capacity(MAX_KEEPER);
        let mut keeper_to_idx = FxHashMap::default();
        let mut keepers_by_count: Vec<Vec<usize>> = vec![Vec::new(); 6];
        let mut kid = 0;

        for a in 0u8..=6 {
            for b in a..=6 {
                for c in b..=6 {
                    for d in c..=6 {
                        for e in d..=6 {
                            let mut counts = [0u8; 6];
                            if a > 0 {
                                counts[(a - 1) as usize] += 1;
                            }
                            if b > 0 {
                                counts[(b - 1) as usize] += 1;
                            }
                            if c > 0 {
                                counts[(c - 1) as usize] += 1;
                            }
                            if d > 0 {
                                counts[(d - 1) as usize] += 1;
                            }
                            if e > 0 {
                                counts[(e - 1) as usize] += 1;
                            }
                            let k = Keeper::new(counts);
                            keepers.push(k);
                            keeper_to_idx.insert(k, kid);
                            keepers_by_count[k.total as usize].push(kid);
                            kid += 1;
                        }
                    }
                }
            }
        }

        // Precompute scores
        let scores = Self::compute_scores(&hands);

        // Precompute hand→keeper mappings
        let mut hand_keeper_map = vec![[0usize; 32]; MAX_HAND + 1];
        let mut hand_to_keeper = vec![0usize; MAX_HAND + 1];

        for hand_idx in 1..=MAX_HAND {
            let hand = &hands[hand_idx];
            for mask in 0u8..32 {
                let mut counts = [0u8; 6];
                for i in 0..5 {
                    if (mask & (1 << (4 - i))) != 0 {
                        counts[(hand.dice[i] - 1) as usize] += 1;
                    }
                }
                let k = Keeper::new(counts);
                hand_keeper_map[hand_idx][mask as usize] = keeper_to_idx[&k];
            }
            // mask=31 (0b11111) = keep all = 5-dice keeper
            hand_to_keeper[hand_idx] = hand_keeper_map[hand_idx][31];
        }

        // Precompute keeper→keeper add transitions
        let mut keeper_add = vec![[0usize; 6]; MAX_KEEPER];
        for kid in 0..MAX_KEEPER {
            let keeper = &keepers[kid];
            if keeper.total < 5 {
                for face in 1u8..=6 {
                    let next = keeper.add(face);
                    keeper_add[kid][(face - 1) as usize] = keeper_to_idx[&next];
                }
            }
        }

        Tables {
            hands,
            hand_to_idx,
            keepers,
            keeper_to_idx,
            keepers_by_count,
            scores,
            hand_keeper_map,
            keeper_add,
            hand_to_keeper,
        }
    }

    fn compute_scores(hands: &[Hand]) -> Vec<[i32; NUM_CATS]> {
        hands
            .iter()
            .map(|h| {
                let c = h.to_counts();
                let d = &h.dice;
                let mut s = [0i32; NUM_CATS];

                // Upper section
                for i in 0..6 {
                    s[i] = c[i] as i32 * (i as i32 + 1);
                }

                // Pair - highest pair
                for i in (0..6).rev() {
                    if c[i] >= 2 {
                        s[6] = 2 * (i as i32 + 1);
                        break;
                    }
                }

                // Two pairs
                let pairs: Vec<i32> = (0..6)
                    .filter(|&i| c[i] >= 2)
                    .map(|i| i as i32 + 1)
                    .collect();
                if pairs.len() >= 2 {
                    s[7] = 2 * (pairs[pairs.len() - 1] + pairs[pairs.len() - 2]);
                }

                // Three of a kind
                for i in (0..6).rev() {
                    if c[i] >= 3 {
                        s[8] = 3 * (i as i32 + 1);
                        break;
                    }
                }

                // Four of a kind
                for i in (0..6).rev() {
                    if c[i] >= 4 {
                        s[9] = 4 * (i as i32 + 1);
                        break;
                    }
                }

                // Small straight (1-2-3-4-5)
                if d == &[1, 2, 3, 4, 5] {
                    s[10] = 15;
                }

                // Large straight (2-3-4-5-6)
                if d == &[2, 3, 4, 5, 6] {
                    s[11] = 20;
                }

                // House (3+2 of different faces)
                let has3 = c.iter().any(|&x| x == 3);
                let has2 = c.iter().any(|&x| x == 2);
                if has3 && has2 {
                    s[12] = d.iter().map(|&x| x as i32).sum();
                }

                // Chance
                s[13] = d.iter().map(|&x| x as i32).sum();

                // Yatzy
                if c.iter().any(|&x| x == 5) {
                    s[14] = 50;
                }

                s
            })
            .collect()
    }
}

// ============================================================================
// ScoreCard state (Java convention)
// ============================================================================

/// ScoreCard follows Java convention exactly:
/// - filled: bit (14-cat) = 1 means category `cat` is filled
/// - upper_total: 0..63, capped
/// - index: (upper << 15) | filled
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct ScoreCard {
    filled: u16,
    upper_total: i32,
}

impl ScoreCard {
    fn new() -> Self {
        ScoreCard {
            filled: 0,
            upper_total: 0,
        }
    }

    fn with_filled_and_upper(filled: u16, upper: i32) -> Self {
        ScoreCard {
            filled,
            upper_total: upper.min(63),
        }
    }

    fn is_filled(&self, cat: usize) -> bool {
        (self.filled & (1 << (14 - cat))) != 0
    }

    fn fill(&mut self, cat: usize) {
        self.filled |= 1 << (14 - cat);
    }

    fn add_upper(&mut self, score: i32) {
        self.upper_total = (self.upper_total + score).min(63);
    }

    fn index(&self) -> u32 {
        ((self.upper_total as u32 & 0x3F) << 15) | (self.filled as u32 & 0x7FFF)
    }

    /// Score for hand in category, including upper bonus if applicable.
    fn value(&self, hand_score: i32, cat: usize) -> i32 {
        let mut score = hand_score;
        if cat < 6 && self.upper_total < 63 && self.upper_total + hand_score >= 63 {
            score += 50;
        }
        score
    }
}

// ============================================================================
// DP Oracle
// ============================================================================

pub struct YatzyDP {
    tables: Tables,
    // expected_scores[scorecard.index()] = expected remaining score
    expected_scores: FxHashMap<u32, f64>,
}

impl YatzyDP {
    pub fn new() -> Self {
        let tables = Tables::new();
        let expected_scores = FxHashMap::default();

        let mut dp = YatzyDP {
            tables,
            expected_scores,
        };
        dp.generate();
        dp
    }

    pub fn expected_score(&self) -> f64 {
        self.expected_scores
            .get(&ScoreCard::new().index())
            .copied()
            .unwrap_or(0.0)
    }

    /// Low-level oracle API. See module docs for full spec.
    pub fn best_action(
        &self,
        avail_mask: u16,
        upper_total: u8,
        dice: [u8; 5],
        rerolls_left: u8,
    ) -> (Action, f64) {
        self.best_action_typed(avail_mask, upper_total as i32, dice, rerolls_left)
    }

    fn generate(&mut self) {
        self.generate_base_cases();
        for filled_count in (0..=13).rev() {
            self.generate_step(filled_count);
        }
    }

    fn generate_base_cases(&mut self) {
        let mut work: Vec<(ScoreCard, usize)> = Vec::new();

        for cat in 0..NUM_CATS {
            for upper in 0..=63 {
                let mut filled: u16 = 0;
                for c in 0..NUM_CATS {
                    if c != cat {
                        filled |= 1 << (14 - c);
                    }
                }
                let sc = ScoreCard::with_filled_and_upper(filled, upper);
                work.push((sc, cat));
            }
        }

        let results: Vec<(u32, f64)> = work
            .par_iter()
            .map(|(sc, cat)| {
                let ev = Self::compute_round_static(&self.tables, sc, *cat);
                (sc.index(), ev)
            })
            .collect();

        for (idx, ev) in results {
            self.expected_scores.insert(idx, ev);
        }
    }

    fn generate_step(&mut self, filled_count: usize) {
        let ways = all_ways_to_fill(filled_count, NUM_CATS);
        let mut work: Vec<ScoreCard> = Vec::new();

        for way in ways {
            for upper in 0..=63 {
                let mut filled: u16 = 0;
                for (cat, &is_filled) in way.iter().enumerate() {
                    if is_filled {
                        filled |= 1 << (14 - cat);
                    }
                }
                let sc = ScoreCard::with_filled_and_upper(filled, upper);
                work.push(sc);
            }
        }

        let expected_scores = &self.expected_scores;
        let results: Vec<(u32, f64)> = work
            .par_iter()
            .map(|sc| {
                let ev = Self::compute_recursive_round_static(&self.tables, expected_scores, sc);
                (sc.index(), ev)
            })
            .collect();

        for (idx, ev) in results {
            self.expected_scores.insert(idx, ev);
        }
    }

    fn compute_round_static(tables: &Tables, sc: &ScoreCard, cat: usize) -> f64 {
        let mut working_vals: [Vec<f64>; 2] = [vec![0.0; MAX_HAND + 1], vec![0.0; MAX_HAND + 1]];

        // Roll 3: must mark
        for hand_idx in 1..=MAX_HAND {
            let raw_score = tables.scores[hand_idx][cat];
            let score = sc.value(raw_score, cat) as f64;
            working_vals[1][hand_idx] = score;
        }

        // Rolls 2, 1, 0: keeper DP
        for roll in (0..=2).rev() {
            let mut k = vec![0.0f64; MAX_KEEPER];
            for hand_idx in 1..=MAX_HAND {
                let kid = tables.hand_to_keeper[hand_idx];
                k[kid] = working_vals[1][hand_idx];
            }

            for held in (0..=4).rev() {
                for &kid in &tables.keepers_by_count[held] {
                    let mut sum = 0.0;
                    for face in 0..6 {
                        let next_kid = tables.keeper_add[kid][face];
                        sum += k[next_kid];
                    }
                    k[kid] = sum / 6.0;
                }
            }

            if roll == 0 {
                return k[0];
            }

            for hand_idx in 1..=MAX_HAND {
                let mut best = 0.0f64;
                for mask in 0..32 {
                    let kid = tables.hand_keeper_map[hand_idx][mask];
                    best = best.max(k[kid]);
                }
                working_vals[0][hand_idx] = best;
            }

            working_vals.swap(0, 1);
        }
        0.0
    }

    fn compute_recursive_round_static(
        tables: &Tables,
        expected_scores: &FxHashMap<u32, f64>,
        sc: &ScoreCard,
    ) -> f64 {
        let mut working_vals: [Vec<f64>; 2] = [vec![0.0; MAX_HAND + 1], vec![0.0; MAX_HAND + 1]];

        // Roll 3: choose best category to mark
        for hand_idx in 1..=MAX_HAND {
            let mut best = 0.0f64;

            for cat in 0..NUM_CATS {
                if sc.is_filled(cat) {
                    continue;
                }

                let raw_score = tables.scores[hand_idx][cat];
                let score = sc.value(raw_score, cat) as f64;

                let mut next_sc = *sc;
                next_sc.fill(cat);
                if cat < 6 {
                    next_sc.add_upper(raw_score);
                }

                let future = expected_scores
                    .get(&next_sc.index())
                    .copied()
                    .unwrap_or(0.0);
                let total = score + future;

                if total > best {
                    best = total;
                }
            }

            working_vals[1][hand_idx] = best;
        }

        // Rolls 2, 1, 0: keeper DP
        for roll in (0..=2).rev() {
            let mut k = vec![0.0f64; MAX_KEEPER];
            for hand_idx in 1..=MAX_HAND {
                let kid = tables.hand_to_keeper[hand_idx];
                k[kid] = working_vals[1][hand_idx];
            }

            for held in (0..=4).rev() {
                for &kid in &tables.keepers_by_count[held] {
                    let mut sum = 0.0;
                    for face in 0..6 {
                        let next_kid = tables.keeper_add[kid][face];
                        sum += k[next_kid];
                    }
                    k[kid] = sum / 6.0;
                }
            }

            if roll == 0 {
                return k[0];
            }

            for hand_idx in 1..=MAX_HAND {
                let mut best = 0.0f64;
                for mask in 0..32 {
                    let kid = tables.hand_keeper_map[hand_idx][mask];
                    best = best.max(k[kid]);
                }
                working_vals[0][hand_idx] = best;
            }

            working_vals.swap(0, 1);
        }

        0.0
    }

    fn best_action_typed(
        &self,
        avail_mask: u16,
        upper: i32,
        mut dice: [u8; 5],
        rerolls: u8,
    ) -> (Action, f64) {
        dice.sort();
        let hand = Hand::new(dice[0], dice[1], dice[2], dice[3], dice[4]);
        let hand_idx = self.tables.hand_to_idx[&hand];

        // filled = FULL_MASK & !avail_mask (both use bit (14-cat))
        let filled: u16 = FULL_MASK & !avail_mask;
        let sc = ScoreCard::with_filled_and_upper(filled, upper.min(63));

        if rerolls == 0 {
            // Must mark - find best category
            let mut best_cat = 0usize;
            let mut best_ev = f64::NEG_INFINITY;

            for cat in 0..NUM_CATS {
                if sc.is_filled(cat) {
                    continue;
                }
                let raw_score = self.tables.scores[hand_idx][cat];
                let score = sc.value(raw_score, cat) as f64;

                let mut next_sc = sc;
                next_sc.fill(cat);
                if cat < 6 {
                    next_sc.add_upper(raw_score);
                }

                let future = self
                    .expected_scores
                    .get(&next_sc.index())
                    .copied()
                    .unwrap_or(0.0);
                let total = score + future;

                if total > best_ev {
                    best_ev = total;
                    best_cat = cat;
                }
            }

            (
                Action::Mark {
                    cat: best_cat as u8,
                },
                best_ev,
            )
        } else {
            // Can reroll - compute K array via keeper DP (mirrors recursion logic)
            let mut k = vec![0.0f64; MAX_KEEPER];

            // Initialize K for all 5-dice states with marking decisions
            for h_idx in 1..=MAX_HAND {
                let mut best = 0.0f64;
                for cat in 0..NUM_CATS {
                    if sc.is_filled(cat) {
                        continue;
                    }

                    let raw_score = self.tables.scores[h_idx][cat];
                    let score = sc.value(raw_score, cat) as f64;

                    let mut next_sc = sc;
                    next_sc.fill(cat);
                    if cat < 6 {
                        next_sc.add_upper(raw_score);
                    }

                    let future = self
                        .expected_scores
                        .get(&next_sc.index())
                        .copied()
                        .unwrap_or(0.0);
                    best = best.max(score + future);
                }
                let kid = self.tables.hand_to_keeper[h_idx];
                k[kid] = best;
            }

            // DP backwards through remaining rerolls
            for roll_iter in 0..rerolls {
                for held in (0..=4).rev() {
                    for &kid in &self.tables.keepers_by_count[held] {
                        let mut sum = 0.0;
                        for face in 0..6 {
                            let next_kid = self.tables.keeper_add[kid][face];
                            sum += k[next_kid];
                        }
                        k[kid] = sum / 6.0;
                    }
                }

                // For intermediate rolls (not the last), update 5-dice keepers with BEST values
                if roll_iter < rerolls - 1 {
                    for h_idx in 1..=MAX_HAND {
                        let mut best_k = 0.0f64;
                        for mask in 0..32 {
                            let kid = self.tables.hand_keeper_map[h_idx][mask];
                            best_k = best_k.max(k[kid]);
                        }
                        let kid = self.tables.hand_to_keeper[h_idx];
                        k[kid] = best_k;
                    }
                }
            }

            // Find best keep mask for the current hand
            let mut best_mask = 0u8;
            let mut best_ev = 0.0f64;
            for mask in 0usize..32 {
                let kid = self.tables.hand_keeper_map[hand_idx][mask];
                if k[kid] > best_ev {
                    best_ev = k[kid];
                    best_mask = mask as u8;
                }
            }

            // If best is hold-all, mark immediately (equivalent)
            if best_mask == 0b1_1111 {
                let mut best_cat = 0usize;
                let mut best_mark_ev = f64::NEG_INFINITY;
                for cat in 0..NUM_CATS {
                    if sc.is_filled(cat) {
                        continue;
                    }
                    let raw_score = self.tables.scores[hand_idx][cat];
                    let score = sc.value(raw_score, cat) as f64;
                    let mut next_sc = sc;
                    next_sc.fill(cat);
                    if cat < 6 {
                        next_sc.add_upper(raw_score);
                    }
                    let future = self
                        .expected_scores
                        .get(&next_sc.index())
                        .copied()
                        .unwrap_or(0.0);
                    let total = score + future;
                    if total > best_mark_ev {
                        best_mark_ev = total;
                        best_cat = cat;
                    }
                }
                (
                    Action::Mark {
                        cat: best_cat as u8,
                    },
                    best_mark_ev,
                )
            } else {
                (Action::KeepMask { mask: best_mask }, best_ev)
            }
        }
    }
}

// ============================================================================
// Combinatorics helper (match Java Utils.allWaysToPut semantics)
// ============================================================================

/// Generate all ways to place `count` true values in `size` positions.
fn all_ways_to_fill(count: usize, size: usize) -> Vec<Vec<bool>> {
    if count == 0 {
        return vec![vec![false; size]];
    }
    if count == size {
        return vec![vec![true; size]];
    }

    let mut result = Vec::new();

    fn generate(
        pos: usize,
        remaining: usize,
        size: usize,
        current: &mut Vec<bool>,
        result: &mut Vec<Vec<bool>>,
    ) {
        if pos == size {
            if remaining == 0 {
                result.push(current.clone());
            }
            return;
        }

        // Don't place here
        current.push(false);
        generate(pos + 1, remaining, size, current, result);
        current.pop();

        // Place here
        if remaining > 0 {
            current.push(true);
            generate(pos + 1, remaining - 1, size, current, result);
            current.pop();
        }
    }

    let mut current = Vec::new();
    generate(0, count, size, &mut current, &mut result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn dp() -> &'static YatzyDP {
        static DP: OnceLock<YatzyDP> = OnceLock::new();
        DP.get_or_init(YatzyDP::new)
    }

    #[test]
    fn state_mask_convention_smoke() {
        // avail_mask uses bit (14-cat)
        let all_avail = FULL_MASK;
        assert_eq!(all_avail & (1 << (14 - 0)), 1 << 14); // ones available
        assert_eq!(all_avail & (1 << (14 - 14)), 1); // yatzy available
    }

    #[test]
    fn expected_score_regression() {
        let ev = dp().expected_score();
        // Literature ~248.63; our port ~248.44
        assert!((ev - 248.44).abs() < 1.0, "ev={}", ev);
    }

    #[test]
    fn best_action_api_smoke() {
        let dice = [1, 1, 1, 1, 2];
        let (a, _ev) = dp().best_action(FULL_MASK, 0, dice, 2);
        match a {
            Action::KeepMask { .. } | Action::Mark { .. } => {}
        }
        let (a2, _ev2) = dp().best_action(FULL_MASK, 0, dice, 0);
        matches!(a2, Action::Mark { .. });
    }
}
