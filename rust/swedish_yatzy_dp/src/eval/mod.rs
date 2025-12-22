//! Evaluation helpers (simulation + histogram) for the oracle.

use std::collections::HashMap;

use crate::game::{FULL_MASK, NUM_CATS};
use crate::oracle::{Action, YatzyDP};

/// SplitMix64 RNG (fast, good statistical quality for simulation).
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn roll_die(&mut self) -> u8 {
        ((self.next_u64() % 6) + 1) as u8
    }

    pub fn roll_5(&mut self) -> [u8; 5] {
        [
            self.roll_die(),
            self.roll_die(),
            self.roll_die(),
            self.roll_die(),
            self.roll_die(),
        ]
    }
}

pub struct SimulationResult {
    pub scores: Vec<i32>,
    pub bonus_count: usize,
    pub upper_totals: Vec<i32>,
}

pub struct ScoreSummary {
    pub mean: f64,
    pub median: i32,
    pub std_dev: f64,
    pub min: i32,
    pub max: i32,
}

pub fn summarize_scores(scores: &[i32]) -> ScoreSummary {
    // Single pass for min/max/mean/std + a frequency table for exact median.
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    let mut sum = 0f64;
    let mut sum_sq = 0f64;

    for &s in scores {
        min = min.min(s);
        max = max.max(s);
        let sf = s as f64;
        sum += sf;
        sum_sq += sf * sf;
    }

    let n = scores.len() as f64;
    let mean = sum / n;
    let var = (sum_sq / n) - mean * mean;
    let std_dev = var.max(0.0).sqrt();

    // Median via frequency table over [min..max]
    let range = (max - min) as usize + 1;
    let mut freq = vec![0usize; range];
    for &s in scores {
        freq[(s - min) as usize] += 1;
    }
    let target = scores.len() / 2;
    let mut cum = 0usize;
    let mut median = min;
    for (i, &c) in freq.iter().enumerate() {
        cum += c;
        if cum > target {
            median = min + i as i32;
            break;
        }
    }

    ScoreSummary {
        mean,
        median,
        std_dev,
        min,
        max,
    }
}

pub fn simulate_games(oracle: &YatzyDP, n: usize, seed: u64) -> SimulationResult {
    let mut rng = Rng::new(seed);
    let mut scores = Vec::with_capacity(n);
    let mut upper_totals = Vec::with_capacity(n);
    let mut bonus_count = 0usize;

    for _ in 0..n {
        let (score, got_bonus, upper) = simulate_game(oracle, &mut rng);
        scores.push(score);
        upper_totals.push(upper);
        if got_bonus {
            bonus_count += 1;
        }
    }

    SimulationResult {
        scores,
        bonus_count,
        upper_totals,
    }
}

/// Simulate a single game using oracle actions.
pub fn simulate_game(oracle: &YatzyDP, rng: &mut Rng) -> (i32, bool, i32) {
    let mut filled: u16 = 0;
    let mut upper_total: i32 = 0;
    let mut total_score: i32 = 0;
    let mut got_bonus = false;

    for _round in 0..NUM_CATS {
        let mut dice = rng.roll_5();
        dice.sort();
        let mut rerolls_left: u8 = 2;

        loop {
            let avail_mask = FULL_MASK & !filled;
            let (action, _ev) =
                oracle.best_action(avail_mask, upper_total.min(63) as u8, dice, rerolls_left);

            match action {
                Action::Mark { cat } => {
                    let cat_idx = cat as usize;
                    // We need raw category score; oracle internal tables know it, but we don’t expose them.
                    // Recompute raw score locally by asking oracle again with rerolls_left=0 (mark) is not OK.
                    // Instead we rely on the fact that oracle’s EV correctness is already validated and we just
                    // need the realized score. For that, we compute raw score via a small local scorer.
                    let raw_score = crate::game::scores_for_dice(dice)[cat_idx];

                    let mut score = raw_score;
                    if cat_idx < 6 {
                        let new_upper = upper_total + raw_score;
                        if upper_total < 63 && new_upper >= 63 {
                            score += 50;
                            got_bonus = true;
                        }
                        upper_total = new_upper;
                    }

                    total_score += score;
                    filled |= 1 << (14 - cat_idx);
                    break;
                }
                Action::KeepMask { mask } => {
                    if rerolls_left == 0 {
                        // Force mark (should not happen)
                        let (a2, _ev2) = oracle.best_action(
                            FULL_MASK & !filled,
                            upper_total.min(63) as u8,
                            dice,
                            0,
                        );
                        if let Action::Mark { cat } = a2 {
                            let cat_idx = cat as usize;
                            let raw_score = crate::game::scores_for_dice(dice)[cat_idx];
                            let mut score = raw_score;
                            if cat_idx < 6 {
                                let new_upper = upper_total + raw_score;
                                if upper_total < 63 && new_upper >= 63 {
                                    score += 50;
                                    got_bonus = true;
                                }
                                upper_total = new_upper;
                            }
                            total_score += score;
                            filled |= 1 << (14 - cat_idx);
                        }
                        break;
                    }

                    let mut next = [0u8; 5];
                    for i in 0..5 {
                        if (mask & (1 << (4 - i))) != 0 {
                            next[i] = dice[i];
                        } else {
                            next[i] = rng.roll_die();
                        }
                    }
                    next.sort();
                    dice = next;
                    rerolls_left -= 1;
                }
            }
        }
    }

    (total_score, got_bonus, upper_total)
}

/// Print a histogram of scores (bucket size = 10).
pub fn print_histogram(scores: &[i32]) {
    let min_score = *scores.iter().min().unwrap();
    let max_score = *scores.iter().max().unwrap();

    let bucket_size = 10;
    let min_bucket = (min_score / bucket_size) * bucket_size;
    // Ensure we include the bucket containing max_score (avoid printing an extra empty bucket).
    let max_bucket = (max_score / bucket_size) * bucket_size;

    let mut buckets: HashMap<i32, usize> = HashMap::new();
    for &score in scores {
        let bucket = (score / bucket_size) * bucket_size;
        *buckets.entry(bucket).or_insert(0) += 1;
    }

    let max_count = *buckets.values().max().unwrap_or(&1);
    let bar_width = 50usize;

    println!("\nScore histogram (N={}, bin=10):", scores.len());
    println!("{}", "─".repeat(70));

    let mut bucket = min_bucket;
    while bucket <= max_bucket {
        let count = *buckets.get(&bucket).unwrap_or(&0);
        let bar_len = (count * bar_width) / max_count.max(1);
        let bar: String = "█".repeat(bar_len);

        println!(
            "{:3}-{:3} │{:<50} {:4} ({:.1}%)",
            bucket,
            bucket + bucket_size - 1,
            bar,
            count,
            (count as f64 / scores.len() as f64) * 100.0
        );

        bucket += bucket_size;
    }

    println!("{}", "─".repeat(70));

    let s = summarize_scores(scores);
    println!(
        "\nSummary: mean={:.2}, median={}, std={:.2}, min={}, max={}",
        s.mean, s.median, s.std_dev, s.min, s.max
    );
}
