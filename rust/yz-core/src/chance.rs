//! Deterministic chance stream for eval/gating mode (PRD §6.1).
//!
//! We define dice outcomes by episode seed + structural event, not by evolving RNG state.
//! Event key: (episode_seed, player, round_idx, roll_idx) where roll_idx ∈ {0,1,2}.
//! For each event key, deterministically generate a sequence of 5 die values (1..=6).
//! When rerolling k dice, take the first k values from that event sequence.

/// Structural event key for deterministic dice generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventKey {
    pub episode_seed: u64,
    pub player: u8,
    pub round_idx: u8,
    pub roll_idx: u8,
}

/// SplitMix64 step (fast, deterministic).
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn mix_seed(key: EventKey) -> u64 {
    // Fixed, stable mixing. Avoid std Hash/RandomState.
    let mut x = key.episode_seed;
    x ^= (key.player as u64).wrapping_mul(0xD6E8FEB86659FD93);
    x ^= (key.round_idx as u64).wrapping_mul(0xA5A35625E4F7C1AD);
    x ^= (key.roll_idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
    // Run through SplitMix once for diffusion
    let mut s = x;
    splitmix64_next(&mut s)
}

/// Deterministically generate 5 dice for the given event key.
pub fn roll5(key: EventKey) -> [u8; 5] {
    let mut state = mix_seed(key);
    let mut out = [0u8; 5];
    for o in &mut out {
        let r = splitmix64_next(&mut state);
        *o = ((r % 6) + 1) as u8;
    }
    out
}

/// Apply a KeepMask to a sorted hand using the event-keyed deterministic stream.
///
/// - `prev_sorted` must be sorted (ascending).
/// - `keep_mask` uses oracle/PRD semantics: bit (4-i) refers to prev_sorted[i].
/// - Rerolled dice values are taken from the first k elements of `roll5(key)`, where
///   k is the number of dice being rerolled (number of 0-bits in keep_mask).
/// - Returns a sorted hand.
pub fn apply_keepmask(prev_sorted: [u8; 5], keep_mask: u8, key: EventKey) -> [u8; 5] {
    debug_assert!(prev_sorted.windows(2).all(|w| w[0] <= w[1]));
    assert!(keep_mask < 32, "keep_mask out of range: {}", keep_mask);

    // Determine positions to reroll (mask bit 0).
    let mut reroll_positions = [0usize; 5];
    let mut k = 0usize;
    for i in 0..5 {
        let bit = 1u8 << (4 - i);
        if (keep_mask & bit) == 0 {
            reroll_positions[k] = i;
            k += 1;
        }
    }

    if k == 0 {
        return prev_sorted;
    }

    let draws = roll5(key);
    let mut next = prev_sorted;
    for (j, &pos) in reroll_positions[..k].iter().enumerate() {
        next[pos] = draws[j];
    }
    next.sort();
    next
}
