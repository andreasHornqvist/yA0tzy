//! yz-oracle: Adapter to swedish_yatzy_dp oracle + oracle suite tools.
//!
//! This crate re-exports the oracle API and provides helper functions
//! for integration with the yA0tzy training system.

use std::time::Instant;

// Re-export core oracle types
pub use swedish_yatzy_dp::{Action, YatzyDP, CAT_NAMES, FULL_MASK, NUM_CATS};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Result of building the oracle and computing expected score.
#[derive(Debug, Clone)]
pub struct OracleInfo {
    /// Optimal expected score for a fresh game (~248.44).
    pub expected_score: f64,
    /// Time taken to build the DP table.
    pub build_time_secs: f64,
}

/// Build the oracle and return the optimal expected score for a fresh game.
///
/// This builds the full DP table (takes a few seconds) and then returns
/// the expected value for the initial state (all categories available,
/// upper total = 0), averaged over all possible dice rolls.
pub fn get_expected_score() -> OracleInfo {
    let start = Instant::now();
    let oracle = YatzyDP::new();
    let build_time = start.elapsed();

    // Use the oracle's expected_score() method which returns the true
    // expected value for a fresh game (averaged over all possible dice)
    let ev = oracle.expected_score();

    OracleInfo {
        expected_score: ev,
        build_time_secs: build_time.as_secs_f64(),
    }
}

/// Get the oracle singleton (builds on first call).
///
/// Note: The oracle is expensive to build (~3-4 seconds), so this
/// function caches the result. Thread-safe via std::sync::OnceLock.
pub fn oracle() -> &'static YatzyDP {
    use std::sync::OnceLock;
    static ORACLE: OnceLock<YatzyDP> = OnceLock::new();
    ORACLE.get_or_init(YatzyDP::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn expected_score_is_reasonable() {
        let info = get_expected_score();
        // Expected score should be around 248.44 (literature value)
        assert!(
            info.expected_score > 248.0,
            "Expected score too low: {}",
            info.expected_score
        );
        assert!(
            info.expected_score < 249.0,
            "Expected score too high: {}",
            info.expected_score
        );
    }

    #[test]
    fn oracle_returns_valid_action() {
        let oracle = oracle();
        let dice = [1, 2, 3, 4, 5];
        let (action, ev) = oracle.best_action(FULL_MASK, 0, dice, 2);

        // Should return some action
        match action {
            Action::Mark { cat } => assert!(cat < NUM_CATS as u8),
            Action::KeepMask { mask } => assert!(mask < 32),
        }

        // EV should be reasonable
        assert!(ev > 200.0 && ev < 300.0);
    }
}
