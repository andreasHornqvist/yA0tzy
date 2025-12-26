//! MCTS implementation (PUCT) for AlphaZero-style search.
//!
//! Epic E4 starts here. The design uses:
//! - Fixed action space `A=47` (yz-core)
//! - Stochastic transitions via `yz_core::apply_action` (dice sampled in self-play RNG mode;
//!   deterministic event-keyed chance in eval mode as configured)
//! - Arena-backed node storage

pub mod arena;
pub mod infer;
pub mod infer_client;
pub mod mcts;
pub mod node;
pub mod state_key;

pub use infer::{Inference, UniformInference};
pub use infer_client::{InferBackend, InferBackendError};
pub use mcts::{
    apply_temperature, ChanceMode, Mcts, MctsConfig, SearchDriver, SearchResult, SearchStats,
};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nonempty() {
        assert!(!VERSION.is_empty());
    }
}

#[cfg(test)]
mod mcts_tests;
