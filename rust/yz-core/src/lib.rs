//! yz-core: Game rules, scoring, state representation, actions, and configuration.

pub mod action;
#[cfg(test)]
mod action_legal_tests;
pub mod chance;
#[cfg(test)]
mod chance_tests;
pub mod config;
pub mod engine;
#[cfg(test)]
mod engine_tests;
pub mod legal;
pub mod scoring;
#[cfg(test)]
mod scoring_tests;
pub mod state;
#[cfg(test)]
mod state_tests;

pub use action::{
    action_to_index, avail_bit_for_cat, canonicalize_keepmask, index_to_action, is_mark_index,
    mark_cat_from_index, Action, A, NUM_CATS,
};
pub use chance::{apply_keepmask, roll5, EventKey};
pub use config::{Config, ConfigError};
pub use engine::{
    apply_action, initial_state, is_terminal, terminal_winner, terminal_z_from_player_to_move,
    terminal_z_from_pov, ApplyError, ChanceMode, TurnContext, FULL_MASK,
};
pub use legal::legal_action_mask;
pub use legal::{is_legal as legal_is_legal, to_u8_array as legal_mask_to_u8_array, LegalMask};
pub use scoring::{apply_mark_score, scores_for_dice};
pub use state::{outcome_to_z, GameState, PlayerState};

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
