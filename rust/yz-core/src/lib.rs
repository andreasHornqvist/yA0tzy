//! yz-core: Game rules, scoring, state representation, actions, and configuration.

pub mod action;
#[cfg(test)]
mod action_legal_tests;
pub mod config;
pub mod legal;

pub use action::{
    action_to_index, avail_bit_for_cat, index_to_action, is_mark_index, mark_cat_from_index,
    Action, A, NUM_CATS,
};
pub use config::{Config, ConfigError};
pub use legal::legal_action_mask;

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
