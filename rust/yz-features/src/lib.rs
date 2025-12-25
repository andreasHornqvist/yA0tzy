//! yz-features: Feature schema + canonical encoding.

pub mod encode;
pub mod schema;

pub use encode::{encode_state_v1, GameStateView, PlayerView};
pub use schema::{F, FEATURE_SCHEMA_ID, SCORE_NORM};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nonempty() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn swap_players_encoding_is_consistent() {
        let s = GameStateView {
            players: [
                PlayerView {
                    avail_mask: 0x7FFF,
                    upper_total_cap: 10,
                    total_score: 50,
                },
                PlayerView {
                    avail_mask: 0x7FFF & !(1u16 << 14), // ones filled for opp
                    upper_total_cap: 20,
                    total_score: 60,
                },
            ],
            dice_sorted: [1, 1, 3, 4, 6],
            rerolls_left: 1,
            player_to_move: 0,
        };

        // Swapping players and flipping player_to_move should yield identical encoding.
        let e1 = encode_state_v1(&s);
        let e2 = encode_state_v1(&s.swap_players());
        assert_eq!(e1, e2);
    }
}
