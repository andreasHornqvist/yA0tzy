//! Canonical game state definitions (PRD §5.3) and POV helpers (PRD Epic E3).

/// Per-player state (oracle-aligned fields + totals for AZ).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerState {
    /// Category availability mask (oracle bit convention: bit (14-cat) = 1 if available).
    pub avail_mask: u16,
    /// Upper total, clamped to 63.
    pub upper_total_cap: u8,
    /// Total score so far (for win/loss/margin).
    pub total_score: i16,
}

/// Canonical 1v1 game state (PRD §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameState {
    pub players: [PlayerState; 2],
    /// Dice are always sorted ascending.
    pub dice_sorted: [u8; 5],
    pub rerolls_left: u8,
    pub player_to_move: u8, // 0 or 1
}

impl GameState {
    /// Swap players as defined in PRD Epic E3 (antisymmetry tests).
    ///
    /// - swap per-player boards/totals
    /// - flip player_to_move (0↔1)
    /// - keep turn state (dice, rerolls_left) identical
    pub fn swap_players(&self) -> GameState {
        GameState {
            players: [self.players[1], self.players[0]],
            dice_sorted: self.dice_sorted,
            rerolls_left: self.rerolls_left,
            player_to_move: 1u8.saturating_sub(self.player_to_move),
        }
    }
}

/// Convert a game outcome into a value target `z` from the POV of `pov_player`.
///
/// Convention:
/// - `winner_player` in {0,1} -> +1 for winner POV, -1 for loser POV
/// - `winner_player == 2` -> draw -> 0
///
/// This is a small helper to lock in the sign convention before MCTS is implemented.
pub fn outcome_to_z(winner_player: u8, pov_player: u8) -> f32 {
    assert!(pov_player <= 1, "pov_player must be 0 or 1");
    match winner_player {
        0 | 1 => {
            if winner_player == pov_player {
                1.0
            } else {
                -1.0
            }
        }
        2 => 0.0,
        _ => panic!("winner_player must be 0,1, or 2 (draw)"),
    }
}
