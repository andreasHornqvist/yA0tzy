#[cfg(test)]
mod tests {
    use crate::{outcome_to_z, GameState, PlayerState};

    #[test]
    fn swap_players_definition_matches_prd() {
        let s = GameState {
            players: [
                PlayerState {
                    avail_mask: 0x7FFF,
                    upper_total_cap: 10,
                    total_score: 50,
                },
                PlayerState {
                    avail_mask: 0x1234,
                    upper_total_cap: 20,
                    total_score: 60,
                },
            ],
            dice_sorted: [1, 1, 3, 4, 6],
            rerolls_left: 1,
            player_to_move: 0,
        };

        let t = s.swap_players();
        assert_eq!(t.players[0], s.players[1]);
        assert_eq!(t.players[1], s.players[0]);
        assert_eq!(t.dice_sorted, s.dice_sorted);
        assert_eq!(t.rerolls_left, s.rerolls_left);
        assert_eq!(t.player_to_move, 1);
    }

    #[test]
    fn outcome_to_z_antisymmetry_under_swap() {
        // If player 0 is winner, then swapping players should flip POV.
        // z(winner=0, pov=0)=+1 and z(winner=0, pov=1)=-1.
        assert_eq!(outcome_to_z(0, 0), 1.0);
        assert_eq!(outcome_to_z(0, 1), -1.0);

        // Similarly for winner=1.
        assert_eq!(outcome_to_z(1, 0), -1.0);
        assert_eq!(outcome_to_z(1, 1), 1.0);

        // Draw is always 0.
        assert_eq!(outcome_to_z(2, 0), 0.0);
        assert_eq!(outcome_to_z(2, 1), 0.0);
    }
}
