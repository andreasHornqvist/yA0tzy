use crate::action::{index_to_action, Action, A};
use crate::engine::{
    apply_action, initial_state, is_terminal, terminal_winner, terminal_z_from_player_to_move,
    ApplyError, TurnContext,
};
use crate::legal::legal_action_mask;

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

fn assert_invariants(s: &crate::GameState) {
    assert!(s.player_to_move <= 1);
    assert!(s.rerolls_left <= 2);
    assert!(s.dice_sorted.windows(2).all(|w| w[0] <= w[1]));
    for &d in &s.dice_sorted {
        assert!((1..=6).contains(&d));
    }
}

#[test]
fn legality_enforcement_basic() {
    let mut ctx = TurnContext::new_deterministic(123);
    let mut s = initial_state(&mut ctx);

    // rerolls_left == 0 => KeepMask illegal
    s.rerolls_left = 0;
    let err = apply_action(s, Action::KeepMask(0), &mut ctx).unwrap_err();
    assert!(matches!(err, ApplyError::IllegalAction { .. }));

    // rerolls_left > 0 => Mark illegal (mark-only-at-roll-3 rule)
    let s2 = initial_state(&mut ctx);
    assert!(s2.rerolls_left > 0);
    let err = apply_action(s2, Action::Mark(0), &mut ctx).unwrap_err();
    assert!(matches!(err, ApplyError::IllegalAction { .. }));

    // rerolls_left > 0 => KeepMask(31) is now legal (keep-all)
    let s3 = initial_state(&mut ctx);
    let result = apply_action(s3, Action::KeepMask(31), &mut ctx);
    assert!(result.is_ok(), "KeepMask(31) should be legal at rerolls>0");
    let s3_next = result.unwrap();
    assert_eq!(s3_next.rerolls_left, 1); // rerolls decremented

    // Marking an unavailable category is illegal at rerolls_left == 0.
    let mut s4 = initial_state(&mut ctx);
    s4.rerolls_left = 0; // must be at roll 3 to mark
    // Make cat 0 unavailable for current player.
    s4.players[s4.player_to_move as usize].avail_mask &= !crate::avail_bit_for_cat(0);
    let err = apply_action(s4, Action::Mark(0), &mut ctx).unwrap_err();
    assert!(matches!(err, ApplyError::IllegalAction { .. }));
}

#[test]
fn keepmask_31_keeps_dice_unchanged() {
    // KeepMask(31) should advance to next roll stage without changing dice.
    let mut ctx = TurnContext::new_deterministic(42);
    let s = crate::GameState {
        players: [
            crate::PlayerState {
                avail_mask: crate::FULL_MASK,
                upper_total_cap: 0,
                total_score: 0,
            },
            crate::PlayerState {
                avail_mask: crate::FULL_MASK,
                upper_total_cap: 0,
                total_score: 0,
            },
        ],
        dice_sorted: [1, 2, 3, 4, 5],
        rerolls_left: 2,
        player_to_move: 0,
    };

    let next = apply_action(s, Action::KeepMask(31), &mut ctx).unwrap();
    assert_eq!(next.dice_sorted, [1, 2, 3, 4, 5], "dice should not change on keep-all");
    assert_eq!(next.rerolls_left, 1);
    assert_eq!(next.player_to_move, 0);
}

#[test]
fn deterministic_reproducibility_same_seed_same_actions() {
    let episode_seed = 999u64;

    // Under mark-only-at-roll-3 rules: must do KeepMask to reach rerolls=0 before Mark.
    let actions = [
        Action::KeepMask(0),  // rerolls 2→1
        Action::KeepMask(0),  // rerolls 1→0
        Action::Mark(0),      // mark (resets to rerolls=2 for next player)
        Action::KeepMask(0),  // rerolls 2→1 (new player's turn)
        Action::KeepMask(0),  // rerolls 1→0
        Action::Mark(1),      // mark (legal at rerolls=0)
    ];

    let mut ctx1 = TurnContext::new_deterministic(episode_seed);
    let mut s1 = initial_state(&mut ctx1);
    for &a in &actions {
        s1 = apply_action(s1, a, &mut ctx1).unwrap();
    }

    let mut ctx2 = TurnContext::new_deterministic(episode_seed);
    let mut s2 = initial_state(&mut ctx2);
    for &a in &actions {
        s2 = apply_action(s2, a, &mut ctx2).unwrap();
    }

    assert_eq!(s1, s2);
}

#[test]
fn golden_keepmask_transition_deterministic_hardcoded() {
    // Golden: lock in deterministic event-key mapping at the engine layer.
    //
    // Setup chosen so round_idx=0 (FULL_MASK) and this KeepMask consumes roll_idx=1
    // (first reroll from rerolls_left=2).
    let episode_seed = 123_456_789u64;
    let mut ctx = TurnContext::new_deterministic(episode_seed);

    let s = crate::GameState {
        players: [
            crate::PlayerState {
                avail_mask: crate::FULL_MASK,
                upper_total_cap: 0,
                total_score: 0,
            },
            crate::PlayerState {
                avail_mask: crate::FULL_MASK,
                upper_total_cap: 0,
                total_score: 0,
            },
        ],
        dice_sorted: [1, 1, 1, 1, 1],
        rerolls_left: 2,
        player_to_move: 0,
    };

    let next = apply_action(s, Action::KeepMask(0), &mut ctx).unwrap();
    assert_eq!(next.rerolls_left, 1);
    assert_invariants(&next);

    // Golden expected dice for EventKey{episode_seed=123456789, player=0, round_idx=0, roll_idx=1}.
    assert_eq!(next.dice_sorted, [2, 3, 5, 5, 6]);
}

#[test]
fn random_playout_terminates_in_30_marks_deterministic_mode() {
    let mut ctx = TurnContext::new_deterministic(1234);
    let mut s = initial_state(&mut ctx);

    let mut chooser = ChaCha8Rng::seed_from_u64(7);
    let mut marks = 0usize;

    for _step in 0..10_000 {
        assert_invariants(&s);
        if is_terminal(&s) {
            break;
        }

        let p = s.player_to_move as usize;
        let legal = legal_action_mask(s.players[p].avail_mask, s.rerolls_left);
        let legal_idxs: Vec<usize> = (0..A).filter(|&i| ((legal >> i) & 1) != 0).collect();
        assert!(!legal_idxs.is_empty());

        let pick = chooser.gen_range(0..legal_idxs.len());
        let idx = legal_idxs[pick] as u8;
        let a = index_to_action(idx);
        if matches!(a, Action::Mark(_)) {
            marks += 1;
        }
        s = apply_action(s, a, &mut ctx).unwrap();
    }

    assert!(is_terminal(&s), "playout did not terminate");
    assert_eq!(marks, 30);
}

#[test]
fn random_playout_terminates_in_30_marks_rng_mode() {
    // Note: RNG mode is still deterministic in tests by seeding both the game RNG and the chooser RNG.
    let mut ctx = TurnContext::new_rng(1234);
    let mut s = initial_state(&mut ctx);

    let mut chooser = ChaCha8Rng::seed_from_u64(7);
    let mut marks = 0usize;

    for _step in 0..10_000 {
        assert_invariants(&s);
        if is_terminal(&s) {
            break;
        }

        let p = s.player_to_move as usize;
        let legal = legal_action_mask(s.players[p].avail_mask, s.rerolls_left);
        let legal_idxs: Vec<usize> = (0..A).filter(|&i| ((legal >> i) & 1) != 0).collect();
        assert!(!legal_idxs.is_empty());

        let pick = chooser.gen_range(0..legal_idxs.len());
        let idx = legal_idxs[pick] as u8;
        let a = index_to_action(idx);
        if matches!(a, Action::Mark(_)) {
            marks += 1;
        }
        s = apply_action(s, a, &mut ctx).unwrap();
    }

    assert!(is_terminal(&s), "playout did not terminate");
    assert_eq!(marks, 30);
}

#[test]
fn golden_mark_transition_bonus_clamp_and_turn_reset() {
    // Golden: mark transition must apply bonus/clamp and advance to next player's fresh turn dice.
    let episode_seed = 123_456_789u64;
    let mut ctx = TurnContext::new_deterministic(episode_seed);

    // Player 0 is about to cross the upper bonus boundary by marking sixes with five 6s.
    let s = crate::GameState {
        players: [
            crate::PlayerState {
                avail_mask: crate::FULL_MASK,
                upper_total_cap: 60,
                total_score: 100,
            },
            crate::PlayerState {
                avail_mask: crate::FULL_MASK,
                upper_total_cap: 0,
                total_score: 0,
            },
        ],
        dice_sorted: [6, 6, 6, 6, 6],
        rerolls_left: 0,
        player_to_move: 0,
    };

    let next = apply_action(s, Action::Mark(5), &mut ctx).unwrap();
    assert_invariants(&next);

    // p0 update: cat 5 filled, +80 delta (30 raw + 50 bonus), upper_total_cap clamped to 63.
    let expected_p0_avail = crate::FULL_MASK & !crate::avail_bit_for_cat(5);
    assert_eq!(next.players[0].avail_mask, expected_p0_avail);
    assert_eq!(next.players[0].upper_total_cap, 63);
    assert_eq!(next.players[0].total_score, 180);

    // Turn reset for p1
    assert_eq!(next.player_to_move, 1);
    assert_eq!(next.rerolls_left, 2);

    // Golden expected dice for EventKey{episode_seed=123456789, player=1, round_idx=0, roll_idx=0}.
    assert_eq!(next.dice_sorted, [2, 5, 5, 6, 6]);
}

fn terminal_state(scores: (i16, i16), player_to_move: u8) -> crate::GameState {
    crate::GameState {
        players: [
            crate::PlayerState {
                avail_mask: 0,
                upper_total_cap: 0,
                total_score: scores.0,
            },
            crate::PlayerState {
                avail_mask: 0,
                upper_total_cap: 0,
                total_score: scores.1,
            },
        ],
        dice_sorted: [1, 1, 1, 1, 1],
        rerolls_left: 0,
        player_to_move,
    }
}

#[test]
fn terminal_winner_and_draw_cases() {
    // p0 wins
    let s = terminal_state((10, 9), 0);
    assert_eq!(terminal_winner(&s).unwrap(), 0);

    // p1 wins
    let s = terminal_state((9, 10), 0);
    assert_eq!(terminal_winner(&s).unwrap(), 1);

    // draw
    let s = terminal_state((10, 10), 0);
    assert_eq!(terminal_winner(&s).unwrap(), 2);
}

#[test]
fn terminal_z_antisymmetry_under_swap_players() {
    // NOTE: `terminal_z_from_player_to_move` returns z from the POV of `player_to_move`.
    // Because `swap_players()` also flips `player_to_move`, the terminal z is invariant under
    // swapping players (we're always measuring from the current player's POV).

    // p0 win
    for player_to_move in [0u8, 1u8] {
        let s = terminal_state((20, 10), player_to_move);
        let z = terminal_z_from_player_to_move(&s).unwrap();
        let z2 = terminal_z_from_player_to_move(&s.swap_players()).unwrap();
        assert_eq!(z, z2);
    }

    // p1 win
    for player_to_move in [0u8, 1u8] {
        let s = terminal_state((10, 20), player_to_move);
        let z = terminal_z_from_player_to_move(&s).unwrap();
        let z2 = terminal_z_from_player_to_move(&s.swap_players()).unwrap();
        assert_eq!(z, z2);
    }

    // draw
    for player_to_move in [0u8, 1u8] {
        let s = terminal_state((10, 10), player_to_move);
        let z = terminal_z_from_player_to_move(&s).unwrap();
        let z2 = terminal_z_from_player_to_move(&s.swap_players()).unwrap();
        assert_eq!(z, 0.0);
        assert_eq!(z2, 0.0);
    }
}

#[test]
fn deterministic_full_game_trajectory_is_reproducible() {
    fn policy(s: &crate::GameState) -> Action {
        if s.rerolls_left > 0 {
            return Action::KeepMask(0);
        }
        // Lowest available category.
        let p = s.player_to_move as usize;
        for cat in 0u8..15 {
            if (s.players[p].avail_mask & crate::avail_bit_for_cat(cat)) != 0 {
                return Action::Mark(cat);
            }
        }
        panic!("no available categories but not terminal?");
    }

    let seed = 2025u64;

    let mut ctx1 = TurnContext::new_deterministic(seed);
    let mut s1 = initial_state(&mut ctx1);
    let mut marks1 = 0usize;
    for _ in 0..10_000 {
        if is_terminal(&s1) {
            break;
        }
        let a = policy(&s1);
        if matches!(a, Action::Mark(_)) {
            marks1 += 1;
        }
        s1 = apply_action(s1, a, &mut ctx1).unwrap();
    }
    assert!(is_terminal(&s1));
    assert_eq!(marks1, 30);

    let mut ctx2 = TurnContext::new_deterministic(seed);
    let mut s2 = initial_state(&mut ctx2);
    let mut marks2 = 0usize;
    for _ in 0..10_000 {
        if is_terminal(&s2) {
            break;
        }
        let a = policy(&s2);
        if matches!(a, Action::Mark(_)) {
            marks2 += 1;
        }
        s2 = apply_action(s2, a, &mut ctx2).unwrap();
    }
    assert!(is_terminal(&s2));
    assert_eq!(marks2, 30);

    assert_eq!(s1, s2);
}
