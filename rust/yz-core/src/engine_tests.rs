use crate::action::{index_to_action, Action, A};
use crate::engine::{apply_action, initial_state, is_terminal, ApplyError, TurnContext};
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

    // rerolls_left > 0 => KeepMask(31) illegal (dominated)
    let s2 = initial_state(&mut ctx);
    let err = apply_action(s2, Action::KeepMask(31), &mut ctx).unwrap_err();
    assert!(matches!(err, ApplyError::IllegalAction { .. }));

    // Marking an unavailable category is illegal.
    let mut s3 = initial_state(&mut ctx);
    // Make cat 0 unavailable for current player.
    s3.players[s3.player_to_move as usize].avail_mask &= !crate::avail_bit_for_cat(0);
    let err = apply_action(s3, Action::Mark(0), &mut ctx).unwrap_err();
    assert!(matches!(err, ApplyError::IllegalAction { .. }));
}

#[test]
fn deterministic_reproducibility_same_seed_same_actions() {
    let episode_seed = 999u64;

    let actions = [
        Action::KeepMask(0),
        Action::KeepMask(0),
        Action::Mark(0),
        Action::KeepMask(0),
        Action::Mark(1),
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
        let legal_idxs: Vec<usize> = (0..A).filter(|&i| legal[i]).collect();
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
        let legal_idxs: Vec<usize> = (0..A).filter(|&i| legal[i]).collect();
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
