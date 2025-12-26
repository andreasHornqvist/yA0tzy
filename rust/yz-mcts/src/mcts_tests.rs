use crate::{ChanceMode, Mcts, MctsConfig, UniformInference};

#[test]
fn pi_is_valid_distribution_and_respects_legality() {
    let mut mcts = Mcts::new(MctsConfig {
        c_puct: 1.5,
        simulations: 64,
    })
    .unwrap();
    let infer = UniformInference;

    let mut ctx = yz_core::TurnContext::new_deterministic(123);
    let root = yz_core::initial_state(&mut ctx);

    let res = mcts.run_search(
        root,
        ChanceMode::Deterministic { episode_seed: 123 },
        &infer,
    );

    let legal = yz_core::legal_action_mask(
        root.players[root.player_to_move as usize].avail_mask,
        root.rerolls_left,
    );
    let mut sum = 0.0f32;
    for a in 0..yz_core::A {
        if legal[a] {
            assert!(res.pi[a].is_finite());
            assert!(res.pi[a] >= 0.0);
            sum += res.pi[a];
        } else {
            assert_eq!(res.pi[a], 0.0);
        }
    }
    assert!((sum - 1.0).abs() < 1e-5, "sum={}", sum);
}

#[test]
fn eval_mode_is_deterministic() {
    let cfg = MctsConfig {
        c_puct: 1.5,
        simulations: 128,
    };
    let infer = UniformInference;

    let mut ctx = yz_core::TurnContext::new_deterministic(999);
    let root = yz_core::initial_state(&mut ctx);

    let mut m1 = Mcts::new(cfg).unwrap();
    let r1 = m1.run_search(
        root,
        ChanceMode::Deterministic { episode_seed: 999 },
        &infer,
    );

    let mut m2 = Mcts::new(cfg).unwrap();
    let r2 = m2.run_search(
        root,
        ChanceMode::Deterministic { episode_seed: 999 },
        &infer,
    );

    assert_eq!(r1.pi, r2.pi);
}
