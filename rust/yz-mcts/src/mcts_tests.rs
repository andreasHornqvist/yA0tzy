use crate::{apply_temperature, ChanceMode, Mcts, MctsConfig, UniformInference};

#[test]
fn pi_is_valid_distribution_and_respects_legality() {
    let mut mcts = Mcts::new(MctsConfig {
        c_puct: 1.5,
        simulations: 64,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
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
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
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

#[test]
fn root_noise_is_only_applied_in_rng_mode() {
    let cfg = MctsConfig {
        c_puct: 1.5,
        simulations: 32,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
    };
    let infer = UniformInference;

    let mut ctx = yz_core::TurnContext::new_deterministic(123);
    let root = yz_core::initial_state(&mut ctx);

    let mut m_det = Mcts::new(cfg).unwrap();
    let r_det = m_det.run_search(
        root,
        ChanceMode::Deterministic { episode_seed: 123 },
        &infer,
    );
    assert!(r_det.root_priors_raw.is_none());
    assert!(r_det.root_priors_noisy.is_none());

    let mut m_rng = Mcts::new(cfg).unwrap();
    let r_rng = m_rng.run_search(root, ChanceMode::Rng { seed: 123 }, &infer);
    assert!(r_rng.root_priors_raw.is_some());
    assert!(r_rng.root_priors_noisy.is_some());
    assert_ne!(
        r_rng.root_priors_raw.unwrap(),
        r_rng.root_priors_noisy.unwrap()
    );
}

#[test]
fn temperature_changes_exec_distribution_but_not_pi_target() {
    let cfg = MctsConfig {
        c_puct: 1.5,
        simulations: 128,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
    };
    let infer = UniformInference;

    let mut ctx = yz_core::TurnContext::new_deterministic(999);
    let root = yz_core::initial_state(&mut ctx);
    let legal = yz_core::legal_action_mask(
        root.players[root.player_to_move as usize].avail_mask,
        root.rerolls_left,
    );

    let mut m1 = Mcts::new(cfg).unwrap();
    let r1 = m1.run_search(
        root,
        ChanceMode::Deterministic { episode_seed: 999 },
        &infer,
    );

    // pi target is identical regardless of how we later apply temperature for executed move.
    let exec_t1 = apply_temperature(&r1.pi, &legal, 1.0);
    let exec_t0 = apply_temperature(&r1.pi, &legal, 0.0);
    assert_ne!(exec_t1, exec_t0);

    // But the replay target itself is unchanged.
    let mut m2 = Mcts::new(cfg).unwrap();
    let r2 = m2.run_search(
        root,
        ChanceMode::Deterministic { episode_seed: 999 },
        &infer,
    );
    assert_eq!(r1.pi, r2.pi);
}
