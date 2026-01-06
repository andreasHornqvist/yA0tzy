use crate::{apply_temperature, ChanceMode, InferBackend, Mcts, MctsConfig, UniformInference};
use yz_core::{GameState, LegalMask, PlayerState, A, FULL_MASK};
use yz_infer::ClientOptions;

struct BadInference;

impl crate::Inference for BadInference {
    fn eval(&self, _features: &[f32], _legal: LegalMask) -> ([f32; A], f32) {
        ([f32::NAN; A], 0.0)
    }
}

fn start_dummy_infer_server_tcp() -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
    use std::net::TcpListener;
    use std::thread;

    use yz_infer::codec::{decode_request_v1, encode_response_v1};
    use yz_infer::frame::{read_frame, write_frame};
    use yz_infer::protocol::{InferResponseV1, ACTION_SPACE_A};

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut sock, _peer) = listener.accept().unwrap();
        loop {
            let payload = match read_frame(&mut sock) {
                Ok(p) => p,
                Err(_) => break,
            };
            let req = match decode_request_v1(&payload) {
                Ok(r) => r,
                Err(_) => break,
            };

            let mut logits = vec![0.0f32; ACTION_SPACE_A as usize];
            for i in 0..(ACTION_SPACE_A as usize) {
                if ((req.legal_mask >> i) & 1) == 0 {
                    logits[i] = -1.0e9;
                }
            }

            let resp = InferResponseV1 {
                request_id: req.request_id,
                policy_logits: logits,
                value: 0.0,
                margin: None,
            };
            let out = encode_response_v1(&resp);
            if write_frame(&mut sock, &out).is_err() {
                break;
            }
        }
    });
    (addr, handle)
}

#[test]
fn pi_is_valid_distribution_and_respects_legality() {
    let mut mcts = Mcts::new(MctsConfig {
        c_puct: 1.5,
        simulations_mark: 64,
        simulations_reroll: 64,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        max_inflight: 8,
        virtual_loss: 1.0,
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
        if ((legal >> a) & 1) != 0 {
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
fn async_leaf_pipeline_works_end_to_end_via_inference_client() {
    let (addr, server) = start_dummy_infer_server_tcp();

    let backend = InferBackend::connect_tcp(
        addr,
        0,
        ClientOptions {
            max_inflight_total: 4096,
            max_outbound_queue: 4096,
            request_id_start: 1,
            protocol_version: yz_infer::protocol::PROTOCOL_VERSION_V1,
            legal_mask_bitset: false,
        },
    )
    .unwrap();

    let mut mcts = Mcts::new(MctsConfig {
        c_puct: 1.5,
        simulations_mark: 64,
        simulations_reroll: 64,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        max_inflight: 8,
        virtual_loss: 1.0,
    })
    .unwrap();

    let mut ctx = yz_core::TurnContext::new_deterministic(123);
    let root = yz_core::initial_state(&mut ctx);
    let res = mcts.run_search_with_backend(
        root,
        ChanceMode::Deterministic { episode_seed: 123 },
        &backend,
    );

    // Valid pi distribution.
    let legal = yz_core::legal_action_mask(
        root.players[root.player_to_move as usize].avail_mask,
        root.rerolls_left,
    );
    let mut sum = 0.0f32;
    for a in 0..yz_core::A {
        if ((legal >> a) & 1) != 0 {
            assert!(res.pi[a].is_finite());
            assert!(res.pi[a] >= 0.0);
            sum += res.pi[a];
        } else {
            assert_eq!(res.pi[a], 0.0);
        }
    }
    assert!((sum - 1.0).abs() < 1e-5, "sum={}", sum);
    // Ensure we actually had multiple in-flight requests at some point.
    assert!(res.stats.pending_count_max > 1);

    drop(backend);
    server.join().unwrap();
}

#[test]
fn eval_mode_is_deterministic() {
    let cfg = MctsConfig {
        c_puct: 1.5,
        simulations_mark: 128,
        simulations_reroll: 128,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_inflight: 8,
        virtual_loss: 1.0,
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
        simulations_mark: 32,
        simulations_reroll: 32,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_inflight: 8,
        virtual_loss: 1.0,
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
        simulations_mark: 128,
        simulations_reroll: 128,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        max_inflight: 8,
        virtual_loss: 1.0,
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
    let exec_t1 = apply_temperature(&r1.pi, legal, 1.0);
    let exec_t0 = apply_temperature(&r1.pi, legal, 0.0);
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

#[test]
fn fallback_can_be_triggered_and_returns_uniform_pi() {
    let cfg = MctsConfig {
        c_puct: 1.5,
        simulations_mark: 64,
        simulations_reroll: 64,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        max_inflight: 8,
        virtual_loss: 1.0,
    };

    let mut ctx = yz_core::TurnContext::new_deterministic(42);
    let root = yz_core::initial_state(&mut ctx);
    let legal = yz_core::legal_action_mask(
        root.players[root.player_to_move as usize].avail_mask,
        root.rerolls_left,
    );

    let mut m = Mcts::new(cfg).unwrap();
    let r = m.run_search(
        root,
        ChanceMode::Deterministic { episode_seed: 42 },
        &BadInference,
    );

    assert!(r.fallbacks > 0, "expected fallback counter to increment");
    let expected = {
        let mut out = [0.0f32; A];
        let mut cnt = 0usize;
        for i in 0..A {
            if ((legal >> i) & 1) != 0 {
                cnt += 1;
            }
        }
        let u = if cnt > 0 { 1.0 / (cnt as f32) } else { 0.0 };
        for i in 0..A {
            if ((legal >> i) & 1) != 0 {
                out[i] = u;
            }
        }
        out
    };
    assert_eq!(r.pi, expected);
}

#[test]
fn virtual_loss_reduces_pending_collisions() {
    let infer = UniformInference;
    let mut ctx = yz_core::TurnContext::new_deterministic(1);
    let root = yz_core::initial_state(&mut ctx);

    let cfg_no_vl = MctsConfig {
        c_puct: 1.5,
        simulations_mark: 64,
        simulations_reroll: 64,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        max_inflight: 8,
        virtual_loss: 0.0,
    };
    let cfg_vl = MctsConfig {
        virtual_loss: 1.0,
        ..cfg_no_vl
    };

    let mut m0 = Mcts::new(cfg_no_vl).unwrap();
    let r0 = m0.run_search(root, ChanceMode::Deterministic { episode_seed: 1 }, &infer);

    let mut m1 = Mcts::new(cfg_vl).unwrap();
    let r1 = m1.run_search(root, ChanceMode::Deterministic { episode_seed: 1 }, &infer);

    assert!(
        r1.stats.pending_collisions < r0.stats.pending_collisions,
        "expected collisions to decrease with virtual loss: no_vl={} vl={}",
        r0.stats.pending_collisions,
        r1.stats.pending_collisions
    );
}

#[test]
fn keepmask_canonicalization_no_duplicates_keeps_all_masks_in_rng() {
    // No duplicates => all KeepMask(0..=30) remain legal in self-play mode.
    let s = GameState {
        players: [
            PlayerState {
                avail_mask: FULL_MASK,
                upper_total_cap: 0,
                total_score: 0,
            },
            PlayerState {
                avail_mask: FULL_MASK,
                upper_total_cap: 0,
                total_score: 0,
            },
        ],
        dice_sorted: [1, 2, 3, 4, 6],
        rerolls_left: 2,
        player_to_move: 0,
    };

    let legal = crate::legal_action_mask_for_mode(&s, ChanceMode::Rng { seed: 123 });
    for mask in 0u8..=30u8 {
        assert!(
            ((legal >> (mask as u64)) & 1) != 0,
            "expected KeepMask({mask}) to be legal for no-duplicate dice"
        );
    }
    // KeepMask(31) is always illegal (dominated).
    assert!(((legal >> 31) & 1) == 0);
}

#[test]
fn keepmask_canonicalization_duplicates_prunes_equivalents_in_rng_only() {
    let s = GameState {
        players: [
            PlayerState {
                avail_mask: FULL_MASK,
                upper_total_cap: 0,
                total_score: 0,
            },
            PlayerState {
                avail_mask: FULL_MASK,
                upper_total_cap: 0,
                total_score: 0,
            },
        ],
        dice_sorted: [1, 1, 3, 4, 6],
        rerolls_left: 2,
        player_to_move: 0,
    };

    // Rng mode prunes redundant KeepMasks (keep the rightmost duplicate).
    let legal_rng = crate::legal_action_mask_for_mode(&s, ChanceMode::Rng { seed: 123 });
    // Keeping exactly one of the two '1's has two index-masks: 0b10000 (idx 16) and 0b01000 (idx 8).
    // Canonicalization keeps the rightmost occurrence => idx 8 stays, idx 16 is pruned.
    assert!(((legal_rng >> 8) & 1) != 0);
    assert!(((legal_rng >> 16) & 1) == 0);

    // Deterministic mode leaves legality unchanged (both are legal).
    let legal_det =
        crate::legal_action_mask_for_mode(&s, ChanceMode::Deterministic { episode_seed: 1 });
    assert!(((legal_det >> 8) & 1) != 0);
    assert!(((legal_det >> 16) & 1) != 0);
}
