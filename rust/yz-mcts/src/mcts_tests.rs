use crate::{
    apply_temperature, ChanceMode, InferBackend, Mcts, MctsConfig, UniformInference, VirtualLossMode,
};
use crate::afterstate::{
    afterstate_from_keepmask, apply_roll_hist_to_afterstate, dice_sorted_from_hist, pack_outcome_key,
    sample_roll_hist, unpack_outcome_key,
};
use yz_core::{GameState, LegalMask, PlayerState, A, FULL_MASK};
use yz_infer::ClientOptions;
use rand::SeedableRng;

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
        virtual_loss_mode: VirtualLossMode::QPenalty,
        virtual_loss: 1.0,
        expansion_lock: false,
        chance_nodes: false,
        chance_pw_enabled: false,
        chance_pw_c: 2.0,
        chance_pw_alpha: 0.6,
        chance_pw_max_children: 64,
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
        virtual_loss_mode: VirtualLossMode::QPenalty,
        virtual_loss: 1.0,
        expansion_lock: false,
        chance_nodes: false,
        chance_pw_enabled: false,
        chance_pw_c: 2.0,
        chance_pw_alpha: 0.6,
        chance_pw_max_children: 64,
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
fn outcome_hist_key_roundtrips_and_dice_materializes_sorted() {
    let h = [2u8, 0, 1, 0, 2, 0]; // sum=5
    let key = pack_outcome_key(h);
    let back = unpack_outcome_key(key);
    assert_eq!(back, h);

    let dice = dice_sorted_from_hist(h);
    assert!(dice.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(dice.len(), 5);
    for &d in &dice {
        assert!((1..=6).contains(&d));
    }
}

#[test]
fn afterstate_and_roll_hist_produce_valid_next_state() {
    let mut ctx = yz_core::TurnContext::new_deterministic(7);
    let s = yz_core::initial_state(&mut ctx);
    assert!(s.rerolls_left > 0);

    // Pick a mask and build afterstate (canonicalization is internal).
    let as_ = afterstate_from_keepmask(&s, 0);
    assert_eq!(as_.player_to_act, s.player_to_move);
    assert_eq!(as_.rerolls_left, s.rerolls_left - 1);
    assert!(as_.k_to_roll <= 5);

    // Deterministic sampling under fixed seed
    let mut rng1 = rand_chacha::ChaCha8Rng::seed_from_u64(123);
    let mut rng2 = rand_chacha::ChaCha8Rng::seed_from_u64(123);
    let r1 = sample_roll_hist(as_.k_to_roll, &mut rng1);
    let r2 = sample_roll_hist(as_.k_to_roll, &mut rng2);
    assert_eq!(r1, r2);

    let s2 = apply_roll_hist_to_afterstate(&as_, r1);
    assert_eq!(s2.players, s.players);
    assert_eq!(s2.player_to_move, s.player_to_move);
    assert_eq!(s2.rerolls_left, s.rerolls_left - 1);
    assert!(s2.dice_sorted.windows(2).all(|w| w[0] <= w[1]));
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
        virtual_loss_mode: VirtualLossMode::QPenalty,
        virtual_loss: 1.0,
        expansion_lock: false,
        chance_nodes: false,
        chance_pw_enabled: false,
        chance_pw_c: 2.0,
        chance_pw_alpha: 0.6,
        chance_pw_max_children: 64,
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
fn chance_nodes_is_deterministic_and_records_stats() {
    let cfg = MctsConfig {
        c_puct: 1.5,
        simulations_mark: 16,
        simulations_reroll: 16,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        max_inflight: 4,
        virtual_loss_mode: VirtualLossMode::QPenalty,
        virtual_loss: 1.0,
        expansion_lock: false,
        chance_nodes: true,
        chance_pw_enabled: false,
        chance_pw_c: 2.0,
        chance_pw_alpha: 0.6,
        chance_pw_max_children: 64,
    };
    let infer = UniformInference;

    let mut ctx = yz_core::TurnContext::new_deterministic(999);
    let root = yz_core::initial_state(&mut ctx);
    assert!(root.rerolls_left > 0, "test requires KeepMask phase");

    let mut m1 = Mcts::new(cfg).unwrap();
    let r1 = m1.run_search(
        root,
        ChanceMode::Deterministic { episode_seed: 999 },
        &infer,
    );
    let s1 = r1.stats.clone();

    let mut m2 = Mcts::new(cfg).unwrap();
    let r2 = m2.run_search(
        root,
        ChanceMode::Deterministic { episode_seed: 999 },
        &infer,
    );

    assert_eq!(r1.pi, r2.pi);
    assert!(s1.chance_nodes_created > 0, "expected chance nodes to be created");
    assert!(s1.chance_visits > 0, "expected chance nodes to be visited");
    assert!(
        s1.chance_children_created > 0,
        "expected some outcome children to be created"
    );
    let hist_sum: u32 = s1.chance_k_hist.iter().sum();
    assert_eq!(hist_sum, s1.chance_visits);
}

#[test]
fn chance_progressive_widening_caps_children_and_uses_transient_eval() {
    let cfg = MctsConfig {
        c_puct: 1.5,
        simulations_mark: 64,
        simulations_reroll: 256,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        max_inflight: 8,
        virtual_loss_mode: VirtualLossMode::QPenalty,
        virtual_loss: 1.0,
        expansion_lock: false,
        chance_nodes: true,
        chance_pw_enabled: true,
        // Keep widening tight so we force transient evals quickly.
        chance_pw_c: 1.0,
        chance_pw_alpha: 0.5,
        chance_pw_max_children: 4,
    };
    let infer = UniformInference;

    let mut ctx = yz_core::TurnContext::new_deterministic(2026);
    let root = yz_core::initial_state(&mut ctx);
    assert!(root.rerolls_left > 0, "test requires KeepMask phase");

    let mut m = Mcts::new(cfg).unwrap();
    let r = m.run_search(
        root,
        ChanceMode::Deterministic { episode_seed: 2026 },
        &infer,
    );

    // With a small max_children, we should see transient evals and blocked stores.
    assert!(r.stats.chance_nodes_created > 0);
    assert!(r.stats.chance_children_created > 0);
    assert!(
        r.stats.chance_children_created
            <= (cfg.chance_pw_max_children as u32) * r.stats.chance_nodes_created.max(1),
        "expected stored chance outcome children to be capped per chance node"
    );
    assert!(
        r.stats.chance_transient_evals > 0,
        "expected transient evals when widening blocks storing"
    );
    assert!(r.stats.chance_pw_blocked > 0);
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
        virtual_loss_mode: VirtualLossMode::QPenalty,
        virtual_loss: 1.0,
        expansion_lock: false,
        chance_nodes: false,
        chance_pw_enabled: false,
        chance_pw_c: 2.0,
        chance_pw_alpha: 0.6,
        chance_pw_max_children: 64,
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
        virtual_loss_mode: VirtualLossMode::QPenalty,
        virtual_loss: 1.0,
        expansion_lock: false,
        chance_nodes: false,
        chance_pw_enabled: false,
        chance_pw_c: 2.0,
        chance_pw_alpha: 0.6,
        chance_pw_max_children: 64,
    };
    let infer = UniformInference;

    let mut ctx = yz_core::TurnContext::new_deterministic(999);
    let root = yz_core::initial_state(&mut ctx);
    let legal = crate::legal_action_mask_for_mode(&root, ChanceMode::Deterministic { episode_seed: 42 });

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
        virtual_loss_mode: VirtualLossMode::QPenalty,
        virtual_loss: 1.0,
        expansion_lock: false,
        chance_nodes: false,
        chance_pw_enabled: false,
        chance_pw_c: 2.0,
        chance_pw_alpha: 0.6,
        chance_pw_max_children: 64,
    };

    let mut ctx = yz_core::TurnContext::new_deterministic(42);
    let root = yz_core::initial_state(&mut ctx);
    let legal =
        crate::legal_action_mask_for_mode(&root, ChanceMode::Deterministic { episode_seed: 42 });

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
fn virtual_loss_modes_affect_pending_collision_tracking() {
    // This test must go through the async backend path; UniformInference is synchronous
    // and never produces pending leaf evaluations (so collisions would always be 0).
    use std::net::TcpListener;
    use std::thread;
    use std::time::Duration;

    use yz_infer::codec::{decode_request_v1, encode_response_v1};
    use yz_infer::frame::{read_frame, write_frame};
    use yz_infer::protocol::{InferResponseV1, ACTION_SPACE_A};

    // Start a *slow* dummy infer server so the client can build up multiple pending requests.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
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
            // Force a window where multiple requests are pending.
            thread::sleep(Duration::from_millis(2));
            let logits = vec![0.0f32; ACTION_SPACE_A as usize];
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

    // Construct a state with exactly ONE legal action (one remaining Mark, no rerolls).
    let mut ctx = yz_core::TurnContext::new_deterministic(7);
    let mut root = yz_core::initial_state(&mut ctx);
    root.rerolls_left = 0;
    root.players[root.player_to_move as usize].avail_mask = 1; // one category available

    let cfg_off = MctsConfig {
        c_puct: 1.5,
        simulations_mark: 64,
        simulations_reroll: 64,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        max_inflight: 8,
        virtual_loss_mode: VirtualLossMode::Off,
        virtual_loss: 0.0,
        expansion_lock: false,
        chance_nodes: false,
        chance_pw_enabled: false,
        chance_pw_c: 2.0,
        chance_pw_alpha: 0.6,
        chance_pw_max_children: 64,
    };
    let cfg_nv = MctsConfig {
        virtual_loss_mode: VirtualLossMode::NVirtualOnly,
        virtual_loss: 1.0,
        ..cfg_off
    };

    let mut m0 = Mcts::new(cfg_off).unwrap();
    let r0 = m0.run_search_with_backend(
        root,
        ChanceMode::Deterministic { episode_seed: 7 },
        &backend,
    );

    let mut m1 = Mcts::new(cfg_nv).unwrap();
    let r1 = m1.run_search_with_backend(
        root,
        ChanceMode::Deterministic { episode_seed: 7 },
        &backend,
    );

    // With mode=off, we never reserve, so the collision counter is disabled by design.
    assert_eq!(r0.stats.pending_collisions, 0);
    // With reservations enabled, repeated selection of the single legal edge will collide.
    assert!(
        r1.stats.pending_collisions > 0,
        "expected collisions to be tracked with reservations enabled; got {}",
        r1.stats.pending_collisions
    );

    drop(backend);
    server.join().unwrap();
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
    for mask in 0u8..=31u8 {
        assert!(
            ((legal >> (mask as u64)) & 1) != 0,
            "expected KeepMask({mask}) to be legal for no-duplicate dice"
        );
    }
    // Mark actions should be illegal at rerolls > 0 (mark-only-at-roll-3).
    for idx in 32..47 {
        assert!(
            ((legal >> idx) & 1) == 0,
            "expected Mark to be illegal at rerolls>0"
        );
    }
}

#[test]
fn keepmask_canonicalization_duplicates_prunes_equivalents_in_rng_and_deterministic() {
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

    // Deterministic mode matches RNG mode (canonical keepmasks only).
    let legal_det =
        crate::legal_action_mask_for_mode(&s, ChanceMode::Deterministic { episode_seed: 1 });
    assert!(((legal_det >> 8) & 1) != 0);
    assert!(((legal_det >> 16) & 1) == 0);
}
