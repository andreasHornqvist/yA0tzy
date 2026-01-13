use criterion::{black_box, criterion_group, criterion_main, Criterion};

use yz_core::A;
use yz_mcts::{bench_select_action_v1, ChanceMode, MctsConfig};

fn make_node() -> yz_mcts::node::Node {
    // Use a fully initialized node with representative priors and visit stats.
    let mut n = yz_mcts::node::Node::new(0);
    n.is_expanded = true;
    n.n_sum = 10_000;
    for a in 0..A {
        // small non-zero priors
        n.p[a] = 1.0 / (A as f32);
        n.n[a] = (a as u32 % 17) as u32;
        n.w[a] = (a as f32).sin();
    }
    n
}

fn make_legal() -> [bool; A] {
    // Roughly half legal.
    let mut legal = [false; A];
    for i in 0..A {
        legal[i] = (i % 2) == 0;
    }
    legal
}

fn bench_puct_select(c: &mut Criterion) {
    let cfg = MctsConfig {
        c_puct: 1.5,
        simulations_mark: 64,
        simulations_reroll: 64,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        max_inflight: 8,
        virtual_loss_mode: yz_mcts::VirtualLossMode::QPenalty,
        virtual_loss: 1.0,
        expansion_lock: false,
    };
    let node = make_node();
    let legal = make_legal();
    let mode = ChanceMode::Deterministic { episode_seed: 0 };

    c.bench_function("yz_mcts_select_action_v1", |b| {
        b.iter(|| {
            black_box(bench_select_action_v1(
                black_box(&cfg),
                black_box(&node),
                black_box(&legal),
                black_box(mode),
            ))
        })
    });
}

criterion_group!(benches, bench_puct_select);
criterion_main!(benches);
