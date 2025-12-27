use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn gen_states(n: usize) -> Vec<(u16, u8)> {
    // Deterministic generator; mask pattern + rerolls.
    let mut out = Vec::with_capacity(n);
    let mut x: u64 = 0xA5A5_A5A5_0123_4567;
    for i in 0..n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let avail = (x as u16) ^ ((i as u16).rotate_left(7));
        let rerolls = (i as u8) % 3; // 0,1,2
        out.push((avail, rerolls));
    }
    out
}

fn bench_legal_action_mask(c: &mut Criterion) {
    let mut g = c.benchmark_group("yz_core_legal");
    for &n in &[256usize, 4096usize] {
        let states = gen_states(n);
        g.bench_with_input(BenchmarkId::new("legal_action_mask_batch", n), &states, |b, s| {
            b.iter(|| {
                for &(avail, rerolls_left) in s.iter() {
                    black_box(yz_core::legal_action_mask(
                        black_box(avail),
                        black_box(rerolls_left),
                    ));
                }
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_legal_action_mask);
criterion_main!(benches);


