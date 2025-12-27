use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn gen_dice_samples(n: usize) -> Vec<[u8; 5]> {
    // Simple deterministic xorshift64, no rand dependency.
    let mut x: u64 = 0x1234_5678_9ABC_DEF0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut d = [0u8; 5];
        for i in 0..5 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            d[i] = (x % 6) as u8 + 1;
        }
        // Sorting isn't required, but representative states keep dice sorted.
        d.sort_unstable();
        out.push(d);
    }
    out
}

fn bench_scores_for_dice(c: &mut Criterion) {
    let mut g = c.benchmark_group("yz_core_scoring");
    for &n in &[256usize, 4096usize] {
        let samples = gen_dice_samples(n);
        g.bench_with_input(BenchmarkId::new("scores_for_dice_batch", n), &samples, |b, s| {
            b.iter(|| {
                for &dice in s.iter() {
                    black_box(yz_core::scores_for_dice(black_box(dice)));
                }
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_scores_for_dice);
criterion_main!(benches);


