use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use diffrel::permanova::permanova;
use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn make_sample(rng: &mut StdRng, nrows: usize, ncols: usize) -> Array2<f64> {
    Array2::from_shape_fn((nrows, ncols), |_| rng.random_range(0u32..=10) as f64)
}

fn bench_permanova_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("permanova/vary_n");
    for n in [20usize, 100, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = StdRng::seed_from_u64(42);
            b.iter_batched(
                || (make_sample(&mut rng, n / 2, 5), make_sample(&mut rng, n / 2, 5)),
                |(s1, s2)| permanova(s1, s2, 999, Some(42)).unwrap(),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_permanova_n);
criterion_main!(benches);
