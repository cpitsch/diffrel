pub fn cliffs_delta(a: &[f64], b: &[f64]) -> f64 {
    let n_a = a.len();
    let n_b = b.len();

    if n_a == 0 || n_b == 0 {
        return 0.0;
    }

    let mut merged: Vec<(f64, u8)> = a
        .iter()
        .map(|&val| (val, 0))
        .chain(b.iter().map(|&val| (val, 1)))
        .collect();
    merged.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));

    let mut rank_sum_a = 0.0;

    let mut rank = 1;
    for group in merged.chunk_by(|x, y| x.0 == y.0) {
        let avg_rank = rank as f64 + (group.len() as f64 - 1.0) / 2.0;
        rank_sum_a += group.iter().filter(|x| x.1 == 0).count() as f64 * avg_rank;
        rank += group.len();
    }

    let u_a = rank_sum_a - (n_a * (n_a + 1)) as f64 / 2.0;
    2.0 * u_a / (n_a as f64 * n_b as f64) - 1.0
}
