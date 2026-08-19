/// Squared euclidean distance: since PERMANOVA uses the squared distance, we can
/// skip the square root entirely
pub fn squared_euclidean_distance(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
}

fn bray_curtis_distance(v1: &[f64], v2: &[f64]) -> f64 {
    let (abs_diffs, sums) = v1
        .iter()
        .zip(v2)
        .map(|(a, b)| ((a - b).abs(), a + b))
        .fold((0.0, 0.0), |(a1, b1), (a2, b2)| (a1 + a2, b1 + b2));

    if sums == 0.0 {
        return 0.0;
    }

    abs_diffs / sums
}

/// The squared Bray-Curtis Dissimilarity.
///
/// The Bray-Curtis Dissimilarity for (only positive entries) is defined as:
///
/// $$
///     \frac{\sum_i\lvert x_i-y_i\rvert}{\sum_i x_i+y_i}
/// $$
pub fn squared_bray_curtis_distance(v1: &[f64], v2: &[f64]) -> f64 {
    bray_curtis_distance(v1, v2).powi(2)
}
