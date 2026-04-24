//! Contains correction methods for multiple testing (using the Holm-Bonferroni
//! correction).

use std::collections::HashMap;

/// Check each p-value for significance using the Holm-Bonferroni procedure
/// to correct for multiple-testing.
///
/// Procedure from Holm et al. "A simple sequentially rejective multiple test
/// procedure" (1979)
///
/// Returns the hashmap mapping each key (event type) to the p-value and a bool
/// indicating significance.
pub fn check_significance_holm_bonferroni(
    pvals: HashMap<&str, f64>,
    alpha: f64,
) -> HashMap<&str, (f64, bool)> {
    let mut vals: Vec<(&str, f64)> = pvals.into_iter().collect();
    vals.sort_by(|(_, p_1), (_, p_2)| p_1.total_cmp(p_2));

    let m = vals.len();
    let mut exit = false;
    vals.into_iter()
        .enumerate()
        .map(|(idx, (key, val))| {
            let significant = !exit && val <= alpha / (m - idx) as f64;
            if !significant {
                exit = true;
            }
            (key, (val, significant))
        })
        .collect()
}
