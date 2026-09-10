//! Contains correction methods for multiple testing (using the Holm-Bonferroni
//! correction).

use std::collections::HashMap;

/// Check each p-value for significance using the Holm-Bonferroni procedure
/// to correct for multiple-testing.
///
/// Procedure from Holm et al. "A simple sequentially rejective multiple test
/// procedure" (1979).
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

pub fn check_significance_benjamini_hochberg(
    pvals: HashMap<&str, f64>,
    alpha: f64,
) -> HashMap<&str, (f64, bool)> {
    let mut vals: Vec<(&str, f64)> = pvals.into_iter().collect();
    vals.sort_by(|(_, p_1), (_, p_2)| p_1.total_cmp(p_2));

    let cutoff = vals
        .iter()
        .enumerate()
        .rev()
        .find(|(k, (_, p))| *p <= ((k + 1) as f64 / vals.len() as f64) * alpha)
        .map(|(k, _p)| k + 1)
        .unwrap_or(0); // 0 = no rejections

    vals.into_iter()
        .enumerate()
        .map(|(i, (name, p))| (name, (p, i < cutoff)))
        .collect()
}

#[test]
fn test_benjamini_hochberg() {
    let vals = vec![
        ("0", 0.3770),
        ("1", 0.0130),
        ("2", 0.6530),
        ("3", 0.4320),
        ("4", 0.0070),
        ("5", 0.0900),
        ("6", 0.1480),
        ("7", 0.1130),
        ("8", 0.4260),
        ("9", 0.3840),
        ("10", 0.1180),
    ];

    let res = check_significance_holm_bonferroni(vals.into_iter().collect(), 0.05);
    println!("{:#?}", res);
    println!("{}", res.values().any(|x| x.1));
}
