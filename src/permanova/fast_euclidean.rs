//! An implementation of the Permutational Analysis of Variance (PERMANOVA) test.
//!
//! Uses the algebraic identity:
//!
//! `$$\sum_{i<j} \left\|x_i-x_j\right\|^2=n\sum_i \left\|x_i\right\|^2-\left\|\sum_i x_i\right\|^2$$`
//!
//! to compute the Pseudo-F statistic in linear time for euclidean distance.

use std::{
    collections::{HashMap, HashSet},
    iter::repeat_n,
};

use ndarray::{Array2, Axis, concatenate};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::permanova::{PermAnovaError, PermAnovaResult};
use crate::relation_extraction::{OcelRelationshipCounter, RelationshipCount};

/// Perform a Permanova test between two multivariate samples. Each row in the matrices
/// corresponds to one instance and columns represent features. As such:
///
/// nrows = number of observations
/// ncols = number of features
///
/// Returns an `Err` if the number of features (columns) in the two matrices doesn't
/// match
pub fn permanova(
    sample_1: &Array2<f64>,
    sample_2: &Array2<f64>,
    num_permuations: usize,
    permutation_seed: Option<u64>,
) -> Result<PermAnovaResult, PermAnovaError> {
    let n1 = sample_1.nrows();
    let n2 = sample_2.nrows();
    let n = n1 + n2;

    if sample_1.ncols() != sample_2.ncols() {
        return Err(PermAnovaError::NumFeaturesMismatch(
            sample_1.ncols(),
            sample_2.ncols(),
        ));
    }

    if n < 3 {
        return Err(PermAnovaError::InsufficientObservations(n));
    }

    let num_features = sample_1.ncols();

    #[allow(non_snake_case)]
    let X = concatenate(Axis(0), &[sample_1.view(), sample_2.view()]).unwrap();

    // Precompute squared norms ||x_i||^2 once — reused across all permutations.
    let sq_norms: Vec<f64> = (0..n)
        .map(|i| X.row(i).iter().map(|&x| x * x).sum())
        .collect();

    // We can compute SS_T as:
    //
    // $$\frac{1}{n}\sum_{i<j}\left\|x_i-x_j\right\|^2$$
    //
    // since it is equivalent to (see compute_pseudo_f_fast documentation):
    //
    // $SS_T=\sum_i\left\|x_i\right\|^2 - \frac{\left\|\sum_i x_i\right\|^2}{n}$$
    let total_sq_norm_sum: f64 = sq_norms.iter().sum();
    let total_vec_sq_sum: f64 = {
        let mut col_sums = vec![0.0f64; num_features];
        for i in 0..n {
            let row = X.row(i);
            for d in 0..num_features {
                col_sums[d] += row[d];
            }
        }
        col_sums.iter().map(|&x| x * x).sum()
    };

    // SS_T
    let ss_total = total_sq_norm_sum - total_vec_sq_sum / n as f64;

    if ss_total == 0.0 {
        // ss_within and ss_between will be 0, giving us pseudo_f NaN
        return Ok(PermAnovaResult {
            pseudo_f: f64::NAN,
            permutation_distribution: vec![f64::NAN; num_permuations],
            p_value: 1.0,
        });
    }

    #[cfg_attr(feature = "rayon", allow(unused_mut))]
    let mut groups: Vec<u8> = repeat_n(0, n1).chain(repeat_n(1, n2)).collect();

    // F
    let pseudo_f = compute_pseudo_f_fast(&X, &sq_norms, &groups, ss_total);

    #[cfg(not(feature = "rayon"))]
    let permutation_distribution: Vec<f64> = {
        let mut rng = if let Some(seed) = permutation_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_os_rng()
        };

        (0..num_permuations)
            .map(|_| {
                groups.shuffle(&mut rng);
                compute_pseudo_f_fast(&X, &sq_norms, &groups, ss_total)
            })
            .collect()
    };

    #[cfg(feature = "rayon")]
    let permutation_distribution: Vec<f64> = {
        (0..num_permuations)
            .into_par_iter()
            .map(|i| {
                let mut rng = if let Some(seed) = permutation_seed {
                    StdRng::seed_from_u64(seed + i as u64)
                } else {
                    StdRng::from_os_rng()
                };
                let mut shuffled_groups = groups.clone();
                shuffled_groups.shuffle(&mut rng);
                compute_pseudo_f_fast(&X, &sq_norms, &shuffled_groups, ss_total)
            })
            .collect()
    };

    let p_value = permutation_distribution
        .iter()
        .filter(|&&perm_f| perm_f >= pseudo_f)
        .count() as f64
        / (num_permuations as f64);

    Ok(PermAnovaResult {
        pseudo_f,
        permutation_distribution,
        p_value,
    })
}

/// Compute the pseudo-F statistic faster using the algebraic identity:
///
/// `$$\sum_{i<j} \left\|x_i-x_j\right\|^2=n\sum_i \left\|x_i\right\|^2-\left\|\sum_i x_i\right\|^2$$`
///
/// So, while the naive implementation (strictly following the definition) would
/// compute for each group `$G_k$`:
///
/// `$$\frac{1}{n_k}\sum_{i<j\in G_k}\left\|x_i-x_j\right\|^2$$`,
///
/// we can equivalently compute:
///
/// `$$\sum_{i\in G_k}\left\|x_i\right\|^2-\frac{\left\|\sum_{i\in G_k} x_i\right\|^2}{n_k}$$`
///
/// This makes the computation of the pseudo-F statistic linear instead of quadratic
///
/// # Arguments
///
/// * `X`: The matrix of observations. Rows = observations, columns = features.
/// * `sq_norms`: The squared norms of rows in X. That is, `sq_norms_i=sum(x^2 for x in X.row(i))`
/// * `groups`: Assignment of rows in X to groups. 0 if the row belongs to log 1,
///   1 if it belongs to log 2
/// * `ss_total`: The `SS_T` measure (`1/n * distances_for_i<j`)
fn compute_pseudo_f_fast(
    #[allow(non_snake_case)] X: &Array2<f64>,
    sq_norms: &[f64],
    groups: &[u8],
    ss_total: f64,
) -> f64 {
    let n = X.nrows();
    let num_features = X.ncols();

    // Two components to compute:
    // $$\sum_{i\in G_k}\left\|x_i\right\|^2$$ --> sum_sq
    // $$\frac{\left\|\sum_{i\in G_k} x_i\right\|^2}{n_k}$$ --> ||group_sums||^2 / group_sizes

    // The running sum of the rows belonging to each group
    let mut group_sums = [vec![0.0f64; num_features], vec![0.0f64; num_features]];
    // The running size of each group
    let mut group_sizes = [0usize; 2];
    // The running sum of the squared norms of the rows for each group
    let mut sum_sq = [0.0f64; 2];

    for i in 0..n {
        let g = groups[i] as usize;
        sum_sq[g] += sq_norms[i];
        group_sizes[g] += 1;

        // Add the row to the respective sum
        let row = X.row(i);
        let vs = &mut group_sums[g];
        for d in 0..num_features {
            vs[d] += row[d];
        }
    }

    // SS_W = sum for all groups of $$\sum_{i\in G_k}\left\|x_i\right\|^2-\frac{\left\|\sum_{i\in G_k} x_i\right\|^2}{n_k}$$
    let ss_within = group_sums
        .into_iter()
        .zip(sum_sq)
        .zip(group_sizes)
        .map(|((group_sum, sum_of_squares), size)| {
            let sq_of_sum: f64 = group_sum.iter().map(|&x| x * x).sum();
            sum_of_squares - (sq_of_sum / size as f64)
        })
        .sum::<f64>();

    let ss_between = ss_total - ss_within;
    ss_between / (ss_within / (n - 2) as f64)
}

pub fn compare_counts<'a>(
    counts_1: &OcelRelationshipCounter<'a>,
    counts_2: &OcelRelationshipCounter<'a>,
    num_permuations: usize,
    permutation_seed: Option<u64>,
) -> Result<HashMap<&'a str, PermAnovaResult>, PermAnovaError> {
    let all_group_keys: HashSet<_> = counts_1.keys().chain(counts_2.keys()).copied().collect();

    let mut group_key_to_pval = HashMap::new();

    // TODO: Could parallelize? But right now, I parallelize across permutations
    for key in all_group_keys {
        let (Some(rel_counts_1), Some(rel_counts_2)) = (counts_1.get(key), counts_2.get(key))
        else {
            log::info!("{key} only in one log");
            continue;
        };

        // Get an index of ordered relation types (tuple of object type and qualifier)
        let ordered_relation_types: Vec<_> = rel_counts_1
            .iter()
            .chain(rel_counts_2.iter())
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>()
            .into_iter()
            .copied()
            .collect();

        let sample_1 = encode_relation_counts(rel_counts_1, &ordered_relation_types);
        let sample_2 = encode_relation_counts(rel_counts_2, &ordered_relation_types);

        let start = std::time::Instant::now();
        let res = permanova(&sample_1, &sample_2, num_permuations, permutation_seed)?;
        log::trace!("{key} took {:?}", start.elapsed());

        group_key_to_pval.insert(key, res);
    }

    Ok(group_key_to_pval)
}

/// Encode relation counts for an event type in a matrix. Each row is an observation
/// (event) and columns correspond to the relation types (I.e., `(qualifier, object_type)` pairs),
/// as given in `ordered_relation_types`.
///
/// Observed relation counts that are not in `ordered_relation_types` are ignored
/// and relation types that were not observed for an event are counted as `0`.
pub fn encode_relation_counts(
    relation_counts: &[RelationshipCount<'_>],
    ordered_relation_types: &[(&'_ str, &'_ str)],
) -> Array2<f64> {
    let mut res = Array2::zeros((relation_counts.len(), ordered_relation_types.len()));
    for (idx, counts) in relation_counts.iter().enumerate() {
        res.row_mut(idx)
            .assign(&counts.encoded_f64(ordered_relation_types));
    }
    res
}
