//! An implementation of the Permutational Analysis of Variance (PERMANOVA) test.
//!
//! Uses the algebraic identity:
//!
//! `$$\sum_{i<j} \left\|x_i-x_j\right\|^2=n\sum_i \left\|x_i\right\|^2-\left\|\sum_i x_i\right\|^2$$`
//!
//! to compute the Pseudo-F statistic in linear time for euclidean distance.

use ndarray::{Array2, Axis, concatenate};

use crate::permanova::{
    PermAnovaError, PermAnovaResult,
    permutation::{SsWithinArgs, run_permanova_permutations},
};

/// Perform a Permanova test between two multivariate samples. Each row in the matrices
/// corresponds to one instance and columns represent features. As such:
///
/// nrows = number of observations
/// ncols = number of features
///
/// Returns an `Err` if the number of features (columns) in the two matrices doesn't
/// match
pub fn restricted_permanova(
    sample_1: &Array2<f64>,
    sample_2: &Array2<f64>,
    component_ids_1: &[usize],
    component_ids_2: &[usize],
    num_permutations: usize,
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
            permutation_distribution: vec![f64::NAN; num_permutations],
            p_value: 1.0,
        });
    }

    let args = SsWithinArgs::Euclidean {
        X: &X,
        sq_norms: &sq_norms,
    };
    Ok(run_permanova_permutations(
        |groups| args.compute_ss_within(groups),
        ss_total,
        component_ids_1,
        component_ids_2,
        num_permutations,
        permutation_seed,
    ))
}

pub fn permanova(
    sample_1: &Array2<f64>,
    sample_2: &Array2<f64>,
    num_permutations: usize,
    permutation_seed: Option<u64>,
) -> Result<PermAnovaResult, PermAnovaError> {
    let component_ids_1: Vec<_> = (0..sample_1.nrows()).collect();
    let component_ids_2: Vec<_> = (0..sample_2.nrows()).collect();

    restricted_permanova(
        sample_1,
        sample_2,
        &component_ids_1,
        &component_ids_2,
        num_permutations,
        permutation_seed,
    )
}

pub(super) fn compute_ss_within_fast(
    #[allow(non_snake_case)] X: &Array2<f64>,
    sq_norms: &[f64],
    groups: &[u8],
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
    group_sums
        .into_iter()
        .zip(sum_sq)
        .zip(group_sizes)
        .map(|((group_sum, sum_of_squares), size)| {
            let sq_of_sum: f64 = group_sum.iter().map(|&x| x * x).sum();
            sum_of_squares - (sq_of_sum / size as f64)
        })
        .sum()
}
