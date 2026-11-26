//! An implementation of the Permutational Analysis of Variance (PERMANOVA) test
//! for an arbitrary cost function.
use std::collections::HashMap;

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
    sq_distance_function: impl Fn(&[f64], &[f64]) -> f64,
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

    // TODO: component_ids shape check?

    #[expect(non_snake_case)]
    let X = concatenate(Axis(0), &[sample_1.view(), sample_2.view()]).unwrap();

    // Compute (squared) distance matrix
    let mut ss_total = 0.0;
    #[expect(non_snake_case)]
    let mut D2 = Array2::zeros((n, n));

    for i in 0..n {
        for j in i + 1..n {
            let dist = sq_distance_function(
                X.row(i).as_slice().expect("X is contiguous"),
                X.row(j).as_slice().expect("X is contiguous"),
            );
            ss_total += dist;
            D2[(i, j)] = dist;
            D2[(j, i)] = dist;
        }
    }

    ss_total /= n as f64;

    let args = SsWithinArgs::DistanceMatrix(&D2);
    Ok(run_permanova_permutations(
        |groups| args.compute_ss_within(groups),
        ss_total,
        component_ids_1,
        component_ids_2,
        num_permutations,
        permutation_seed,
    ))
}

pub(super) fn compact_component_ids(component_ids: &[usize]) -> (Vec<usize>, usize) {
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let compact = component_ids
        .iter()
        .map(|&c| {
            let next = remap.len();
            *remap.entry(c).or_insert(next)
        })
        .collect();

    (compact, remap.len())
}

pub fn permanova(
    sample_1: &Array2<f64>,
    sample_2: &Array2<f64>,
    sq_distance_function: impl Fn(&[f64], &[f64]) -> f64,
    num_permuations: usize,
    permutation_seed: Option<u64>,
) -> Result<PermAnovaResult, PermAnovaError> {
    let component_ids_1: Vec<_> = (0..sample_1.nrows()).collect();
    let component_ids_2: Vec<_> = (0..sample_2.nrows()).collect();

    restricted_permanova(
        sample_1,
        sample_2,
        &component_ids_1,
        &component_ids_2,
        sq_distance_function,
        num_permuations,
        permutation_seed,
    )
}

/// Compute the `SS_W`statistic for the Pseudo-F statistic. For two groups.
///
/// * `groups`: 0 if that value belongs to group 1, 1 if it belongs to group two
///
/// TODO: The counts within groups never really change. Could pass them to skip
/// some computation
pub(super) fn compute_sum_of_squares_within(
    #[expect(non_snake_case)] D2: &Array2<f64>,
    groups: &[u8],
) -> f64 {
    let mut group_sums: [f64; 2] = [0.0, 0.0];
    let mut group_sizes: [usize; 2] = [0, 0];

    let n = D2.nrows();
    for i in 0..n - 1 {
        let group_i = groups[i] as usize;
        group_sizes[group_i] += 1;
        for j in (i + 1)..n {
            let d = D2[(i, j)];
            let group_j = groups[j] as usize;
            if group_i == group_j {
                group_sums[group_i] += d;
            }
        }
    }

    // The last element wasnt counted in the loop
    group_sizes[groups[n - 1] as usize] += 1;

    group_sums
        .into_iter()
        .zip(group_sizes)
        .map(|(sum, size)| sum / size as f64)
        .sum()
}
