//! A specialized implementation of the Permutational Analysis of Variance (PERMANOVA)
//! test for drilling down on a significant test result.
//!
//! Intended use: performing an individual PERMANOVA test on each dimension of the
//! data that yielded a significant result. Using the same permutations as the
//! significant test allows us to use a stronger multiple testing correction.

use std::{collections::HashMap, iter::repeat_n};

use itertools::Itertools;
use ndarray::{Array2, Axis, concatenate};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::{
    permanova::{PermAnovaError, PermAnovaResult, base::compact_component_ids},
    relation_extraction::Qualifier,
};

/// Drill down on a significant result from [`crate::permanova::base::compare_counts`]
/// or [`crate::permanova::fast_euclidean::compare_counts`] by performing separate
/// tests on each relation type. The seed is necessary to ensure that the tests
/// use the same permutations as the original test.
///
/// * `ordered_relation_types` is the index used to create the matrices, i.e., a
///   mapping of column index to relation type
///
/// Returns a hashmap of relationsip type `(qualifier, object_type)` pair to a pvalue
// TODO: Could remember the permutations to not need the seed here. But depending
// on log size and number of permutations this can quickly become infeasible.
// TODO: Could just return a 1xn matrix of p-values and leave the mapping to relation
// type to the user.
pub fn drilldown<'a>(
    sample_1: &Array2<f64>,
    sample_2: &Array2<f64>,
    component_ids_1: &[usize],
    component_ids_2: &[usize],
    num_permuations: usize,
    permutation_seed: u64,
    ordered_relation_types: &[(Qualifier<'a>, &'a str)],
) -> Result<HashMap<(Qualifier<'a>, &'a str), PermAnovaResult>, PermAnovaError> {
    let n1 = sample_1.nrows();
    let n2 = sample_2.nrows();
    let n = n1 + n2;

    if n < 3 {
        return Err(PermAnovaError::InsufficientObservations(n));
    }

    #[expect(non_snake_case)]
    let X = concatenate(Axis(0), &[sample_1.view(), sample_2.view()])
        .map_err(|_| PermAnovaError::NumFeaturesMismatch(sample_1.ncols(), sample_2.ncols()))?;

    let ss_totals = ss_total_per_dim(&X);

    let (component_ids_1, num_components_1) = compact_component_ids(component_ids_1);
    let (component_ids_2, num_components_2) = compact_component_ids(component_ids_2);
    let component_index_of_row: Vec<usize> = component_ids_1
        .iter()
        .copied()
        .chain(component_ids_2.iter().map(|&c| c + num_components_1))
        .collect();

    let component_label: Vec<u8> = repeat_n(0u8, num_components_1)
        .chain(repeat_n(1u8, num_components_2))
        .collect();

    let groups: Vec<u8> = repeat_n(0, n1).chain(repeat_n(1, n2)).collect();
    let f_statistics = f_per_dim_fast(&X, &groups, &ss_totals);

    let sorted_dims: Vec<usize> = f_statistics
        .iter()
        .enumerate()
        .sorted_by(|x, y| x.1.total_cmp(y.1))
        .map(|x| x.0)
        .collect();

    #[cfg(not(feature = "rayon"))]
    let permutation_distribution: Vec<Vec<f64>> = {
        let mut rng = StdRng::seed_from_u64(permutation_seed);
        let mut shuffled_labels = component_label;
        let mut shuffled_groups = vec![0u8; n];

        (0..num_permuations)
            .map(|_| {
                shuffled_labels.shuffle(&mut rng);
                shuffled_groups
                    .iter_mut()
                    .zip(&component_index_of_row)
                    .for_each(|(group_of_row, &component)| {
                        *group_of_row = shuffled_labels[component];
                    });
                let f_stats = f_per_dim_fast(&X, &shuffled_groups, &ss_totals);
                let mut q: Vec<_> = sorted_dims.iter().map(|&i| f_stats[i]).collect();

                for i in 1..q.len() {
                    q[i] = q[i].max(q[i - 1])
                }
                q
            })
            .collect()
    };

    #[cfg(feature = "rayon")]
    let permutation_distribution: Vec<Vec<f64>> = {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        (0..num_permuations)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(permutation_seed + i as u64);
                let mut shuffled_labels = component_label.clone();
                shuffled_labels.shuffle(&mut rng);
                let shuffled_groups: Vec<_> = component_index_of_row
                    .iter()
                    .map(|&c| shuffled_labels[c])
                    .collect();
                let f_stats = f_per_dim_fast(&X, &shuffled_groups, &ss_totals);
                let mut q: Vec<_> = sorted_dims.iter().map(|&i| f_stats[i]).collect();

                for i in 1..q.len() {
                    q[i] = q[i].max(q[i - 1]);
                }
                // TODO: Also return f so we can report un-corrected p-value?
                q
            })
            .collect()
    };

    let mut results: Vec<_> = (0..sorted_dims.len())
        .map(|i| {
            let vals: Vec<f64> = permutation_distribution.iter().map(|v| v[i]).collect();

            let original_index = sorted_dims[i];
            let pseudo_f = f_statistics[original_index];

            let key = ordered_relation_types[original_index].clone();

            let result = if pseudo_f.is_nan() {
                PermAnovaResult {
                    pseudo_f,
                    permutation_distribution: vals,
                    p_value: 1.0,
                }
            } else {
                let p_value = vals.iter().filter(|&&perm_f| perm_f >= pseudo_f).count() as f64
                    / (num_permuations as f64);
                PermAnovaResult {
                    pseudo_f,
                    permutation_distribution: vals,
                    p_value,
                }
            };

            (key, result)
        })
        .collect();

    // Still sorted by their F-Statistic - now need to ensure that less significant
    // dimensions don't get a smaller p-value than more significant dimensions
    for i in (0..results.len() - 1).rev() {
        results[i].1.p_value = results[i].1.p_value.max(results[i + 1].1.p_value);
    }

    Ok(results.into_iter().collect())

    // Ok(permutation_distribution
    //     .into_iter()
    //     .enumerate()
    //     .map(|(i, vals)| {
    //         let original_index = sorted_dims[i];
    //         let pseudo_f = f_statistics[original_index];
    //
    //         let p_value = vals.iter().filter(|&&perm_f| perm_f >= pseudo_f).count() as f64
    //             / (num_permuations as f64);
    //
    //         let key = ordered_relation_types[original_index];
    //         (
    //             key,
    //             PermAnovaResult {
    //                 pseudo_f,
    //                 permutation_distribution: vals,
    //                 p_value,
    //             },
    //         )
    //     })
    //     .collect())
}

fn ss_total_per_dim(#[expect(non_snake_case)] X: &Array2<f64>) -> Vec<f64> {
    let n = X.nrows();
    let dims = X.ncols();

    let mut col_sums = vec![0.0; dims];
    let mut col_sq_sums = vec![0.0; dims];

    for row in X.rows() {
        for (j, &v) in row.iter().enumerate() {
            col_sums[j] += v;
            col_sq_sums[j] += v * v;
        }
    }

    col_sums
        .into_iter()
        .zip(col_sq_sums)
        .map(|(sum, sq_sum)| sq_sum - (sum * sum / n as f64))
        .collect()
}

fn f_per_dim_fast(
    #[expect(non_snake_case)] X: &Array2<f64>,
    groups: &[u8],
    ss_total_per_dim: &[f64],
) -> Vec<f64> {
    let n = X.nrows();
    let dims = X.ncols();

    let mut group_sums = [vec![0.0f64; dims], vec![0.0f64; dims]];
    let mut group_sq_sums = [vec![0.0f64; dims], vec![0.0f64; dims]];
    let mut group_sizes = [0usize; 2];

    for (i, row) in X.rows().into_iter().enumerate() {
        let group = groups[i] as usize;
        group_sizes[group] += 1;

        for (j, &v) in row.iter().enumerate() {
            group_sums[group][j] += v;
            group_sq_sums[group][j] += v * v;
        }
    }

    (0..dims)
        .map(|i| {
            let ss_within: f64 = group_sums
                .iter()
                .zip(group_sq_sums.iter())
                .zip(group_sizes.iter())
                .map(|((group_sum, group_sq_sum), group_size)| {
                    group_sq_sum[i] - (group_sum[i] * group_sum[i]) / (*group_size as f64)
                })
                .sum();

            let ss_between = ss_total_per_dim[i] - ss_within;

            ss_between / (ss_within / (n - 2) as f64)
        })
        .collect()
}
