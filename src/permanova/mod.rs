//! An implementation of the Permutational Analysis of Variance (PERMANOVA) test
//! for an arbitrary cost function.

use std::{
    collections::{HashMap, HashSet},
    iter::repeat_n,
};

use ndarray::{Array2, Axis, concatenate};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use thiserror::Error;

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::relation_extraction::{OcelRelationshipCounter, RelationshipCount};

pub mod drilldown;
pub mod fast_euclidean;

#[derive(Debug, Clone)]
pub struct PermAnovaResult {
    /// The measured test statistic between the inputs
    pub pseudo_f: f64,
    pub permutation_distribution: Vec<f64>,
    pub p_value: f64,
}

#[derive(Debug, Clone, Error)]
pub enum PermAnovaError {
    #[error(
        "Both samples must have the same number of features. `sample_1` has {0} columns and `sample_2`has {1}"
    )]
    NumFeaturesMismatch(usize, usize),
    #[error(
        "Insufficient observations to compute the F-Statistic in the  Permanova test. Need at least 3 observations between both samples. Found {0}"
    )]
    InsufficientObservations(usize),
}

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
    sq_distance_function: impl Fn(&[f64], &[f64]) -> f64,
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

    #[expect(non_snake_case)]
    let X = concatenate(Axis(0), &[sample_1.view(), sample_2.view()]).unwrap();

    // Compute (squared) distance matrix
    let mut ss_total = 0.0;
    #[expect(non_snake_case)]
    let mut D2 = Array2::zeros((n, n));

    for i in 0..n {
        for j in i + 1..n {
            // let dist = squared_euclidean_distance(
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
    let pseudo_f = compute_pseudo_f(&D2, &groups, ss_total);

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
                compute_pseudo_f(&D2, &groups, ss_total)
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
                compute_pseudo_f(&D2, &shuffled_groups, ss_total)
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

/// Compute the `SS_W`statistic for the Pseudo-F statistic. For two groups.
///
/// * `groups`: 0 if that value belongs to group 1, 1 if it belongs to group two
///
/// TODO: The counts within groups never really change. Could pass them to skip
/// some computation
fn compute_sum_of_squares_within(#[expect(non_snake_case)] D2: &Array2<f64>, groups: &[u8]) -> f64 {
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

/// Compute the `SS_A` statistic as the difference between `SS_T` and `SS_W`
#[inline(always)]
fn compute_sum_of_squares_between(ss_total: f64, ss_within: f64) -> f64 {
    ss_total - ss_within
}

/// Compute the Pseudo-F Statistic
/// * `D2`: The matrix of squared distances.
/// * `groups`: Assignment of rows in X to groups. 0 if the row belongs to log 1,
///   1 if it belongs to log 2
/// * `ss_total`: The `SS_T` measure (`1/n * distances_for_i<j`)
fn compute_pseudo_f(
    #[expect(non_snake_case)] D2: &Array2<f64>,
    groups: &[u8],
    ss_total: f64,
) -> f64 {
    // p is number of groups, i.e., 2. Thus, `ss_between`, i.e., SS_A is divided by 2-1=1

    // SS_W
    let ss_within = compute_sum_of_squares_within(D2, groups);
    // SS_A
    let ss_between = compute_sum_of_squares_between(ss_total, ss_within);

    ss_between / (ss_within / (D2.nrows() - 2) as f64)
}

pub fn compare_counts<'a>(
    counts_1: &OcelRelationshipCounter<'a>,
    counts_2: &OcelRelationshipCounter<'a>,
    sq_distance_function: impl Fn(&[f64], &[f64]) -> f64,
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
        let res = permanova(
            &sample_1,
            &sample_2,
            &sq_distance_function,
            num_permuations,
            permutation_seed,
        )?;
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
