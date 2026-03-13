use std::{
    collections::{HashMap, HashSet},
    iter::repeat_n,
};

use ndarray::{Array2, Axis, concatenate};
use process_mining::OCEL;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use thiserror::Error;

use crate::per_relation::{OcelEventRelationshipCount, extract_e2o_relation_counts};

#[derive(Debug, Clone)]
pub struct PermAnovaResult {
    /// The test-statistic measured between the t
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
}

/// Perform a Permanova test between two multivariate samples. Each row in the matrices
/// corresponds to one instance and columns represent features. As such:
///
///     nrows = number of observations
///     ncols = number of features
pub fn permanova(
    mut sample_1: Array2<f64>,
    mut sample_2: Array2<f64>,
    num_permuations: usize,
    permutation_seed: Option<u64>,
) -> Result<PermAnovaResult, PermAnovaError> {
    if sample_1.ncols() != sample_2.ncols() {
        return Err(PermAnovaError::NumFeaturesMismatch(
            sample_1.ncols(),
            sample_2.ncols(),
        ));
    }

    if sample_1.nrows() > sample_2.nrows() {
        std::mem::swap(&mut sample_1, &mut sample_2)
    }

    let n1 = sample_1.nrows();
    let n2 = sample_2.nrows();
    let n = n1 + n2;

    #[allow(non_snake_case)]
    let X = concatenate(Axis(0), &[sample_1.view(), sample_2.view()]).unwrap();

    #[cfg_attr(feature = "rayon", allow(unused_mut))]
    let mut groups: Vec<u8> = repeat_n(0, n1).chain(repeat_n(1, n2)).collect();

    // Compute squared distance matrix
    // TODO: could use Array2::uninit if n is very large
    let mut ss_total = 0.0;
    #[allow(non_snake_case)]
    let mut D2 = Array2::zeros((n, n));
    for i in 0..n {
        for j in i + 1..n {
            let dist = squared_euclidean_distance(
                X.row(i).as_slice().unwrap(),
                X.row(j).as_slice().unwrap(),
            );
            ss_total += dist;
            D2[(i, j)] = dist;
            D2[(j, i)] = dist;
        }
    }

    // SS_T
    ss_total /= D2.nrows() as f64;
    assert_eq!(ss_total, compute_total_sum_of_squares(&D2));

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
                groups.partial_shuffle(&mut rng, sample_1.nrows());
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
                shuffled_groups.partial_shuffle(&mut rng, n1);
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

fn squared_euclidean_distance(v1: &[f64], v2: &[f64]) -> f64
where
{
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
}

/// Takes D2, the matrix containing squared distances, and computes the total
/// sum of squares.
#[allow(dead_code)]
fn compute_total_sum_of_squares(#[allow(non_snake_case)] D2: &Array2<f64>) -> f64 {
    let mut total = 0.0;
    let n = D2.nrows();
    for i in 0..n - 1 {
        for j in (i + 1)..n {
            total += D2[(i, j)]
        }
    }

    total / (n as f64)
    // D.iter().map(|&d| d * d).sum::<f64>() / (D.nrows() as f64)
}

/// Compute the `SS_W`statistic for the Pseudo-F statistic. For two groups.
///
/// * `groups`: 0 if that value belongs to group 1, 1 if it belongs to group two
///
/// TODO: The counts within groups never really change. Could pass them to skip
/// some computation
fn compute_sum_of_squares_within(#[allow(non_snake_case)] D2: &Array2<f64>, groups: &[u8]) -> f64 {
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

fn compute_pseudo_f(
    #[allow(non_snake_case)] D2: &Array2<f64>,
    groups: &[u8],
    ss_total: f64,
) -> f64 {
    // p is number of groups, i.e., 2. Thus, `ss_between`, i.e., SS_A is divided py 2-1=1

    // SS_W
    let ss_within = compute_sum_of_squares_within(D2, groups);
    // SS_A
    let ss_between = compute_sum_of_squares_between(ss_total, ss_within);

    ss_between / (ss_within / (D2.nrows() - 2) as f64)
}

/// Compare the E2O relationship cardinalities between two event logs using a
/// Permanova test. Returns a hashmap of event type -> p-value
///
/// If an event type only occurs in one of the event logs, no p-value is computed,
/// i.e., the event type is _not_ contained as a key in the hashmap.
pub fn compare<'a>(
    log_1: &'a OCEL,
    log_2: &'a OCEL,
    num_permuations: usize,
    permutation_seed: Option<u64>,
) -> Result<HashMap<&'a str, PermAnovaResult>, PermAnovaError> {
    let counts_1 = extract_e2o_relation_counts(log_1);
    let counts_2 = extract_e2o_relation_counts(log_2);

    let all_event_types: HashSet<_> = counts_1.keys().chain(counts_2.keys()).copied().collect();

    let mut event_type_to_pval = HashMap::new();

    // TODO: Could parallelize? But right now, I parallelize across permutations
    for event_type in all_event_types {
        let (Some(rel_counts_1), Some(rel_counts_2)) =
            (counts_1.get(event_type), counts_2.get(event_type))
        else {
            log::info!("{event_type} only in one log");
            continue;
        };

        let ordered_relation_types: Vec<_> = rel_counts_1
            .iter()
            .chain(rel_counts_2.iter())
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>()
            .into_iter()
            .cloned()
            .collect();

        let sample_1 = encode_relation_counts(rel_counts_1, &ordered_relation_types);
        let sample_2 = encode_relation_counts(rel_counts_2, &ordered_relation_types);

        let start = std::time::Instant::now();
        let res = permanova(sample_1, sample_2, num_permuations, permutation_seed)?;
        log::trace!("{event_type} took {:?}", start.elapsed());

        event_type_to_pval.insert(event_type, res);
    }

    Ok(event_type_to_pval)
}

fn encode_relation_counts(
    relation_counts: &[OcelEventRelationshipCount<'_>],
    ordered_relation_types: &[(&'_ str, &'_ str)],
) -> Array2<f64> {
    let mut res = Array2::zeros((relation_counts.len(), ordered_relation_types.len()));
    for (idx, counts) in relation_counts.iter().enumerate() {
        res.row_mut(idx)
            .assign(&counts.encoded_f64(ordered_relation_types));
    }
    res
}
