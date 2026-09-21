//! An implementation of the Permutational Analysis of Variance (PERMANOVA) test
//! for an arbitrary cost function.
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    iter::repeat_n,
};

use ndarray::{Array2, Axis, concatenate};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    permanova::{PermAnovaError, PermAnovaResult},
    relation_extraction::{Qualifier, RelationshipCount},
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

    if ss_total == 0.0 {
        // ss_within and ss_between will be 0, giving us pseudo_f NaN
        return Ok(PermAnovaResult {
            pseudo_f: f64::NAN,
            permutation_distribution: vec![f64::NAN; num_permuations],
            p_value: 1.0,
        });
    }

    let groups: Vec<u8> = repeat_n(0, n1).chain(repeat_n(1, n2)).collect();

    // F
    let pseudo_f = compute_pseudo_f(&D2, &groups, ss_total);

    // Precompute translation from vec of component labels to vec of items
    let (component_ids_1, num_components_1) = compact_component_ids(component_ids_1);
    let (component_ids_2, num_components_2) = compact_component_ids(component_ids_2);
    let component_index_of_row: Vec<usize> = component_ids_1
        .iter()
        .copied()
        .chain(component_ids_2.iter().map(|&c| c + num_components_1))
        .collect();

    // This is what we shuffle
    let component_label: Vec<u8> = repeat_n(0u8, num_components_1)
        .chain(repeat_n(1u8, num_components_2))
        .collect();

    #[cfg(not(feature = "rayon"))]
    let permutation_distribution: Vec<f64> = {
        let mut rng = if let Some(seed) = permutation_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_os_rng()
        };

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
                compute_pseudo_f(&D2, &shuffled_groups, ss_total)
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
                let mut shuffled_labels = component_label.clone();
                shuffled_labels.shuffle(&mut rng);
                let shuffled_groups: Vec<_> = component_index_of_row
                    .iter()
                    .map(|&c| shuffled_labels[c])
                    .collect();
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

pub fn compare_counts<'a, T>(
    counts_1: &HashMap<&'a str, Vec<(T, RelationshipCount<'a>)>>,
    counts_2: &HashMap<&'a str, Vec<(T, RelationshipCount<'a>)>>,
    components_1: &HashMap<T, usize>,
    components_2: &HashMap<T, usize>,
    sq_distance_function: impl Fn(&[f64], &[f64]) -> f64,
    num_permuations: usize,
    permutation_seed: Option<u64>,
) -> Result<HashMap<&'a str, PermAnovaResult>, PermAnovaError>
where
    T: Eq + Hash + Copy,
{
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
            .flat_map(|(_, x)| x.keys())
            .collect::<HashSet<_>>()
            .into_iter()
            .cloned()
            .collect();

        let sample_1 = encode_relation_counts(rel_counts_1, &ordered_relation_types);
        let sample_2 = encode_relation_counts(rel_counts_2, &ordered_relation_types);

        let component_ids_1: Vec<usize> =
            rel_counts_1.iter().map(|(t, _)| components_1[t]).collect();
        let component_ids_2: Vec<usize> =
            rel_counts_2.iter().map(|(t, _)| components_2[t]).collect();

        let start = std::time::Instant::now();
        let res = restricted_permanova(
            &sample_1,
            &sample_2,
            &component_ids_1,
            &component_ids_2,
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
pub fn encode_relation_counts<T>(
    relation_counts: &[(T, RelationshipCount<'_>)],
    ordered_relation_types: &[(Qualifier<'_>, &'_ str)],
) -> Array2<f64> {
    let mut res = Array2::zeros((relation_counts.len(), ordered_relation_types.len()));
    for (idx, (_, counts)) in relation_counts.iter().enumerate() {
        res.row_mut(idx)
            .assign(&counts.encoded_f64(ordered_relation_types));
    }
    res
}
