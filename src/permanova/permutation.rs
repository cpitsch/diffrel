//! Logic for the permutation loop of the Permanova test
//! Shared by the generic distance function version and the fast euclidean implementation

use std::iter::repeat_n;

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::permanova::{PermAnovaResult, base::compact_component_ids};

/// Arguments used to compute `ss_within` in the permutation loop
pub(super) enum SsWithinArgs<'a> {
    DistanceMatrix(&'a Array2<f64>),
    #[allow(non_snake_case)]
    Euclidean {
        X: &'a Array2<f64>,
        sq_norms: &'a [f64],
    },
}

impl<'a> SsWithinArgs<'a> {
    pub(super) fn compute_ss_within(&self, groups: &[u8]) -> f64 {
        match self {
            Self::DistanceMatrix(d2) => super::base::compute_sum_of_squares_within(d2, groups),
            Self::Euclidean { X, sq_norms } => {
                super::fast_euclidean::compute_ss_within_fast(X, sq_norms, groups)
            }
        }
    }
}

pub(super) fn run_permanova_permutations(
    ss_within: impl Fn(&[u8]) -> f64 + Sync,
    ss_total: f64,
    component_ids_1: &[usize],
    component_ids_2: &[usize],
    num_permutations: usize,
    permutation_seed: Option<u64>,
) -> PermAnovaResult {
    let n1 = component_ids_1.len();
    let n2 = component_ids_2.len();
    let n = n1 + n2;

    if ss_total == 0.0 {
        // ss_within and ss_between will be 0, giving us pseudo_f NaN
        return PermAnovaResult {
            pseudo_f: f64::NAN,
            permutation_distribution: vec![f64::NAN; num_permutations],
            p_value: 1.0,
        };
    }

    let groups: Vec<u8> = repeat_n(0, n1).chain(repeat_n(1, n2)).collect();
    let pseudo_f = compute_pseudo_f(ss_within(&groups), ss_total, n);

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

        (0..num_permutations)
            .map(|_| {
                shuffled_labels.shuffle(&mut rng);
                shuffled_groups
                    .iter_mut()
                    .zip(&component_index_of_row)
                    .for_each(|(group_of_row, &component)| {
                        *group_of_row = shuffled_labels[component];
                    });
                let ss_within_value = ss_within(&shuffled_groups);
                compute_pseudo_f(ss_within_value, ss_total, n)
            })
            .collect()
    };

    #[cfg(feature = "rayon")]
    let permutation_distribution: Vec<f64> = {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        (0..num_permutations)
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
                let ss_within_value = ss_within(&shuffled_groups);
                compute_pseudo_f(ss_within_value, ss_total, n)
            })
            .collect()
    };

    let p_value = permutation_distribution
        .iter()
        .filter(|&&perm_f| perm_f >= pseudo_f)
        .count() as f64
        / (num_permutations as f64);

    PermAnovaResult {
        pseudo_f,
        permutation_distribution,
        p_value,
    }
}

fn compute_pseudo_f(ss_within: f64, ss_total: f64, n: usize) -> f64 {
    let ss_between = ss_total - ss_within;
    ss_between / (ss_within / (n - 2) as f64)
}
