use ndarray::Array2;
use thiserror::Error;

use crate::distances::squared_bray_curtis_distance;

pub mod base;
pub mod drilldown;
pub mod fast_euclidean;
mod permutation;

#[derive(Debug, Clone)]
pub struct PermAnovaResult {
    /// The measured test statistic between the inputs
    pub pseudo_f: f64,
    pub permutation_distribution: Vec<f64>,
    pub p_value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Distance {
    BrayCurtis,
    Euclidean,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

pub fn restricted_permanova(
    sample_1: &Array2<f64>,
    sample_2: &Array2<f64>,
    component_ids_1: &[usize],
    component_ids_2: &[usize],
    distance: Distance,
    num_permutations: usize,
    permutation_seed: Option<u64>,
) -> Result<PermAnovaResult, PermAnovaError> {
    match distance {
        Distance::BrayCurtis => base::restricted_permanova(
            sample_1,
            sample_2,
            component_ids_1,
            component_ids_2,
            squared_bray_curtis_distance,
            num_permutations,
            permutation_seed,
        ),
        Distance::Euclidean => fast_euclidean::restricted_permanova(
            sample_1,
            sample_2,
            component_ids_1,
            component_ids_2,
            num_permutations,
            permutation_seed,
        ),
    }
}

pub fn permanova(
    sample_1: &Array2<f64>,
    sample_2: &Array2<f64>,
    distance: Distance,
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
        distance,
        num_permutations,
        permutation_seed,
    )
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::{
        distances::squared_euclidean_distance,
        permanova::{base, fast_euclidean},
    };

    /// Fast euclidean just uses some tricks to compute the results faster.
    /// (specifically `ss_within` and, by extension, `pseudo_f`). The result, however,
    /// should be identical provided the same permutations are used.
    #[test]
    fn test_euclidean_fast_vs_base_same_result() {
        let sample_1 = array![[1.0, 2.0], [2.0, 1.0], [5.0, 5.0], [6.0, 4.0]];
        let sample_2 = array![[1.5, 2.5], [2.5, 1.5], [5.5, 4.5], [6.5, 5.5]];
        let component_ids_1 = [0, 0, 1, 1];
        let component_ids_2 = [0, 0, 1, 1];

        let fast = fast_euclidean::restricted_permanova(
            &sample_1,
            &sample_2,
            &component_ids_1,
            &component_ids_2,
            500,
            Some(42),
        )
        .unwrap();
        let generic = base::restricted_permanova(
            &sample_1,
            &sample_2,
            &component_ids_1,
            &component_ids_2,
            squared_euclidean_distance,
            500,
            Some(42),
        )
        .unwrap();

        assert!((fast.pseudo_f - generic.pseudo_f).abs() < 1e-9);
        assert!((fast.p_value - generic.p_value).abs() < 1e-9);
    }
}
