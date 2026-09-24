use thiserror::Error;

pub mod base;
pub mod drilldown;
pub mod fast_euclidean;

#[derive(Debug, Clone)]
pub struct PermAnovaResult {
    /// The measured test statistic between the inputs
    pub pseudo_f: f64,
    pub permutation_distribution: Vec<f64>,
    pub p_value: f64,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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
