use crate::relation_extraction::E2ERelationType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Distance {
    BrayCurtis,
    Euclidean,
}

// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// #[cfg_attr(feature = "serde", derive(serde::Serialize))]
// #[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
// pub enum Correction {
//     BonferroniHolm,
//     BenjaminiHochberg,
// }

// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// #[cfg_attr(feature = "serde", derive(serde::Serialize))]
// #[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
// pub enum QualifierMode {
//     Single,
//     Grouped,
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Grouping {
    Executions,
    Singletons,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonConfig {
    pub permutations: usize,
    pub alpha: f64,
    pub seed: Option<u64>,
    pub distance: Distance,
    // pub correction: Correction,
    pub grouping: Grouping,
    pub follows: E2ERelationType,
    // pub qualifier: QualifierMode,
    /// The object types considered for E2E relationships
    pub e2e_object_types: Option<Vec<String>>,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        ComparisonConfig {
            permutations: 10_000,
            alpha: 0.05,
            seed: None,
            distance: Distance::BrayCurtis,
            // correction: Correction::BonferroniHolm,
            grouping: Grouping::Executions,
            follows: E2ERelationType::DirectlyFollows,
            // qualifier: QualifierMode::Single,
            e2e_object_types: None,
        }
    }
}
