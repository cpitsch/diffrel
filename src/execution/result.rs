use thiserror::Error;

use crate::{execution::instance::ComparisonInstance, permanova::PermAnovaError};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ComparisonResult {
    pub results: Vec<ComparisonInstanceResult>,
    pub skipped: Vec<SkippedInstanceResult>,
    /// If no seed was provided, one is generated.
    pub seed_used: u64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ComparisonInstanceResult {
    pub instance: ComparisonInstance,
    pub n1: usize,
    pub n2: usize,
    pub test: TestResult,
    pub effect_size: Option<f64>,
    #[cfg_attr(
        feature = "serde",
        serde(serialize_with = "serialize_drilldown_result_option")
    )]
    pub drilldown: Option<DrilldownResult>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TestResult {
    pub pseudo_f: f64,
    pub p_value: f64,
    pub significant: bool,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SkippedInstanceResult {
    pub instance: ComparisonInstance,
    // TODO: Make it an enum.
    pub reason: SkipReason,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum SkipReason {
    OnlyInLogOne,
    OnlyInLogTwo,
    NoRelationsObserved,
    TestFailed(PermAnovaError),
}

pub type DrilldownResult = Result<Vec<DrilldownRelationResult>, DrilldownError>;
// #[derive(Debug, Clone, PartialEq)]
// #[cfg_attr(feature = "serde", derive(serde::Serialize))]
// pub struct DrilldownResult {
//     pub relations: Vec<DrilldownRelationResult>,
//     pub error: Option<PermAnovaError>,
// }

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum DrilldownError {
    #[error("Group key '{0}' has no entry in the extracted profile for log one.")]
    GroupKeyNotInProfileOne(String),
    #[error("Group key '{0}' has no entry in the extracted profile for log two.")]
    GroupKeyNotInProfileTwo(String),
    #[error(transparent)]
    PermAnovaError(#[from] PermAnovaError),
}

#[cfg(feature = "serde")]
fn serialize_drilldown_result_option<S: serde::Serializer>(
    drilldown_result: &Option<DrilldownResult>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::Serialize;

    #[derive(serde::Serialize)]
    struct Helper<'a> {
        relations: &'a [DrilldownRelationResult],
        error: Option<&'a DrilldownError>,
    }

    let item = drilldown_result.as_ref().map(|result| match result {
        Ok(relations) => Helper {
            relations,
            error: None,
        },
        Err(e) => Helper {
            relations: &[],
            error: Some(e),
        },
    });
    item.serialize(serializer)
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct DrilldownRelationResult {
    pub qualifier: String,
    pub target: String,
    pub test: TestResult,
    pub effect_size: f64,
}
