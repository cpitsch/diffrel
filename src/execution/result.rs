use thiserror::Error;

use crate::{execution::instance::ComparisonInstance, permanova::PermAnovaError};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonResult {
    pub results: Vec<ComparisonInstanceResult>,
    pub skipped: Vec<SkippedInstanceResult>,
    /// If no seed was provided, one is generated.
    pub seed_used: u64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonInstanceResult {
    pub instance: ComparisonInstance,
    pub n1: usize,
    pub n2: usize,
    pub test: TestResult,
    pub effect_size: Option<f64>,
    #[cfg_attr(
        feature = "serde",
        serde(
            serialize_with = "serialize_drilldown_result_option",
            deserialize_with = "deserialize_drilldown_result_option"
        )
    )]
    pub drilldown: Option<DrilldownResult>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TestResult {
    #[cfg_attr(feature = "serde", serde(deserialize_with = "deserialize_f64_or_nan"))]
    pub pseudo_f: f64,
    pub p_value: f64,
    pub significant: bool,
}

/// f64::NAN serializes to `null`, but `null` does not deserialize to a float...
#[cfg(feature = "serde")]
fn deserialize_f64_or_nan<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    use serde::Deserialize;
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SkippedInstanceResult {
    pub instance: ComparisonInstance,
    // TODO: Make it an enum.
    pub reason: SkipReason,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum SkipReason {
    OnlyInLogOne,
    OnlyInLogTwo,
    NoRelationsObserved,
    TestFailed(PermAnovaError),
}

pub type DrilldownResult = Result<Vec<DrilldownRelationResult>, DrilldownError>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum DrilldownError {
    #[error("Group key '{0}' has no entry in the extracted profile for log one.")]
    GroupKeyNotInProfileOne(String),
    #[error("Group key '{0}' has no entry in the extracted profile for log two.")]
    GroupKeyNotInProfileTwo(String),
    #[error(transparent)]
    PermAnovaError(#[from] PermAnovaError),
}

/// Just a helper struct for conversion from and to the more "json-native" format
/// I'd prefer for exports (better than Ok(...) and Err(...) in the export)
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct DrilldownResultSerdeHelper {
    relations: Vec<DrilldownRelationResult>,
    error: Option<DrilldownError>,
}

impl From<DrilldownResult> for DrilldownResultSerdeHelper {
    fn from(value: DrilldownResult) -> Self {
        match value {
            Ok(relations) => Self {
                relations,
                error: None,
            },
            Err(e) => Self {
                relations: vec![],
                error: Some(e),
            },
        }
    }
}

impl From<DrilldownResultSerdeHelper> for DrilldownResult {
    fn from(value: DrilldownResultSerdeHelper) -> Self {
        if let Some(e) = value.error {
            Err(e)
        } else {
            Ok(value.relations)
        }
    }
}

#[cfg(feature = "serde")]
fn serialize_drilldown_result_option<S: serde::Serializer>(
    v: &Option<DrilldownResult>,
    s: S,
) -> Result<S::Ok, S::Error> {
    use serde::Serialize;
    v.clone().map(DrilldownResultSerdeHelper::from).serialize(s)
}

#[cfg(feature = "serde")]
fn deserialize_drilldown_result_option<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<DrilldownResult>, D::Error> {
    use serde::Deserialize;
    Ok(Option::<DrilldownResultSerdeHelper>::deserialize(d)?.map(DrilldownResult::from))
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DrilldownRelationResult {
    pub qualifier: String,
    pub target: String,
    pub test: TestResult,
    pub effect_size: f64,
}
