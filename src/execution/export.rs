//! Struct definitions for the comparison result export. This is used by the UI
//! and cli (and can, of course, be used via the crate).

use std::collections::HashSet;

use process_mining::core::event_data::object_centric::linked_ocel::LinkedOCELAccess;
use serde::{Deserialize, Serialize};

use crate::{
    execution::{config::ComparisonConfig, result::ComparisonResult},
    joint_ocdfg::JointOcdfg,
};

const CURRENT_SCHEMA_VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonResultExport {
    /// The version of the document format.
    /// Will (hopefully) be used to resolve incompatibilities of exports of different
    /// versions.
    pub schema_version: u8,
    pub params: ComparisonConfig,
    pub logs: ExportLogsInfo,
    pub disjoint_types: ExportLogsDisjointTypes,
    #[serde(flatten)]
    pub result: ComparisonResult,
    pub summary: ExportSummary,

    #[serde(
        rename = "processModel",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub process_model: Option<ProcessModel>,
}

impl ComparisonResultExport {
    pub fn new<'a, T>(
        result: ComparisonResult,
        config: &ComparisonConfig,
        log_1: &'a T,
        log_1_path: Option<String>,
        log_2: &'a T,
        log_2_path: Option<String>,
    ) -> Self
    where
        T: LinkedOCELAccess<'a> + Sync,
        T::ObjectRepr: Sync,
        T::EventRepr: Sync,
    {
        let e2e_types: Option<HashSet<&str>> = config
            .e2e_object_types
            .as_ref()
            .map(|types| types.iter().map(String::as_str).collect());

        let total_results = result.results.len();
        let significant_results = result.results.iter().filter(|r| r.test.significant).count();
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            params: config.clone(),
            logs: ExportLogsInfo {
                log_1: LogInfo::for_log(log_1, log_1_path),
                log_2: LogInfo::for_log(log_2, log_2_path),
            },
            disjoint_types: ExportLogsDisjointTypes::for_logs(log_1, log_2),
            result,
            summary: ExportSummary {
                total_results,
                significant_results,
            },

            process_model: Some(ProcessModel {
                dfg: JointOcdfg::discover(log_1, log_2, e2e_types.as_ref()),
                layout: None,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportLogsInfo {
    pub log_1: LogInfo,
    pub log_2: LogInfo,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogInfo {
    pub path: Option<String>,
    pub num_events: usize,
    pub num_event_types: usize,
    pub event_types: Vec<String>,
    pub num_objects: usize,
    pub num_object_types: usize,
    pub object_types: Vec<String>,
}

impl LogInfo {
    pub fn for_log<'a, T: LinkedOCELAccess<'a>>(log: &'a T, path: Option<String>) -> LogInfo {
        let event_types: Vec<_> = log.get_ev_types().map(str::to_string).collect();
        let object_types: Vec<_> = log.get_ob_types().map(str::to_string).collect();

        Self {
            path,
            num_events: log.get_num_evs(),
            num_event_types: event_types.len(),
            event_types,
            num_objects: log.get_num_obs(),
            num_object_types: object_types.len(),
            object_types,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportLogsDisjointTypes {
    pub event_types: DisjointTypeSet,
    pub object_types: DisjointTypeSet,
}

impl ExportLogsDisjointTypes {
    pub fn for_logs<'a, T: LinkedOCELAccess<'a>>(log_1: &'a T, log_2: &'a T) -> Self {
        /// Returns the items in a but not b, sorted and owned
        fn sorted_diff<'a>(a: HashSet<&'a str>, b: HashSet<&'a str>) -> Vec<String> {
            let mut v: Vec<_> = a.difference(&b).map(|s| s.to_string()).collect();
            v.sort_unstable();
            v
        }
        Self {
            event_types: DisjointTypeSet {
                log_1: sorted_diff(
                    log_1.get_ev_types().collect(),
                    log_2.get_ev_types().collect(),
                ),
                log_2: sorted_diff(
                    log_2.get_ev_types().collect(),
                    log_1.get_ev_types().collect(),
                ),
            },
            object_types: DisjointTypeSet {
                log_1: sorted_diff(
                    log_1.get_ob_types().collect(),
                    log_2.get_ob_types().collect(),
                ),
                log_2: sorted_diff(
                    log_2.get_ob_types().collect(),
                    log_1.get_ob_types().collect(),
                ),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisjointTypeSet {
    pub log_1: Vec<String>,
    pub log_2: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportSummary {
    pub total_results: usize,
    pub significant_results: usize,
}

/// The `processModel` section of an export. Contains the joint OC-DFG, and the
/// layouting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessModel {
    pub dfg: JointOcdfg,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<serde_json::Value>,
}
