//! Implementation of joint OCDFG by discovering an OCDFG for each log. Nodes and arcs
//! are then merged.
//!
//! [`BTreeMap`]s are used to ensure deterministic output

use std::collections::{BTreeMap, BTreeSet, HashSet};

use process_mining::core::{
    event_data::object_centric::linked_ocel::LinkedOCELAccess,
    process_models::ocdfg::discover_ocdfg_from,
};

/// A directly-follows arc of an object type, with its count in each log (0 if it
/// doesn't occur in that log)
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct JointDfArc {
    pub from: String,
    pub to: String,
    /// The object type of the arc. A one-element list for backwards-compatibility
    pub qualifier: Vec<String>,
    pub count_log_1: u32,
    pub count_log_2: u32,
}

/// Merged activity counts and start/end activities of an object type
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct ObjectTypeNodes {
    /// Maps each activity to the larger count between the two logs
    pub activities: BTreeMap<String, u32>,
    /// Sorted union of both logs' start activities
    pub start_activities: Vec<String>,
    /// Sorted union of both logs' end activities
    pub end_activities: Vec<String>,
}

/// Joint OCDFG: union of the OCDFGs of two logs, with per-log counts of arcs, and
/// max(l1, l2) for activity counts.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct JointOcdfg {
    pub object_type_to_dfg: BTreeMap<String, ObjectTypeNodes>,
    /// Arcs of the joint OCDFG, sorted by (qualifier, from, to)
    #[cfg_attr(feature = "serde", serde(alias = "edges"))]
    pub arcs: Vec<JointDfArc>,
}

impl JointOcdfg {
    /// Discover the joint OCDFG from two logs. Arc counts are stored per log, and
    /// activity counts are the maximum between the two OCDFGs.
    pub fn discover<'a, T>(log_1: &'a T, log_2: &'a T, object_types: Option<&HashSet<&str>>) -> Self
    where
        T: LinkedOCELAccess<'a> + Sync,
        T::ObjectRepr: Sync,
        T::EventRepr: Sync,
    {
        let ocdfg_1 = discover_ocdfg_from(log_1);
        let ocdfg_2 = discover_ocdfg_from(log_2);

        let otypes: BTreeSet<_> = ocdfg_1
            .object_type_to_dfg
            .keys()
            .chain(ocdfg_2.object_type_to_dfg.keys())
            .map(|x| x.as_str())
            .filter(|&t| object_types.is_none_or(|x| x.contains(t)))
            .collect();

        let mut object_type_to_dfg = BTreeMap::new();
        let mut arcs = Vec::new();

        for t in otypes {
            let dfg_1 = ocdfg_1.object_type_to_dfg.get(t);
            let dfg_2 = ocdfg_2.object_type_to_dfg.get(t);
            let all = || dfg_1.into_iter().chain(dfg_2);

            let mut activities = BTreeMap::new();
            for (act, &count) in all().flat_map(|d| &d.activities) {
                let entry = activities.entry(act.clone()).or_default();
                *entry = u32::max(*entry, count);
            }

            let start_activities: BTreeSet<&String> =
                all().flat_map(|d| d.start_activities.keys()).collect();
            let end_activities: BTreeSet<&String> =
                all().flat_map(|d| d.end_activities.keys()).collect();

            object_type_to_dfg.insert(
                t.to_string(),
                ObjectTypeNodes {
                    activities,
                    start_activities: start_activities.into_iter().cloned().collect(),
                    end_activities: end_activities.into_iter().cloned().collect(),
                },
            );

            // from,to -> count_1, count_2
            let mut arc_counts: BTreeMap<(&str, &str), (u32, u32)> = BTreeMap::new();
            for ((from, to), &count) in dfg_1
                .into_iter()
                .flat_map(|d| &d.directly_follows_relations)
            {
                arc_counts.entry((from, to)).or_default().0 = count;
            }
            for ((from, to), &count) in dfg_2
                .into_iter()
                .flat_map(|d| &d.directly_follows_relations)
            {
                arc_counts.entry((from, to)).or_default().1 = count;
            }

            arcs.extend(
                arc_counts
                    .into_iter()
                    .map(|((from, to), (count_log_1, count_log_2))| JointDfArc {
                        from: from.to_string(),
                        to: to.to_string(),
                        qualifier: vec![t.to_string()],
                        count_log_1,
                        count_log_2,
                    }),
            );
        }

        Self {
            object_type_to_dfg,
            arcs,
        }
    }
}
