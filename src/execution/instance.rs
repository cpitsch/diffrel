use std::collections::HashSet;

use process_mining::core::event_data::object_centric::linked_ocel::LinkedOCELAccess;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RelationMode {
    E2O,
    O2O,
    O2E,
    E2E,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComparisonInstance {
    pub mode: RelationMode,
    /// The name of the "central entity" of the comparison. For E2O/E2E this is an
    /// event type. For O2O/O2E this is an object type.
    pub group_key: String,
    /// A filter on the targets of relationships. `None` imposes no filter.
    pub target_types: Option<Vec<String>>,
}

impl ComparisonInstance {
    pub fn all<'a, T: LinkedOCELAccess<'a>>(log_1: &'a T, log_2: &'a T) -> Vec<Self> {
        let mut instances = Self::all_e2o(log_1, log_2);
        instances.extend(Self::all_o2o(log_1, log_2));
        instances.extend(Self::all_o2e(log_1, log_2));
        instances.extend(Self::all_e2e(log_1, log_2));
        instances
    }

    pub fn all_e2o<'a, T: LinkedOCELAccess<'a>>(log_1: &'a T, log_2: &'a T) -> Vec<Self> {
        Self::from_event_types(log_1, log_2, RelationMode::E2O)
    }
    pub fn all_o2o<'a, T: LinkedOCELAccess<'a>>(log_1: &'a T, log_2: &'a T) -> Vec<Self> {
        Self::from_object_types(log_1, log_2, RelationMode::O2O)
    }
    pub fn all_o2e<'a, T: LinkedOCELAccess<'a>>(log_1: &'a T, log_2: &'a T) -> Vec<Self> {
        Self::from_object_types(log_1, log_2, RelationMode::O2E)
    }
    pub fn all_e2e<'a, T: LinkedOCELAccess<'a>>(log_1: &'a T, log_2: &'a T) -> Vec<Self> {
        Self::from_event_types(log_1, log_2, RelationMode::E2E)
    }

    fn from_object_types<'a, T: LinkedOCELAccess<'a>>(
        log_1: &'a T,
        log_2: &'a T,
        mode: RelationMode,
    ) -> Vec<Self> {
        let types_1: HashSet<_> = log_1.get_ob_types().collect();
        let types_2: HashSet<_> = log_2.get_ob_types().collect();
        let mut types: Vec<_> = types_1.intersection(&types_2).copied().collect();
        types.sort_unstable();
        types
            .into_iter()
            .map(|otype| Self {
                mode,
                group_key: otype.to_string(),
                target_types: None,
            })
            .collect()
    }

    fn from_event_types<'a, T: LinkedOCELAccess<'a>>(
        log_1: &'a T,
        log_2: &'a T,
        mode: RelationMode,
    ) -> Vec<Self> {
        let types_1: HashSet<_> = log_1.get_ev_types().collect();
        let types_2: HashSet<_> = log_2.get_ev_types().collect();
        let mut types: Vec<_> = types_1.intersection(&types_2).copied().collect();
        types.sort_unstable();
        types
            .into_iter()
            .map(|etype| Self {
                mode,
                group_key: etype.to_string(),
                target_types: None,
            })
            .collect()
    }
}
