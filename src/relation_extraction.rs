use std::{
    collections::{BTreeSet, HashMap},
    hash::Hash,
    ops::{Deref, DerefMut},
};

use ndarray::Array1;
use process_mining::core::event_data::object_centric::linked_ocel::LinkedOCELAccess;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Qualifier<'a> {
    Single(&'a str),
    Multiple(BTreeSet<&'a str>),
}

impl std::fmt::Display for Qualifier<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Qualifier::Single(s) => write!(f, "{s}"),
            Qualifier::Multiple(set) => {
                write!(f, "{}", set.iter().copied().collect::<Vec<_>>().join(","))
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RelationshipCount<'a>(pub HashMap<(Qualifier<'a>, &'a str), usize>);

impl<'a> Deref for RelationshipCount<'a> {
    type Target = HashMap<(Qualifier<'a>, &'a str), usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RelationshipCount<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl RelationshipCount<'_> {
    /// Encode the [`RelationshipCount`] as an Array of usize. Only considers
    /// relationship types contained in `relationship_types`, and of they are not in
    /// `self`, their count is assumed to be 0 (not observed).
    pub fn encoded(&self, relationship_types: &[(Qualifier<'_>, &str)]) -> Array1<usize> {
        relationship_types
            .iter()
            .map(|rel_type| self.get(rel_type).copied().unwrap_or_default())
            .collect()
    }

    /// Encode the [`RelationshipCount`] as an Array of floats. Only considers
    /// relationship types contained in `relationship_types`, and of they are not in
    /// `self`, their count is assumed to be 0 (not observed).
    ///
    /// (To be used to set a row of a matrix in permanova.)
    pub fn encoded_f64(&self, relationship_types: &[(Qualifier<'_>, &str)]) -> Array1<f64> {
        relationship_types
            .iter()
            .map(|rel_type| self.get(rel_type).copied().unwrap_or_default() as f64)
            .collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcelRelationshipCounter<'a>(pub HashMap<&'a str, Vec<RelationshipCount<'a>>>);

impl<'a> Deref for OcelRelationshipCounter<'a> {
    type Target = HashMap<&'a str, Vec<RelationshipCount<'a>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for OcelRelationshipCounter<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub fn extract_e2o_relation_counts<'a>(
    linked_ocel: &'a impl LinkedOCELAccess<'a>,
) -> OcelRelationshipCounter<'a> {
    OcelRelationshipCounter(
        linked_ocel
            .get_ev_types()
            .map(|etype| {
                (
                    etype,
                    linked_ocel
                        .get_evs_of_type(etype)
                        .map(|event| {
                            let mut counts = RelationshipCount::default();
                            linked_ocel.get_e2o(event).for_each(|(qualifier, obj)| {
                                let otype = linked_ocel.get_ob_type_of(obj);
                                *counts
                                    .entry((Qualifier::Single(qualifier), otype))
                                    .or_default() += 1;
                            });
                            counts
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

pub fn extract_o2o_relation_counts<'a>(
    linked_ocel: &'a impl LinkedOCELAccess<'a>,
) -> OcelRelationshipCounter<'a> {
    OcelRelationshipCounter(
        linked_ocel
            .get_ob_types()
            .map(|otype| {
                (
                    otype,
                    linked_ocel
                        .get_obs_of_type(otype)
                        .map(|object| {
                            let mut counts = RelationshipCount::default();

                            linked_ocel.get_o2o(object).for_each(|(qualifier, obj)| {
                                let otype = linked_ocel.get_ob_type_of(obj);
                                *counts
                                    .entry((Qualifier::Single(qualifier), otype))
                                    .or_default() += 1;
                            });
                            counts
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

pub fn extract_o2e_relation_counts<'a>(
    linked_ocel: &'a impl LinkedOCELAccess<'a>,
) -> OcelRelationshipCounter<'a> {
    OcelRelationshipCounter(
        linked_ocel
            .get_ob_types()
            .map(|otype| {
                (
                    otype,
                    linked_ocel
                        .get_obs_of_type(otype)
                        .map(|object| {
                            let mut counts = RelationshipCount::default();

                            linked_ocel
                                .get_e2o_rev(object)
                                .for_each(|(qualifier, evt)| {
                                    let etype = linked_ocel.get_ev_type_of(evt);
                                    *counts
                                        .entry((Qualifier::Single(qualifier), etype))
                                        .or_default() += 1;
                                });
                            counts
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

// /// Two events are in an e2e relationship if they share an object and the first
// /// event occurred (directly) before the second.
// /// The relation is counted for _each shared object_. That is, if, e.g., an order is
// /// placed for multiple items, and later shipped (for those items), that eventually
// /// follows relation is counted for _each one of the items_.
// /// The qualifier of that relationship is the object type.
// // TODO: Option for directly follows?
// // TODO: Or distance? 1, ... (negative / 0 means infinite?)
// pub fn extract_e2e_relation_counts<'a, T>(linked_ocel: &'a T) -> OcelRelationshipCounter<'a>
// where
//     T: LinkedOCELAccess<'a>,
//     T::EventRepr: Eq + Hash + Copy,
// {
//     // Basically, check every object's O2E relationships
//     // TODO: event -> event index -> relcount (simplifies conversion)
//     let mut event_to_rels: HashMap<T::EventRepr, RelationshipCount<'a>> = HashMap::new();
//
//     for obj in linked_ocel.get_all_obs() {
//         let otype = linked_ocel.get_ob_type_of(&obj);
//         let mut o2e: Vec<_> = linked_ocel.get_e2o_rev(obj).collect();
//         // Sort events by timestamp -> every event is related to all following ones.
//         o2e.sort_by_key(|ev| linked_ocel.get_ev_time(ev.1));
//
//         for i in 0..(o2e.len() - 1) {
//             let ev1 = o2e[i];
//             for ev2 in &o2e[(i + 1)..] {
//                 let etype_2 = linked_ocel.get_ev_type_of(ev2.1);
//
//                 *event_to_rels
//                     .entry(*ev1.1)
//                     .or_default()
//                     .entry((Qualifier::Single(otype), etype_2))
//                     .or_default() += 1;
//             }
//         }
//     }
//
//     let mut counter = OcelRelationshipCounter::default();
//     for (ev_index, rels) in event_to_rels {
//         let etype = linked_ocel.get_ev_type_of(ev_index);
//         counter.entry(etype).or_default().push(rels);
//     }
//
//     counter
// }

/// Two events are in an e2e relationship if they share an object and the first
/// event occurred  before the second.
/// In that case, an E2E relationship is extracted with the exact combination of
/// object types as qualifier.
pub fn extract_e2e_relation_counts<'a, T>(linked_ocel: &'a T) -> OcelRelationshipCounter<'a>
where
    T: LinkedOCELAccess<'a>,
    T::EventRepr: Eq + Hash + Copy,
{
    let mut paired_events: HashMap<
        (T::EventRepr, T::EventRepr),
        std::collections::BTreeSet<&'a str>,
    > = HashMap::new();

    // TODO: Could get all object types then get_obs_of_type, skips the repeated
    // type lookup? Depends on how get_obs_of_type is implemented - might have separate
    // sets for each type, or might do a type lookup for each object, in which case
    // this is worse
    for obj in linked_ocel.get_all_obs() {
        let otype = linked_ocel.get_ob_type_of(&obj);
        let mut o2e: Vec<_> = linked_ocel.get_e2o_rev(obj).collect();
        // Sort events by timestamp -> every event is related to all following ones.
        o2e.sort_by_key(|ev| linked_ocel.get_ev_time(ev.1));

        for i in 0..o2e.len().saturating_sub(1) {
            let ev1 = *o2e[i].1;
            for ev2 in &o2e[(i + 1)..] {
                paired_events
                    .entry((ev1, *ev2.1))
                    .or_default()
                    .insert(otype);
            }
        }
    }

    let mut event_to_rels: HashMap<T::EventRepr, RelationshipCount<'a>> = HashMap::new();
    for ((ev1, ev2), otypes) in paired_events {
        let etype_2 = linked_ocel.get_ev_type_of(ev2);

        *event_to_rels
            .entry(ev1)
            .or_default()
            .entry((Qualifier::Multiple(otypes), etype_2))
            .or_default() += 1;
    }

    let mut counter = OcelRelationshipCounter::default();
    for (ev_index, rels) in event_to_rels {
        let etype = linked_ocel.get_ev_type_of(ev_index);
        counter.entry(etype).or_default().push(rels);
    }

    counter
}
