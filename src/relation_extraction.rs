use std::{
    collections::{BTreeSet, HashMap, HashSet},
    hash::Hash,
    ops::{Deref, DerefMut},
};

use ndarray::Array1;
use process_mining::core::event_data::object_centric::linked_ocel::LinkedOCELAccess;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

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
                write!(f, "{}", set.iter().copied().collect::<Vec<_>>().join(", "))
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

#[derive(Debug, Clone)]
pub struct OcelEventRelationshipCounter<'a, E>(
    pub HashMap<&'a str, Vec<(E, RelationshipCount<'a>)>>,
);

impl<'a, E> Deref for OcelEventRelationshipCounter<'a, E> {
    type Target = HashMap<&'a str, Vec<(E, RelationshipCount<'a>)>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<E> DerefMut for OcelEventRelationshipCounter<'_, E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, E> Default for OcelEventRelationshipCounter<'a, E> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

#[derive(Debug, Clone)]
pub struct OcelObjectRelationshipCounter<'a, O>(
    pub HashMap<&'a str, Vec<(O, RelationshipCount<'a>)>>,
);

impl<'a, O> Deref for OcelObjectRelationshipCounter<'a, O> {
    type Target = HashMap<&'a str, Vec<(O, RelationshipCount<'a>)>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<O> DerefMut for OcelObjectRelationshipCounter<'_, O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, O> Default for OcelObjectRelationshipCounter<'a, O> {
    fn default() -> Self {
        Self(HashMap::new())
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

pub fn extract_e2o_relation_counts<'a, T>(
    linked_ocel: &'a T,
) -> OcelEventRelationshipCounter<'a, T::EventRepr>
where
    T: LinkedOCELAccess<'a> + Sync,
    T::EventRepr: Copy + Sync + Send,
{
    OcelEventRelationshipCounter(
        linked_ocel
            .get_ev_types()
            .map(|etype| {
                #[cfg(not(feature = "rayon"))]
                let evs = linked_ocel.get_evs_of_type(etype);
                #[cfg(feature = "rayon")]
                let evs = linked_ocel
                    .get_evs_of_type(etype)
                    .collect::<Vec<_>>()
                    .into_par_iter();
                (
                    etype,
                    evs.map(|event| {
                        let mut counts = RelationshipCount::default();
                        linked_ocel.get_e2o(event).for_each(|(qualifier, obj)| {
                            let otype = linked_ocel.get_ob_type_of(obj);
                            *counts
                                .entry((Qualifier::Single(qualifier), otype))
                                .or_default() += 1;
                        });
                        (*event, counts)
                    })
                    .collect(),
                )
            })
            .collect(),
    )
}

pub fn extract_o2o_relation_counts<'a, T>(
    linked_ocel: &'a T,
) -> OcelObjectRelationshipCounter<'a, T::ObjectRepr>
where
    T: LinkedOCELAccess<'a> + Sync,
    T::ObjectRepr: Copy + Send + Sync,
{
    OcelObjectRelationshipCounter(
        linked_ocel
            .get_ob_types()
            .map(|otype| {
                #[cfg(not(feature = "rayon"))]
                let objs = linked_ocel.get_obs_of_type(otype);
                #[cfg(feature = "rayon")]
                let objs = linked_ocel
                    .get_obs_of_type(otype)
                    .collect::<Vec<_>>()
                    .into_par_iter();

                (
                    otype,
                    objs.map(|object| {
                        let mut counts = RelationshipCount::default();

                        linked_ocel.get_o2o(object).for_each(|(qualifier, obj)| {
                            let otype = linked_ocel.get_ob_type_of(obj);
                            *counts
                                .entry((Qualifier::Single(qualifier), otype))
                                .or_default() += 1;
                        });
                        (*object, counts)
                    })
                    .collect(),
                )
            })
            .collect(),
    )
}

pub fn extract_o2e_relation_counts<'a, T>(
    linked_ocel: &'a T,
) -> OcelObjectRelationshipCounter<'a, T::ObjectRepr>
where
    T: LinkedOCELAccess<'a> + Sync,
    T::ObjectRepr: Copy + Send + Sync,
{
    OcelObjectRelationshipCounter(
        linked_ocel
            .get_ob_types()
            .map(|otype| {
                #[cfg(not(feature = "rayon"))]
                let objs = linked_ocel.get_obs_of_type(otype);
                #[cfg(feature = "rayon")]
                let objs = linked_ocel
                    .get_obs_of_type(otype)
                    .collect::<Vec<_>>()
                    .into_par_iter();

                (
                    otype,
                    objs.map(|object| {
                        let mut counts = RelationshipCount::default();

                        linked_ocel
                            .get_e2o_rev(object)
                            .for_each(|(qualifier, evt)| {
                                let etype = linked_ocel.get_ev_type_of(evt);
                                *counts
                                    .entry((Qualifier::Single(qualifier), etype))
                                    .or_default() += 1;
                            });
                        (*object, counts)
                    })
                    .collect(),
                )
            })
            .collect(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum E2ERelationType {
    DirectlyFollows,
    #[default]
    EventuallyFollows,
}

pub fn extract_e2e_relation_counts<'a, T>(
    linked_ocel: &'a T,
    object_types: Option<&HashSet<&str>>,
    e2e_type: Option<E2ERelationType>,
    merge_object_types: bool,
) -> OcelEventRelationshipCounter<'a, T::EventRepr>
where
    T: LinkedOCELAccess<'a> + Sync,
    T::EventRepr: Eq + Hash + Copy + Sync + Send,
{
    if merge_object_types {
        extract_e2e_relation_counts_merged(linked_ocel, object_types, e2e_type)
    } else {
        extract_e2e_relation_counts_dedup_separate(linked_ocel, object_types, e2e_type)
    }
}

/// Two events are in an e2e relationship if they share an object and the first
/// event occurred (directly) before the second.
/// The relation is counted for each shared _object type_. That is, if, e.g.,
/// an order is placed for multiple items, and later shipped (for those items),
/// that eventually follows relation is only counted _once_. If the items are shipped
/// separately due to their large amount, this is counted twice, once per unique event.
/// The qualifier of that relationship is the object type.
pub fn extract_e2e_relation_counts_dedup_separate<'a, T>(
    linked_ocel: &'a T,
    object_types: Option<&HashSet<&str>>,
    e2e_type: Option<E2ERelationType>,
) -> OcelEventRelationshipCounter<'a, T::EventRepr>
where
    T: LinkedOCELAccess<'a> + Sync,
    T::EventRepr: Eq + Hash + Copy + Sync + Send,
{
    let e2e_type = e2e_type.unwrap_or_default();
    // let mut event_to_rels: HashMap<T::EventRepr, RelationshipCount<'a>> = HashMap::new();

    #[cfg(not(feature = "rayon"))]
    let evs = linked_ocel.get_all_evs();
    #[cfg(feature = "rayon")]
    let evs = linked_ocel
        .get_all_evs()
        .collect::<Vec<_>>()
        .into_par_iter();

    let event_to_rels: HashMap<T::EventRepr, RelationshipCount<'a>> = evs
        .map(|ev1| {
            let ev1_time = linked_ocel.get_ev_time(ev1);

            let mut shared: HashMap<T::EventRepr, HashSet<&'a str>> = HashMap::new();

            for (_q, o) in linked_ocel.get_e2o(ev1) {
                let otype = linked_ocel.get_ob_type_of(o);

                if !object_types.is_none_or(|x| x.contains(otype)) {
                    continue;
                }

                // If there was some way to guarantee that the LinkedOcel was sorted
                // "by o2e"..
                match e2e_type {
                    E2ERelationType::DirectlyFollows => {
                        let mut o2e: Vec<_> = linked_ocel.get_e2o_rev(o).collect();
                        o2e.sort_by_key(|(_, ev)| linked_ocel.get_ev_time(**ev));

                        if let Some(pos) = o2e.iter().position(|(_, ev)| **ev == ev1)
                        // If the object is related via multiple qualifiers, ev1
                        // will be in its trace multiple times, so skip it
                            && let Some((_, ev2)) =
                                o2e[pos + 1..].iter().find(|(_, ev)| **ev != ev1 && linked_ocel.get_ev_time(**ev) > ev1_time)
                        {
                            shared.entry(**ev2).or_default().insert(otype);
                        }
                    }
                    E2ERelationType::EventuallyFollows => {
                        for (_q2, ev2) in linked_ocel.get_e2o_rev(o) {
                            if linked_ocel.get_ev_time(*ev2) > ev1_time {
                                shared.entry(*ev2).or_default().insert(otype);
                            }
                        }
                    }
                };
            }

            let mut rels = RelationshipCount::default();
            for (ev2, otypes) in shared {
                let etype_2 = linked_ocel.get_ev_type_of(ev2);
                for otype in otypes {
                    *rels.entry((Qualifier::Single(otype), etype_2)).or_default() += 1;
                }
            }
            (ev1, rels)
        })
        .collect();

    let mut counter = OcelEventRelationshipCounter::default();
    for (ev_index, rels) in event_to_rels {
        let etype = linked_ocel.get_ev_type_of(ev_index);
        counter.entry(etype).or_default().push((ev_index, rels));
    }

    counter
}

/// Two events are in an E2E relationship if they share an object and the first
/// event occurred (directly) before the second.
/// The relation is counted for _each shared object_. That is, if, e.g., an order is
/// placed for multiple items, and later shipped (for those items), that E2E
/// relation is counted for _each one of the items_.
/// The qualifier of that relationship is the object type.
pub fn extract_e2e_relation_counts_separate<'a, T>(
    linked_ocel: &'a T,
    object_types: Option<&HashSet<&str>>,
    e2e_type: Option<E2ERelationType>,
) -> OcelEventRelationshipCounter<'a, T::EventRepr>
where
    T: LinkedOCELAccess<'a>,
    T::EventRepr: Eq + Hash + Copy,
{
    let e2e_type = e2e_type.unwrap_or_default();
    // Basically, check every object's O2E relationships
    // TODO: event -> event index -> relcount (simplifies conversion)
    let mut event_to_rels: HashMap<T::EventRepr, RelationshipCount<'a>> = HashMap::new();

    for obj in linked_ocel.get_all_obs() {
        let otype = linked_ocel.get_ob_type_of(&obj);
        if !object_types.is_none_or(|x| x.contains(otype)) {
            continue;
        }

        let mut o2e: Vec<_> = linked_ocel.get_e2o_rev(obj).collect();
        // Sort events by timestamp -> every event is related to all following ones.
        o2e.sort_by_key(|ev| linked_ocel.get_ev_time(ev.1));

        for i in 0..o2e.len().saturating_sub(1) {
            let ev1 = o2e[i];
            match e2e_type {
                E2ERelationType::DirectlyFollows => {
                    let ev1_time = linked_ocel.get_ev_time(ev1.1);
                    // If the object is related via multiple qualifiers, ev1 will be in its trace multiple times, so skip it
                    if let Some(ev2) = o2e[i + 1..]
                        .iter()
                        .find(|(_, ev)| *ev != ev1.1 && linked_ocel.get_ev_time(*ev) > ev1_time)
                    {
                        let etype_2 = linked_ocel.get_ev_type_of(ev2.1);
                        *event_to_rels
                            .entry(*ev1.1)
                            .or_default()
                            .entry((Qualifier::Single(otype), etype_2))
                            .or_default() += 1;
                    }
                }
                E2ERelationType::EventuallyFollows => {
                    // If the object is related via multiple qualifiers, ev1 will be in its trace multiple times, so skip it
                    for ev2 in o2e[(i + 1)..].iter().filter(|(_, ev)| *ev != ev1.1) {
                        let etype_2 = linked_ocel.get_ev_type_of(ev2.1);
                        *event_to_rels
                            .entry(*ev1.1)
                            .or_default()
                            .entry((Qualifier::Single(otype), etype_2))
                            .or_default() += 1;
                    }
                }
            };
        }
    }

    let mut counter = OcelEventRelationshipCounter::default();
    for (ev_index, rels) in event_to_rels {
        let etype = linked_ocel.get_ev_type_of(ev_index);
        counter.entry(etype).or_default().push((ev_index, rels));
    }

    counter
}

/// Two events are in an e2e relationship if they share an object and the first
/// event occurred  before the second.
/// In that case, an E2E relationship is extracted with the exact combination of
/// object types as qualifier.
pub fn extract_e2e_relation_counts_merged<'a, T>(
    linked_ocel: &'a T,
    object_types: Option<&HashSet<&str>>,
    e2e_type: Option<E2ERelationType>,
) -> OcelEventRelationshipCounter<'a, T::EventRepr>
where
    T: LinkedOCELAccess<'a>,
    T::EventRepr: Eq + Hash + Copy + Sync + Send,
{
    let e2e_type = e2e_type.unwrap_or_default();

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
        if !object_types.is_none_or(|x| x.contains(otype)) {
            continue;
        }

        let mut o2e: Vec<_> = linked_ocel.get_e2o_rev(obj).collect();
        // Sort events by timestamp -> every event is related to all following ones.
        o2e.sort_by_key(|ev| linked_ocel.get_ev_time(ev.1));

        for i in 0..o2e.len().saturating_sub(1) {
            let ev1 = *o2e[i].1;
            match e2e_type {
                E2ERelationType::DirectlyFollows => {
                    let ev1_time = linked_ocel.get_ev_time(ev1);
                    // If the object is related via multiple qualifiers, ev1 will be in its trace multiple times, so skip it
                    if let Some(ev2) = o2e[i + 1..]
                        .iter()
                        .find(|(_, ev)| **ev != ev1 && linked_ocel.get_ev_time(**ev) > ev1_time)
                    {
                        paired_events
                            .entry((ev1, *ev2.1))
                            .or_default()
                            .insert(otype);
                    }
                }
                E2ERelationType::EventuallyFollows => {
                    // If the object is related via multiple qualifiers, ev1 will be in its trace multiple times, so skip it
                    for ev2 in o2e[(i + 1)..].iter().filter(|(_, ev)| **ev != ev1) {
                        paired_events
                            .entry((ev1, *ev2.1))
                            .or_default()
                            .insert(otype);
                    }
                }
            };
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

    let mut counter = OcelEventRelationshipCounter::default();
    for (ev_index, rels) in event_to_rels {
        let etype = linked_ocel.get_ev_type_of(ev_index);
        counter.entry(etype).or_default().push((ev_index, rels));
    }

    counter
}
