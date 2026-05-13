use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use ndarray::Array1;
use process_mining::{OCEL, core::event_data::object_centric::OCELEvent};

#[derive(Default, Debug, Clone)]
pub struct OcelIndex<'a> {
    events_by_type: HashMap<&'a str, Vec<&'a OCELEvent>>,
    object_id_to_type: HashMap<&'a str, &'a str>,
    /// WARN: If an object is not related to any events, it will not be included as a key.
    object_id_to_events: HashMap<&'a str, Vec<(&'a str, &'a OCELEvent)>>,
    /// (`qualifier`, `other_object_id`)
    object_id_to_o2o: HashMap<&'a str, Vec<(&'a str, &'a str)>>,
}

impl<'a> From<&'a OCEL> for OcelIndex<'a> {
    fn from(ocel: &'a OCEL) -> Self {
        let mut res = Self::default();
        for event in &ocel.events {
            res.events_by_type
                .entry(&event.event_type)
                .or_default()
                .push(event);
            for rel in &event.relationships {
                res.object_id_to_events
                    .entry(&rel.object_id)
                    .or_default()
                    .push((&rel.qualifier, event));
            }
        }

        for obj in &ocel.objects {
            res.object_id_to_type.insert(&obj.id, &obj.object_type);
            obj.relationships.iter().for_each(|rel| {
                res.object_id_to_o2o
                    .entry(&obj.id)
                    .or_default()
                    .push((&rel.qualifier, &rel.object_id));
            });
        }
        res
    }
}

#[derive(Debug, Clone, Default)]
pub struct RelationshipCount<'a>(pub HashMap<(&'a str, &'a str), usize>);

impl<'a> Deref for RelationshipCount<'a> {
    type Target = HashMap<(&'a str, &'a str), usize>;

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
    pub fn encoded(&self, relationship_types: &[(&str, &str)]) -> Array1<usize> {
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
    pub fn encoded_f64(&self, relationship_types: &[(&str, &str)]) -> Array1<f64> {
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
    indexed_ocel: &OcelIndex<'a>,
) -> OcelRelationshipCounter<'a> {
    OcelRelationshipCounter(
        indexed_ocel
            .events_by_type
            .iter()
            .map(|(&etype, events)| {
                (
                    etype,
                    events
                        .iter()
                        .map(|evt| {
                            let mut counts = RelationshipCount::default();
                            for rel in &evt.relationships {
                                // Otype is only `None` if an event has a relationship to
                                // a non-existent object
                                if let Some(otype) =
                                    indexed_ocel.object_id_to_type.get(rel.object_id.as_str())
                                {
                                    *counts.entry((&rel.qualifier, otype)).or_default() += 1;
                                }
                            }
                            counts
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

pub fn extract_o2o_relation_counts<'a>(
    indexed_ocel: &OcelIndex<'a>,
) -> OcelRelationshipCounter<'a> {
    let mut counter = OcelRelationshipCounter::default();

    for (&oid, &otype) in &indexed_ocel.object_id_to_type {
        let mut counts = RelationshipCount::default();
        if let Some(o2o_rels) = indexed_ocel.object_id_to_o2o.get(oid) {
            for (qualifier, other_oid) in o2o_rels {
                // other_type is only None if the OCEL contains relationships to
                // non-existent objects
                if let Some(other_type) = indexed_ocel.object_id_to_type.get(other_oid) {
                    *counts.entry((qualifier, other_type)).or_default() += 1;
                }
                // TODO: Print a warning?
            }
        };
        // Push even if the object has no o2o relations.
        counter.entry(otype).or_default().push(counts);
    }
    counter
}

pub fn extract_o2e_relation_counts<'a>(
    indexed_ocel: &OcelIndex<'a>,
) -> OcelRelationshipCounter<'a> {
    let mut counter = OcelRelationshipCounter::default();

    for (&oid, &otype) in &indexed_ocel.object_id_to_type {
        let mut counts = RelationshipCount::default();

        // Push empty counts?
        if let Some(events) = indexed_ocel.object_id_to_events.get(oid) {
            for (qualifier, evt) in events {
                *counts.entry((qualifier, &evt.event_type)).or_default() += 1;
            }
        }
        // Push even if object was not related to any events
        counter.entry(otype).or_default().push(counts);
    }

    counter
}
