use std::{
    collections::{HashMap, HashSet},
    ops::{Deref, DerefMut},
};

use ndarray::Array1;
use process_mining::core::event_data::object_centric::{OCEL, OCELEvent};

/// Built for an [`OCELEvent`], maps qualifier and object type to a count (the number
/// of objects with that object type that are in a relation with this event with
/// the given qualifier).
#[derive(Debug, Clone, Default)]
pub struct OcelEventRelationshipCount<'a>(pub HashMap<(&'a str, &'a str), usize>);

impl<'a> Deref for OcelEventRelationshipCount<'a> {
    type Target = HashMap<(&'a str, &'a str), usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> DerefMut for OcelEventRelationshipCount<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<'a> OcelEventRelationshipCount<'a> {
    pub fn from_event(
        event: &'a OCELEvent,
        object_id_to_object_type: &HashMap<&'a str, &'a str>,
    ) -> Self {
        let mut ret = Self::default();
        event.relationships.iter().for_each(|rel| {
            let object_type = *object_id_to_object_type
                .get(rel.object_id.as_str())
                .expect("All object ids in map");

            *ret.0
                .entry((rel.qualifier.as_str(), object_type))
                .or_default() += 1
        });
        ret
    }

    pub fn encoded(&self, relationship_types: &[(&str, &str)]) -> Array1<usize> {
        relationship_types
            .iter()
            .map(|rel_type| self.get(rel_type).copied().unwrap_or_default())
            .collect()
    }
    pub fn encoded_f64(&self, relationship_types: &[(&str, &str)]) -> Array1<f64> {
        relationship_types
            .iter()
            .map(|rel_type| self.get(rel_type).copied().unwrap_or_default() as f64)
            .collect()
    }
}

/// Tracks, per event type, a list encodings of those events w.r.t. the counts
/// of E20 relationships it has with which object types and which qualifiers.
#[derive(Debug, Clone, Default)]
pub struct OcelEventRelationshipCounter<'a>(
    pub HashMap<&'a str, Vec<OcelEventRelationshipCount<'a>>>,
);

impl<'a> Deref for OcelEventRelationshipCounter<'a> {
    type Target = HashMap<&'a str, Vec<OcelEventRelationshipCount<'a>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> DerefMut for OcelEventRelationshipCounter<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> OcelEventRelationshipCounter<'a> {
    /// Get all qualifiers observed in E20 relationships involving events with
    /// a certain event type.
    pub fn all_relation_types(&self, event_type: &str) -> Option<HashSet<&(&'a str, &'a str)>> {
        Some(
            self.get(event_type)?
                .iter()
                .flat_map(|x| x.keys())
                .collect(),
        )
    }
}

/// Extract E20 relation counts. That is an [`OcelEventRelationshipCounter`] is
/// extracted, which is a hashmap mapping each event type to a vec containing a (qualifier, count)
/// mapping for each individual event with this type.
pub fn extract_e2o_relation_counts<'a>(ocel: &'a OCEL) -> OcelEventRelationshipCounter<'a> {
    let object_id_to_type: HashMap<&str, &str> = ocel
        .objects
        .iter()
        .map(|obj| (obj.id.as_str(), obj.object_type.as_str()))
        .collect();

    let mut ret = OcelEventRelationshipCounter::default();
    ocel.events.iter().for_each(|evt| {
        let event_type = evt.event_type.as_str();
        ret.entry(event_type)
            .or_default()
            .push(OcelEventRelationshipCount::from_event(
                evt,
                &object_id_to_type,
            ));
    });
    ret
}
