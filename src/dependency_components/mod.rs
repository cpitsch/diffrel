use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use process_mining::core::event_data::object_centric::linked_ocel::LinkedOCELAccess;
use union_find::UnionFind;

mod union_find;

/// Group ids for events and objects of one log.
///
/// Group ids are shared between `events` and `objects`. That is, if an event
/// and object share a group id, they are in the same process execution / case notion
#[derive(Debug, Clone)]
pub struct DependencyComponents<E, O> {
    pub events: HashMap<E, usize>,
    pub objects: HashMap<O, usize>,
}

/// Place every event and object in its own component. That is, consider every
/// event and object independent
pub fn singleton_components<'a, T>(
    locel: &'a T,
) -> DependencyComponents<T::EventRepr, T::ObjectRepr>
where
    T: LinkedOCELAccess<'a>,
    T::EventRepr: Eq + Hash,
    T::ObjectRepr: Eq + Hash,
{
    let events = locel
        .get_all_evs()
        .enumerate()
        .map(|(i, e)| (e, i))
        .collect();
    let n_evs = locel.get_num_evs();

    let objects = locel
        .get_all_obs()
        .enumerate()
        .map(|(i, o)| (o, i + n_evs))
        .collect();

    DependencyComponents { events, objects }
}

/// Compute groups of interdependent events and objects based on a simplified version
/// of advanced case notions by van Detten et al. (ICPM 2024).
///
/// The event-object graph is pruned by removing edges between a diverging pair
/// of event and object and two events / two objects are considered "dependent"
/// iff they are in the same connected component of the pruned event-object graph.
pub fn compute_dependency_components<'a, T>(
    locel: &'a T,
) -> DependencyComponents<T::EventRepr, T::ObjectRepr>
where
    T: LinkedOCELAccess<'a>,
    T::EventRepr: Eq + Hash + Copy,
    T::ObjectRepr: Eq + Hash + Copy,
{
    // Event type/Object type pairs to _not_ consider for dependencies.
    let divergent = compute_divergent_pairs(locel);

    // Compute connected components based on divergence information

    let DependencyComponents {
        events: ev_enum,
        objects: ob_enum,
    } = singleton_components(locel);

    // Compute groups based on connected components, considering only connections
    // via non-divergent relationships.
    let mut uf = UnionFind::new(ev_enum.len() + ob_enum.len());
    locel.get_all_evs().for_each(|e| {
        let etype = locel.get_ev_type_of(e);
        let e_group = ev_enum[&e];
        for (_q, o) in locel.get_e2o(&e) {
            let otype = locel.get_ob_type_of(o);
            // If we remove the check, we are using connected components.
            if !divergent.contains(&(etype, otype)) {
                uf.union(e_group, ob_enum[o]);
            }
        }
    });

    // Compact ("unsparse") group (component) ids so we can use them as indices later
    let mut remap = HashMap::new();
    let events: HashMap<_, _> = ev_enum
        .into_iter()
        .map(|(e, group)| {
            let root = uf.find(group);
            let next = remap.len();
            let new_group = *remap.entry(root).or_insert(next);
            (e, new_group)
        })
        .collect();
    let objects: HashMap<_, _> = ob_enum
        .into_iter()
        .map(|(o, group)| {
            let root = uf.find(group);
            let next = remap.len();
            let new_group = *remap.entry(root).or_insert(next);
            (o, new_group)
        })
        .collect();

    DependencyComponents { events, objects }
}

pub fn compute_divergent_pairs<'a, T>(locel: &'a T) -> HashSet<(&'a str, &'a str)>
where
    T: LinkedOCELAccess<'a>,
    T::EventRepr: Eq,
{
    let mut divergent = HashSet::new();
    for ot in locel.get_ob_types() {
        for et in locel.get_ev_types() {
            let pair_is_divergent = locel.get_obs_of_type(ot).any(|o| {
                let mut ids = locel.get_e2o_rev_of_type(o, et).map(|(_q, e)| e);
                ids.next().is_some_and(|first| ids.any(|e| e != first))
            });

            if pair_is_divergent {
                divergent.insert((et, ot));
            }
        }
    }

    divergent
}
