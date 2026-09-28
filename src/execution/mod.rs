use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use process_mining::core::event_data::object_centric::linked_ocel::LinkedOCELAccess;

use crate::{
    dependency_components::compute_dependency_components,
    distances::squared_bray_curtis_distance,
    effect_size::cliffs_delta,
    execution::{
        config::{ComparisonConfig, Distance, Grouping},
        instance::{ComparisonInstance, RelationMode},
        result::{
            ComparisonInstanceResult, ComparisonResult, DrilldownError, DrilldownRelationResult,
            DrilldownResult, SkipReason, SkippedInstanceResult, TestResult,
        },
    },
    multiple_testing::check_significance_holm_bonferroni,
    permanova::{
        base::{encode_relation_counts, restricted_permanova},
        drilldown::drilldown,
        fast_euclidean::permanova as permanova_fast,
    },
    relation_extraction::{
        RelationshipCount, compute_ordered_relation_types, extract_e2e_relation_counts,
        extract_e2o_relation_counts, extract_o2e_relation_counts, extract_o2o_relation_counts,
    },
};

pub mod config;
pub mod export;
pub mod instance;
pub mod result;

pub fn run_comparison<'a, T>(
    log_1: &'a T,
    log_2: &'a T,
    instances: &[ComparisonInstance],
    config: &ComparisonConfig,
) -> ComparisonResult
where
    T: LinkedOCELAccess<'a> + Sync,
    T::EventRepr: Eq + Hash + Copy + Sync + Send,
    T::ObjectRepr: Eq + Hash + Copy + Sync + Send,
{
    let seed = config.seed.unwrap_or_else(rand::random::<u64>);
    let used_modes: HashSet<_> = instances.iter().map(|instance| instance.mode).collect();

    let e2e_object_types: Option<HashSet<_>> = config
        .e2e_object_types
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());

    let e2o = used_modes.contains(&RelationMode::E2O).then(|| {
        (
            extract_e2o_relation_counts(log_1),
            extract_e2o_relation_counts(log_2),
        )
    });
    let o2o = used_modes.contains(&RelationMode::O2O).then(|| {
        (
            extract_o2o_relation_counts(log_1),
            extract_o2o_relation_counts(log_2),
        )
    });
    let o2e = used_modes.contains(&RelationMode::O2E).then(|| {
        (
            extract_o2e_relation_counts(log_1),
            extract_o2e_relation_counts(log_2),
        )
    });
    let e2e = used_modes.contains(&RelationMode::E2E).then(|| {
        (
            extract_e2e_relation_counts(
                log_1,
                e2e_object_types.as_ref(),
                Some(config.follows),
                false,
            ),
            extract_e2e_relation_counts(
                log_2,
                e2e_object_types.as_ref(),
                Some(config.follows),
                false,
            ),
        )
    });

    let components = match config.grouping {
        Grouping::Executions => Some((
            compute_dependency_components(log_1),
            compute_dependency_components(log_2),
        )),
        Grouping::Singletons => None,
    };

    let mut skipped = Vec::new();
    let mut test_results = Vec::new();

    for instance in instances {
        let res = match instance.mode {
            RelationMode::E2O => {
                let (profile_1, profile_2) = e2o.as_ref().expect("E2O Contained -> E2O extracted");

                let component_events = components.as_ref().map(|(c1, c2)| (&c1.events, &c2.events));
                run_instance(
                    instance,
                    &profile_1.0,
                    &profile_2.0,
                    component_events,
                    seed,
                    config,
                )
            }
            RelationMode::O2O => {
                let (profile_1, profile_2) = o2o.as_ref().expect("O2O Contained -> O2O extracted");

                let component_objects = components
                    .as_ref()
                    .map(|(c1, c2)| (&c1.objects, &c2.objects));
                run_instance(
                    instance,
                    &profile_1.0,
                    &profile_2.0,
                    component_objects,
                    seed,
                    config,
                )
            }
            RelationMode::O2E => {
                let (profile_1, profile_2) = o2e.as_ref().expect("O2E Contained -> O2E extracted");

                let component_objects = components
                    .as_ref()
                    .map(|(c1, c2)| (&c1.objects, &c2.objects));
                run_instance(
                    instance,
                    &profile_1.0,
                    &profile_2.0,
                    component_objects,
                    seed,
                    config,
                )
            }
            RelationMode::E2E => {
                let (profile_1, profile_2) = e2e.as_ref().expect("E2E Contained -> E2E extracted");

                let component_events = components.as_ref().map(|(c1, c2)| (&c1.events, &c2.events));
                run_instance(
                    instance,
                    &profile_1.0,
                    &profile_2.0,
                    component_events,
                    seed,
                    config,
                )
            }
        };

        match res {
            RawInstanceResult::Skipped(reason) => skipped.push(SkippedInstanceResult {
                instance: instance.clone(),
                reason,
            }),

            RawInstanceResult::Result {
                n1,
                n2,
                pseudo_f,
                p_value,
            } => {
                test_results.push((instance.clone(), n1, n2, pseudo_f, p_value));
            }
        }
    }

    let ids: Vec<String> = test_results
        .iter()
        .map(|(instance, ..)| format!("{:?}::{}", instance.mode, instance.group_key))
        .collect();
    let pvals: HashMap<&str, f64> = ids
        .iter()
        .zip(&test_results)
        .map(|(id, (.., p))| (id.as_str(), *p))
        .collect();
    let significance: HashMap<&str, bool> = check_significance_holm_bonferroni(pvals, config.alpha)
        .into_iter()
        .map(|(id, (_, sig))| (id, sig))
        .collect();

    let results = ids
        .iter()
        .zip(test_results)
        .map(|(id, (instance, n1, n2, pseudo_f, p_value))| {
            let significant = significance.get(id.as_str()).copied().unwrap_or(false);
            ComparisonInstanceResult {
                instance,
                n1,
                n2,
                test: TestResult {
                    pseudo_f,
                    p_value,
                    significant,
                },
                effect_size: None,
                drilldown: None,
            }
        })
        .collect();

    ComparisonResult {
        results,
        skipped,
        seed_used: seed,
    }
}

/// Auxilliary raw output of a comparison, with no multiple testing correction
/// applied.
#[derive(Debug, Clone, PartialEq)]
pub enum RawInstanceResult {
    Skipped(SkipReason),
    Result {
        n1: usize,
        n2: usize,
        pseudo_f: f64,
        p_value: f64,
    },
}

#[allow(clippy::type_complexity)]
pub fn run_instance<'a, R: Eq + Hash + Copy>(
    instance: &ComparisonInstance,
    profile_1: &HashMap<&'a str, Vec<(R, RelationshipCount<'a>)>>,
    profile_2: &HashMap<&'a str, Vec<(R, RelationshipCount<'a>)>>,
    components: Option<(&HashMap<R, usize>, &HashMap<R, usize>)>,
    seed: u64,
    config: &ComparisonConfig,
) -> RawInstanceResult {
    let in_log_1 = profile_1.contains_key(instance.group_key.as_str());
    if !in_log_1 {
        return RawInstanceResult::Skipped(SkipReason::OnlyInLogOne);
    }
    let in_log_2 = profile_2.contains_key(instance.group_key.as_str());
    if !in_log_2 {
        return RawInstanceResult::Skipped(SkipReason::OnlyInLogTwo);
    }

    let counts_1 = &profile_1[instance.group_key.as_str()];
    let counts_2 = &profile_2[instance.group_key.as_str()];

    let allowed_targets: Option<HashSet<_>> = instance
        .target_types
        .as_ref()
        .map(|target| target.iter().map(String::as_str).collect());

    let ordered_relation_types =
        compute_ordered_relation_types(counts_1, counts_2, allowed_targets.as_ref());

    if ordered_relation_types.is_empty() {
        return RawInstanceResult::Skipped(SkipReason::NoRelationsObserved);
    }

    let n1 = counts_1.len();
    let n2 = counts_2.len();

    let sample_1 = encode_relation_counts(counts_1, &ordered_relation_types);
    let sample_2 = encode_relation_counts(counts_2, &ordered_relation_types);

    let res = match config.distance {
        Distance::BrayCurtis => {
            let (component_ids_1, component_ids_2) = match components {
                Some((c1, c2)) => (
                    counts_1.iter().map(|(r, _)| c1[r]).collect::<Vec<_>>(),
                    counts_2.iter().map(|(r, _)| c2[r]).collect::<Vec<_>>(),
                ),
                None => ((0..n1).collect::<Vec<_>>(), (0..n2).collect::<Vec<_>>()),
            };

            restricted_permanova(
                &sample_1,
                &sample_2,
                &component_ids_1,
                &component_ids_2,
                squared_bray_curtis_distance,
                config.permutations,
                Some(seed),
            )
        }
        Distance::Euclidean => match config.grouping {
            Grouping::Singletons => {
                permanova_fast(&sample_1, &sample_2, config.permutations, Some(seed))
            }
            Grouping::Executions => {
                todo!("fast euclidean does not yet support component-restricted permutation")
            }
        },
    };

    match res {
        Ok(result) => RawInstanceResult::Result {
            n1,
            n2,
            pseudo_f: result.pseudo_f,
            p_value: result.p_value,
        },
        Err(e) => RawInstanceResult::Skipped(SkipReason::TestFailed(e)),
    }
}

pub fn drilldown_for_instances<'a, T>(
    log_1: &'a T,
    log_2: &'a T,
    instances: &[ComparisonInstance],
    config: &ComparisonConfig,
    seed: u64,
) -> Vec<(ComparisonInstance, DrilldownResult)>
where
    T: LinkedOCELAccess<'a> + Sync,
    T::EventRepr: Eq + Hash + Copy + Sync + Send,
    T::ObjectRepr: Eq + Hash + Copy + Sync + Send,
{
    let used_modes: HashSet<_> = instances.iter().map(|instance| instance.mode).collect();

    let e2e_object_types: Option<HashSet<_>> = config
        .e2e_object_types
        .as_ref()
        .map(|otype| otype.iter().map(String::as_str).collect());

    let e2o = used_modes.contains(&RelationMode::E2O).then(|| {
        (
            extract_e2o_relation_counts(log_1),
            extract_e2o_relation_counts(log_2),
        )
    });
    let o2o = used_modes.contains(&RelationMode::O2O).then(|| {
        (
            extract_o2o_relation_counts(log_1),
            extract_o2o_relation_counts(log_2),
        )
    });
    let o2e = used_modes.contains(&RelationMode::O2E).then(|| {
        (
            extract_o2e_relation_counts(log_1),
            extract_o2e_relation_counts(log_2),
        )
    });
    let e2e = used_modes.contains(&RelationMode::E2E).then(|| {
        (
            extract_e2e_relation_counts(
                log_1,
                e2e_object_types.as_ref(),
                Some(config.follows),
                false,
            ),
            extract_e2e_relation_counts(
                log_2,
                e2e_object_types.as_ref(),
                Some(config.follows),
                false,
            ),
        )
    });

    let components = match config.grouping {
        Grouping::Executions => Some((
            compute_dependency_components(log_1),
            compute_dependency_components(log_2),
        )),
        Grouping::Singletons => None,
    };

    instances
        .iter()
        .map(|instance| {
            let res = match instance.mode {
                RelationMode::E2O => {
                    let (profile_1, profile_2) =
                        e2o.as_ref().expect("E2O contained -> E2O extracted");
                    let component_events =
                        components.as_ref().map(|(c1, c2)| (&c1.events, &c2.events));

                    drilldown_for_instance(
                        instance,
                        &profile_1.0,
                        &profile_2.0,
                        component_events,
                        config,
                        seed,
                    )
                }
                RelationMode::O2O => {
                    let (profile_1, profile_2) =
                        o2o.as_ref().expect("O2O contained -> O2O extracted");
                    let component_objects = components
                        .as_ref()
                        .map(|(c1, c2)| (&c1.objects, &c2.objects));

                    drilldown_for_instance(
                        instance,
                        &profile_1.0,
                        &profile_2.0,
                        component_objects,
                        config,
                        seed,
                    )
                }
                RelationMode::O2E => {
                    let (profile_1, profile_2) =
                        o2e.as_ref().expect("O2E contained -> O2E extracted");
                    let component_objects = components
                        .as_ref()
                        .map(|(c1, c2)| (&c1.objects, &c2.objects));

                    drilldown_for_instance(
                        instance,
                        &profile_1.0,
                        &profile_2.0,
                        component_objects,
                        config,
                        seed,
                    )
                }
                RelationMode::E2E => {
                    let (profile_1, profile_2) =
                        e2e.as_ref().expect("E2E contained -> E2E extracted");
                    let component_events =
                        components.as_ref().map(|(c1, c2)| (&c1.events, &c2.events));

                    drilldown_for_instance(
                        instance,
                        &profile_1.0,
                        &profile_2.0,
                        component_events,
                        config,
                        seed,
                    )
                }
            };
            (instance.clone(), res)
        })
        .collect()
}

#[allow(clippy::type_complexity)]
pub fn drilldown_for_instance<'a, R>(
    instance: &ComparisonInstance,
    profile_1: &HashMap<&'a str, Vec<(R, RelationshipCount<'a>)>>,
    profile_2: &HashMap<&'a str, Vec<(R, RelationshipCount<'a>)>>,
    components: Option<(&HashMap<R, usize>, &HashMap<R, usize>)>,
    config: &ComparisonConfig,
    seed: u64,
) -> DrilldownResult
where
    R: Eq + Hash + Copy,
{
    // If called from `drilldown_for_instances`, the corresponding profile is
    // definitely present.
    let Some(counts_1) = profile_1.get(instance.group_key.as_str()) else {
        return Err(DrilldownError::GroupKeyNotInProfileOne(
            instance.group_key.clone(),
        ));
    };
    let Some(counts_2) = profile_2.get(instance.group_key.as_str()) else {
        return Err(DrilldownError::GroupKeyNotInProfileTwo(
            instance.group_key.clone(),
        ));
    };

    let allowed_targets: Option<HashSet<_>> = instance
        .target_types
        .as_ref()
        .map(|target| target.iter().map(String::as_str).collect());

    let ordered_relation_types =
        compute_ordered_relation_types(counts_1, counts_2, allowed_targets.as_ref());

    if ordered_relation_types.is_empty() {
        return Ok(vec![]);
    }

    let n1 = counts_1.len();
    let n2 = counts_2.len();

    let (component_ids_1, component_ids_2) = match components {
        Some((c1, c2)) => (
            counts_1.iter().map(|(r, _)| c1[r]).collect::<Vec<_>>(),
            counts_2.iter().map(|(r, _)| c2[r]).collect::<Vec<_>>(),
        ),
        None => ((0..n1).collect::<Vec<_>>(), (0..n2).collect::<Vec<_>>()),
    };

    let sample_1 = encode_relation_counts(counts_1, &ordered_relation_types);
    let sample_2 = encode_relation_counts(counts_2, &ordered_relation_types);

    let mut drilldown_results = drilldown(
        &sample_1,
        &sample_2,
        &component_ids_1,
        &component_ids_2,
        config.permutations,
        seed,
        &ordered_relation_types,
    )?;

    let relations = ordered_relation_types
        .iter()
        .enumerate()
        .map(|(idx, key)| {
            let res = drilldown_results
                .remove(key)
                .expect("drilldown was called with this relation type");

            let col_1: Vec<_> = sample_1.column(idx).iter().copied().collect();
            let col_2: Vec<_> = sample_2.column(idx).iter().copied().collect();

            DrilldownRelationResult {
                qualifier: key.0.to_string(),
                target: key.1.to_string(),
                test: TestResult {
                    pseudo_f: res.pseudo_f,
                    p_value: res.p_value,
                    significant: res.p_value <= config.alpha,
                },
                effect_size: cliffs_delta(&col_1, &col_2),
            }
        })
        .collect();

    Ok(relations)
}
