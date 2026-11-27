use std::collections::HashSet;

use diffrel::{
    dependency_components::{
        compute_dependency_components, compute_divergent_pairs, singleton_components,
    },
    execution::result::{ComparisonInstanceResult, SkippedInstanceResult},
};
use process_mining::core::event_data::object_centric::linked_ocel::{
    LinkedOCELAccess, SlimLinkedOCEL,
};

use crate::{Cli, Grouping};

pub fn print_header(args: &Cli, log_1: &SlimLinkedOCEL, log_2: &SlimLinkedOCEL) {
    let ev_types_1: HashSet<&str> = log_1.get_ev_types().collect();
    let ev_types_2: HashSet<&str> = log_2.get_ev_types().collect();
    let ob_types_1: HashSet<&str> = log_1.get_ob_types().collect();
    let ob_types_2: HashSet<&str> = log_2.get_ob_types().collect();

    println!("diffrel-cli: full E2O/O2O/O2E/E2E comparison\n");
    println!(
        "  log 1: {} ({} events, {} event types, {} objects, {} object types)",
        args.log_1.display(),
        log_1.get_num_evs(),
        ev_types_1.len(),
        log_1.get_num_obs(),
        ob_types_1.len()
    );
    println!(
        "  log 2: {} ({} events, {} event types, {} objects, {} object types)",
        args.log_2.display(),
        log_2.get_num_evs(),
        ev_types_2.len(),
        log_2.get_num_obs(),
        ob_types_2.len()
    );
    println!(
        "  distance: {:?}   permutations: {}   alpha: {}   seed: {}",
        args.distance, args.permutations, args.alpha, args.seed
    );
    println!("  E2E follows: {:?}", args.follows);
    println!("  grouping: {:?}", args.grouping);

    let components_1 = match args.grouping {
        Grouping::Executions => compute_dependency_components(log_1),
        Grouping::Singletons => singleton_components(log_1),
    };
    let components_2 = match args.grouping {
        Grouping::Executions => compute_dependency_components(log_2),
        Grouping::Singletons => singleton_components(log_2),
    };
    let num_components_1 = components_1
        .events
        .values()
        .chain(components_1.objects.values())
        .copied()
        .collect::<HashSet<_>>()
        .len();
    let num_components_2 = components_2
        .events
        .values()
        .chain(components_2.objects.values())
        .copied()
        .collect::<HashSet<_>>()
        .len();
    println!(
        "  log 1 components: {}   log 2 components: {}",
        num_components_1, num_components_2
    );

    let divergent_pairs_1 = compute_divergent_pairs(log_1);
    let divergent_pairs_2 = compute_divergent_pairs(log_2);
    let mut div_1: Vec<_> = divergent_pairs_1.iter().collect();
    div_1.sort();
    println!(
        "  log 1 divergent (event, object) pairs ({}):{}",
        divergent_pairs_1.len(),
        if div_1.is_empty() { " none" } else { "" }
    );
    for (et, ot) in &div_1 {
        println!("    ({et}, {ot})");
    }
    let mut div_2: Vec<_> = divergent_pairs_2.iter().collect();
    div_2.sort();
    println!(
        "  log 2 divergent (event, object) pairs ({}):{}",
        divergent_pairs_2.len(),
        if div_2.is_empty() { " none" } else { "" }
    );
    for (et, ot) in &div_2 {
        println!("    ({et}, {ot})");
    }
}
pub fn print_results(args: &Cli, results: &[ComparisonInstanceResult]) {
    let num_significant = results.iter().filter(|res| res.test.significant).count();

    let id_width = results
        .iter()
        .map(|r| block_id(r).len())
        .max()
        .unwrap_or(5)
        .max(5);

    println!("\n=== Results ({} instances tested) ===", results.len());
    println!(
        "{:<id_width$}  {:>6}  {:>6}  {:>12}  {:>10}  {:>3}",
        "BLOCK",
        "n1",
        "n2",
        "pseudo-F",
        "p",
        "sig",
        id_width = id_width
    );

    for res in results {
        println!(
            "{:<id_width$}  {:>6}  {:>6}  {:>12.4}  {:>10.4}  {:>3}",
            block_id(res),
            res.n1,
            res.n2,
            res.test.pseudo_f,
            res.test.p_value,
            if res.test.significant { "*" } else { "" },
            id_width = id_width
        );
    }

    println!(
        "\n{num_significant} / {} instances significant at alpha={} (Holm-Bonferroni correction).",
        results.len(),
        args.alpha
    );
}

pub fn print_drilldowns(results: &[ComparisonInstanceResult]) {
    let num_significant = results.iter().filter(|res| res.test.significant).count();

    println!("\n=== Drilldown ===");
    if num_significant == 0 {
        println!("(no significant instances - nothing to drill into)");
    }
    for res in results {
        let Some(drill) = &res.drilldown else {
            continue;
        };
        match drill {
            Err(e) => println!(
                "\n{} (p={:.4}): drilldown failed: {e}",
                block_id(res),
                res.test.p_value
            ),
            Ok(rows) => {
                println!(
                    "\n--- {} (p={:.4}, n1={}, n2={}) ---",
                    block_id(res),
                    res.test.p_value,
                    res.n1,
                    res.n2
                );
                let mut sorted_rows: Vec<_> = rows.iter().collect();
                sorted_rows.sort_by(|a, b| a.test.p_value.total_cmp(&b.test.p_value));
                for row in sorted_rows {
                    let sig_mark = if row.test.significant { "*" } else { "" };
                    println!(
                        "  {:<20} -> {:<20}  p={:>7.4}{:<1}  pseudo-F={:>10.4}  cliffs_delta={:>7.3}",
                        row.qualifier,
                        row.target,
                        row.test.p_value,
                        sig_mark,
                        row.test.pseudo_f,
                        row.effect_size
                    );
                }
            }
        }
    }
}

pub fn print_skipped(skipped: &[SkippedInstanceResult]) {
    if !skipped.is_empty() {
        println!("\n=== Skipped instances ({}) ===", skipped.len());
        for s in skipped {
            println!(
                "  {:?}::{}: {:?}",
                s.instance.mode, s.instance.group_key, s.reason
            );
        }
    }
}

pub fn print_type_differences(log_1: &SlimLinkedOCEL, log_2: &SlimLinkedOCEL) {
    let ev_types_1: HashSet<&str> = log_1.get_ev_types().collect();
    let ev_types_2: HashSet<&str> = log_2.get_ev_types().collect();
    let ob_types_1: HashSet<&str> = log_1.get_ob_types().collect();
    let ob_types_2: HashSet<&str> = log_2.get_ob_types().collect();
    let evs_only_in_log_1 = sorted_diff(&ev_types_1, &ev_types_2);
    let evs_only_in_log_2 = sorted_diff(&ev_types_2, &ev_types_1);
    let obs_only_in_log_1 = sorted_diff(&ob_types_1, &ob_types_2);
    let obs_only_in_log_2 = sorted_diff(&ob_types_2, &ob_types_1);

    if !evs_only_in_log_1.is_empty() || !evs_only_in_log_2.is_empty() {
        println!("\nEvent types only in log 1: {evs_only_in_log_1:?}");
        println!("Event types only in log 2: {evs_only_in_log_2:?}");
    }
    if !obs_only_in_log_1.is_empty() || !obs_only_in_log_2.is_empty() {
        println!("Object types only in log 1: {obs_only_in_log_1:?}");
        println!("Object types only in log 2: {obs_only_in_log_2:?}");
    }
}

fn block_id(r: &ComparisonInstanceResult) -> String {
    format!("{:?}::{}", r.instance.mode, r.instance.group_key)
}

fn sorted_diff<'a>(a: &HashSet<&'a str>, b: &HashSet<&'a str>) -> Vec<&'a str> {
    let mut v: Vec<&str> = a.difference(b).copied().collect();
    v.sort_unstable();
    v
}
