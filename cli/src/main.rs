use anyhow::anyhow;
use clap::{Parser, ValueEnum};
use diffrel::{
    execution::{
        config::{ComparisonConfig, Grouping as LibGrouping},
        drilldown_for_instances,
        export::ComparisonResultExport,
        instance::{ComparisonInstance, RelationMode},
        result::ComparisonResult,
        run_comparison,
    },
    permanova::Distance as LibDistance,
    relation_extraction::E2ERelationType,
};
use process_mining::{Importable, core::event_data::object_centric::linked_ocel::SlimLinkedOCEL};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

pub mod text;

#[derive(Parser)]
#[command(version, about)]
pub struct Cli {
    /// The first OCEL (xml, xml.gz, json, json.gz)
    log_1: PathBuf,
    /// The second OCEL (xml, xml.gz, json, json.gz)
    log_2: PathBuf,
    /// Number of permutations for each PERMANOVA test
    #[arg(long, default_value_t = 1_000)]
    permutations: usize,
    /// The significance level applied after multiple-testing correction
    #[arg(long, default_value_t = 0.05)]
    alpha: f64,
    /// The seed to use for permutations. Reused by drilldowns
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Distance measure for the PERMANOVA tests
    #[arg(long, value_enum, default_value_t = Distance::BrayCurtis)]
    distance: Distance,
    /// Type of follows-relations to use for E2E relations.
    #[arg(long, value_enum, default_value_t = Follows::Directly)]
    follows: Follows,
    /// Row-grouping based on dependency. `executions` uses divergence-pruned
    /// connected components (inspired by van Detten et al.). `singletons` treats
    /// all rows as exchangeable.
    #[arg(long, value_enum, default_value_t = Grouping::Executions)]
    grouping: Grouping,
    /// The object types to consider for E2E relations and O2O/O2E blocks
    /// (comma-separated).
    #[arg(long, value_delimiter = ',')]
    object_types: Vec<String>,
    /// Output format
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,
}

impl Cli {
    pub fn comparison_config(&self) -> ComparisonConfig {
        ComparisonConfig {
            permutations: self.permutations,
            alpha: self.alpha,
            seed: Some(self.seed),
            distance: self.distance.into(),
            grouping: self.grouping.into(),
            follows: self.follows.into(),
            e2e_object_types: if self.object_types.is_empty() {
                None
            } else {
                Some(self.object_types.clone())
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Distance {
    BrayCurtis,
    Euclidean,
}

impl From<Distance> for LibDistance {
    fn from(d: Distance) -> Self {
        match d {
            Distance::BrayCurtis => LibDistance::BrayCurtis,
            Distance::Euclidean => LibDistance::Euclidean,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Follows {
    Directly,
    Eventually,
}

impl From<Follows> for E2ERelationType {
    fn from(f: Follows) -> Self {
        match f {
            Follows::Directly => E2ERelationType::DirectlyFollows,
            Follows::Eventually => E2ERelationType::EventuallyFollows,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Grouping {
    Executions,
    Singletons,
}

impl From<Grouping> for LibGrouping {
    fn from(g: Grouping) -> Self {
        match g {
            Grouping::Executions => LibGrouping::Executions,
            Grouping::Singletons => LibGrouping::Singletons,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Format {
    Text,
    Json,
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    let log_1 = SlimLinkedOCEL::import_from_path(&args.log_1)
        .map_err(|e| anyhow!("failed to load log {}: {e}", args.log_1.display()))?;
    let log_2 = SlimLinkedOCEL::import_from_path(&args.log_2)
        .map_err(|e| anyhow!("failed to load log {}: {e}", args.log_2.display()))?;

    let config = args.comparison_config();
    let mut instances = ComparisonInstance::all(&log_1, &log_2);
    if !args.object_types.is_empty() {
        let allowed: HashSet<_> = args.object_types.iter().map(String::as_str).collect();
        instances.retain(|x| {
            // Either doesnt have objects as center or has an allowed object as center
            !matches!(x.mode, RelationMode::O2O | RelationMode::O2E)
                || allowed.contains(x.group_key.as_str())
        });
    }

    if args.format == Format::Text {
        text::print_header(&args, &log_1, &log_2);
    }

    let comparison = run_comparison(&log_1, &log_2, &instances, &config);
    let mut results = comparison.results;

    // Drilldowns
    let significant_instances: Vec<_> = results
        .iter()
        .filter(|res| res.test.significant)
        .map(|res| res.instance.clone())
        .collect();

    let mut drilldown_by_key: HashMap<_, _> = drilldown_for_instances(
        &log_1,
        &log_2,
        &significant_instances,
        &config,
        comparison.seed_used,
    )
    .into_iter()
    .map(|(instance, res)| ((instance.mode, instance.group_key), res))
    .collect();

    results
        .iter_mut()
        .filter(|res| res.test.significant)
        .for_each(|res| {
            res.drilldown =
                drilldown_by_key.remove(&(res.instance.mode, res.instance.group_key.clone()));
        });

    match args.format {
        Format::Json => {
            let result = ComparisonResult {
                results,
                skipped: comparison.skipped,
                seed_used: comparison.seed_used,
            };
            let output = ComparisonResultExport::new(
                result,
                &config,
                &log_1,
                Some(args.log_1.display().to_string()),
                &log_2,
                Some(args.log_2.display().to_string()),
            );
            serde_json::to_writer_pretty(std::io::stdout(), &output)?;
            println!()
        }
        Format::Text => {
            // Show significant results first, then by pvalue
            results.sort_by(|a, b| {
                b.test
                    .significant
                    .cmp(&a.test.significant)
                    .then_with(|| a.test.p_value.total_cmp(&b.test.p_value))
            });
            text::print_results(&args, &results);
            text::print_drilldowns(&results);
            text::print_skipped(&comparison.skipped);
            text::print_type_differences(&log_1, &log_2);
        }
    }
    Ok(())
}
