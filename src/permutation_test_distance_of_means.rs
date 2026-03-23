use std::collections::{HashMap, HashSet};

use ndarray::Array1;
use process_mining::OCEL;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::relation_extraction::extract_e2o_relation_counts;

/// Compare the E2O relationships between the two event logs. Returns a hashmap
/// of event type -> p-value
///
/// If an event type only occurs in one of the event logs, no p-value is computed,
/// i.e., the event type is _not_ contained as a key in the hashmap.
pub fn compare<'a>(
    log_1: &'a OCEL,
    log_2: &'a OCEL,
    num_permutations: usize,
    permutation_seed: Option<u64>,
) -> HashMap<&'a str, f64> {
    let counts_1 = extract_e2o_relation_counts(log_1);
    let counts_2 = extract_e2o_relation_counts(log_2);

    let all_event_types: HashSet<_> = counts_1.keys().chain(counts_2.keys()).copied().collect();

    let mut event_type_to_pval = HashMap::new();
    for event_type in all_event_types {
        let (Some(rel_counts_1), Some(rel_counts_2)) =
            (counts_1.get(event_type), counts_2.get(event_type))
        else {
            println!("{event_type} only in one log");
            continue;
        };

        let ordered_relation_types: Vec<_> = rel_counts_1
            .iter()
            .chain(rel_counts_2.iter())
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>()
            .into_iter()
            .cloned()
            .collect();

        let encodings_1: Vec<_> = rel_counts_1
            .iter()
            .map(|c| c.encoded(&ordered_relation_types))
            .collect();
        let encodings_2: Vec<_> = rel_counts_2
            .iter()
            .map(|c| c.encoded(&ordered_relation_types))
            .collect();

        let sum_1: Array1<usize> = encodings_1.iter().fold(
            Array1::<usize>::zeros(ordered_relation_types.len()),
            |mut acc, arr| {
                acc += arr;
                acc
            },
        );
        let mean_1 = sum_1.mapv(|x| x as f64) / (encodings_1.len() as f64);
        let sum_2: Array1<usize> = encodings_2.iter().fold(
            Array1::<usize>::zeros(ordered_relation_types.len()),
            |mut acc, arr| {
                acc += arr;
                acc
            },
        );
        let mean_2 = sum_2.mapv(|x| x as f64) / (encodings_2.len() as f64);
        let test_stat_1 = euclidean_distance(&mean_1, &mean_2);
        let total_sum = sum_1 + sum_2;

        // TODO: Might be able to do this slightly more elegantly
        let all_encodings: Vec<_> = encodings_1
            .into_iter()
            .chain(encodings_2.into_iter())
            .collect();

        // TODO: Can use the smaller group as the first one to make it less to do
        let mut shuffled: Vec<_> = (0..all_encodings.len()).collect();
        let mut rng = match permutation_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };
        let permutation_distribution: Vec<f64> = (0..num_permutations)
            .map(|_| {
                // 1. Get indices to be used for the first group
                // 2. Compute its sum
                // 3. The sum of the other group (rest) is total_sum - sum_group_1
                // 4. Use them to compute the means
                // 5. Compute the difference in means
                // TODO: Are the samples independent if I re-shuffle the shuffled version?
                shuffled.partial_shuffle(&mut rng, rel_counts_1.len());
                let sum_1 = shuffled.iter().take(rel_counts_1.len()).fold(
                    Array1::<usize>::zeros(ordered_relation_types.len()),
                    |mut acc, idx| {
                        acc += &all_encodings[*idx];
                        acc
                    },
                );
                let sum_2 = total_sum.clone() - sum_1.clone();

                let mean_1 = sum_1.mapv(|x| x as f64) / rel_counts_1.len() as f64;
                let mean_2 = sum_2.mapv(|x| x as f64) / rel_counts_2.len() as f64;

                euclidean_distance(&mean_1, &mean_2)
            })
            .collect();

        let p = permutation_distribution
            .iter()
            .filter(|ts| **ts > test_stat_1)
            .count() as f64
            / num_permutations as f64;
        event_type_to_pval.insert(event_type, p);
    }

    event_type_to_pval
}

fn euclidean_distance(v1: &Array1<f64>, v2: &Array1<f64>) -> f64 {
    let diff = v1 - v2;
    diff.dot(&diff).sqrt()
}
