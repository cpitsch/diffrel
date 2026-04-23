#![warn(
    clippy::redundant_clone,
    clippy::doc_markdown,
    clippy::perf,
    clippy::implicit_clone,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::manual_let_else,
    clippy::clone_on_copy,
    missing_debug_implementations,
    rust_2018_idioms
)]

pub mod multiple_testing;
pub mod permanova;
pub mod permanova_drilldown;
pub mod permutation_test_distance_of_means;
pub mod relation_extraction;
