use clap::Parser;
use once_cell::sync::Lazy;

mod args;
mod data_loader;
mod data_structures;
mod sgd;

use args::Args;
use sgd::{matrix_completion, schedule_simulation::run_simulation};

#[allow(clippy::redundant_closure)]
static ARGS: Lazy<Args> = Lazy::new(|| Args::parse());

fn main() {
    if ARGS.simulation_only {
        let (cycle_count, _) = run_simulation(&ARGS, ARGS.num_samples);
        println!("{}", cycle_count);
        // logs.into_iter()
        //     .for_each(|s| println!("{}, {}", s.time, s.weight_version));
        return;
    }

    match ARGS.model.as_str() {
        "mat_comp" => matrix_completion::run(&ARGS),
        m => panic!("Unknown model: {}", m),
    }
}
