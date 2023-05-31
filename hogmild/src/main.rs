use clap::Parser;
use once_cell::sync::Lazy;

mod args;
mod data_loader;
mod data_structures;
mod sgd;

use args::Args;
use sgd::{matrix_completion, schedule_simulation::run_simulation};

pub type Tick = u64;

#[allow(clippy::redundant_closure)]
static ARGS: Lazy<Args> = Lazy::new(|| Args::parse());

fn main() {
    let logs = run_simulation(&ARGS, ARGS.num_samples);
    println!("{}", logs.last().unwrap().time);
    // match args.model.as_str() {
    //     "mat_comp" => matrix_completion::run(args),
    //     m => panic!("Unknown model: {}", m),
    // }
}
