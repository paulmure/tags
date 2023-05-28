use clap::Parser;

mod args;
mod data_loader;
mod data_structures;
mod sgd;

use args::Args;
use sgd::matrix_completion;

fn main() {
    let args = Args::parse();
    match args.model.as_str() {
        "mat_comp" => matrix_completion::run(args),
        m => panic!("Unknown model: {}", m),
    }
}
