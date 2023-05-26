use clap::Parser;

mod args;
mod data_loader;
mod data_structures;
mod sgd;

mod mat_comp;
use args::Args;
use mat_comp::matrix_completion;

fn main() {
    let args = Args::parse();
    match args.model.as_str() {
        "mat_comp" => matrix_completion(args),
        m => panic!("Unknown model: {}", m),
    }
}
