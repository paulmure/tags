use clap::Parser;

mod args;
mod data_loader;
mod data_structures;
mod simulator;

use args::Args;
use simulator::{run_simulation, Sample};

fn print_update_logs(mut update_logs: Vec<Sample>) {
    update_logs.sort_by_key(|s| s.weight_version);

    for sample in update_logs {
        println!("{},{}", sample.sample_id, sample.weight_version);
    }
}

fn main() {
    let args = Args::parse();
    if args.simulation {
        let (cycle_count, update_logs) = run_simulation(&args, args.num_samples);
        print_update_logs(update_logs);
        println!("{}", cycle_count);
        return;
    }

    match args.dataset.as_str() {
        "netflix" => {
            let matrix = data_loader::netflix::load_netflix_dataset(args.n_movies);
            println!("{}", matrix.nnz());
            println!("{}", matrix);
        }
        d => {
            panic!("Unknown dataset {}", d)
        }
    }
}
