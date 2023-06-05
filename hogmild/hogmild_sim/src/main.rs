use clap::Parser;

mod args;
mod data_loader;
mod data_structures;
mod simulator;

use args::Args;
use simulator::run_simulation;

fn main() {
    let args = Args::parse();

    if args.simulation {
        let (cycle_count, update_logs) = run_simulation(&args, args.num_samples);
        println!("{}", cycle_count);
        if args.print_data {
            println!("{}", update_logs);
        }
        return;
    }

    match args.dataset.as_str() {
        "netflix" => {
            let matrix = data_loader::netflix::load_netflix_dataset(args.n_movies);
            println!("{}", matrix.nnz());
            if args.print_data {
                println!("{}", matrix);
            }
        }
        d => {
            panic!("Unknown dataset {}", d)
        }
    }
}
