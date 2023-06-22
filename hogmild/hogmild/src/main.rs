use clap::Parser;

mod args;
mod data_loader;
mod data_structures;
mod mat_comp;
mod simulator;

use args::Args;
use simulator::run_simulation;

fn main() {
    let args = Args::parse();

    if args.simulation {
        let (cycle_count, _) = run_simulation(&args, args.num_samples);
        println!("{}", cycle_count);
        return;
    }

    match args.dataset.as_str() {
        "netflix" => {
            let matrix = data_loader::netflix::load_netflix_dataset(args.n_movies);
            let num_samples = matrix.nnz();
            let (cycle_count, updates) = run_simulation(&args, num_samples);
            println!("cycles per epoch: {}", cycle_count);
            let mut matrix_completion =
                mat_comp::MatrixCompletion::new(&args, matrix, updates.samples);
            let _ = matrix_completion.train();
        }
        d => {
            panic!("Unknown dataset {}", d)
        }
    }
}
