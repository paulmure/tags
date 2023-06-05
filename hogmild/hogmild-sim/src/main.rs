use clap::Parser;
use std::{fs, io::Write, path::PathBuf};

mod args;
mod data_loader;
mod data_structures;
mod simulator;

use args::Args;
use simulator::{run_simulation, Sample};

fn write_update_logs(path: PathBuf, mut update_logs: Vec<Sample>) {
    update_logs.sort_by_key(|s| s.weight_version);

    if path.exists() {
        fs::remove_file(&path).unwrap();
    }

    let mut file = fs::File::create(path).unwrap();
    writeln!(&mut file, "Sample,WeightVersion").unwrap();

    for sample in update_logs {
        writeln!(&mut file, "{},{}", sample.sample_id, sample.weight_version).unwrap();
    }
}

fn main() {
    let args = Args::parse();
    if args.simulation {
        let (cycle_count, update_logs) = run_simulation(&args, args.num_samples);
        write_update_logs(args.data_path, update_logs);
        println!("{}", cycle_count);
        return;
    }

    match args.dataset.as_str() {
        "netflix" => {
            let matrix = data_loader::netflix::load_netflix_dataset(args.n_movies);
            matrix.save(args.data_path);
            println!("{}", matrix.nnz());
        }
        d => {
            panic!("Unknown dataset {}", d)
        }
    }
}
