use clap::Parser;

mod data_loader;

use data_loader::netflix::load_netflix_dataset;

#[derive(Parser, Debug)]
struct Args {
    /// Number of movies to load
    #[arg(short, long, default_value_t = 10)]
    num_movies: usize,
}

fn main() {
    let args = Args::parse();

    let (m, _) = load_netflix_dataset(args.num_movies as u32);
    let (rows, cols) = m.shape();
    println!("matrix dim = {} x {}", rows, cols);
    println!("matrix[0, 0]= {}", m[(0, 0)]);
}
