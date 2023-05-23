use clap::Parser;

mod data_loader;

use data_loader::get_netflix_data;

#[derive(Parser, Debug)]
struct Args {
    /// Number of movies to load
    #[arg(short, long, default_value_t = 10)]
    num_movies: usize,
}

fn main() {
    let args = Args::parse();

    let df = get_netflix_data(args.num_movies);
    println!("{}", df);
}
