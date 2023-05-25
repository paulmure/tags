use clap::Parser;

mod data_loader;
mod mat_comp;

use data_loader::netflix::load_netflix_dataset;
use mat_comp::{train, train_async, HyperParams};

#[derive(Parser, Debug)]
struct Args {
    /// Number of movies to load
    #[arg(short, long, default_value_t = 100)]
    n_movies: usize,
    /// Number of features in the decomposition matrix
    #[arg(long, default_value_t = 10)]
    n_features: usize,
    /// Model hyper parameter mu
    #[arg(long, default_value_t = 1.)]
    mu: f32,
    /// Model hyper parameter lambda_xf
    #[arg(long, default_value_t = 1.)]
    lam_xf: f32,
    /// Model hyper parameter lambda_yf
    #[arg(long, default_value_t = 1.)]
    lam_yf: f32,
    /// Model hyper parameter lambda_xb
    #[arg(long, default_value_t = 1.)]
    lam_xb: f32,
    /// Model hyper parameter lambda_yb
    #[arg(long, default_value_t = 1.)]
    lam_yb: f32,
    /// Model hyper parameter initial learning rate
    #[arg(long, default_value_t = 0.1)]
    alpha_0: f32,
    /// Model hyper parameter initial learning rate
    #[arg(long, default_value_t = 5.)]
    decay_rate: f32,
    /// Maximum number of epochs to run for
    #[arg(long, default_value_t = 1000)]
    max_epoch: usize,
    /// When to stop training
    #[arg(long, default_value_t = 0.001)]
    stopping_criterion: f32,
    /// Whether or not to do hogwild
    #[arg(long, default_value_t = true)]
    hogwild: bool,
    /// Number of worker threads in async sgd
    #[arg(long, default_value_t = 8)]
    n_workers: usize,
    /// Fifo depth in async sgd
    #[arg(long, default_value_t = 8)]
    fifo_depth: usize,
}

fn main() {
    let args = Args::parse();

    let (m, _) = load_netflix_dataset(args.n_movies);

    let h = HyperParams::new(
        args.n_features,
        args.mu,
        args.lam_xf,
        args.lam_yf,
        args.lam_xb,
        args.lam_yb,
        args.alpha_0,
        args.decay_rate,
        args.max_epoch,
        args.stopping_criterion,
    );

    if args.hogwild {
        train_async(&m, &h, args.n_workers, args.fifo_depth);
    } else {
        train(&m, &h);
    }
}
