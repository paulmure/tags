use clap::{Parser, ValueHint};
use std::path::PathBuf;

use crate::simulator::Tick;

#[derive(Parser, Debug)]
pub struct Args {
    /// Run the simulation, otherwise load the data
    #[arg(long, default_value_t = false)]
    pub simulation: bool,
    /// Number of samples to use for simulation only
    #[arg(long, default_value_t = 128)]
    pub num_samples: usize,
    /// Path to store data or update_log
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub data_path: PathBuf,

    // <<<< Common args across data sets and models >>>>
    /// Model hyper parameter initial learning rate
    #[arg(long, default_value_t = 0.1)]
    pub alpha_0: f32,
    /// Model hyper parameter initial learning rate
    #[arg(long, default_value_t = 5.)]
    pub decay_rate: f32,
    /// Maximum number of epochs to run for
    #[arg(long, default_value_t = 1000)]
    pub max_epoch: usize,
    /// When to stop training
    #[arg(long, default_value_t = 0.001)]
    pub stopping_criterion: f32,
    /// Whether or not to do hogwild
    #[arg(long, default_value_t = false)]
    pub hogwild: bool,
    /// The model to use
    #[arg(long, default_value = "mat_comp")]
    pub model: String,
    /// The dataset to use
    #[arg(long, default_value = "netflix")]
    pub dataset: String,
    /// RNG seed for weights initialization
    #[arg(long, default_value_t = 4102000)]
    pub rng_seed: u64,
    /// Number of banks to separate the weights into
    #[arg(long, default_value_t = 8)]
    pub n_weight_banks: usize,
    /// Number of worker threads in async sgd
    #[arg(long, default_value_t = 8)]
    pub n_workers: usize,
    /// Number of gradient folds that can happen in parallel
    #[arg(long, default_value_t = 8)]
    pub n_folders: usize,
    /// Fifo depth in async sgd
    #[arg(long, default_value_t = 8)]
    pub fifo_depth: usize,

    // <<<< Timing Related >>>>
    /// Time to send a sample/update
    #[arg(long, default_value_t = 4)]
    pub send_delay: Tick,
    /// Time to deliver a sample/update
    #[arg(long, default_value_t = 8)]
    pub network_delay: Tick,
    /// Time to receive a sample/update
    #[arg(long, default_value_t = 4)]
    pub receive_delay: Tick,
    /// Initiation interval of gradient calculation
    #[arg(long, default_value_t = 8)]
    pub gradient_ii: Tick,
    /// Latency of calculating one gradient
    #[arg(long, default_value_t = 32)]
    pub gradient_latency: Tick,
    /// Initiation interval of folding gradient updates
    #[arg(long, default_value_t = 8)]
    pub fold_ii: Tick,
    /// Latency of folding one gradient update
    #[arg(long, default_value_t = 32)]
    pub fold_latency: Tick,

    // <<<< Matrix completion specific >>>>
    /// Number of features in the decomposition matrix
    #[arg(long, default_value_t = 10)]
    pub n_features: usize,
    /// Model hyper parameter mu
    #[arg(long, default_value_t = 1.)]
    pub mu: f32,
    /// Model hyper parameter lambda_xf
    #[arg(long, default_value_t = 1.)]
    pub lam_xf: f32,
    /// Model hyper parameter lambda_yf
    #[arg(long, default_value_t = 1.)]
    pub lam_yf: f32,
    /// Model hyper parameter lambda_xb
    #[arg(long, default_value_t = 1.)]
    pub lam_xb: f32,
    /// Model hyper parameter lambda_yb
    #[arg(long, default_value_t = 1.)]
    pub lam_yb: f32,

    // <<<< Netflix dataset specific >>>>
    /// Number of movies to load
    #[arg(short, long, default_value_t = 100)]
    pub n_movies: usize,
}
