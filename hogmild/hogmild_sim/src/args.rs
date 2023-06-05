use clap::Parser;

use crate::simulator::Tick;

#[derive(Parser, Debug)]
pub struct Args {
    /// Run the simulation, otherwise load the data
    #[arg(long, default_value_t = false)]
    pub simulation: bool,
    /// Number of samples to use for simulation only
    #[arg(long, default_value_t = 128)]
    pub num_samples: usize,
    /// The dataset to use
    #[arg(long, default_value = "netflix")]
    pub dataset: String,
    /// Whether to print the data associated with the data load or simulation
    #[arg(long, default_value_t = false)]
    pub print_data: bool,

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

    // <<<< Netflix dataset specific >>>>
    /// Number of movies to load
    #[arg(short, long, default_value_t = 100)]
    pub n_movies: usize,
}
