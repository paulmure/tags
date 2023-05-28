use clap::Parser;

#[derive(Parser, Debug)]
pub struct Args {
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
    /// Fifo depth in async sgd
    #[arg(long, default_value_t = 8)]
    pub fifo_depth: usize,
    /// Time to deliver a sample/update
    #[arg(long, default_value_t = 32)]
    pub network_delay: usize,
    /// Initiation interval of gradient calculation
    #[arg(long, default_value_t = 8)]
    pub gradient_ii: usize,
    /// Latency of calculating one gradient
    #[arg(long, default_value_t = 32)]
    pub gradient_latency: usize,
    /// Initiation interval of folding gradient updates
    #[arg(long, default_value_t = 8)]
    pub fold_ii: usize,
    /// Latency of folding one gradient update
    #[arg(long, default_value_t = 32)]
    pub fold_latency: usize,

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
