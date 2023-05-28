use atomic_float::AtomicF32;
use std::{marker::PhantomData, sync::Arc};

use crate::{args::Args, data_loader::DataLoader};

pub mod matrix_completion;

pub trait HasLoss {
    fn loss(&self) -> f32;
}

pub trait Sgd<Data, Sample, Update>
where
    Update: HasLoss,
{
    fn take_samples(&self, data: Vec<Data>) -> Vec<Sample>;
    fn gradient(&self, samples: Vec<Sample>) -> Vec<Update>;
    fn fold(&mut self, updates: Vec<Update>);
}

pub struct Config {
    n_weight_banks: usize,
    n_workers: usize,
    fifo_depth: usize,
    network_delay: usize,
    gradient_ii: usize,
    gradient_latency: usize,
    fold_ii: usize,
    fold_latency: usize,

    alpha_0: f32,
    decay_rate: f32,
    stopping_criterion: f32,
    max_epoch: usize,
}

impl Config {
    pub fn new(args: &Args) -> Self {
        Self {
            n_weight_banks: args.n_weight_banks,
            n_workers: args.n_workers,
            fifo_depth: args.fifo_depth,
            network_delay: args.network_delay,
            gradient_ii: args.gradient_ii,
            gradient_latency: args.gradient_latency,
            fold_ii: args.fold_ii,
            fold_latency: args.fold_latency,
            alpha_0: args.alpha_0,
            decay_rate: args.decay_rate,
            stopping_criterion: args.stopping_criterion,
            max_epoch: args.max_epoch,
        }
    }
}

pub struct Orchestrator<Data, Loader>
where
    Loader: DataLoader<Data>,
{
    config: Config,
    learning_rate: Arc<AtomicF32>,
    epoch: usize,
    loader: Loader,
    phantom: PhantomData<Data>,
}

impl<Data, Loader> Orchestrator<Data, Loader>
where
    Loader: DataLoader<Data>,
{
    pub fn new(config: Config, learning_rate: Arc<AtomicF32>, loader: Loader) -> Self {
        Self {
            config,
            learning_rate,
            epoch: 0,
            loader,
            phantom: Default::default(),
        }
    }

    pub fn run<Sample, Update>(self, sgd: impl Sgd<Data, Sample, Update>)
    where
        Update: HasLoss,
    {
    }
}
