use atomic_float::AtomicF32;
use crossbeam::channel::{bounded, Receiver, Sender};
use std::{marker::PhantomData, sync::Arc};

use crate::{args::Args, data_loader::DataLoader};

pub mod matrix_completion;
mod schedule_simulation;

pub trait HasLoss {
    fn loss(&self) -> f32;
}

pub trait HasTime {
    fn time(&self) -> usize;
    fn add_time(&mut self, inc: usize);
}

pub trait Sgd<Data, Sample, Update>
where
    Sample: HasTime,
    Update: HasTime + HasLoss,
{
    fn take_sample(&self, time: usize, data: Data) -> Sample;
    fn gradient(&self, sample: Sample) -> Update;
    fn fold(&mut self, update: Update);
}

pub struct Config {
    alpha_0: f32,
    decay_rate: f32,
    stopping_criterion: f32,
    max_epoch: usize,
}

impl Config {
    pub fn new(args: &Args) -> Self {
        Self {
            alpha_0: args.alpha_0,
            decay_rate: args.decay_rate,
            stopping_criterion: args.stopping_criterion,
            max_epoch: args.max_epoch,
        }
    }
}

pub struct Orchestrator<Data, Loader, Sample, Update>
where
    Loader: DataLoader<Data>,
{
    config: Config,
    learning_rate: Arc<AtomicF32>,
    epoch: usize,
    loader: Loader,
    phantom: PhantomData<(Data, Sample, Update)>,
}

fn worker<Data, Sample, Update>(
    config: &Config,
    sample_rx: Receiver<Sample>,
    update_tx: Sender<Update>,
    sgd: &impl Sgd<Data, Sample, Update>,
) where
    Sample: HasTime,
    Update: HasTime + HasLoss,
{
    for sample in sample_rx {
        let mut update = sgd.gradient(sample);
        update.add_time(config.gradient_latency + config.network_delay);
        update_tx.send(update).unwrap();
    }
}

impl<Data, Loader, Sample, Update> Orchestrator<Data, Loader, Sample, Update>
where
    Loader: DataLoader<Data>,
    Sample: HasTime,
    Update: HasTime + HasLoss,
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

    fn train_epoch(&self) {}

    pub fn run(self, sgd: impl Sgd<Data, Sample, Update>) {
        for _ in 0..self.config.max_epoch {
            self.train_epoch();
        }
    }
}
