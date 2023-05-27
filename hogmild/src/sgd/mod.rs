use std::marker::PhantomData;

use crate::data_loader::DataLoader;

pub mod matrix_completion;

pub trait SGD<Data, Params, Sample, Update> {
    fn take_samples(&self, params: &Params, data: Vec<Data>) -> Vec<Sample>;
    fn gradient(&self, samples: &Vec<Sample>) -> Vec<Update>;
    fn fold(&self, params: &mut Params, updates: Vec<Update>);
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
}

pub struct Orchestrator<Data, Loader>
where
    Loader: DataLoader<Data>,
{
    epoch: usize,
    loader: Loader,
    phantom: PhantomData<Data>,
}
