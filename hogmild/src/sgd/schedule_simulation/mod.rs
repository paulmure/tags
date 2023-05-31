use crossbeam::channel::{bounded, Receiver, Sender};
use rayon::scope;

mod params_server;
mod worker;

use crate::{
    args::Args,
    sgd::schedule_simulation::{params_server::run_params_server, worker::run_worker},
    Tick,
};

#[derive(Clone, Copy)]
pub struct Sample {
    pub time: Tick,
    pub sample_id: usize,
    pub weight_version: usize,
}

pub fn max_sample_time(samples: &[Sample]) -> Tick {
    samples.iter().map(|s| s.time).max().unwrap_or(Tick::MAX)
}

pub fn run_simulation(args: &'static Args, num_samples: usize) -> Vec<Sample> {
    let mut res: Option<Vec<Sample>> = None;

    scope(|s| {
        let mut sample_txs: Vec<Sender<Sample>> = vec![];
        let mut update_rxs: Vec<Receiver<Sample>> = vec![];

        for _ in 0..args.n_workers {
            let (sample_tx, sample_rx) = bounded(args.fifo_depth);
            let (update_tx, update_rx) = bounded(args.fifo_depth);

            sample_txs.push(sample_tx);
            update_rxs.push(update_rx);

            s.spawn(|_| run_worker(args, sample_rx, update_tx));
        }

        s.spawn(|_| res = Some(run_params_server(args, num_samples, sample_txs, update_rxs)));
    });

    res.unwrap()
}
