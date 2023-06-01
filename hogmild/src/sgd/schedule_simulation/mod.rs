use crossbeam::channel::{bounded, Receiver, Select, SelectedOperation, Sender};
use std::cmp::max;

mod channel;
mod params_server;

use crate::{args::Args, sgd::schedule_simulation::params_server::run_params_server, Tick};

trait Node {
    fn advance_to_time(&mut self, time: Tick);
}

trait HasTime {
    fn time(&self) -> Tick;
}

#[derive(Clone, Copy)]
pub struct Sample {
    pub time: Tick,
    pub sample_id: usize,
    pub weight_version: usize,
}

impl HasTime for Sample {
    fn time(&self) -> Tick {
        self.time
    }
}

pub fn max_sample_time(samples: &[Sample]) -> Tick {
    samples.iter().map(|s| s.time).max().unwrap_or(Tick::MAX)
}

fn run_worker(args: &Args, sample_rx: Receiver<Sample>, update_tx: Sender<Sample>) {
    let mut tick: Tick = 0;
    for mut sample in sample_rx {
        tick = max(sample.time, tick);
        let arrival_time = args.gradient_latency + args.send_delay + args.network_delay + tick;
        sample.time = arrival_time;
        update_tx.send(sample).unwrap();
        tick += args.gradient_ii;
    }
}

pub fn run_simulation(args: &'static Args, num_samples: usize) -> (Tick, Vec<Sample>) {
    let mut res: Option<(Tick, Vec<Sample>)> = None;

    rayon::scope(|s| {
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

fn make_receive_select<T>(update_rxs: &[Receiver<T>]) -> Select {
    let mut sel = Select::new();
    for r in update_rxs {
        sel.recv(r);
    }
    sel
}

/// Unwrap an operation and remove the index if it is error
fn unwrap_oper<T>(oper: SelectedOperation, rxs: &[Receiver<T>], sel: &mut Select) -> Option<T> {
    let index = oper.index();
    match oper.recv(&rxs[index]) {
        Ok(packet) => Some(packet),
        Err(_) => {
            sel.remove(index);
            None
        }
    }
}

/// Try to receive as many as we can without blocking
/// But will not receive more than `limit`.
fn try_receive_all<T>(rxs: &[Receiver<T>], sel: &mut Select, limit: usize) -> Vec<T> {
    let mut res = vec![];
    while res.len() < limit {
        if let Ok(oper) = sel.try_select() {
            if let Some(packet) = unwrap_oper(oper, rxs, sel) {
                res.push(packet);
            }
        } else {
            break;
        }
    }
    res
}

/// Block until one sample is received
fn receive_one<T>(rxs: &[Receiver<T>], sel: &mut Select) -> T {
    // Must sit in a loop because chanel closure can be received as an error.
    loop {
        let oper = sel.select();
        if let Some(packet) = unwrap_oper(oper, rxs, sel) {
            return packet;
        }
    }
}

/// Blocking until at least one sample is received,
/// then try to collect as many more as we can.
/// But will not receive more than `limit`.
fn receive_at_least_one<T>(rxs: &[Receiver<T>], sel: &mut Select, limit: usize) -> Vec<T> {
    let first_packet = receive_one(rxs, sel);
    let mut bonus = try_receive_all(rxs, sel, limit - 1);
    bonus.push(first_packet);
    bonus
}
