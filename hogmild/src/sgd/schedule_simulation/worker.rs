use crossbeam::channel::{Receiver, Sender};
use std::cmp::max;

use crate::{args::Args, sgd::schedule_simulation::Sample, Tick};

pub fn run_worker(args: &Args, sample_rx: Receiver<Sample>, update_tx: Sender<Sample>) {
    let mut tick: Tick = 0;
    for mut sample in sample_rx {
        tick = max(sample.time, tick);
        let arrival_time = args.send_delay + args.network_delay + args.gradient_latency + tick;
        sample.time = arrival_time;
        update_tx.send(sample).unwrap();
        tick += args.gradient_ii;
    }
}
