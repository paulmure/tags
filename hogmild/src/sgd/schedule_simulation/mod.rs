use std::collections::VecDeque;

mod params_server;

use self::params_server::ParamsServerState;
use crate::args::Args;

pub type Tick = u64;

#[derive(Clone, Copy)]
pub struct Sample {
    pub time: Tick,
    pub sample_id: usize,
    pub weight_version: usize,
}

pub fn max_sample_time(samples: &[Sample]) -> Tick {
    samples.iter().map(|s| s.time).max().unwrap_or(Tick::MAX)
}

struct WorkerState<'a> {
    args: &'a Args,
    tick: Tick,
    next_ready: Tick,
}

impl<'a> WorkerState<'a> {
    fn new(args: &'a Args) -> Self {
        Self {
            args,
            tick: 0,
            next_ready: 0,
        }
    }

    pub fn tick_worker(
        &mut self,
        sample_rx: &mut VecDeque<Sample>,
        update_tx: &mut VecDeque<Sample>,
    ) {
        if self.tick >= self.next_ready && update_tx.len() < self.args.fifo_depth {
            let can_pop = sample_rx
                .front()
                .map(|s| self.tick >= s.time)
                .unwrap_or(false);
            if can_pop {
                let mut s = sample_rx.pop_front().unwrap();
                s.time = self.tick
                    + self.args.gradient_latency
                    + self.args.network_delay
                    + self.args.send_delay;
                update_tx.push_back(s);
                self.next_ready = self.tick + self.args.gradient_ii;
            }
        }
        self.tick += 1;
    }
}

pub fn run_simulation(args: &'static Args, num_samples: usize) -> (Tick, Vec<Sample>) {
    let mut params_server = ParamsServerState::new(args, num_samples);
    let mut sample_chans = vec![];
    let mut update_chans = vec![];
    let mut workers = vec![];
    for _ in 0..args.n_workers {
        sample_chans.push(VecDeque::with_capacity(args.fifo_depth));
        update_chans.push(VecDeque::with_capacity(args.fifo_depth));
        workers.push(WorkerState::new(args));
    }

    while !params_server.finished_receiving() {
        let samples = params_server.tick_server(&sample_chans, &mut update_chans);
        for i in 0..args.n_workers {
            workers[i].tick_worker(&mut sample_chans[i], &mut update_chans[i]);
        }
        for (i, sample) in samples {
            sample_chans[i].push_back(sample);
        }
    }

    params_server.cleanup();
    (params_server.curr_tick(), params_server.get_update_logs())
}
