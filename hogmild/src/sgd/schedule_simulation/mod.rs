use crossbeam::channel::{bounded, Receiver, Select, SelectedOperation, Sender};
use std::{cmp::max, collections::VecDeque, sync::Arc};

mod channel;
mod params_server;

use crate::args::Args;

use self::params_server::ParamsServerState;

trait HasTime {
    fn time(&self) -> Tick;
}

trait Node: HasTime {
    fn advance_to_time(&mut self, time: Tick);
}

pub type Tick = u64;

impl HasTime for Tick {
    fn time(&self) -> Tick {
        *self
    }
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

// fn run_worker(args: &Args, sample_rx: Receiver<Sample>, update_tx: Sender<Sample>) {
//     let mut tick: Tick = 0;
//     for mut sample in sample_rx {
//         tick = max(sample.time, tick);
//         let arrival_time = args.gradient_latency + args.send_delay + args.network_delay + tick;
//         sample.time = arrival_time;
//         update_tx.send(sample).unwrap();
//         tick += args.gradient_ii;
//     }
// }

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

// pub fn run_simulation(args: &'static Args, num_samples: usize) -> (Tick, Vec<Sample>) {
//     let mut res: Option<(Tick, Vec<Sample>)> = None;
//
//     rayon::scope(|s| {
//         let mut sample_txs: Vec<Sender<Sample>> = vec![];
//         let mut update_rxs: Vec<Receiver<Sample>> = vec![];
//
//         for _ in 0..args.n_workers {
//             let (sample_tx, sample_rx) = bounded(args.fifo_depth);
//             let (update_tx, update_rx) = bounded(args.fifo_depth);
//
//             sample_txs.push(sample_tx);
//             update_rxs.push(update_rx);
//
//             s.spawn(|_| run_worker(args, sample_rx, update_tx));
//         }
//
//         s.spawn(|_| res = Some(run_params_server(args, num_samples, sample_txs, update_rxs)));
//     });
//
//     res.unwrap()
// }

// fn make_receive_select<T>(update_rxs: &[Receiver<T>]) -> Select {
//     let mut sel = Select::new();
//     for r in update_rxs {
//         sel.recv(r);
//     }
//     sel
// }
//
// /// Unwrap an operation and remove the index if it is error
// fn unwrap_oper<T>(oper: SelectedOperation, rxs: &[Receiver<T>], sel: &mut Select) -> Option<T> {
//     let index = oper.index();
//     match oper.recv(&rxs[index]) {
//         Ok(packet) => Some(packet),
//         Err(_) => {
//             sel.remove(index);
//             None
//         }
//     }
// }
//
// /// Try to receive as many as we can without blocking
// /// But will not receive more than `limit`.
// fn try_receive_all<T>(rxs: &[Receiver<T>], sel: &mut Select, limit: usize) -> Vec<T> {
//     let mut res = vec![];
//     while res.len() < limit {
//         if let Ok(oper) = sel.try_select() {
//             if let Some(packet) = unwrap_oper(oper, rxs, sel) {
//                 res.push(packet);
//             }
//         } else {
//             break;
//         }
//     }
//     res
// }
//
// /// Block until one sample is received
// fn receive_one<T>(rxs: &[Receiver<T>], sel: &mut Select) -> T {
//     // Must sit in a loop because chanel closure can be received as an error.
//     loop {
//         let oper = sel.select();
//         if let Some(packet) = unwrap_oper(oper, rxs, sel) {
//             return packet;
//         }
//     }
// }
//
// /// Blocking until at least one sample is received,
// /// then try to collect as many more as we can.
// /// But will not receive more than `limit`.
// fn receive_at_least_one<T>(rxs: &[Receiver<T>], sel: &mut Select, limit: usize) -> Vec<T> {
//     let first_packet = receive_one(rxs, sel);
//     let mut bonus = try_receive_all(rxs, sel, limit - 1);
//     bonus.push(first_packet);
//     bonus
// }
