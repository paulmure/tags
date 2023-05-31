use crossbeam::channel::{Receiver, Select, SelectedOperation, Sender};
use std::collections::VecDeque;

use crate::{
    args::Args,
    sgd::schedule_simulation::{max_sample_time, Sample},
    Tick,
};

struct ParamsServerState<'a> {
    tick: Tick,
    args: &'a Args,
    num_samples: usize,
    /// The id of the next sample to be sent
    next_sample: usize,
    curr_weight_version: usize,
    /// When the banks will be ready to send another sample.
    /// Earliest ready is at the front
    bank_states: VecDeque<Tick>,
    /// A new weight version to be used at a future time.
    /// Always push new versions to the back
    weight_version_queue: VecDeque<(Tick, usize)>,
    /// When the folding unit will be ready again
    fold_ready_at: Tick,
    /// The sequence of updates made to weights
    update_logs: Vec<Sample>,
}

impl<'a> ParamsServerState<'a> {
    fn new(args: &'a Args, num_samples: usize) -> Self {
        Self {
            tick: 0,
            args,
            num_samples,
            next_sample: 0,
            curr_weight_version: 0,
            bank_states: VecDeque::with_capacity(args.n_weight_banks),
            weight_version_queue: VecDeque::new(),
            fold_ready_at: 0,
            update_logs: vec![],
        }
    }

    fn has_free_weight_banks(&self) -> bool {
        self.bank_states.len() < self.args.n_weight_banks
    }

    fn has_more_samples(&self) -> bool {
        self.next_sample < self.num_samples
    }

    fn can_send(&self) -> bool {
        self.has_more_samples() && self.has_free_weight_banks()
    }

    fn can_fold(&self) -> bool {
        self.tick >= self.fold_ready_at
    }

    fn finished_receiving(&self) -> bool {
        self.update_logs.len() == self.num_samples
    }

    fn clear_free_banks(&mut self) {
        while let Some(&t) = self.bank_states.front() {
            if self.tick >= t {
                self.bank_states.pop_front();
            } else {
                return;
            }
        }
    }

    fn send_sample(&mut self, sample_tx: &Sender<Sample>) {
        debug_assert!(self.has_free_weight_banks());

        let arrival_time = self.tick + self.args.send_delay + self.args.network_delay;
        let sample = Sample {
            time: arrival_time,
            sample_id: self.next_sample,
            weight_version: self.curr_weight_version,
        };
        sample_tx.send(sample).unwrap();

        self.next_sample += 1;
        let next_ready_at = self.tick + self.args.send_delay;
        self.bank_states.push_back(next_ready_at);
    }

    /// The latest weight version we know of in the future.
    fn spear_head_weight_version(&self) -> usize {
        match self.weight_version_queue.back() {
            Some(&(_, v)) => v,
            None => self.curr_weight_version,
        }
    }

    /// Add a new weight version to be incorporated in the future.
    fn push_new_weight_version(&mut self, num_updates: usize) {
        let update_at_tick = self.tick + self.args.fold_latency;
        let new_version = self.spear_head_weight_version() + num_updates;
        self.weight_version_queue
            .push_back((update_at_tick, new_version));
    }

    /// Check if any weight version update should be patched.
    fn update_weight_version(&mut self) {
        while let Some(&(t, v)) = self.weight_version_queue.front() {
            if self.tick >= t {
                self.curr_weight_version = v;
                self.weight_version_queue.pop_front();
            } else {
                return;
            }
        }
    }

    fn fold_gradient(&mut self, updates: Vec<Sample>) {
        debug_assert!(self.can_fold());
        debug_assert!(updates.len() <= self.args.n_folders);

        self.push_new_weight_version(updates.len());
        self.fold_ready_at = self.tick + self.args.fold_ii;

        updates.into_iter().for_each(|mut update| {
            update.time += self.args.fold_latency;
            self.update_logs.push(update);
        });
    }

    /// TODO: This receive granularity is not right, this does not account
    /// for the simulated current timestamps of workers
    fn try_receive_samples(&mut self, update_rxs: &[Receiver<Sample>], recv_sel: &mut Select) {
        if !self.can_fold() {
            return;
        }

        // TODO: peek from every channel and choose the lowest one
        let updates = try_receive_all(update_rxs, recv_sel, self.args.n_folders);

        self.fold_gradient(updates);
    }

    /// TODO: This receive granularity is not right, this does not account
    /// for the simulated current timestamps of workers
    fn receive_all_updates(&mut self, update_rxs: &[Receiver<Sample>], recv_sel: &mut Select) {
        while !self.finished_receiving() {
            let updates = receive_at_least_one(update_rxs, recv_sel, self.args.n_folders);
            debug_assert!(!updates.is_empty());

            self.tick = max_sample_time(&updates);
            if !self.can_fold() {
                self.tick = self.fold_ready_at;
            }

            self.fold_gradient(updates);
        }
    }

    fn send_samples(&mut self, sample_txs: &[Sender<Sample>]) {
        if !self.has_free_weight_banks() {
            return;
        }

        for sample_tx in sample_txs {
            if sample_tx.is_full() {
                continue;
            }

            self.send_sample(sample_tx);
            if !self.can_send() {
                return;
            }
        }
    }

    fn sending_phase(
        &mut self,
        sample_txs: Vec<Sender<Sample>>,
        update_rxs: &[Receiver<Sample>],
        recv_sel: &mut Select,
    ) {
        while self.has_more_samples() {
            self.update_weight_version();
            self.clear_free_banks();
            self.send_samples(&sample_txs);
            self.try_receive_samples(update_rxs, recv_sel);
            self.tick += 1;
        }
    }

    fn run_server(&mut self, sample_txs: Vec<Sender<Sample>>, update_rxs: Vec<Receiver<Sample>>) {
        let mut recv_sel = make_receive_select(&update_rxs);

        // Move `sample_txs` to a function ensure they are dropped after all samples are sent.
        // This is not necessary, but kinda nice to shut down workers ASAP.
        self.sending_phase(sample_txs, &update_rxs, &mut recv_sel);

        self.receive_all_updates(&update_rxs, &mut recv_sel);
    }
}

pub fn run_params_server(
    args: &Args,
    num_samples: usize,
    sample_txs: Vec<Sender<Sample>>,
    update_rxs: Vec<Receiver<Sample>>,
) -> Vec<Sample> {
    let mut state = ParamsServerState::new(args, num_samples);
    state.run_server(sample_txs, update_rxs);
    state.update_logs
}

fn make_receive_select<T>(update_rxs: &[Receiver<T>]) -> Select {
    let mut sel = Select::new();
    for r in update_rxs {
        sel.recv(r);
    }
    sel
}

/// Unwrap and operation and remove the index if it is error
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

/// Blocking until one sample is received
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
/// But will not receive more than `n_folders`.
fn receive_at_least_one<T>(rxs: &[Receiver<T>], sel: &mut Select, limit: usize) -> Vec<T> {
    let first_packet = receive_one(rxs, sel);
    let mut bonus = try_receive_all(rxs, sel, limit - 1);
    bonus.push(first_packet);
    bonus
}
