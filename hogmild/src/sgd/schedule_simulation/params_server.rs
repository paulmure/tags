use std::collections::VecDeque;

use super::*;

pub struct ParamsServerState<'a> {
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
    pub fn new(args: &'a Args, num_samples: usize) -> Self {
        Self {
            tick: 0,
            args,
            num_samples,
            next_sample: 0,
            curr_weight_version: 0,
            bank_states: VecDeque::with_capacity(args.n_weight_banks),
            weight_version_queue: VecDeque::with_capacity(args.n_folders),
            fold_ready_at: 0,
            update_logs: Vec::with_capacity(num_samples),
        }
    }

    pub fn curr_tick(&self) -> Tick {
        self.tick
    }

    pub fn get_update_logs(self) -> Vec<Sample> {
        self.update_logs
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

    pub fn finished_receiving(&self) -> bool {
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

    /// The latest weight version we know of in the future.
    fn spearhead_weight_version(&self) -> usize {
        self.weight_version_queue
            .back()
            .map_or(self.curr_weight_version, |&(_, v)| v)
    }

    /// Add a new weight version to be incorporated in the future.
    fn push_new_weight_version(&mut self, num_updates: usize) {
        let update_at_tick = self.tick + self.args.fold_latency;
        let new_version = self.spearhead_weight_version() + num_updates;
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

    fn send_next_sample(&mut self) -> Sample {
        debug_assert!(self.has_free_weight_banks() && self.has_more_samples());

        let arrival_time = self.tick + self.args.send_delay + self.args.network_delay;
        let sample = Sample {
            time: arrival_time,
            sample_id: self.next_sample,
            weight_version: self.curr_weight_version,
        };

        self.next_sample += 1;
        let next_ready_at = self.tick + self.args.send_delay;
        self.bank_states.push_back(next_ready_at);

        sample
    }

    fn try_send_samples(&mut self, sample_txs: &[VecDeque<Sample>]) -> Vec<(usize, Sample)> {
        if !self.can_send() {
            return vec![];
        }

        let mut res = vec![];

        for (i, sample_tx) in sample_txs.iter().enumerate() {
            if sample_tx.len() == self.args.fifo_depth {
                continue;
            }

            res.push((i, self.send_next_sample()));
            if !self.can_send() {
                return res;
            }
        }

        res
    }

    // fn send_all_samples(
    //     &mut self,
    //     sample_txs: Vec<Sender<Sample>>,
    //     update_rxs: &[Receiver<Sample>],
    //     recv_sel: &mut Select,
    // ) {
    //     while self.has_more_samples() {
    //         self.update_weight_version();
    //         self.clear_free_banks();
    //         self.try_send_samples(&sample_txs);
    //         self.try_receive_samples(update_rxs, recv_sel);
    //         self.tick += 1;
    //     }
    // }

    fn fold_gradient(&mut self, updates: Vec<Sample>) {
        debug_assert!(self.can_fold());
        debug_assert!(updates.len() <= self.args.n_folders);

        self.push_new_weight_version(updates.len());
        self.fold_ready_at = self.tick + self.args.fold_ii;

        for mut update in updates {
            update.time = self.tick + self.args.fold_latency;
            self.update_logs.push(update);
        }
    }

    fn try_receive_samples(&mut self, update_rxs: &mut [VecDeque<Sample>]) {
        if !self.can_fold() {
            return;
        }

        let mut updates = vec![];
        for update_rx in update_rxs {
            let can_pop = update_rx
                .front()
                .map(|s| self.tick >= s.time)
                .unwrap_or(false);
            if can_pop {
                updates.push(update_rx.pop_front().unwrap());
            }
        }
        self.fold_gradient(updates);
    }

    // /// TODO: This receive granularity is not right, this does not account
    // /// for the simulated current timestamps of workers
    // fn receive_all_updates(&mut self, update_rxs: &[Receiver<Sample>], recv_sel: &mut Select) {
    //     while !self.finished_receiving() {
    //         let updates = receive_at_least_one(update_rxs, recv_sel, self.args.n_folders);
    //         debug_assert!(!updates.is_empty());
    //
    //         self.tick = max_sample_time(&updates);
    //         if !self.can_fold() {
    //             self.tick = self.fold_ready_at;
    //         }
    //
    //         self.fold_gradient(updates);
    //     }
    // }

    pub fn cleanup(&mut self) {
        self.tick = max_sample_time(&self.update_logs);
        self.update_weight_version();
        assert!(
            self.weight_version_queue.is_empty()
                && self.curr_weight_version == self.num_samples
                && self.update_logs.len() == self.num_samples
        );
    }

    // fn run_server(
    //     &mut self,
    //     sample_txs: Vec<Sender<Sample>>,
    //     update_rxs: Vec<Receiver<Sample>>,
    // ) -> Tick {
    //     let mut recv_sel = make_receive_select(&update_rxs);
    //     self.send_all_samples(sample_txs, &update_rxs, &mut recv_sel);
    //     self.receive_all_updates(&update_rxs, &mut recv_sel);
    //
    //     self.cleanup();
    //     self.tick
    // }

    pub fn tick_server(
        &mut self,
        sample_txs: &[VecDeque<Sample>],
        update_rxs: &mut [VecDeque<Sample>],
    ) -> Vec<(usize, Sample)> {
        self.clear_free_banks();
        self.update_weight_version();
        let samples = self.try_send_samples(sample_txs);
        self.try_receive_samples(update_rxs);
        self.tick += 1;
        samples
    }
}

// pub fn run_params_server(
//     args: &Args,
//     num_samples: usize,
//     sample_txs: Vec<Sender<Sample>>,
//     update_rxs: Vec<Receiver<Sample>>,
// ) -> (Tick, Vec<Sample>) {
//     let state = ParamsServerState::new(args, num_samples);
//     // let cycle_count = state.run_server(sample_txs, update_rxs);
//     (0, state.update_logs)
// }
