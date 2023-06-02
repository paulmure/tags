use std::collections::VecDeque;

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

pub fn run_simulation(args: &'static Args, num_samples: usize) -> (Tick, Vec<Sample>) {
    let mut params_server = ParamsServerState::new(args, num_samples);
    let (mut workers, mut sample_chans, mut update_chans) = (vec![], vec![], vec![]);
    for _ in 0..args.n_workers {
        workers.push(WorkerState::new(args));
        sample_chans.push(VecDeque::with_capacity(args.fifo_depth));
        update_chans.push(VecDeque::with_capacity(args.fifo_depth));
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
    (params_server.tick, params_server.update_logs)
}

struct WorkerState<'a> {
    args: &'a Args,
    tick: Tick,
    next_ready: Tick,
}

fn can_pop(tick: Tick, fifo: &VecDeque<Sample>) -> bool {
    fifo.front().map(|s| tick >= s.time).unwrap_or(false)
}

impl<'a> WorkerState<'a> {
    fn new(args: &'a Args) -> Self {
        Self {
            args,
            tick: 0,
            next_ready: 0,
        }
    }

    fn ready(&self, update_tx: &VecDeque<Sample>) -> bool {
        self.tick >= self.next_ready && update_tx.len() < self.args.fifo_depth
    }

    fn tick_worker(&mut self, sample_rx: &mut VecDeque<Sample>, update_tx: &mut VecDeque<Sample>) {
        if self.ready(update_tx) && can_pop(self.tick, sample_rx) {
            let mut s = sample_rx.pop_front().unwrap();
            s.time = self.tick
                + self.args.gradient_latency
                + self.args.network_delay
                + self.args.send_delay;
            update_tx.push_back(s);
            self.next_ready = self.tick + self.args.gradient_ii;
        }
        self.tick += 1;
    }
}

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
            weight_version_queue: VecDeque::with_capacity(args.n_folders),
            fold_ready_at: 0,
            update_logs: Vec::with_capacity(num_samples),
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
            if can_pop(self.tick, update_rx) {
                updates.push(update_rx.pop_front().unwrap());
            }
        }
        self.fold_gradient(updates);
    }

    fn cleanup(&mut self) {
        self.tick = max_sample_time(&self.update_logs);
        self.update_weight_version();
        assert!(
            self.weight_version_queue.is_empty()
                && self.curr_weight_version == self.num_samples
                && self.update_logs.len() == self.num_samples
        );
    }

    fn tick_server(
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
