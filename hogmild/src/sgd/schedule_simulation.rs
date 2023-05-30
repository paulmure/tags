use std::collections::VecDeque;

use crate::args::Args;

pub struct Config {
    n_weight_banks: usize,
    n_workers: usize,
    fifo_depth: usize,
    send_delay: usize,
    network_delay: usize,
    receive_delay: usize,
    gradient_ii: usize,
    gradient_latency: usize,
    fold_ii: usize,
    fold_latency: usize,
    num_data: usize,
}

impl Config {
    fn new(num_data: usize, args: &Args) -> Self {
        Self {
            n_weight_banks: args.n_weight_banks,
            n_workers: args.n_workers,
            fifo_depth: args.fifo_depth,
            send_delay: args.send_delay,
            network_delay: args.network_delay,
            receive_delay: args.receive_delay,
            gradient_ii: args.gradient_ii,
            gradient_latency: args.gradient_latency,
            fold_ii: args.fold_ii,
            fold_latency: args.fold_latency,
            num_data,
        }
    }
}

enum PacketType {
    SAMPLE,
    UPDATE,
}

struct Packet {
    data: usize,
    time: usize,
    packet_type: PacketType,
}

pub enum EventType {
    GRADIENT,
    UPDATE,
}

pub struct Event {
    data: usize,
    kind: EventType,
}

pub fn get_schedule(config: &Config) -> Vec<Event> {
    let mut updates: Vec<Event> = vec![];

    let sample_fifos: Vec<VecDeque<Packet>> = Vec::with_capacity(config.n_workers);
    let update_fifos: Vec<VecDeque<Packet>> = Vec::with_capacity(config.n_workers);

    let mut curr_data: usize = 0;
    let mut time: usize = 0;

    loop {
        for _ in 0..config.n_weight_banks {}

        if curr_data == config.num_data {
            break;
        }
    }

    updates
}
