use crossbeam::channel::{bounded, Receiver, Select, SelectedOperation, Sender};
use ndarray::prelude::*;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::{rand::rngs::StdRng, rand_distr::Uniform, RandomExt};
use std::iter::Peekable;
use std::thread;
use std::time::Instant;

use crate::args::Args;
use crate::data_loader::netflix::{load_netflix_dataset, NetflixMatrix};

struct HyperParams {
    n_features: usize,
    mu: f32,
    lam_xf: f32,
    lam_yf: f32,
    lam_xb: f32,
    lam_yb: f32,
    alpha_0: f32,
    decay_rate: f32,
    max_epoch: usize,
    stopping_criterion: f32,
}

#[allow(clippy::too_many_arguments)]
impl HyperParams {
    fn new(
        n_features: usize,
        mu: f32,
        lam_xf: f32,
        lam_yf: f32,
        lam_xb: f32,
        lam_yb: f32,
        alpha_0: f32,
        decay_rate: f32,
        max_epoch: usize,
        stopping_criterion: f32,
    ) -> Self {
        Self {
            n_features,
            mu,
            lam_xf,
            lam_yf,
            lam_xb,
            lam_yb,
            alpha_0,
            decay_rate,
            max_epoch,
            stopping_criterion,
        }
    }
}

struct ModelParams {
    x: Array2<f32>,
    y: Array2<f32>,
    xb: Array1<f32>,
    yb: Array1<f32>,
}

impl ModelParams {
    fn new(n_rows: usize, n_cols: usize, n_features: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            x: Array::random_using((n_rows, n_features), Uniform::new(-1., 1.), &mut rng),
            y: Array::random_using((n_cols, n_features), Uniform::new(-1., 1.), &mut rng),
            xb: Array::random_using(n_rows, Uniform::new(-1., 1.), &mut rng),
            yb: Array::random_using(n_cols, Uniform::new(-1., 1.), &mut rng),
        }
    }
}

fn pred(x: &ArrayView1<f32>, y: &ArrayView1<f32>, xb: f32, yb: f32, mu: f32) -> f32 {
    x.dot(y) + xb + yb + mu
}

fn error(r: f32, x: &ArrayView1<f32>, y: &ArrayView1<f32>, xb: f32, yb: f32, mu: f32) -> f32 {
    r - pred(x, y, xb, yb, mu)
}

fn regu(x: &ArrayView1<f32>, lam: f32, nnz: usize) -> f32 {
    let x2 = x.mapv(|n| n.powi(2));
    x2.sum() * lam / (nnz as f32)
}

#[allow(dead_code)]
fn sample_loss(
    r: &NetflixMatrix,
    p: &ModelParams,
    h: &HyperParams,
    u: usize,
    i: usize,
    z: f32,
) -> f32 {
    let nnzrow = r.nnz_row(u);
    let nnzcol = r.nnz_col(i);

    let xrow = p.x.slice(s![u, ..]);
    let ycol = p.y.slice(s![i, ..]);
    let xb = p.xb[u];
    let yb = p.yb[i];

    let e = error(z, &xrow, &ycol, xb, yb, h.mu);
    let x_regu = regu(&xrow, h.lam_xf, nnzrow);
    let y_regu = regu(&ycol, h.lam_yf, nnzcol);
    let xb_regu = xb * h.lam_xb / (nnzrow as f32);
    let yb_regu = yb * h.lam_yb / (nnzcol as f32);

    e.powi(2) + x_regu + y_regu + xb_regu + yb_regu
}

#[allow(dead_code)]
fn batch_loss(r: &NetflixMatrix, p: &ModelParams, h: &HyperParams) -> f32 {
    r.entries
        .iter()
        .map(|(k, v)| sample_loss(r, p, h, k.0, k.1, *v))
        .sum()
}

fn sgd_step(
    r: &NetflixMatrix,
    p: &mut ModelParams,
    h: &HyperParams,
    u: usize,
    i: usize,
    z: f32,
    learning_rate: f32,
) -> f32 {
    // Forward prop
    let nnzrow = r.nnz_row(u);
    let nnzcol = r.nnz_col(i);

    let xrow = p.x.slice(s![u, ..]);
    let ycol = p.y.slice(s![i, ..]);
    let xb = p.xb[u];
    let yb = p.yb[i];

    let e = error(z, &xrow, &ycol, xb, yb, h.mu);
    let x_regu = regu(&xrow, h.lam_xf, nnzrow);
    let y_regu = regu(&ycol, h.lam_yf, nnzcol);
    let xb_regu = xb * h.lam_xb / (nnzrow as f32);
    let yb_regu = yb * h.lam_yb / (nnzcol as f32);

    let loss = e.powi(2) + x_regu + y_regu + xb_regu + yb_regu;

    // Backward prop
    p.xb[u] += learning_rate * (e - xb_regu);
    p.yb[i] += learning_rate * (e - yb_regu);

    let new_x = &xrow + (learning_rate * ((e * &ycol) - (x_regu * &xrow)));
    let new_y = &ycol + (learning_rate * ((e * &xrow) - (y_regu * &ycol)));

    p.x.slice_mut(s![u, ..]).assign(&new_x);
    p.y.slice_mut(s![i, ..]).assign(&new_y);

    loss
}

fn batch_sgd(r: &NetflixMatrix, p: &mut ModelParams, h: &HyperParams, learning_rate: f32) -> f32 {
    r.entries
        .iter()
        .map(|(k, v)| sgd_step(r, p, h, k.0, k.1, *v, learning_rate))
        .sum()
}

fn train(r: &NetflixMatrix, h: &HyperParams, rng_seed: u64) -> Vec<f32> {
    let mut p = ModelParams::new(r.shape().0, r.shape().1, h.n_features, rng_seed);

    let mut res = vec![];

    for epoch in 0..h.max_epoch {
        let start_time = Instant::now();

        let learning_rate = 1. / (1. + (h.decay_rate * (epoch as f32))) * h.alpha_0;
        let curr_loss = batch_sgd(r, &mut p, h, learning_rate);
        let last_loss = *res.last().unwrap_or(&f32::MAX);
        res.push(curr_loss);

        let elapsed = start_time.elapsed();
        println!("{}, {}", curr_loss, elapsed.as_secs_f64());

        let delta = (last_loss - curr_loss) / last_loss;
        if delta < h.stopping_criterion {
            break;
        }
    }

    res
}

/// A packet send by control node to compute node to perform a step of sgd
struct SamplePacket {
    u: usize,
    i: usize,
    xrow: Array1<f32>,
    ycol: Array1<f32>,
    xb: f32,
    yb: f32,
    z: f32,
    learning_rate: f32,
}

struct UpdatePacket {
    u: usize,
    i: usize,
    x_update: Array1<f32>,
    y_update: Array1<f32>,
    xb_update: f32,
    yb_update: f32,
    loss: f32,
}

fn sgd_step_async(r: &NetflixMatrix, h: &HyperParams, sample: &SamplePacket) -> UpdatePacket {
    // Forward prop
    let nnzrow = r.nnz_row(sample.u);
    let nnzcol = r.nnz_col(sample.i);

    let xrow = sample.xrow.view();
    let ycol = sample.ycol.view();

    let e = error(sample.z, &xrow, &ycol, sample.xb, sample.yb, h.mu);
    let x_regu = regu(&xrow, h.lam_xf, nnzrow);
    let y_regu = regu(&ycol, h.lam_yf, nnzcol);
    let xb_regu = sample.xb * h.lam_xb / (nnzrow as f32);
    let yb_regu = sample.yb * h.lam_yb / (nnzcol as f32);

    let loss = e.powi(2) + x_regu + y_regu + xb_regu + yb_regu;

    // Backward prop
    let xb_update = sample.learning_rate * (e - xb_regu);
    let yb_update = sample.learning_rate * (e - yb_regu);

    let x_update = sample.learning_rate * ((e * &ycol) - (x_regu * &xrow));
    let y_update = sample.learning_rate * ((e * &xrow) - (y_regu * &ycol));

    UpdatePacket {
        u: sample.u,
        i: sample.i,
        x_update,
        y_update,
        xb_update,
        yb_update,
        loss,
    }
}

fn sgd_async_worker(
    r: &NetflixMatrix,
    h: &HyperParams,
    sample_rx: Receiver<SamplePacket>,
    update_tx: Sender<UpdatePacket>,
) {
    for sample in sample_rx {
        let loss = sgd_step_async(r, h, &sample);
        update_tx.send(loss).unwrap();
    }
}

fn make_sample_packet(
    p: &ModelParams,
    u: usize,
    i: usize,
    z: f32,
    learning_rate: f32,
) -> SamplePacket {
    SamplePacket {
        u,
        i,
        xrow: p.x.slice(s![u, ..]).to_owned(),
        ycol: p.y.slice(s![i, ..]).to_owned(),
        xb: p.xb[u],
        yb: p.yb[i],
        z,
        learning_rate,
    }
}

fn send_samples<'a>(
    sample_sel: &mut Select,
    sample_txs: &[Sender<SamplePacket>],
    data: &mut Peekable<impl Iterator<Item = (&'a (usize, usize), &'a f32)>>,
    p: &mut ModelParams,
    learning_rate: f32,
) -> usize {
    let mut samples_sent = 0;
    while data.peek().is_some() {
        if let Ok(oper) = sample_sel.try_select() {
            let index = oper.index();
            if let Some(((u, i), z)) = data.next() {
                oper.send(
                    &sample_txs[index],
                    make_sample_packet(p, *u, *i, *z, learning_rate),
                )
                .unwrap();
                samples_sent += 1;
            }
        } else {
            break;
        }
    }
    samples_sent
}

fn update_gradient(update: UpdatePacket, p: &mut ModelParams) {
    let x_idx = s![update.u, ..];
    let new_x = &p.x.slice(x_idx) + &update.x_update;
    p.x.slice_mut(x_idx).assign(&new_x);

    let y_idx = s![update.i, ..];
    let new_y = &p.y.slice(y_idx) + &update.y_update;
    p.y.slice_mut(y_idx).assign(&new_y);

    p.xb[update.u] += update.xb_update;
    p.yb[update.i] += update.yb_update;
}

fn receive_updates(
    update_sel: &mut Select,
    update_rxs: &[Receiver<UpdatePacket>],
    p: &mut ModelParams,
    minimum: usize,
) -> (usize, f32) {
    let mut updates_received = 0;
    let mut total_loss = 0.;

    let mut pull_update = |oper: SelectedOperation| {
        let index = oper.index();
        let update = oper.recv(&update_rxs[index]).unwrap();
        total_loss += update.loss;
        update_gradient(update, p);
        updates_received += 1;
    };

    for _ in 0..minimum {
        let oper = update_sel.select();
        pull_update(oper);
    }

    while let Ok(oper) = update_sel.try_select() {
        pull_update(oper);
    }

    (updates_received, total_loss)
}

fn batch_sgd_async(
    r: &NetflixMatrix,
    p: &mut ModelParams,
    learning_rate: f32,
    sample_txs: &[Sender<SamplePacket>],
    update_rxs: &[Receiver<UpdatePacket>],
) -> f32 {
    let mut total_loss: f32 = 0.;

    let mut sample_sel = Select::new();
    for stx in sample_txs {
        sample_sel.send(stx);
    }

    let mut update_sel = Select::new();
    for urx in update_rxs {
        update_sel.recv(urx);
    }

    let mut data = r.entries.iter().peekable();
    let mut total_sent = 0;

    loop {
        if data.peek().is_none() {
            break;
        }

        let samples_sent = send_samples(&mut sample_sel, sample_txs, &mut data, p, learning_rate);
        assert!(samples_sent >= 1, "Failed to send at least one sample");
        total_sent += samples_sent;

        let (updates_received, new_loss) = receive_updates(&mut update_sel, update_rxs, p, 1);
        total_sent -= updates_received;
        total_loss += new_loss;
    }

    receive_updates(&mut update_sel, update_rxs, p, total_sent);

    total_loss
}

fn train_async(
    r: &NetflixMatrix,
    h: &HyperParams,
    n_workers: usize,
    fifo_depth: usize,
    rng_seed: u64,
) -> Vec<f32> {
    let mut p = ModelParams::new(r.shape().0, r.shape().1, h.n_features, rng_seed);
    let mut history = vec![];

    thread::scope(|s| {
        let mut sample_txs = vec![];
        let mut update_rxs = vec![];

        for _ in 0..n_workers {
            let (sample_tx, sample_rx): (Sender<SamplePacket>, Receiver<SamplePacket>) =
                bounded(fifo_depth);
            let (update_tx, update_rx): (Sender<UpdatePacket>, Receiver<UpdatePacket>) =
                bounded(fifo_depth);

            sample_txs.push(sample_tx);
            update_rxs.push(update_rx);

            s.spawn(move || sgd_async_worker(r, h, sample_rx, update_tx));
        }

        for epoch in 0..h.max_epoch {
            let start_time = Instant::now();

            let learning_rate = 1. / (1. + (h.decay_rate * (epoch as f32))) * h.alpha_0;
            let curr_loss = batch_sgd_async(r, &mut p, learning_rate, &sample_txs, &update_rxs);
            let last_loss = *history.last().unwrap_or(&f32::MAX);
            history.push(curr_loss);

            let elapsed = start_time.elapsed();
            println!("{}, {}", curr_loss, elapsed.as_secs_f64());

            let delta = (last_loss - curr_loss) / last_loss;
            if delta < h.stopping_criterion {
                break;
            }
        }
    });

    history
}

pub fn matrix_completion(args: Args) {
    let m = match args.data_set.as_str() {
        "netflix" => load_netflix_dataset(args.n_movies),
        ds => panic!("Unknown dataset: {}", ds),
    };

    // let num_entries: f64 = m.entries.len() as f64;
    // let matrix_size: f64 = (m.n_row * m.n_col) as f64;
    // let sparsity = (1. - num_entries / matrix_size) * 100.;
    // println!(
    //     "Dataset has {} samples with sparsity {:.2}%",
    //     m.entries.len(),
    //     sparsity
    // );

    let h = HyperParams::new(
        args.n_features,
        args.mu,
        args.lam_xf,
        args.lam_yf,
        args.lam_xb,
        args.lam_yb,
        args.alpha_0,
        args.decay_rate,
        args.max_epoch,
        args.stopping_criterion,
    );

    if args.hogwild {
        train_async(&m, &h, args.n_workers, args.fifo_depth, args.rng_seed);
    } else {
        train(&m, &h, args.rng_seed);
    }
}
