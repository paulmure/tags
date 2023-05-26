use ndarray::prelude::*;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::{rand::rngs::StdRng, rand_distr::Uniform, RandomExt};
use std::time::Instant;

use crate::data_structures::SparseMatrixView;

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
    r: &impl SparseMatrixView<f32>,
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
fn batch_loss(r: &impl SparseMatrixView<f32>, p: &ModelParams, h: &HyperParams) -> f32 {
    r.iter()
        .map(|(k, v)| sample_loss(r, p, h, k.0, k.1, *v))
        .sum()
}

fn sgd_step(
    r: &impl SparseMatrixView<f32>,
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

fn batch_sgd(
    r: &impl SparseMatrixView<f32>,
    p: &mut ModelParams,
    h: &HyperParams,
    learning_rate: f32,
) -> f32 {
    r.iter()
        .map(|(k, v)| sgd_step(r, p, h, k.0, k.1, *v, learning_rate))
        .sum()
}

fn train(r: &impl SparseMatrixView<f32>, h: &HyperParams, rng_seed: u64) -> Vec<f32> {
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
