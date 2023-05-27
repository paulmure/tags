use atomic_float::AtomicF32;
use ndarray::prelude::*;
use ndarray_rand::{
    rand::{rngs::StdRng, SeedableRng},
    rand_distr::Uniform,
    RandomExt,
};
use std::time::Instant;

use crate::{data_structures::SparseMatrixView, sgd::SGD};

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

// #[allow(clippy::too_many_arguments)]
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

struct Weights {
    x: Array2<f32>,
    y: Array2<f32>,
    xb: Array1<f32>,
    yb: Array1<f32>,
}

impl Weights {
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

struct Sample<'a> {
    u: usize,
    v: usize,
    z: f32,
    xrow: ArrayView1<'a, f32>,
    ycol: ArrayView1<'a, f32>,
    xb: f32,
    yb: f32,
}

struct Update {
    u: usize,
    v: usize,
    xrow_grd: Array1<f32>,
    ycol_grd: Array1<f32>,
    xb_grad: f32,
    yb_grad: f32,
    loss: f32,
}

struct MatrixCompletion<'a, SpM>
where
    SpM: SparseMatrixView<f32>,
{
    hyper_params: HyperParams,
    weights: Weights,
    data: &'a SpM,
    learning_rate: AtomicF32,
}

type MatrixData = ((usize, usize), f32);

impl<'a> SGD<MatrixData, Weights, Sample<'a>, Update> for MatrixCompletion {
    fn take_samples(&self, params: &Weights, data: Vec<MatrixData>) -> Vec<Sample<'a>> {
        data.iter()
            .map(|&((u, v), z)| Sample {
                u,
                v,
                z,
                xrow: params.x.slice(s![u, ..]),
                ycol: params.y.slice(s![v, ..]),
                xb: params.xb[u],
                yb: params.yb[v],
            })
            .collect()
    }

    fn gradient(&self, samples: &Vec<Sample<'a>>) -> Vec<Update> {}

    fn fold(&self, params: &mut Weights, updates: Vec<Update>) {}
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

impl MatrixCompletion {
    fn gradient(&self, sample: Sample) -> Update {
        // Forward prop
        let nnzrow = self..nnz_row(u);
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
}
