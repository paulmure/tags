use atomic_float::AtomicF32;
use ndarray::prelude::*;
use ndarray_rand::{
    rand::{rngs::StdRng, SeedableRng},
    rand_distr::Uniform,
    RandomExt,
};
use std::sync::{atomic::Ordering, Arc};

use crate::{
    args::Args,
    data_loader,
    data_structures::SparseMatrixView,
    sgd::{Config, HasLoss, HasTime, Orchestrator, Sgd},
};

struct HyperParams {
    mu: f32,
    lam_xf: f32,
    lam_yf: f32,
    lam_xb: f32,
    lam_yb: f32,
}

impl HyperParams {
    fn new(args: &Args) -> Self {
        Self {
            mu: args.mu,
            lam_xf: args.lam_xf,
            lam_yf: args.lam_yf,
            lam_xb: args.lam_xb,
            lam_yb: args.lam_yb,
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

struct Sample {
    time: usize,
    u: usize,
    v: usize,
    z: f32,
    xrow: Array1<f32>,
    ycol: Array1<f32>,
    xb: f32,
    yb: f32,
}

impl HasTime for Sample {
    fn time(&self) -> usize {
        self.time
    }

    fn add_time(&mut self, inc: usize) {
        self.time += inc;
    }
}

struct Update {
    time: usize,
    u: usize,
    v: usize,
    xrow_grad: Array1<f32>,
    ycol_grad: Array1<f32>,
    xb_grad: f32,
    yb_grad: f32,
    loss: f32,
}

impl HasTime for Update {
    fn time(&self) -> usize {
        self.time
    }

    fn add_time(&mut self, inc: usize) {
        self.time += inc;
    }
}

impl HasLoss for Update {
    fn loss(&self) -> f32 {
        self.loss
    }
}

struct MatrixCompletion<SpM>
where
    SpM: SparseMatrixView<f32>,
{
    hyper_params: HyperParams,
    weights: Weights,
    data: SpM,
    learning_rate: Arc<AtomicF32>,
}

type MatrixData = ((usize, usize), f32);

impl<SpM> Sgd<MatrixData, Sample, Update> for MatrixCompletion<SpM>
where
    SpM: SparseMatrixView<f32>,
{
    fn take_sample(&self, time: usize, ((u, v), z): MatrixData) -> Sample {
        Sample {
            time,
            u,
            v,
            z,
            xrow: self.weights.x.slice(s![u, ..]).to_owned(),
            ycol: self.weights.y.slice(s![v, ..]).to_owned(),
            xb: self.weights.xb[u],
            yb: self.weights.yb[v],
        }
    }

    fn gradient(&self, sample: Sample) -> Update {
        // Forward prop
        let nnzrow = self.data.nnz_row(sample.u);
        let nnzcol = self.data.nnz_col(sample.v);

        let e = error(
            sample.z,
            &sample.xrow.view(),
            &sample.ycol.view(),
            sample.xb,
            sample.yb,
            self.hyper_params.mu,
        );
        let x_regu = regu(&sample.xrow.view(), self.hyper_params.lam_xf, nnzrow);
        let y_regu = regu(&sample.ycol.view(), self.hyper_params.lam_yf, nnzcol);
        let xb_regu = sample.xb * self.hyper_params.lam_xb / (nnzrow as f32);
        let yb_regu = sample.yb * self.hyper_params.lam_yb / (nnzcol as f32);

        let loss = e.powi(2) + x_regu + y_regu + xb_regu + yb_regu;

        // Backward prop
        let learning_rate = self.learning_rate.load(Ordering::Relaxed);
        let xb_grad = learning_rate * (e - xb_regu);
        let yb_grad = learning_rate * (e - yb_regu);

        let x_grad = learning_rate * ((e * &sample.ycol) - (x_regu * &sample.xrow));
        let y_grad = learning_rate * ((e * &sample.xrow) - (y_regu * &sample.ycol));

        Update {
            time: sample.time(),
            u: sample.u,
            v: sample.v,
            xrow_grad: x_grad,
            ycol_grad: y_grad,
            xb_grad,
            yb_grad,
            loss,
        }
    }

    fn fold(&mut self, update: Update) {
        self.weights.xb[update.u] += update.xb_grad;
        self.weights.yb[update.v] += update.yb_grad;

        let x_idx = s![update.u, ..];
        let new_x = &self.weights.x.slice(x_idx) + &update.xrow_grad;
        self.weights.x.slice_mut(x_idx).assign(&new_x);

        let y_idx = s![update.v, ..];
        let new_y = &self.weights.y.slice(y_idx) + &update.ycol_grad;
        self.weights.y.slice_mut(y_idx).assign(&new_y);
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

impl<SpM> MatrixCompletion<SpM>
where
    SpM: SparseMatrixView<f32>,
{
    fn new(
        hyper_params: HyperParams,
        weights: Weights,
        data: SpM,
        learning_rate: Arc<AtomicF32>,
    ) -> Self {
        Self {
            hyper_params,
            weights,
            data,
            learning_rate,
        }
    }
}

fn run_matrix_completion(args: &Args) {
    let matrix = data_loader::netflix::load_netflix_dataset(args.n_movies);
    let data_loader = data_loader::netflix::NetflixDataLoader::new(&matrix);
    let config = Config::new(&args);
    let learning_rate = Arc::new(AtomicF32::new(args.alpha_0));
    let orchestrator = Orchestrator::new(config, Arc::clone(&learning_rate), data_loader);

    let weights = Weights::new(
        matrix.n_rows_base(),
        matrix.n_cols_base(),
        args.n_features,
        args.rng_seed,
    );
    let hyper_params = HyperParams::new(args);
    let mat_comp = MatrixCompletion::new(hyper_params, weights, &matrix, learning_rate);

    orchestrator.run(mat_comp);
}

pub fn run(args: &Args) {
    match args.dataset.as_str() {
        "netflix" => run_matrix_completion(&args),
        d => {
            panic!("Unknown dataset {}", d)
        }
    }
}
