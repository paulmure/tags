use ndarray::prelude::*;
use ndarray_rand::{
    rand::{rngs::StdRng, SeedableRng},
    rand_distr::Uniform,
    RandomExt,
};

use crate::{
    args::Args,
    data_structures::CoordListSparseMatrix,
    simulator::{Sample, UpdateLogs},
};

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

struct GradUpdate {
    u: usize,
    v: usize,
    xrow_grad: Array1<f32>,
    ycol_grad: Array1<f32>,
    xb_grad: f32,
    yb_grad: f32,
    loss: f32,
}

pub struct MatrixCompletion {
    matrix: CoordListSparseMatrix<f32>,
    weights: Weights,
    updates: Vec<Sample>,

    /// Model hyper parameter initial learning rate
    pub alpha_0: f32,
    /// Model hyper parameter initial learning rate
    pub decay_rate: f32,
    /// Maximum number of epochs to run for
    pub max_epoch: usize,
    /// When to stop training
    pub stopping_criterion: f32,

    /// Number of features in the decomposition matrix
    pub n_features: usize,
    /// Model hyper parameter mu
    pub mu: f32,
    /// Model hyper parameter lambda_xf
    pub lam_xf: f32,
    /// Model hyper parameter lambda_yf
    pub lam_yf: f32,
    /// Model hyper parameter lambda_xb
    pub lam_xb: f32,
    /// Model hyper parameter lambda_yb
    pub lam_yb: f32,
}

impl MatrixCompletion {
    pub fn new(args: &Args, matrix: CoordListSparseMatrix<f32>, updates: Vec<Sample>) -> Self {
        let nrows = matrix.n_rows();
        let ncols = matrix.n_cols();
        Self {
            matrix,
            weights: Weights::new(nrows, ncols, args.n_features, args.rng_seed),
            updates,
            alpha_0: args.alpha_0,
            decay_rate: args.decay_rate,
            max_epoch: args.max_epoch,
            stopping_criterion: args.stopping_criterion,
            n_features: args.n_features,
            mu: args.mu,
            lam_xf: args.lam_xf,
            lam_yf: args.lam_yf,
            lam_xb: args.lam_xb,
            lam_yb: args.lam_yb,
        }
    }

    fn total_loss(&self) -> f32 {
        self.matrix
            .iter()
            .map(|&(row, col, entry)| {
                let xrow = self.weights.x.slice(s![row, ..]);
                let ycol = self.weights.y.slice(s![col, ..]);
                let xb = self.weights.xb[row];
                let yb = self.weights.yb[col];

                let nnzrow = self.matrix.nnz_row(row);
                let nnzcol = self.matrix.nnz_col(col);

                let e = error(entry, &xrow, &ycol, xb, yb, self.mu);
                let x_regu = regu(&xrow.view(), self.lam_xf, nnzrow);
                let y_regu = regu(&ycol.view(), self.lam_yf, nnzcol);
                let xb_regu = xb * self.lam_xb / (nnzrow as f32);
                let yb_regu = yb * self.lam_yb / (nnzcol as f32);

                let loss = e.powi(2) + x_regu + y_regu + xb_regu + yb_regu;
                loss
            })
            .sum()
    }

    fn gradient(&self, sample_id: usize, learning_rate: f32) -> GradUpdate {
        // Forward prop
        let (row, col, entry) = self.matrix[sample_id];

        let xrow = self.weights.x.slice(s![row, ..]);
        let ycol = self.weights.y.slice(s![col, ..]);
        let xb = self.weights.xb[row];
        let yb = self.weights.yb[col];

        let nnzrow = self.matrix.nnz_row(row);
        let nnzcol = self.matrix.nnz_col(col);

        let e = error(entry, &xrow, &ycol, xb, yb, self.mu);
        let x_regu = regu(&xrow.view(), self.lam_xf, nnzrow);
        let y_regu = regu(&ycol.view(), self.lam_yf, nnzcol);
        let xb_regu = xb * self.lam_xb / (nnzrow as f32);
        let yb_regu = yb * self.lam_yb / (nnzcol as f32);

        let loss = e.powi(2) + x_regu + y_regu + xb_regu + yb_regu;

        // Backward prop
        let xb_grad = learning_rate * (e - xb_regu);
        let yb_grad = learning_rate * (e - yb_regu);

        let x_grad = learning_rate * ((e * &ycol) - (x_regu * &xrow));
        let y_grad = learning_rate * ((e * &xrow) - (y_regu * &ycol));

        GradUpdate {
            u: row,
            v: col,
            xrow_grad: x_grad,
            ycol_grad: y_grad,
            xb_grad,
            yb_grad,
            loss,
        }
    }

    fn fold(&mut self, updates: &[GradUpdate]) {
        for update in updates {
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

    pub fn train(&mut self) -> Vec<f32> {
        let mut history = vec![self.total_loss()];
        println!("{}", history.last().unwrap());

        for i in 0..self.max_epoch {
            let mut curr_loss = 0.;

            let learning_rate = self.alpha_0 / (1. + self.decay_rate * (i as f32));

            let mut updates_idx = 0;
            while updates_idx < self.updates.len() {
                let mut gradients = vec![];
                let curr_version = self.updates[updates_idx].weight_version;
                while updates_idx < self.updates.len()
                    && self.updates[updates_idx].weight_version == curr_version
                {
                    gradients
                        .push(self.gradient(self.updates[updates_idx].sample_id, learning_rate));
                    updates_idx += 1;
                }
                curr_loss += gradients.iter().map(|grad| grad.loss).sum::<f32>();
                self.fold(&gradients);
            }

            println!("{}", curr_loss);

            let last_loss = *history.last().unwrap();
            history.push(curr_loss);

            if (last_loss - curr_loss) / last_loss < self.stopping_criterion {
                break;
            }
        }

        history
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
