use std::fmt::{self, Display};

pub struct CoordListSparseMatrix<Elem: Copy + Display> {
    data: Vec<(usize, usize, Elem)>,
    n_rows: usize,
    n_cols: usize,
}

impl<Elem: Copy + Display> CoordListSparseMatrix<Elem> {
    pub fn new_empty() -> Self {
        Self {
            data: vec![],
            n_rows: 0,
            n_cols: 0,
        }
    }

    pub fn add_row(&mut self) {
        self.n_rows += 1;
    }

    pub fn add_col(&mut self) {
        self.n_cols += 1;
    }

    pub fn insert(&mut self, row: usize, col: usize, elem: Elem) {
        self.data.push((row, col, elem));
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

impl<Elem: Copy + Display> fmt::Display for CoordListSparseMatrix<Elem> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Row,Column,Entry")?;
        for &(i, j, e) in &self.data {
            writeln!(f, "{},{},{}", i, j, e)?;
        }
        Ok(())
    }
}
