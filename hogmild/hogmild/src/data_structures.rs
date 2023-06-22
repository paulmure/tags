use std::{
    collections::HashMap,
    fmt::{self, Display},
    ops::Index,
};

pub struct CoordListSparseMatrix<Elem: Copy + Display> {
    data: Vec<(usize, usize, Elem)>,
    n_rows: usize,
    n_cols: usize,
    nnz_rows: HashMap<usize, usize>,
    nnz_cols: HashMap<usize, usize>,
}

impl<Elem: Copy + Display> CoordListSparseMatrix<Elem> {
    pub fn new_empty() -> Self {
        Self {
            data: vec![],
            n_rows: 0,
            n_cols: 0,
            nnz_rows: HashMap::new(),
            nnz_cols: HashMap::new(),
        }
    }

    pub fn add_row(&mut self) {
        self.n_rows += 1;
    }

    pub fn add_col(&mut self) {
        self.n_cols += 1;
    }

    pub fn insert(&mut self, row: usize, col: usize, elem: Elem) {
        *self.nnz_rows.entry(row).or_default() += 1;
        *self.nnz_cols.entry(col).or_default() += 1;
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

    pub fn nnz_row(&self, row: usize) -> usize {
        self.nnz_rows.get(&row).map_or(0, |&val| val)
    }

    pub fn nnz_col(&self, col: usize) -> usize {
        self.nnz_cols.get(&col).map_or(0, |&val| val)
    }

    pub fn iter(&self) -> std::slice::Iter<(usize, usize, Elem)> {
        self.data.iter()
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

impl<Elem> Index<usize> for CoordListSparseMatrix<Elem>
where
    Elem: Copy + Display,
{
    type Output = (usize, usize, Elem);
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
