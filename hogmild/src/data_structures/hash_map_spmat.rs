use std::{
    collections::{hash_map, HashMap},
    ops::Index,
};

use crate::data_structures::SparseMatrixView;

#[derive(Debug)]
pub struct HashMapSparseMatrix<Elem>
where
    Elem: Copy,
{
    n_rows: usize,
    n_cols: usize,
    /// `entries[(u, v)]` = revealed entry of the matrix at (`u`, `v`)
    entries: HashMap<(usize, usize), Elem>,
    /// `nnz_row[u]` = number of non-zeros in row `u`
    nnz_row: HashMap<usize, usize>,
    /// `nnz_col[m]` = number of non-zeros in col `u`
    nnz_col: HashMap<usize, usize>,
}

impl<Elem> HashMapSparseMatrix<Elem>
where
    Elem: Copy,
{
    pub fn new_empty() -> Self {
        Self {
            n_rows: 0,
            n_cols: 0,
            entries: HashMap::new(),
            nnz_row: HashMap::new(),
            nnz_col: HashMap::new(),
        }
    }

    pub fn add_row(&mut self) {
        self.n_rows += 1;
    }

    pub fn add_col(&mut self) {
        self.n_cols += 1;
    }

    pub fn insert(&mut self, row: usize, col: usize, elem: Elem) {
        self.entries.insert((row, col), elem);
        *self.nnz_row.entry(row).or_insert(0) += 1;
        *self.nnz_col.entry(col).or_insert(0) += 1;
    }

    pub fn n_rows_base(&self) -> usize {
        self.n_rows
    }

    pub fn n_cols_base(&self) -> usize {
        self.n_cols
    }

    pub fn shape_base(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    pub fn nnz_base(&self) -> usize {
        self.entries.len()
    }

    pub fn nnz_row_base(&self, row: usize) -> usize {
        *self.nnz_row.get(&row).unwrap()
    }

    pub fn nnz_col_base(&self, col: usize) -> usize {
        *self.nnz_col.get(&col).unwrap()
    }

    pub fn iter_base(&self) -> Iter<Elem> {
        Iter {
            iter: self.entries.iter(),
        }
    }
}

impl<'a, Elem> Index<(usize, usize)> for &'a HashMapSparseMatrix<Elem>
where
    Elem: Copy + Default,
{
    type Output = Elem;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.entries.get(&index).unwrap()
    }
}

pub struct Iter<'a, Elem>
where
    Elem: Copy,
{
    iter: hash_map::Iter<'a, (usize, usize), Elem>,
}

impl<'a, Elem> Iterator for Iter<'a, Elem>
where
    Elem: Copy,
{
    type Item = ((usize, usize), Elem);
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(&(i, j), &v)| ((i, j), v))
    }
}

impl<'a, Elem> SparseMatrixView<Elem> for &'a HashMapSparseMatrix<Elem>
where
    Elem: Copy + Default,
{
    type Iter = Iter<'a, Elem>;

    fn n_rows(&self) -> usize {
        self.n_rows_base()
    }

    fn n_cols(&self) -> usize {
        self.n_cols_base()
    }

    fn shape(&self) -> (usize, usize) {
        self.shape_base()
    }

    fn nnnz(&self) -> usize {
        self.nnz_base()
    }

    fn nnz_row(&self, row: usize) -> usize {
        self.nnz_row_base(row)
    }

    fn nnz_col(&self, col: usize) -> usize {
        self.nnz_col_base(col)
    }

    fn iter(&self) -> Self::Iter {
        self.iter_base()
    }
}
