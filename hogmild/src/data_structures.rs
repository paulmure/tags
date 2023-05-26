use std::{
    collections::{hash_map, HashMap},
    default::Default,
    ops::Index,
};

pub struct Iter<'a, Elem> {
    iter: Box<dyn Iterator<Item = (&'a (usize, usize), &'a Elem)>>,
}

impl<'a, Elem> Iterator for Iter<'a, Elem> {
    type Item = (&'a (usize, usize), &'a Elem);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub trait SparseMatrixView<Elem>: Index<(usize, usize)> {
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn shape(&self) -> (usize, usize);

    fn nnz_row(&self, row: usize) -> usize;
    fn nnz_col(&self, row: usize) -> usize;

    fn iter(&self) -> Iter<Elem>;
}

#[derive(Debug)]
pub struct HashMapSparseMatrix<Elem> {
    n_rows: usize,
    n_cols: usize,
    /// `entries[(u, v)]` = revealed entry of the matrix at (`u`, `v`)
    entries: HashMap<(usize, usize), Elem>,
    /// `nnz_row[u]` = number of non-zeros in row `u`
    nnz_row: HashMap<usize, usize>,
    /// `nnz_col[m]` = number of non-zeros in col `u`
    nnz_col: HashMap<usize, usize>,
}

impl<Elem> HashMapSparseMatrix<Elem> {
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

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    pub fn nnz_row(&self, row: usize) -> usize {
        *self.nnz_row.get(&row).unwrap()
    }

    pub fn nnz_col(&self, col: usize) -> usize {
        *self.nnz_col.get(&col).unwrap()
    }
}

impl<'a, Elem> Index<(usize, usize)> for &'a HashMapSparseMatrix<Elem>
where
    Elem: Default,
{
    type Output = Elem;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.entries.get(&index).unwrap()
    }
}

impl<'a, Elem> IntoIterator for &'a HashMapSparseMatrix<Elem> {
    type Item = (&'a (usize, usize), &'a Elem);
    type IntoIter = hash_map::Iter<'a, (usize, usize), Elem>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

impl<'a, Elem> SparseMatrixView<Elem> for &'a HashMapSparseMatrix<Elem>
where
    Elem: Default,
{
    fn n_rows(&self) -> usize {
        self.n_rows()
    }

    fn n_cols(&self) -> usize {
        self.n_cols()
    }

    fn shape(&self) -> (usize, usize) {
        self.shape()
    }

    fn nnz_row(&self, row: usize) -> usize {
        self.nnz_row(row)
    }

    fn nnz_col(&self, col: usize) -> usize {
        self.nnz_col(col)
    }

    fn iter(&self) -> Iter<Elem> {
        Iter {
            iter: Box::new(self.entries.iter()),
        }
    }
}
