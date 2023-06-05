use std::{collections::HashMap, fmt::Display, fs, io::Write, path::PathBuf};

pub struct CoordListSparseMatrix<Elem: Copy + Display> {
    data: Vec<(usize, usize, Elem)>,
    n_rows: usize,
    n_cols: usize,
    /// `nnz_row[u]` = number of non-zeros in row `u`
    nnz_row: HashMap<usize, usize>,
    /// `nnz_col[m]` = number of non-zeros in col `u`
    nnz_col: HashMap<usize, usize>,
}

impl<Elem: Copy + Display> CoordListSparseMatrix<Elem> {
    pub fn new_empty() -> Self {
        Self {
            data: vec![],
            n_rows: 0,
            n_cols: 0,
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
        self.data.push((row, col, elem));
        *self.nnz_row.entry(row).or_insert(0) += 1;
        *self.nnz_col.entry(col).or_insert(0) += 1;
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

    pub fn save(&self, path: PathBuf) {
        if path.exists() {
            fs::remove_file(&path).unwrap();
        }

        let mut file = fs::File::create(path).unwrap();

        for &(row, col, entry) in &self.data {
            writeln!(&mut file, "{},{},{}", row, col, entry).unwrap();
        }
    }
}
