use std::ops::Index;
pub mod hash_map_spmat;

pub trait SparseMatrixView<Elem>: Index<(usize, usize)> {
    type Iter: Iterator<Item = ((usize, usize), Elem)>;

    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn shape(&self) -> (usize, usize);

    fn nnnz(&self) -> usize;
    fn nnz_row(&self, row: usize) -> usize;
    fn nnz_col(&self, row: usize) -> usize;

    fn iter(&self) -> Self::Iter;
}
