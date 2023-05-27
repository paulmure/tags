pub mod netflix;

pub trait DataLoader<Data> {
    fn len(&self) -> usize;
    fn start_new_epoch(&mut self);
    /// Return the next `n` data and `true` if there is no more data for this epoch.
    fn take(&mut self, n: usize) -> (Vec<Data>, bool);
}
