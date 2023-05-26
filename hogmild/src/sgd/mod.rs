pub mod matrix_completion;

pub trait SGD<Sample, Update> {
    fn next_sample() -> Sample;
    fn update(update: Update);
}
