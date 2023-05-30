use indicatif::ProgressIterator;
use std::{
    collections::{hash_map::Entry, HashMap},
    fs::{read_dir, DirEntry, File},
    io::{BufRead, BufReader},
    iter::Peekable,
    path::PathBuf,
};

use crate::{
    data_loader::DataLoader,
    data_structures::hash_map_spmat::{self, HashMapSparseMatrix},
};

type NetflixMatrix = HashMapSparseMatrix<f32>;

fn get_data_dir() -> PathBuf {
    let mut base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    base_dir.push("..");
    base_dir.push("data");
    base_dir.push("netflix");
    base_dir.push("training_set");
    base_dir
}

fn load_one_movie(
    dir_entry: DirEntry,
    m: &mut NetflixMatrix,
    user_to_row: &mut HashMap<usize, usize>,
) {
    let path = dir_entry.path();

    let col = m.n_cols_base();
    m.add_col();

    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    for res_line in reader.lines().skip(1) {
        let line = res_line.unwrap();
        let mut toks = line.split(',');

        let user_id: usize = toks.next().unwrap().parse().unwrap();
        let rating: f32 = toks.next().unwrap().parse().unwrap();
        let rating_norm: f32 = (rating / 2.5) - 1.;

        let row: usize = match user_to_row.entry(user_id) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let row = m.n_rows_base();
                m.add_row();
                *v.insert(row)
            }
        };

        m.insert(row, col, rating_norm);
    }
}

pub fn load_netflix_dataset(n_movies: usize) -> NetflixMatrix {
    println!("Loading Netflix dataset with {} movies", n_movies);

    let mut m = NetflixMatrix::new_empty();
    let mut user_to_row: HashMap<usize, usize> = HashMap::new();
    let paths = read_dir(get_data_dir()).unwrap();

    paths
        .take(n_movies)
        .progress_count(n_movies as u64)
        .for_each(|de| load_one_movie(de.unwrap(), &mut m, &mut user_to_row));

    println!("Done!");
    m
}

pub struct NetflixDataLoader<'a> {
    matrix: &'a NetflixMatrix,
    iter: Peekable<hash_map_spmat::Iter<'a, f32>>,
}

impl<'a> NetflixDataLoader<'a> {
    pub fn new(matrix: &'a NetflixMatrix) -> Self {
        Self {
            matrix,
            iter: matrix.iter_base().peekable(),
        }
    }
}

impl<'a> DataLoader<((usize, usize), f32)> for NetflixDataLoader<'a> {
    fn len(&self) -> usize {
        self.matrix.nnz_base()
    }

    fn start_new_epoch(&mut self) {
        self.iter = self.matrix.iter_base().peekable()
    }

    fn take_one(&mut self) -> Option<((usize, usize), f32)> {
        self.iter.next()
    }
}
