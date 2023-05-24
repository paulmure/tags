use std::{
    cmp::Eq,
    collections::{hash_map::Entry, HashMap},
    default::Default,
    hash::Hash,
    ops::Index,
};

#[derive(Debug)]
pub struct SparseMatrix<Idx, E> {
    n_row: Idx,
    n_col: Idx,
    /// `entries[(u, v)]` = revealed entry of the matrix at (`u`, `v`)
    pub entries: HashMap<(Idx, Idx), E>,
    /// `nnz_row[u]` = number of non-zeros in row `u`
    nnz_row: HashMap<Idx, Idx>,
    /// `nnz_col[m]` = number of non-zeros in col `u`
    nnz_col: HashMap<Idx, Idx>,
}

impl<Idx, E> SparseMatrix<Idx, E>
where
    Idx: Eq + Hash + Default + Copy,
{
    fn new_empty(n_row: Idx, n_col: Idx) -> Self {
        Self {
            n_row,
            n_col,
            entries: HashMap::new(),
            nnz_row: HashMap::new(),
            nnz_col: HashMap::new(),
        }
    }

    pub fn shape(&self) -> (Idx, Idx) {
        (self.n_row, self.n_col)
    }

    pub fn row_occupancy(&self, row: Idx) -> Idx {
        *self.nnz_row.get(&row).unwrap()
    }

    pub fn col_occupancy(&self, col: Idx) -> Idx {
        *self.nnz_col.get(&col).unwrap()
    }
}

impl<Idx, E> Index<(Idx, Idx)> for SparseMatrix<Idx, E>
where
    Idx: Eq + Hash,
    E: Default,
{
    type Output = E;
    fn index(&self, index: (Idx, Idx)) -> &Self::Output {
        self.entries.get(&index).unwrap()
    }
}

pub mod netflix {
    use super::*;

    use indicatif::ProgressIterator;
    use std::fs::{read_dir, DirEntry, File};
    use std::io::{BufRead, BufReader};
    use std::path::{Path, PathBuf};

    type nfidx = usize;
    pub type NetflixMatrix = SparseMatrix<nfidx, f32>;

    pub struct IdMapping {
        user_to_row: HashMap<nfidx, nfidx>,
        movie_to_col: HashMap<nfidx, nfidx>,
    }

    impl IdMapping {
        fn new_empty() -> Self {
            Self {
                user_to_row: HashMap::new(),
                movie_to_col: HashMap::new(),
            }
        }

        pub fn user_to_row(&self, user: nfidx) -> nfidx {
            *self.user_to_row.get(&user).unwrap()
        }

        pub fn movie_to_col(&self, movie: nfidx) -> nfidx {
            *self.movie_to_col.get(&movie).unwrap()
        }
    }

    fn get_data_dir() -> PathBuf {
        let mut base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        base_dir.push("..");
        base_dir.push("data");
        base_dir.push("netflix");
        base_dir.push("training_set");
        base_dir
    }

    fn parse_movie_id(path: &Path) -> nfidx {
        let file_name = path.file_stem().unwrap().to_str().unwrap();
        file_name.parse().unwrap()
    }

    fn load_one_movie(
        dir_entry: DirEntry,
        m: &mut NetflixMatrix,
        im: &mut IdMapping,
        next_row: &mut nfidx,
        next_col: &mut nfidx,
    ) {
        let path = dir_entry.path();
        let movie_id = parse_movie_id(&path);

        let col = *next_col;
        *next_col += 1;
        im.movie_to_col.insert(movie_id, col);

        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);

        for res_line in reader.lines().skip(1) {
            let line = res_line.unwrap();
            let mut toks = line.split(',');

            let user_id: nfidx = toks.next().unwrap().parse().unwrap();
            let rating: f32 = toks.next().unwrap().parse().unwrap();
            let rating_norm: f32 = (rating / 2.5) - 1.;

            let row: nfidx = match im.user_to_row.entry(user_id) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let row = *next_row;
                    *next_row += 1;
                    m.n_row += 1;
                    *v.insert(row)
                }
            };

            m.entries.insert((row, col), rating_norm);
            *m.nnz_row.entry(row).or_insert(0) += 1;
            *m.nnz_col.entry(col).or_insert(0) += 1;
        }
    }

    pub fn load_netflix_dataset(n_movies: nfidx) -> (NetflixMatrix, IdMapping) {
        let mut m = NetflixMatrix::new_empty(0, n_movies);
        let mut im = IdMapping::new_empty();
        let mut next_row: nfidx = 0;
        let mut next_col: nfidx = 0;

        let paths = read_dir(get_data_dir()).unwrap();

        paths
            .take(n_movies)
            .progress_count(n_movies as u64)
            .for_each(|de| {
                load_one_movie(de.unwrap(), &mut m, &mut im, &mut next_row, &mut next_col)
            });

        (m, im)
    }
}
