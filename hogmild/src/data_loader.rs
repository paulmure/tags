use indicatif::ProgressBar;
use once_cell::sync::Lazy;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{read_dir, DirEntry};
use std::path::{Path, PathBuf};

static NETFLIX_FILE_SCHEMA: Lazy<SchemaRef> = Lazy::new(|| {
    let mut s = Schema::new();
    s.insert_at_index(0, Into::into("User"), DataType::UInt32)
        .unwrap();
    s.insert_at_index(1, Into::into("Rating"), DataType::UInt32)
        .unwrap();
    s.insert_at_index(2, Into::into("Date"), DataType::Date)
        .unwrap();
    s.into()
});

fn get_data_dir() -> PathBuf {
    let mut base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    base_dir.push("..");
    base_dir.push("data");
    base_dir.push("training_set");
    base_dir
}

fn get_movie_id(path: &Path) -> u32 {
    let file_name = path.file_stem().unwrap().to_str().unwrap();
    file_name.parse().unwrap()
}

#[allow(unused_must_use)]
fn load_one_movie(dir_entry: DirEntry) -> DataFrame {
    let path = dir_entry.path();
    let id = get_movie_id(&path);

    let mut df = CsvReader::from_path(path)
        .unwrap()
        .with_schema(NETFLIX_FILE_SCHEMA.clone())
        .has_header(true)
        .finish()
        .unwrap();

    df.drop_in_place("Date").unwrap();

    let ids = Series::new("Movie", vec![id; df.shape().0]);
    df.with_column(ids).unwrap();

    df
}

fn remap_indices(data: &Series) -> Series {
    let unique_ids = data.unique().unwrap();
    let unique_ids_int = unique_ids.u32().unwrap();

    let mut id_lookup: HashMap<u32, u32> = HashMap::new();
    for (i, val) in unique_ids_int.into_iter().enumerate() {
        id_lookup.insert(val.unwrap(), i as u32);
    }

    let chunked_array = data.u32().unwrap();
    let vec_option_u32: Vec<Option<u32>> = chunked_array.into_iter().collect();

    vec_option_u32
        .into_par_iter()
        .map(|opt_id| opt_id.map(|id| id_lookup[&id]))
        .collect::<UInt32Chunked>()
        .into_series()
}

/// Reset a field to be indices from 0 to n
fn tidy_indices(df: &mut DataFrame, field: &str) {
    df.apply(field, remap_indices).unwrap();
}

pub fn get_netflix_data(n: usize) -> DataFrame {
    assert_ne!(n, 0, "Must load at least one movie");

    println!("Loading netflix data...");
    let bar = ProgressBar::new(n as u64);
    bar.inc(1);

    let mut paths = read_dir(get_data_dir()).unwrap();
    let mut next_dir_entry = || paths.next().unwrap().unwrap();

    let mut res: DataFrame = load_one_movie(next_dir_entry());

    let join_ons = ["User", "Movie", "Rating"];
    for _ in 1..n {
        bar.inc(1);
        let df = load_one_movie(next_dir_entry());
        res = res.outer_join(&df, join_ons, join_ons).unwrap();
    }

    tidy_indices(&mut res, "User");
    tidy_indices(&mut res, "Movie");
    bar.finish();

    res
}
