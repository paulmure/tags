use once_cell::sync::Lazy;
use polars::lazy::dsl::col;
use polars::prelude::*;
use std::collections::HashMap;
use std::fs::{self, ReadDir};
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
fn load_one_movie(paths: &mut ReadDir) -> DataFrame {
    let path = paths.next().unwrap().unwrap().path();
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

/// Reset a field to be indices from 0 to n
fn tidy_indices(df: &mut DataFrame, field: &str) {
    let unique_ids = df.column(field).unwrap().unique().unwrap();
    let unique_ids_int = unique_ids.u32().unwrap();

    let mut unique_ids_dict: HashMap<u32, u32> = HashMap::new();
    for (i, val) in unique_ids_int.into_iter().enumerate() {
        unique_ids_dict.insert(val.unwrap(), i as u32);
    }

    df.apply(field, |id_series| {
        id_series
            .u32()
            .unwrap()
            .into_iter()
            .map(|opt_id| opt_id.map(|id| unique_ids_dict[&id]))
            .collect::<UInt32Chunked>()
            .into_series()
    })
    .unwrap();
}

pub fn get_netflix_data(n: usize) -> DataFrame {
    assert_ne!(n, 0, "Must load at least one movie");

    let mut paths = fs::read_dir(get_data_dir()).unwrap();
    let mut res = load_one_movie(&mut paths);
    let join_ons = ["User", "Movie", "Rating"];

    for _ in 1..n {
        let df = load_one_movie(&mut paths);
        res = res.outer_join(&df, join_ons, join_ons).unwrap();
    }

    tidy_indices(&mut res, "User");
    tidy_indices(&mut res, "Movie");

    res
}
