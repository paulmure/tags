module DataLoader

export load_netflix_data, MatrixData, NNZ

using CSV
using DataFrames
using DataStructures
using SparseArrays

const data_dir::String = joinpath(@__DIR__, "data", "training_set")

function load_movie(file::String)::DataFrame
    id::Int = parse(Int, split(file, '.')[1])
    data::DataFrame = CSV.read(joinpath(data_dir, file), DataFrame)
    select!(data, Not(:Date))
    insertcols!(data, :Movie => id)
    data
end

function tidy_field!(df::DataFrame, field::Symbol)
    ids::Vector{Int} = unique(df[!, field])
    ids_dict::Dict{Int,Int} = Dict(ids[i] => i for i in 1:length(ids))
    f(i) = ids_dict[i]
    df[!, field] = f.(df[!, field])
end

function tidy_indices!(df::DataFrame)
    tidy_field!(df, :User)
    tidy_field!(df, :Movie)
end

struct NNZ
    row::DefaultDict{Int,Int}
    col::DefaultDict{Int,Int}
end

struct MatrixData{T}
    data::SparseMatrixCSC{T,Int}
    nnz::NNZ
    nrows::Int
    ncols::Int
end

function build_nnz(df::DataFrame, row_name::Symbol, col_name::Symbol)::NNZ
    nnz::NNZ = NNZ(DefaultDict{Int,Int}(0), DefaultDict{Int,Int}(0))
    for row in eachrow(df)
        nnz.row[row[row_name]] += 1
        nnz.col[row[col_name]] += 1
    end
    nnz
end

function build_sparse_matrix(df::DataFrame,
    nrows, ncols,
    row_name::Symbol, col_name::Symbol,
    val_name::Symbol)::SparseMatrixCSC{Float32,Int}
    mat = spzeros(nrows, ncols)
    for row in eachrow(df)
        mat[row[row_name], row[col_name]] = convert(Float32, row[val_name])
    end
    mat
end

function load_netflix_data(nmovies::Int)::MatrixData{Float32}
    files::Vector{String} = readdir(data_dir)[1:nmovies]
    movies::Vector{DataFrame} = map(load_movie, files)
    df = reduce((x, y) -> outerjoin(x, y, on=[:User, :Movie, :Rating]), movies)
    tidy_indices!(df)
    nnz = build_nnz(df, :User, :Movie)
    nusers = length(unique(df.User))
    mat = build_sparse_matrix(df, nusers, nmovies, :User, :Movie, :Rating)
    MatrixData{Float32}(mat, nnz, nusers, nmovies)
end

end