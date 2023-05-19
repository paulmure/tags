module DataLoader

export load_data, NNZ

using CSV
using DataFrames
using DataStructures

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
    user::DefaultDict{Int,Int}
    movie::DefaultDict{Int,Int}
end

function build_nnz(df::DataFrame)::NNZ
    nnz::NNZ = NNZ(DefaultDict{Int,Int}(0), DefaultDict{Int,Int}(0))
    for row in eachrow(df)
        nnz.user[row.User] += 1
        nnz.movie[row.Movie] += 1
    end
    nnz
end

function load_data(nrows::Int)::Tuple{DataFrame,NNZ}
    files::Vector{String} = readdir(data_dir)[1:nrows]
    movies::Vector{DataFrame} = map(load_movie, files)
    df = reduce((x, y) -> outerjoin(x, y, on=[:User, :Movie, :Rating]), movies)
    select!(df, :User, :Movie, :Rating)
    tidy_indices!(df)
    nnz = build_nnz(df)
    df, nnz
end

end