using ReverseDiff
using LinearAlgebra
using SparseArrays
using Plots

include("./DataLoader.jl")
using .DataLoader



struct Params
    x::Matrix{Float32}
    y::Matrix{Float32}
    xb::Vector{Float32}
    yb::Vector{Float32}
    μ::Float32
    λxf::Float32
    λyf::Float32
    λxb::Float32
    λyb::Float32
    η::Float32
end

pred(x, y, xb, yb, μ) = (x ⋅ y) + xb + yb + μ
error(r, x, y, xb, yb, μ) = r - pred(x, y, xb, yb, μ)
vec_regu(x, λ, nnz) = λ * sum(x .^ 2) / nnz

function sample_loss(r::MatrixData{Float32}, p::Params, u::Int, i::Int)::Float32
    nnzrow = r.nnz.row[u]
    nnzcol = r.nnz.col[i]
    e = error(r.data[u, i], p.x[u, :], p.y[:, i], p.xb[u], p.yb[i], p.μ)
    x_regu = vec_regu(p.x[u, :], p.λxf, nnzrow)
    y_regu = vec_regu(p.y[:, i], p.λyf, nnzcol)
    xb_regu = p.λxb * p.xb[u] / nnzrow
    yb_regu = p.λyb * p.yb[i] / nnzcol
    e^2 + x_regu + y_regu + xb_regu + yb_regu
end

function batch_loss(r::MatrixData{Float32}, p::Params)::Float32
    sum(sample_loss(r, p, u, i) for (u, i, _) in zip(findnz(r.data)...))
end

function process_sample!(r::MatrixData{Float32}, p::Params, u::Int, i::Int)::Float32
    # forward prop
    nnzrow = r.nnz.row[u]
    nnzcol = r.nnz.col[i]
    # There is numerical instability,
    # y might become NaN if nfeatures is too big
    # println("($(u), $(i)), y = $(sum(p.y[:, i])), yb = $(p.yb[i])")
    e = error(r.data[u, i], p.x[u, :], p.y[:, i], p.xb[u], p.yb[i], p.μ)
    x_regu = vec_regu(p.x[u, :], p.λxf, nnzrow)
    y_regu = vec_regu(p.y[:, i], p.λyf, nnzcol)
    xb_regu = p.λxb * p.xb[u] / nnzrow
    yb_regu = p.λyb * p.yb[i] / nnzcol
    partial_loss = e^2 + x_regu + y_regu + xb_regu + yb_regu
    # backward prop
    p.xb[u] += p.η * (e - xb_regu)
    p.yb[i] += p.η * (e - yb_regu)
    p.x[u, :] += p.η .* (e .* p.y[:, i] - x_regu .* p.x[u, :])
    p.y[:, i] += p.η .* (e .* p.x[u, :] - y_regu .* p.y[:, i])
    # return
    partial_loss
end

function process_batch!(r::MatrixData{Float32}, p::Params)::Float32
    loss = 0
    for (u, i, _) in zip(findnz(r.data)...)
        loss += process_sample!(r, p, u, i)
    end
    loss
end

function train(r::MatrixData{Float32}, p::Params)
    history = []
    initial_loss = batch_loss(r, p)
    println(initial_loss)
    push!(history, initial_loss)
    for i in 1:1000
        curr_loss = process_batch!(r, p)
        last_loss = history[end]
        push!(history, curr_loss)
        println(curr_loss)
        if (i > 100) && (curr_loss > last_loss)
            break
        end
    end
    plot(1:length(history), history, label="loss")
    xlabel!("Iteration")
    ylabel!("Loss")
end

function new_params(
    r::MatrixData{Float32},
    nfeatures::Int,
    μ::Float32,
    λxf::Float32,
    λyf::Float32,
    λxb::Float32,
    λyb::Float32,
    η::Float32)::Params
    x = randn(Float32, (r.nrows, nfeatures))
    y = randn(Float32, (nfeatures, r.ncols))
    xb = randn(Float32, (r.nrows))
    yb = randn(Float32, (r.ncols))
    Params(x, y, xb, yb, μ, λxf, λyf, λxb, λyb, η)
end

function run()
    r = load_netflix_data(128)
    p = new_params(
        r,
        20, # nfeatures
        convert(Float32, 1), # μ
        convert(Float32, 1), # λxf
        convert(Float32, 1), # λyf
        convert(Float32, 1), # λxb
        convert(Float32, 1), # λyb
        convert(Float32, 0.01), # η
    )
    train(r, p)
end

run()
