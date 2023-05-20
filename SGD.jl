using ReverseDiff
using LinearAlgebra
using SparseArrays
using Plots

include("./DataLoader.jl")
using .DataLoader

##

function train(data::MatrixData{Float32}, nfeatures::Int, μ::Float32, learning_rate::Float32)
    function loss(L, R)
        pred(u, v) = L[u, :] ⋅ R[:, v]
        diff_sq(u, v, z) = (pred(u, v) - z)^2
        l_reg(u) = (μ * sum(L[u, :] .^ 2)) / (2 * data.nnz.row[u])
        r_reg(v) = (μ * sum(R[:, v] .^ 2)) / (2 * data.nnz.col[v])
        entry_loss(u, v, z) = diff_sq(u, v, z) + l_reg(u) + r_reg(v)

        sum(entry_loss(u, v, z) for (u, v, z) in zip(findnz(data.data)...))
    end

    gt = ReverseDiff.GradientTape(loss, (rand(data.nrows, nfeatures), rand(nfeatures, data.ncols)))
    backprop = ReverseDiff.compile(gt)

    L = rand(data.nrows, nfeatures)
    R = rand(nfeatures, data.ncols)

    inputs = (L, R)
    results = (similar(L), similar(R))

    history = []
    initial_loss = loss(L, R)
    println(initial_loss)
    push!(history, initial_loss)

    for i in 1:100
        ReverseDiff.gradient!(results, backprop, inputs)
        L -= results[1] .* learning_rate
        R -= results[2] .* learning_rate

        curr_loss = loss(L, R)
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

data = load_netflix_data(32);

train(data, 100, Float32(1.5), Float32(0.01))
##

