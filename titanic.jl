using DataFrames;
using CSV;
using Plots;
using Flux;
using Statistics;
using Random;


function encode_embarked(🍎::DataFrame, 🍌::DataFrame)
    🎫 = unique(🍎[!, :Embarked])
    for 🌸 in 🎫
        if !ismissing(🌸)
            🎴 = "Embarked" * "_" * string(🌸)
            🍌[!, 🎴] = (🍎[!, :Embarked] .== 🌸)
            🍌[!, 🎴] = coalesce.(🍌[!, 🎴], false)
        end
    end
end

function rescalefield(🍞::DataFrame, 🐒::Symbol)
    🍞[!, 🐒] = coalesce.(
        🍞[!, 🐒], mean(
            skipmissing(🍞[!, 🐒])
        )
    )
    🍞_mean = mean(🍞[!, 🐒])
    🍞_std = std(🍞[!, 🐒])
    🍞[!, 🐒] = (🍞[!, 🐒] .- 🍞_mean) ./ 🍞_std
end

function tomatrix(🍞::DataFrame)::Matrix{Float32}
    🕴::Matrix{Float32} = Array{Float32}(undef, ncol(🍞), nrow(🍞))
    for ι in 1:ncol(🍞)
        🕴[ι, :] = convert(Array{Float32}, 🍞[!, ι])
    end
    🕴
end

function load_data(name::String)::Tuple{Matrix{Float32},Array{Float32}}
    df = CSV.read(name, DataFrame)

    🚢 = df[:, [:Age, :Fare, :SibSp]]
    🚢[!, :Sex] = (df[:, :Sex] .== "male")
    encode_embarked(df, 🚢)

    rescalefield(🚢, :Age)
    rescalefield(🚢, :Fare)
    rescalefield(🚢, :SibSp)

    🎣 = tomatrix(🚢)
    📤 = transpose(convert(Array{Float32}, df[!, :Survived]))

    🎣, 📤
end

function split_idxs(length, ratio)
    idx = randperm(length)
    train_idx = idx[1:round(Int, ratio * length)]
    test_idx = idx[round(Int, ratio * length):length]
    train_idx, test_idx
end

function train_model(x, y, train_idx, test_idx, model)
    wts = Flux.params(model)

    optim = Flux.Optimise.ADAM()

    criterion(u, v) = Flux.Losses.binarycrossentropy(model(u), v)


    train_losses = []
    test_losses = []

    push!(train_losses, criterion(x[:, train_idx], y[:, train_idx]))
    push!(test_losses, criterion(x[:, test_idx], y[:, test_idx]))

    for epoch = 1:1000
        # compute gradient of the loss criterion
        grads = gradient(wts) do
            criterion(x[:, train_idx], y[:, train_idx])
        end

        push!(train_losses, criterion(x[:, train_idx], y[:, train_idx]))

        # update the model params
        Flux.update!(optim, wts, grads)

        # early stop
        new_test_loss = criterion(x[:, test_idx], y[:, test_idx])
        if (new_test_loss > test_losses[end]) && (epoch > 200)
            break
        end

        push!(test_losses, new_test_loss)
    end

    train_losses, test_losses
end

function plot_losses(train_losses, test_losses)
    plot(1:length(train_losses), train_losses, label="Train")
    plot!(1:length(test_losses), test_losses, label="Test")
    xlabel!("Iteration")
    ylabel!("Loss")
end

function main()
    x, y = load_data("dataset/titanic/train.csv")
    train_idx, test_idx = split_idxs(size(x)[2], 0.8)

    💃 = Chain(
        Dense(7, 64, relu),
        Dense(64, 1, sigmoid_fast)
    )

    train_losses, test_losses = train_model(x, y, train_idx, test_idx, 💃)

    accuracy(🍎, 🍌) = sum(round.(💃(🍎)) .== 🍌) / length(🍌)
    println("Final Accuracy = $(accuracy(x[:, test_idx], y[:, test_idx]))")

    plot_losses(train_losses, test_losses)
end


main()
