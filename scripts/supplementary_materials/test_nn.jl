using DrWatson
@quickactivate "ASP_model_comparison"
using DataFrames, Statistics
using StateSpaceIdentification
using Plots
using StatsPlots
using Measures
using StateSpaceIdentification
using NearestNeighbors
using Measures

using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches, Zygote

using ComponentArrays, Lux, Zygote

import Base: convert
function Base.convert(::Type{StateSpaceIdentification.LocalLinearRegressor}, input)
    StateSpaceIdentification.LocalLinearRegressor(input.index_analogs, input.analogs, input.successors, KDTree(input.analogs), n_neighbors=input.k, min_lag_in_days=input.lag_x, kernel_type=input.kernel)
end

# Set a random seed for reproducible behaviour
using StableRNGs
rng = StableRNG(1111)

include(srcdir("dataset.jl"))
include(srcdir("model.jl"))
include(srcdir("utils.jl"))

# Comparison for S1
df = collect_results(datadir("rolling_evaluation"), subfolders=true)
df = df[in.(df.name_dataset, Ref(["S1"])),:]
df = df[.∈(df.model, Ref([:nonparametric_black])),:]

for i in 1:size(df)[1]
    if  ~(typeof(df.llrs[i][1]) <: LocalLinearRegressor)
        convert_new_interface(df.llrs[i])
    end
end


knn_error = []
nn_error = []
for id in 1:50
    @info "Start training for id=$(id)"
    dataset = load_dataset(df.name_dataset[id], seed=df.seed[id])
    adapt_signals_to_model!(dataset, df.model[id], df.dt_model[id])
    (x_train, y_train, U_train, E_train), (x_test, y_test, U_test, E_test) = train_test_split(dataset)


    ###########################################################################################
    #################################### DATASET CONSTRUCTION #################################
    ###########################################################################################

    T_1 = df.llrs[id][1].analog_times_in_days
    T_2 = df.llrs[id][2].analog_times_in_days
    X_1 = Matrix{Float32}(df.llrs[id][1].analog_inputs)
    n_1 = size(X_1, 2)
    U_1 = repeat([1.0], n_1)
    X_2 = Matrix{Float32}(df.llrs[id][2].analog_inputs)
    n_2 = size(X_2, 2)
    U_2 = repeat([0.0], n_2)
    Y_1 = Matrix{Float32}(df.llrs[id][1].analog_outputs)
    Y_2 = Matrix{Float32}(df.llrs[id][2].analog_outputs)

    p_test = 0.2
    n_test_1 = Int((n_1 + n_2)*p_test*(n_1/(n_1+n_2)))
    n_test_2 = Int((n_1 + n_2)*p_test*(n_2/(n_1+n_2)))
    selected_test_indexes_1 = partialsortperm(T_1, 1:n_test_1, rev=true)
    selected_test_indexes_2 = partialsortperm(T_2, 1:n_test_2, rev=true)
    selected_train_indexes_1 = setdiff(1:n_1, selected_test_indexes_1)
    selected_train_indexes_2 = setdiff(1:n_2, selected_test_indexes_2)

    X_1_train, U_1_train, Y_1_train, T1_train = X_1[:, selected_train_indexes_1], U_1[selected_train_indexes_1], Y_1[selected_train_indexes_1], T_1[selected_train_indexes_1]
    X_2_train, U_2_train, Y_2_train, T2_train = X_2[:, selected_train_indexes_2], U_2[selected_train_indexes_2], Y_2[selected_train_indexes_2], T_2[selected_train_indexes_2]

    X_1_test, U_1_test, Y_1_test, T1_test = X_1[:, selected_test_indexes_1], U_1[selected_test_indexes_1], Y_1[selected_test_indexes_1], T_1[selected_test_indexes_1]
    X_2_test, U_2_test, Y_2_test, T2_test = X_2[:, selected_test_indexes_2], U_2[selected_test_indexes_2], Y_2[selected_test_indexes_2], T_2[selected_test_indexes_2]

    Random.seed!(1234)
    ind_train = shuffle(1:(size(X_1_train, 2) + size(X_2_train, 2)))
    ind_test = shuffle(1:size(X_1_test, 2) + size(X_2_test, 2))
    T_train = vcat([T1_train, T2_train]...)[ind_train]
    T_test = vcat([T1_test, T2_test]...)[ind_test]
    X_train = hcat([X_1_train, X_2_train]...)[:, ind_train]
    X_test = hcat([X_1_test, X_2_test]...)[:, ind_test]
    Y_train = vcat([Y_1_train, Y_2_train]...)[ind_train]
    Y_test = vcat([Y_1_test, Y_2_test]...)[ind_test]
    U_train = vcat([U_1_train, U_2_train]...)[ind_train]
    U_test = vcat([U_1_test, U_2_test]...)[ind_test]

    ###########################################################################################
    #################################### NEURAL NETWORK TRAIN #################################
    ###########################################################################################

    X_NN_train = vcat([X_train, reshape(U_train, (1, :))]...)
    X_NN_test = vcat([X_test, reshape(U_test, (1, :))]...)

    sigmoid(x) = 1 ./ (1 .+ exp.(-x))
    rbf(x) = exp.(-(x .^ 2))
    relu(x) = max.(0, x)

    # Multilayer FeedForward
    U = Lux.Chain(
        Lux.Dense(4, 10, relu),  
        Lux.Dropout(0.2),  
        Lux.Dense(10, 10, relu),
        Lux.Dropout(0.3),  
        Lux.Dense(10, 10, relu),
        Lux.Dropout(0.3),  
        Lux.Dense(10, 1)   
    )
    # Get the initial parameters and state variables of the model
    p, st = Lux.setup(rng, U)
    _st = st
    LuxCore.trainmode(_st)

    function loss_nn(θ, X=X_NN_train, Y=Y_train)
        Ŷ = U(X, θ, _st)[1]'
        mean(abs2, Y .- Ŷ)
    end

    losses = Float64[]

    callback = function (state, l)
        push!(losses, l)
        if length(losses) % 1000 == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_nn(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

    res1 = Optimization.solve(
        optprob, OptimizationOptimisers.Adam(), callback = callback, maxiters = 20_000)
    println("Training loss after $(length(losses)) iterations: $(losses[end])")

    plot(1:20_000, losses[1:20_000], yaxis = :log10, xaxis = :log10,
        xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)

    p_trained = res1.u
    LuxCore.testmode(_st)
    println("Erreur NN Test : $(loss_nn(p_trained, X_NN_test, Y_test))")
    LuxCore.trainmode(_st)

    ###########################################################################################
    ######################################### LLR TRAIN #######################################
    ###########################################################################################

    selected_aeration_on = findall(==(1), U_train)
    selected_aeration_off = findall(==(0), U_train)

    knn_on = StateSpaceIdentification.LocalLinearRegressor(
        T_train[selected_aeration_on],
        Float64.(X_train[:, selected_aeration_on]),
        Float64.(reshape(Y_train[selected_aeration_on], (1, :))),
        KDTree(X_train[:, selected_aeration_on]),
        n_neighbors = df.llrs[id][1].n_neighbors,
        min_lag_in_days = df.llrs[id][1].min_lag_in_days,
        kernel_type = df.llrs[id][1].kernel_type,
        max_cyclic_lag = df.llrs[id][1].max_cyclic_lag
    )

    knn_off = StateSpaceIdentification.LocalLinearRegressor(
        T_train[selected_aeration_off],
        Float64.(X_train[:, selected_aeration_off]),
        Float64.(reshape(Y_train[selected_aeration_off], (1, :))),
        KDTree(X_train[:, selected_aeration_off]),
        n_neighbors = df.llrs[id][2].n_neighbors,
        min_lag_in_days = df.llrs[id][2].min_lag_in_days,
        kernel_type = df.llrs[id][2].kernel_type,
        max_cyclic_lag = df.llrs[id][2].max_cyclic_lag
    )

    function loss_knn(X=X_train, Y=Y_train, U=U_train)
        Y_pred = Float64[]
        for i in 1:size(U, 1)
            if U[i] > 0.5
                push!(Y_pred, knn_on(Float64.(X[:, i:i]), 0.0)[1,1])
            else
                push!(Y_pred, knn_off(Float64.(X[:, i:i]), 0.0)[1,1])
            end
        end
        mean(abs2, Y .- Y_pred)
    end

    loss_knn(X_test, Y_test, U_test)

    ###########################################################################################
    #################################### Compare Performance ##################################
    ###########################################################################################

    begin
        LuxCore.testmode(_st)
        @info "Erreur NN Test : $(loss_nn(p_trained, X_NN_test, Y_test))"
        LuxCore.trainmode(_st)
        @info "Erreur KNN Test : $(loss_knn(X_test, Y_test, U_test))"
    end
    push!(nn_error, loss_nn(p_trained, X_NN_test, Y_test))
    push!(knn_error, loss_knn(X_test, Y_test, U_test))
end


Plots.backend(:gr)

plot_font = "Computer Modern"
font_size = 12

default(fontfamily=plot_font, 
        guidefontsize=font_size,
        tickfontsize = font_size, 
        legendfontsize = font_size - 2,
        linewidth=1.1, 
        framestyle=:box, 
        label=nothing, 
        grid=false
)

@save datadir("errors.jld2") knn_error nn_error


# Creating the boxplot with labels in the x-axis
boxplot([knn_error, nn_error], 
    ylabel="Test Error", 
    xticks=([1, 2], ["Local Linear Regression", "Neural Network"]), top_margin=5mm, left_margin=2mm,
    size=(700, 450), title="Distribution of Test Errors: \n Local Linear Regression vs. Fully Connected Neural Network")
safesave(savefig(plot!(size=(700, 450)), plotsdir("paper", "nn_vs_llr.pdf")))