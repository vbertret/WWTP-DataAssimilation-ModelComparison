include(srcdir("dataset.jl"))
include(srcdir("model.jl"))

using DataFrames, Statistics
using Plots
using StatsPlots
using LaTeXStrings
using Measures
using NearestNeighbors
using Latexify

function custom_quantile(x, alpha)
    x = hcat(x...)
    return [quantile(x[i, :], alpha) for i in 1:size(x, 1)]'
end

function get_nh4_metrics(name, total_metrics, observed_var)
    idx_nh4 = findall(x-> x == :nh4, observed_var)[1]
    @match name begin
        :white => @match_return  total_metrics[:, 4]
        :grey => @match_return total_metrics[:, 1]
        :linear_black => @match_return total_metrics[:, idx_nh4]
        :nonparametric_black => @match_return total_metrics[:, idx_nh4]
        :linear_grey => @match_return total_metrics[:, 1]
        _ => throw(ArgumentError("Non available symbol. Model has to be in $available_model ."))
    end
end

function cat_horizon_op(x, op)
    return Ref(NamedTuple{Tuple([Symbol("H$i") for i in 1:24])}(Tuple(op(x))))
end

function calculate_grouped_aggregations(sub_df::DataFrame, col_name::Symbol; grouping_cols::Vector{Symbol} = [:model, :name_dataset])

    # Group the DataFrame
    gdf = groupby(sub_df, grouping_cols)

    # Calculate aggregations (Mean, Low/High Quantiles)
    aw_df = combine(gdf, col_name => (x -> cat_horizon_op(x, y -> mean(y, dims=1)[1])) => AsTable)
    q_low_aw_df = combine(gdf, col_name => (x -> cat_horizon_op(x, y -> custom_quantile(y, 0.025))) => AsTable)
    q_high_aw_df = combine(gdf, col_name => (x -> cat_horizon_op(x, y -> custom_quantile(y, 0.975))) => AsTable)
    result_cols = names(aw_df, Not(grouping_cols))

    # Extract values into matrices
    μ_aw_values = Matrix{Float64}(aw_df[!, result_cols])'
    q_low_aw_values = Matrix{Float64}(q_low_aw_df[!, result_cols])'
    q_high_aw_values = Matrix{Float64}(q_high_aw_df[!, result_cols])'

    # Generate labels for the legend and concatenate the values of grouping columns for each group
    aw_label = hcat(join.(eachrow(aw_df[!, grouping_cols]), "_")...) 

    # Return the calculated data as a NamedTuple
    return (
        mean_values = μ_aw_values,
        low_quantile_values = q_low_aw_values,
        high_quantile_values = q_high_aw_values,
        labels = aw_label
    )
end

function extract_noise_obs(params, name_dataset)
    if name_dataset == "S1"
        return abs.(params[end:end])
    else
        return abs.(params[end-1:end])
    end
end

function extract_noise_model(params, model, name_dataset)
    if name_dataset == "S1"
        if model == :grey
            return abs.(params[4:4])
        elseif model == :linear_black
            return abs.(params[(end-1):(end-1)])
        elseif model == :nonparametric_black
            return abs.(params[(end-1):(end-1)])
        elseif model == :white
            return abs.(params[20:24])
        end
    else
        return abs.(params[end-3:end-2])
    end
end


function convert_new_interface(row)
    for i in [1, 2]
        input = row[i]
        row[i] = StateSpaceIdentification.LocalLinearRegressor(input.index_analogs, input.analogs, permutedims(input.successors, (2, 1)), KDTree(input.analogs), n_neighbors=input.k, min_lag_in_days=input.lag_x, kernel_type=input.kernel)
    end
end

function get_prediction(tasks_parameters; H=288, lag=1, name_directory=nothing, model=nothing, n_particles=1_000, training=true)

    # Get arguments tasks_parameters
    symbol_model = tasks_parameters["model"]
    name_dataset = tasks_parameters["name_dataset"]
    seed = tasks_parameters["seed"]
    dt_model = tasks_parameters["dt_model"]

    if isnothing(name_directory)
        name_directory = datadir("training", String(symbol_model))
    end

    # Load and Split dataset
    dataset = load_dataset(name_dataset, seed=seed)
    adapt_signals_to_model!(dataset, symbol_model, dt_model)
    (x_train, y_train, U_train, E_train), (x_test, y_test, U_test, E_test) = train_test_split(dataset)

    # Adapt testsize
    x_test = x_test[:, 1:((H+lag-1)*dt_model)]
    y_test = y_test[1:(H+lag-1), :]
    U_test = U_test[1:(H+lag-1), :]
    E_test = E_test[1:(H+lag-1), :]
    
    # Get training_results
    sname = savename((@dict name_dataset seed))
    res = collect_results(name_directory; rinclude = [Regex(".*($sname)[.|_].*")])

    if isnothing(model)
        # Build model
        if symbol_model == :nonparametric_black
            # Fix compatibility problems with NearestNeighbors
            if  ~(typeof(res.llrs[1][1]) <: LocalLinearRegressor)
                convert_new_interface(res.llrs[1])
            end
            # res.llrs[1][1].neighbor_tree = KDTree(res.llrs[1][1].analog_inputs)
            # res.llrs[1][2].neighbor_tree = KDTree(res.llrs[1][2].analog_inputs)

            model, _ = get_model(symbol_model, dataset.init_t, dataset.init_state, typeof(dataset); dt_model=dt_model, opt_params_linear_black_box_model = res.opt_parameters[1], llrs = res.llrs[1], μ=res.μ[1], σ=res.σ[1], observed_var=dataset.obs_var)
        else
            model = get_model(symbol_model, dataset.init_t, dataset.init_state, typeof(dataset); dt_model=dt_model, observed_var=dataset.obs_var)
        end

        if training==false
            return model
        end
        
        # Set optimal parameters
        model.parameters = res.opt_parameters[1]

        y_train_nan = similar(y_train)
        y_train_nan .= NaN
        @info "Update model with training set."
        if symbol_model ∈ [:linear_black, :linear_grey]
            filter_output_train = update!(model, y_train, E_train, U_train) 
        else
            filter_output_train = update!(model, y_train, E_train, U_train, positive=true, n_particles=300)
        end
    end

    if symbol_model ∈ [:linear_black, :linear_grey]
        new_model, _ = update(model, y_test[1:lag, :], E_test[1:lag, :], U_test[1:lag, :]) 
    else
        new_model, _ = update(model, y_test[1:lag, :], E_test[1:lag, :], U_test[1:lag, :], positive=true, n_particles=300) 
    end

    y_test_nan = similar(y_test)
    y_test_nan .= NaN
    if symbol_model ∈ [:linear_black, :linear_grey]
        filter_output_test = filtering(new_model, y_test_nan[(lag+1):end, :], E_test[(lag+1):end, :], U_test[(lag+1):end, :]) 
    else
        filter_output_test = filtering(new_model, y_test_nan[(lag+1):end, :], E_test[(lag+1):end, :], U_test[(lag+1):end, :], n_particles=n_particles, positive=true) 
    end

    output_dict = Dict([i => j[1] for (i,j) in zip(names(res), eachcol(res))])
    output_dict["filter_output_test"] = filter_output_test
    if symbol_model ∈ [:linear_black, :linear_grey]
        output_dict["filtered_state_test"] = filter_output_test.filtered_state
    else
        output_dict["filtered_state_test"] = filter_output_test.filtered_particles_swarm
    end
    output_dict["y_train"] = y_train
    output_dict["y_test"] = y_test

    return output_dict, output_dict["filtered_state_test"], x_test, U_test, E_test, model, lag

end

