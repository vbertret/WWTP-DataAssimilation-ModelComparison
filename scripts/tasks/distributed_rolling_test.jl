using Distributed

addprocs(16; exeflags=`--threads=1 --heap-size-hint=5G`);

@everywhere using DrWatson
@everywhere @quickactivate "ASP_model_comparison"

@everywhere include(srcdir("dataset.jl"))
@everywhere include(srcdir("model.jl"))


using DataFrames
using ArgParse
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--overwrite", "-o"
            help = "Do you want to overwrite results that have already been generated ? Default is false."
            default=false
            arg_type = Bool
        "--name_dataset", "-n"
            help = "the name of the dataset to use."
            default="S1"
            arg_type = String
        "--seed", "-s"
            help = "What is the seed to use ?"
            arg_type = Int
        "model"
            help = "the model you want to test."
            required = true
            arg_type = String
    end

    return parse_args(s)
end


@everywhere function test_plot_save_model(tasks_parameters)

    # Get arguments tasks_parameters
    symbol_model = tasks_parameters["model"]
    name_dataset = tasks_parameters["name_dataset"]
    seed = tasks_parameters["seed"]
    dt_model = tasks_parameters["dt_model"]

    # Load and Split dataset
    dataset = load_dataset(name_dataset, seed=seed)
    adapt_signals_to_model!(dataset, symbol_model, dt_model)
    (x_train, y_train, U_train, E_train), (x_test, y_test, U_test, E_test) = train_test_split(dataset)

    # Get training_results
    sname = savename((@dict name_dataset seed))
    res = collect_results(datadir("training", String(symbol_model)); rinclude = [Regex(".*($sname)[.|_].*")])

    # Build model
    if symbol_model == :nonparametric_black
        model, _ = get_model(symbol_model, dataset.init_t, dataset.init_state, typeof(dataset); dt_model=dt_model, opt_params_linear_black_box_model = res.opt_parameters[1], llrs = res.llrs[1], μ=res.μ[1], σ=res.σ[1], observed_var=dataset.obs_var)
    else
        model = get_model(symbol_model, dataset.init_t, dataset.init_state, typeof(dataset); dt_model=dt_model, observed_var=dataset.obs_var)
    end

    # Set optimal parameters
    model.parameters = res.opt_parameters[1]
    output_dict = Dict([i => j[1] for (i,j) in zip(names(res), eachcol(res))])
    output_dict["obs_var"] = dataset.obs_var

    _ = StateSpaceIdentification.update!(model, y_train, E_train, U_train, positive=true, n_particles=300) 

    n_test = size(y_test, 1)
    size_window = Int(ceil(1440/dt_model))
    size_1h = Int(ceil(60/dt_model))
    y_test_nan = similar(y_test[1:size_window])
    y_test_nan .= NaN

    n_X = model.system.n_X
    subsample_x_test = @view x_test[:, 1:dt_model:end]
    subsample_x_test = adapt_x_test_model(symbol_model, subsample_x_test, dataset.obs_var, typeof(dataset))
    dt_window = 10
    n_tab = Int(ceil((n_test - size_window)/dt_window))
    rmse_tab = zeros(Float64, n_tab, 24, n_X)
    ic_tab = zeros(Float64, n_tab, 24, n_X)
    aw_tab = zeros(Float64, n_tab, 24, n_X)
    @info "Evaluation : Start evaluation $symbol_model model with seed=$seed."
    for (i_tab, i) in enumerate(1:dt_window:(n_test - size_window))

        if i%100 == 1
            @info "Evaluation : $(round((i/(n_test - size_window))*100, digits=2))% progress."
        end

        filter_output_test = StateSpaceIdentification.filter(model, y_test_nan, E_test[i:(i+size_window-1), :], U_test[i:(i+size_window-1), :], positive=true, n_particles=300) 

        if symbol_model ∈ [:linear_black, :linear_grey]
            filtered_state_test = filter_output_test.filtered_state
        else
            filtered_state_test = filter_output_test.filtered_particles_swarm
        end

        for j in 1:24
            rmse_tab[i_tab, j, :] = StateSpaceIdentification.rmse(subsample_x_test[:, i:(i+size_1h*j-1)], filtered_state_test[1:(size_1h*j)])
            ic_tab[i_tab, j, :] = StateSpaceIdentification.coverage_probability(subsample_x_test[:, i:(i+size_1h*j-1)], filtered_state_test[1:(size_1h*j)])
            aw_tab[i_tab, j, :] = StateSpaceIdentification.average_width(filtered_state_test[1:(size_1h*j)])
        end

        # Update model
        _ = StateSpaceIdentification.update!(model, y_test[i:(i+dt_window-1), :], E_test[i:(i+dt_window-1), :], U_test[i:(i+dt_window-1), :], positive=true, n_particles=300) 

    end
    output_dict["rolling_rmse"] = mean(rmse_tab, dims=1)[1, :, :]
    output_dict["rolling_cp"] = mean(ic_tab, dims=1)[1, :, :]
    output_dict["rolling_aw"] = mean(aw_tab, dims=1)[1, :, :]


    # Specific name
    prefix = datadir("rolling_evaluation", String(output_dict["model"]))
    sname = savename((@dict name_dataset seed), "jld2")
    mkpath(prefix)
    wsave(datadir(prefix, sname), output_dict)

end


# If used in interactive mode, don't execute this part.
if !isinteractive()

    parsed_args = parse_commandline()

    if isnothing(parsed_args["seed"])
        list_files = readdir(datadir("dataset", parsed_args["name_dataset"]); join=true) 
        list_seeds = sort(map(x-> parse(Int64, x[:seed]), match.(r"seed=(?<seed>[0-9]*)[_.]*", list_files)))
        parsed_args["seed"] = list_seeds
    end

    # Get parsed args
    general_args = Dict(
        "model" => Symbol(parsed_args["model"]),
        "seed" => parsed_args["seed"],
        "name_dataset" => parsed_args["name_dataset"],
        "dt_model" => 5
    )

    # overwrite = parsed_args["overwrite"]
    @everywhere function on_error(error)
        println(error)
        return 0
    end

    # Load dataset and preprocess
    dicts = dict_list(general_args)

    # Train instances
    pmap(test_plot_save_model, dicts; on_error=ex->on_error(ex))

end