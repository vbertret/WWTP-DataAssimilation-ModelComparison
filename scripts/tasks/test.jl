using DrWatson
@quickactivate "ASP_model_comparison"

include(srcdir("dataset.jl"))
include(srcdir("model.jl"))


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


function test_plot_save_model(tasks_parameters)

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
        model, _ = get_model(symbol_model, dataset.init_t, dataset.init_state; dt_model=dt_model, opt_params_linear_black_box_model = res.opt_parameters[1], llrs = res.llrs[1], μ=res.μ[1], σ=res.σ[1], observed_var=dataset.obs_var)
    else
        model = get_model(symbol_model, dataset.init_t, dataset.init_state; dt_model=dt_model, observed_var=dataset.obs_var)
    end
    
    # Set optimal parameters
    model.parameters = res.opt_parameters[1]

    filter_output_train = StateSpaceIdentification.update!(model, y_train, E_train, U_train, positive=true, n_particles=300) 

    y_test_nan = similar(y_test)
    y_test_nan .= NaN
    filter_output_test = StateSpaceIdentification.filter(model, y_test_nan, E_test, U_test, positive=true, n_particles=300) 

    output_dict = Dict([i => j[1] for (i,j) in zip(names(res), eachcol(res))])
    if symbol_model ∈ [:linear_black, :linear_grey]
        output_dict["filtered_state_train"] = get_nh4_timeseries_from_model(symbol_model, filter_output_train.filtered_state, dataset.obs_var)
        output_dict["filtered_state_test"] = get_nh4_timeseries_from_model(symbol_model, filter_output_test.filtered_state, dataset.obs_var)
    else
        output_dict["filtered_state_train"] = get_nh4_timeseries_from_model(symbol_model, filter_output_train.filtered_particles_swarm, dataset.obs_var)
        output_dict["filtered_state_test"] = get_nh4_timeseries_from_model(symbol_model, filter_output_test.filtered_particles_swarm, dataset.obs_var)
    end

    # Specific score
    output_dict["rmse_train"] = StateSpaceIdentification.rmse(x_train[10:10, 1:dt_model:end], output_dict["filtered_state_train"])
    output_dict["rmse_test"] = StateSpaceIdentification.rmse(x_test[10:10, 1:dt_model:end], output_dict["filtered_state_test"])
    output_dict["cp_train"] = StateSpaceIdentification.coverage_probability(x_train[10:10, 1:dt_model:end], output_dict["filtered_state_train"])
    output_dict["cp_test"] = StateSpaceIdentification.coverage_probability(x_test[10:10, 1:dt_model:end], output_dict["filtered_state_test"])
    output_dict["aw_train"] = StateSpaceIdentification.average_width(output_dict["filtered_state_train"])
    output_dict["aw_test"] = StateSpaceIdentification.average_width(output_dict["filtered_state_test"])


    # Specific name
    prefix = datadir("evaluation", String(output_dict["model"]))
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

    # Load dataset and preprocess
    dicts = dict_list(general_args)

    # Train instances
    map(test_plot_save_model, dicts)

end