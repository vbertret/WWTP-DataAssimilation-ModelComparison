using DrWatson
@quickactivate "ASP_model_comparison"

include(srcdir("tasks.jl"))

using ArgParse
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--overwrite", "-o"
        help = "Do you want to overwrite results that have already been generated ? Default is false."
        default = false
        arg_type = Bool
        "--name_dataset", "-n"
        help = "the name of the dataset to use."
        default = "S1"
        arg_type = String
        "--seed", "-s"
        help = "Do you want to regenerate/generate a dataset with a specific seed ?"
        arg_type = Int
        "model"
        help = "the model you want to train."
        required = true
        arg_type = String
    end

    return parse_args(s)
end

# If used in interactive mode, don't execute this part.
if !isinteractive()
    parsed_args = parse_commandline()

    if isnothing(parsed_args["seed"])
        list_files = readdir(datadir("dataset", parsed_args["name_dataset"]); join = true)
        list_seeds = sort(map(
            x -> parse(Int64, x[:seed]), match.(r"seed=(?<seed>[0-9]*)[_.]*", list_files)))
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
    map(train_save_model, dicts)
end