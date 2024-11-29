using DrWatson
@quickactivate "ASP_model_comparison"

include(srcdir("dataset.jl"))
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--overwrite", "-o"
            help = "Do you want to overwrite datasets that have already been generated ? Default is false."
            default=false
            arg_type = Bool
        "--seed", "-s"
            help = "Do you want to regenerate/generate a dataset with a specific seed ?"
            arg_type = Int
        "name_dataset"
            help = "the name of the dataset to be generated."
            required = true
            arg_type = String
        "nb_dataset"
            help = "the number of dataset to to be generated."
            required = true
            arg_type = Int
    end

    return parse_args(s)
end

function generate_save_dataset(name_dataset, nb_dataset; seed=nothing, overwrite=false)

    if isnothing(seed)
        @info "Seed is not specified. Generate $name_dataset dataset $nb_dataset times."
        for i in 1:nb_dataset
    
            # Get dataset
            data = get_dataset(name_dataset; seed = i)
    
            # Save dataset
            save(data; overwrite = overwrite)
    
        end
    else
        @info "Seed is specified. Generate $name_dataset dataset with the seed $seed ."
    
        # Get dataset
        data = get_dataset(name_dataset; seed = seed)
    
        # Save dataset
        save(data; overwrite = overwrite)
    
    end

end

# If used in interactive mode, don't execute this part.
if !isinteractive()

    parsed_args = parse_commandline()

    # Get parsed args
    nb_dataset = parsed_args["nb_dataset"]
    name_dataset = parsed_args["name_dataset"]
    seed = parsed_args["seed"]
    overwrite = parsed_args["overwrite"]

    generate_save_dataset(name_dataset, nb_dataset; seed=seed, overwrite=overwrite)

end