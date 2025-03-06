using DrWatson
@quickactivate "ASP_model_comparison"

include(srcdir("dataset.jl"))
include(srcdir("model.jl"))

using DataFrames
using OptimizationNLopt
using NearestNeighbors

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
        """
        model
        """
        help = "the model you want to train."
        required = true
        arg_type = String
    end

    return parse_args(s)
end


function train_save_model(tasks_parameters)

    # Get arguments tasks_parameters
    symbol_model = tasks_parameters["model"]
    name_dataset = tasks_parameters["name_dataset"]
    seed = tasks_parameters["seed"]
    dt_model = tasks_parameters["dt_model"]

    # Load and Split dataset
    dataset = load_dataset(name_dataset, seed = seed)
    adapt_signals_to_model!(dataset, symbol_model, dt_model)
    (_, y_train, U_train, E_train), (_, _, _, _) = train_test_split(dataset)
    n_Y = size(dataset.obs_var)[1]

    # Build model
    if symbol_model == :nonparametric_black
        @info "Load results obtained with linear_black and Kalman Filter."
        sname = savename((@dict name_dataset seed))
        res = collect_results(
            datadir("training", "linear_black"); rinclude = [Regex(".*($sname)[.|_].*")])

        if size(res, 1) != 1
            @error "One training with the linear_black model on the same dataset has to be done !"
        end
        trained_model = get_model(
            :linear_black, dataset.init_t, dataset.init_state, typeof(dataset);
            dt_model = dt_model, observed_var = dataset.obs_var)
        trained_model.parameters = res.opt_parameters[1]

        dataset_bis = load_dataset(name_dataset, seed = seed)
        adapt_signals_to_model!(dataset_bis, :linear_black, dt_model)
        (_, y_train_bis, U_train_bis, E_train_bis), (_, _, _, _) = train_test_split(dataset_bis)

        filter_output_train = filtering(
            trained_model, y_train_bis, E_train_bis, U_train_bis)
        smoother_output_train = smoothing(
            trained_model, y_train_bis, E_train_bis, U_train_bis, filter_output_train)
        μ_t = Matrix(hcat(map((x) -> x.μ_t, smoother_output_train.smoothed_state)...)')

        knn_data = transpose(hcat([
            StateSpaceIdentification.standard_scaler(μ_t)[1:(end - 2), :],
            E_train[1:(end - 1), :]]...))
        succesors_data = μ_t[2:(end - 1), :] - μ_t[1:(end - 2), :]
        t_data = [dataset.init_t + (dt_model / 1440) * (i - 1) for i in 1:size(knn_data, 2)]

        ind_aeration_on = (U_train[1:(end), :] .> 0.5)
        ind_aeration_on_whithout_end = @view ind_aeration_on[1:(end - 1), :]
        knn_data_aeration_on = knn_data[:, ind_aeration_on_whithout_end]
        knn_data_aeration_off = knn_data[:, .!ind_aeration_on_whithout_end]
        succesors_data_aeration_on = reshape(
            succesors_data[ind_aeration_on_whithout_end[:, 1], :], (:, n_Y))
        succesors_data_aeration_off = reshape(
            succesors_data[.!ind_aeration_on_whithout_end[:, 1], :], (:, n_Y))

        lag_time = 0
        if name_dataset == "AcigneDataset"
            lag_time = 60 * 3 / 1440
        end

        kdtree_aeration_on = KDTree(knn_data_aeration_on)
        kdtree_aeration_off = KDTree(knn_data_aeration_off)
        llr_aeration_on = StateSpaceIdentification.LLR(
            t_data[ind_aeration_on[1:(end - 1), 1]], knn_data_aeration_on,
            succesors_data_aeration_on, kdtree_aeration_on, Set(); k = 100,
            lag_x = 10 / 1440, kernel = "rectangular", lag_time = lag_time)
        llr_aeration_off = StateSpaceIdentification.LLR(
            t_data[.!ind_aeration_on[1:(end - 1), 1]], knn_data_aeration_off,
            succesors_data_aeration_off, kdtree_aeration_off, Set(); k = 100,
            lag_x = 10 / 1440, kernel = "rectangular", lag_time = lag_time)

        model, update_M = get_model(symbol_model, dataset.init_t, dataset.init_state,
            typeof(dataset); dt_model = dt_model,
            opt_params_linear_black_box_model = res.opt_parameters[1][(end - 1 * n_Y):end],
            llrs = [llr_aeration_on, llr_aeration_off],
            μ = mean(μ_t[1:(end - 2), :], dims = 1)[1, :],
            σ = std(μ_t[1:(end - 2), :], dims = 1)[1, :], observed_var = dataset.obs_var)

    else
        model = get_model(
            symbol_model, dataset.init_t, dataset.init_state, typeof(dataset);
            dt_model = dt_model, observed_var = dataset.obs_var)
    end

    @info "Start Training model."
    if symbol_model ∈ [:linear_grey, :linear_black]
        max_iter_em = 200
        p_optim_method = Dict(:maxiters => 100)
        optim_method = Opt(:LD_LBFGS, size(model.parameters, 1))
        abstol_em = 1e-4
        reltol_em = 1e-4
        t_training = @elapsed begin
            optim_params = ExpectationMaximization(
                model, y_train, E_train, U_train, maxiters_em = max_iter_em,
                p_optim_method = p_optim_method, optim_method = optim_method,
                abstol_em = abstol_em, reltol_em = reltol_em, verbose = true)
        end
    elseif symbol_model ∈ [:grey]
        optim_method = Opt(:LD_LBFGS, size(model.parameters, 1))
        lb, ub = get_bounds_parameters_model(symbol_model)
        p_opt_problem = Dict(:lb => lb, :ub => ub)
        filter_method = ParticleFilter(model, n_particles = 300, positive = true)
        smoother_method = BackwardSimulationSmoother(model, n_particles = 300)
        max_iter_em = 100
        abstol_em = 1e-4
        reltol_em = 1e-4
        iter_saem = 10
        alpha = 0.1
        p_optim_method = Dict(:maxiters => 100)
        t_training = @elapsed begin
            optim_params = ExpectationMaximization(
                model, y_train, E_train, U_train; filter_method = filter_method,
                smoother_method = smoother_method, optim_method = optim_method,
                p_opt_problem = p_opt_problem, iter_saem = iter_saem,
                alpha = alpha, abstol_em = abstol_em, reltol_em = reltol_em,
                maxiters_em = max_iter_em, p_optim_method = p_optim_method, verbose = true)
        end
    elseif symbol_model ∈ [:white]

        # Get first conditionnal particle
        n_filtering = 100
        filter_output = filtering(model,
            y_train,
            E_train,
            U_train,
            filter_method = ParticleFilter(
                model, n_particles = n_filtering, positive = true))
        smoother_output = smoothing(model, y_train, E_train, U_train, filter_output;
            smoother_method = BackwardSimulationSmoother(model, n_particles = n_filtering))
        conditional_particle = hcat(map(
            (x) -> x.particles_state[:, end], smoother_output.smoothed_particles_swarm)...)'

        # Train model
        optim_method = Opt(:LD_LBFGS, size(model.parameters, 1))
        lb, ub = get_bounds_parameters_model(symbol_model, observed_var = dataset.obs_var)
        p_opt_problem = Dict(:lb => lb, :ub => ub)
        filter_method = ConditionalParticleFilter(model, n_particles = 10, positive = true,
            conditional_particle = conditional_particle)
        smoother_method = BackwardSimulationSmoother(model, n_particles = 5)
        max_iter_em = 1
        abstol_em = 1e-4
        reltol_em = 1e-4
        iter_saem = 0
        alpha = 0.1
        p_optim_method = Dict(:maxiters => 100)
        t_training = @elapsed begin
            optim_params = ExpectationMaximization(
                model, y_train, E_train, U_train; filter_method = filter_method,
                smoother_method = smoother_method, optim_method = optim_method,
                p_opt_problem = p_opt_problem, iter_saem = iter_saem,
                alpha = alpha, abstol_em = abstol_em, reltol_em = reltol_em,
                maxiters_em = max_iter_em, p_optim_method = p_optim_method, verbose = true)
        end

    elseif symbol_model ∈ [:nonparametric_black]

        # Train model
        optim_method = Opt(:LD_LBFGS, size(model.parameters, 1))
        lb, ub = get_bounds_parameters_model(symbol_model)
        p_opt_problem = Dict() #Dict(:lb => lb, :ub => ub)
        p_filter = Dict(:n_particles => 50, :positive => true)
        p_smoothing = Dict(:n_particles => 50)
        max_iter_em = 10
        p_optim_method = Dict(:maxiters => 100)
        t_training = @elapsed begin
            optim_params, _ = StateSpaceIdentification.npSEM_CPF(
                model, y_train, E_train, U_train, update_M; optim_method = optim_method,
                p_opt_problem = p_opt_problem, p_filter = p_filter,
                p_smoothing = p_smoothing, maxiters = max_iter_em,
                p_optim_method = p_optim_method, conditional_particle = μ_t)
        end
        tasks_parameters["llrs"] = model.system.llrs
        tasks_parameters["μ"] = model.system.μ
        tasks_parameters["σ"] = model.system.σ
    end
    @info "End Training model. Current optimal parameters=$(optim_params). Took $(t_training)s."
    tasks_parameters["opt_parameters"] = optim_params
    tasks_parameters["training_time"] = t_training

    # Specific name
    prefix = datadir("training", String(tasks_parameters["model"]))
    sname = savename((@dict name_dataset seed), "jld2")
    mkpath(prefix)
    wsave(datadir(prefix, sname), tasks_parameters)
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

# tasks_parameters = Dict("model"=>:nonparametric_black, "seed"=>1, "name_dataset"=>"AcigneDataset", "dt_model"=>5)
# using Profile
# # Profile.clear()
# # @profview train_save_model(tasks_parameters);

# using PProf
# Profile.Allocs.clear()
# Profile.Allocs.@profile sample_rate=0.01 train_save_model(tasks_parameters);
# PProf.Allocs.pprof()

# # @code_warntype train_save_model(tasks_parameters)
# # using JET
# # @report_opt train_save_model(tasks_parameters)