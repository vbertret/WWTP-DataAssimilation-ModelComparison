include(srcdir("dataset.jl"))
include(srcdir("model.jl"))

using DataFrames
using OptimizationNLopt
using NearestNeighbors

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

        kdtree_aeration_on = KDTree(knn_data_aeration_on)
        kdtree_aeration_off = KDTree(knn_data_aeration_off)
        llr_aeration_on = StateSpaceIdentification.LocalLinearRegressor(
            t_data[ind_aeration_on[1:(end - 1), 1]], knn_data_aeration_on,
            permutedims(succesors_data_aeration_on, (2, 1)), kdtree_aeration_on, n_neighbors = 100,
            min_lag_in_days = 20 / 1440, kernel_type = "rectangular")
        llr_aeration_off = StateSpaceIdentification.LocalLinearRegressor(
            t_data[.!ind_aeration_on[1:(end - 1), 1]], knn_data_aeration_off,
            permutedims(succesors_data_aeration_off, (2, 1)), kdtree_aeration_off, n_neighbors = 100,
            min_lag_in_days = 20 / 1440, kernel_type = "rectangular")

        model, callback_llrs = get_model(symbol_model, dataset.init_t, dataset.init_state,
            typeof(dataset); dt_model = dt_model,
            opt_params_linear_black_box_model =  res.opt_parameters[1][(end - 2 * n_Y+1):end],
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

    elseif symbol_model ∈ [:nonparametric_black]

        # Train model
        optim_method = Opt(:LD_LBFGS, size(model.parameters, 1))
        filter_method = filter_method = ConditionalParticleFilter(model, n_particles = 50, positive = true, conditional_particle = μ_t)
        smoother_method = BackwardSimulationSmoother(model, n_particles = 50)
        max_iter_em = 100
        abstol_em = 1e-5
        reltol_em = 1e-5
        iter_saem = 10
        alpha = 0.1
        p_optim_method = Dict(:maxiters => 100)
        t_training = @elapsed begin
            optim_params = ExpectationMaximization(
                model, y_train, E_train, U_train; filter_method = filter_method,
                smoother_method = smoother_method, optim_method = optim_method,
                iter_saem = iter_saem,
                alpha = alpha, abstol_em = abstol_em, reltol_em = reltol_em,
                maxiters_em = max_iter_em, p_optim_method = p_optim_method, verbose = true, custom_llrs_update_callback! = callback_llrs)
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
        model, _ = get_model(symbol_model, dataset.init_t, dataset.init_state, typeof(dataset); dt_model=dt_model, opt_params_linear_black_box_model = res.opt_parameters[1], llrs = res.llrs[1], μ=res.μ[1], σ=res.σ[1], observed_var=dataset.obs_var)
    else
        model = get_model(symbol_model, dataset.init_t, dataset.init_state, typeof(dataset); dt_model=dt_model, observed_var=dataset.obs_var)
    end

    # Set optimal parameters
    model.parameters = res.opt_parameters[1]
    output_dict = Dict([i => j[1] for (i,j) in zip(names(res), eachcol(res))])
    output_dict["obs_var"] = dataset.obs_var

    if symbol_model ∈ [:linear_black, :linear_grey]
        _ = update!(model, y_train, E_train, U_train)
    else
        _ = update!(model, y_train, E_train, U_train, positive=true, n_particles=300)
    end

    n_test = size(y_test, 1)
    size_window = Int(ceil(1440/dt_model))
    size_1h = Int(ceil(60/dt_model))
    y_test_nan = similar(y_test[1:size_window, :])
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

        if symbol_model ∈ [:linear_black, :linear_grey]
            filter_output_test = filtering(model, y_test_nan, E_test[i:(i+size_window-1), :], U_test[i:(i+size_window-1), :]) 
        else
            filter_output_test = filtering(model, y_test_nan, E_test[i:(i+size_window-1), :], U_test[i:(i+size_window-1), :], positive=true, n_particles=300) 
        end

        if symbol_model ∈ [:linear_black, :linear_grey]
            filtered_state_test = filter_output_test.filtered_state
        else
            filtered_state_test = filter_output_test.filtered_particles_swarm
        end

        for j in 1:24
            rmse_tab[i_tab, j, :] = StateSpaceIdentification.rmse(subsample_x_test[:, (i+size_1h*(j-1)):(i+size_1h*j-1)], filtered_state_test[(1+size_1h*(j-1)):(size_1h*j)])
            ic_tab[i_tab, j, :] = StateSpaceIdentification.coverage_probability(subsample_x_test[:, (i+size_1h*(j-1)):(i+size_1h*j-1)], filtered_state_test[(1+size_1h*(j-1)):(size_1h*j)])
            aw_tab[i_tab, j, :] = StateSpaceIdentification.average_width(filtered_state_test[(1+size_1h*(j-1)):(size_1h*j)])
        end

        # Update model
        if symbol_model ∈ [:linear_black, :linear_grey]
            _ = StateSpaceIdentification.update!(model, y_test[i:(i+dt_window-1), :], E_test[i:(i+dt_window-1), :], U_test[i:(i+dt_window-1), :]) 
        else
            _ = StateSpaceIdentification.update!(model, y_test[i:(i+dt_window-1), :], E_test[i:(i+dt_window-1), :], U_test[i:(i+dt_window-1), :], positive=true, n_particles=300) 
        end

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