using Match, SparseArrays, PDMats
using StateSpaceIdentification
using ASPSimulator
using LinearAlgebra
using DifferentialEquations
using NearestNeighbors

available_models = [:white, :grey, :linear_black, :nonparametric_black, :linear_grey]
function get_system(name::Symbol; kwargs...)
    @match name begin
        :white => @match_return get_white_box_model(; kwargs...)
        :grey => @match_return get_non_linear_grey_box_system(; kwargs...)
        :linear_black => @match_return get_linear_black_box_model(; kwargs...)
        :nonparametric_black => @match_return get_nonparametric_black_box_model(; kwargs...)
        :linear_grey => @match_return get_linear_grey_box_system(; kwargs...)
        _ => throw(ArgumentError("Non available symbol. Model has to be in $available_models ."))
    end
end

function get_initial_parameters_system(name::Symbol; kwargs...)
    @match name begin
        :white => begin
            whitebox_params, _ = ASPSimulator.get_default_parameters_simplified_asm1()
            model_params = convert(Array{Float64}, collect(whitebox_params)[1:19])
            cov_params = repeat([0.4], 5 + kwargs[:n_Y])
            params = vcat([model_params, cov_params]...)

            @match_return params
        end
        :grey => @match_return [1099, 262, 0.71, 0.4, 0.4]
        :linear_black => @match_return vcat([
            repeat([0.000001],
                kwargs[:n_Y] * kwargs[:n_Y] + kwargs[:n_Y] + kwargs[:n_Y] * kwargs[:n_E]),
            repeat([-2.30], 2 * kwargs[:n_Y])]...)
        :nonparametric_black => @match_return kwargs[:opt_params_linear_black_box_model]
        :linear_grey => @match_return [1333.0, 200.0, 0.4, 0.4]
        _ => throw(ArgumentError("Non available symbol. Model has to be in $available_models ."))
    end
end

function get_bounds_parameters_model(name::Symbol; observed_var = [:nh4], kwargs...)
    n_Y = size(observed_var, 1)
    @match name begin
        :white => @match_return vcat([repeat([1e-2], 19), repeat([0], 5 + n_Y)]...),
        vcat([repeat([10e5], 19), repeat([100], 5 + n_Y)]...)
        :grey => @match_return [1e-2, 1e-2, 1e-8, 1e-8, 1e-2], [Inf, Inf, Inf, Inf, Inf]
        :linear_black || :linear_grey => throw(ArgumentError("No bounds for linear systems."))
        :nonparametric_black => @match_return repeat([0], 2 * n_Y), repeat([Inf], 2 * n_Y)
        _ => throw(ArgumentError("Non available symbol. Model has to be in $available_models ."))
    end
end

function get_model(
        name::Symbol, init_t, x_asm1_init, type_dataset; observed_var = [:nh4], kwargs...)
    @match name begin
        :grey || :linear_grey => begin

            # Get system
            system = get_system(name; kwargs...)

            # Define initial state
            x_init = [x_asm1_init[10]]
            init_P = (zeros(1, 1) .+ 0.2) .* x_init
            init_state = GaussianStateStochasticProcess(init_t, x_init, init_P)

            parameters = get_initial_parameters_system(name)
            return ForecastingModel(system, init_state, parameters)
        end

        :linear_black | :white | :nonparametric_black => begin
            ASPSimulator.set_default_system(:asm1)
            n_Y = size(observed_var)[1]
            if type_dataset ∈ [S1Dataset, S2Dataset]
                n_E = n_Y + 1
            elseif type_dataset ∈ [AcigneDataset]
                if name == :linear_black
                    n_E = 2
                else
                    n_E = 1
                end
            end

            # Get system
            if name == :linear_black
                system = get_system(name, n_Y = n_Y, n_E = n_E; kwargs...)
            elseif name == :nonparametric_black
                system, update_M = get_system(name, n_Y = n_Y; kwargs...)
            elseif name == :white
                system = get_system(name, observed_var = observed_var; kwargs...)
            end

            # Define initial state
            if name ∈ [:linear_black, :nonparametric_black]
                if type_dataset ∈ [S1Dataset, S2Dataset]
                    x_init = reshape(
                        vcat([x_asm1_init[
                                  ASPSimulator.get_indexes_from_symbols(species), :]
                              for species in observed_var]...), (:))
                elseif type_dataset ∈ [AcigneDataset]
                    x_init = x_asm1_init
                else
                    @error "Not implemented"
                end
                init_P = x_init .* Matrix(I(n_Y) .* 0.2)
            elseif name == :white
                x_init = [x_asm1_init[2, 1] + x_asm1_init[4, 1], x_asm1_init[8, 1],
                    x_asm1_init[9, 1], x_asm1_init[10, 1], x_asm1_init[11, 1]]
                init_P = x_init .* Matrix(I(5) .* 0.2)
            end
            init_state = GaussianStateStochasticProcess(init_t, x_init, init_P)

            parameters = get_initial_parameters_system(
                name, n_Y = n_Y, n_E = n_E; kwargs...)

            if name == :linear_black
                return ForecastingModel(
                    system, init_state, parameters)
            elseif name == :white
                return ForecastingModel(
                    system, init_state, parameters)
            elseif name == :nonparametric_black
                return ForecastingModel(
                    system, init_state, parameters),
                update_M
            end
        end
        _ => throw(ArgumentError("Non available symbol. Model has to be in $available_models ."))
    end
end

function get_nh4_timeseries_from_model(name::Symbol, complete_timeseries, observed_var)
    idx_nh4 = findall(x -> x == :nh4, observed_var)
    @match name begin
        :white => @match_return TimeSeries(map(x -> x[4], complete_timeseries))
        :grey => @match_return complete_timeseries
        :linear_black => @match_return TimeSeries(map(x -> x[idx_nh4], complete_timeseries))
        :nonparametric_black => @match_return TimeSeries(map(x -> x[idx_nh4], complete_timeseries))
        :linear_grey => @match_return complete_timeseries
        _ => throw(ArgumentError("Non available symbol. Model has to be in $available_models ."))
    end
end

function get_linear_grey_box_system(; dt_model::Int64 = 1)
    @inline A_t(exogenous, params, t) = sparse(hcat([[1 -
                                                      exogenous[1] / params[1] *
                                                      (dt_model / 1440)]]...))
    @inline B_t(exogenous, params, t) = sparse(hcat([[-params[2] * (dt_model / 1440)]]...))
    @inline c_t(exogenous, params, t) = sparse([(exogenous[2] * exogenous[1]) / params[1] *
                                                (dt_model / 1440)])

    @inline H_t(exogenous, params, t) = sparse(Matrix{Float64}(I, 1, 1))
    @inline d_t(exogenous, params, t) = sparse(zeros(1))

    @inline R_t(exogenous, params, t) = PDiagMat([params[3]^2])
    @inline Q_t(exogenous, params, t) = PDiagMat([params[4]^2])

    n_X = 1
    n_Y = 1

    return GaussianLinearStateSpaceSystem{Float64}(
        A_t, B_t, c_t, H_t, d_t, R_t, Q_t, n_X, n_Y, dt_model / 1440)
end

function get_non_linear_grey_box_system(; dt_model::Int64 = 1)
    @inline M_t(x, exogenous, u, params, t) = sparse([1 -
                                                      exogenous[1] / params[1] *
                                                      (dt_model / 1440);;]) * x +
                                              sparse([-params[2] * (dt_model / 1440);;]) *
                                              u .* (x ./ (x .+ params[3])) .+
                                              sparse([(exogenous[2] * exogenous[1]) /
                                                      params[1] * (dt_model / 1440)])

    @inline H_t(x, exogenous, params, t) = x

    @inline R_t(exogenous, params, t) = PDiagMat([params[4]^2])
    @inline Q_t(exogenous, params, t) = PDiagMat([params[5]^2])

    n_X = 1
    n_Y = 1

    return GaussianNonLinearStateSpaceSystem{Float64}(
        M_t, H_t, R_t, Q_t, n_X, n_Y, dt_model / 1440)
end

function get_white_box_model(; dt_model::Int64 = 1, observed_var = [:nh4])
    n_Y = size(observed_var, 1)
    greybox_problem = ODEProblem(
        ASPSimulator.simplified_asm1!, zeros(6), (0, 1), repeat([0.0, 25]))

    function M_t(x, exogenous, u, params, t)

        # To overcome stabilities issues
        params = max.(params, 0.01)

        ode_params = vcat(params[1:19], exogenous[1:6])

        problem_ite = remake(greybox_problem, u0 = zeros(6),
            tspan = (t, t + dt_model / 1440), p = ode_params)

        n_particules = size(x, 2)
        states = vcat(x, repeat(u, n_particules)')

        function prob_func(prob, i, repeat)
            remake(prob, u0 = states[:, i])
        end
        monte_prob = EnsembleProblem(problem_ite, prob_func = prob_func)

        sim_results = solve(monte_prob, AutoTsit5(Rosenbrock23()),
            trajectories = n_particules, saveat = [t + dt_model / 1440],
            maxiters = 10e5, reltol = 10e-8, abstol = 10e-8)

        return hcat([max.(sim_results[i].u[1][1:5], 0.0) for i in 1:n_particules]...)
    end

    index_obs_var = ASPSimulator.get_indexes_from_symbols(observed_var, :asm1_simplified)
    H_matrix = zeros(5, 5)
    H_matrix[index_obs_var, index_obs_var] = Matrix(I, n_Y, n_Y)
    H_matrix = H_matrix[index_obs_var, :]
    @inline H_t(x, exogenous, params, t) = sparse(H_matrix * x)

    ϵ = 10e-6
    @inline R_t(exogenous, params, t) = PDiagMat(params[20:24] .^ 2) + ϵ .* I(5)
    @inline Q_t(exogenous, params, t) = PDiagMat(params[25:(25 + n_Y - 1)] .^ 2) +
                                        ϵ .* I(n_Y)

    n_X = 5

    return GaussianNonLinearStateSpaceSystem{Float64}(
        M_t, H_t, R_t, Q_t, n_X, n_Y, dt_model / 1440)
end

function get_linear_black_box_model(; dt_model::Int64 = 1, n_Y = 1, n_E = 2)
    @inline A_t(exogenous, params, t) = sparse(reshape(params[1:(n_Y * n_Y)], n_Y, n_Y))
    @inline B_t(exogenous, params, t) = sparse(reshape(
        params[(n_Y * n_Y + 1):(n_Y * n_Y + n_Y)], n_Y, 1))
    @inline c_t(exogenous, params, t) = sparse(reshape(
        params[(n_Y * n_Y + n_Y + 1):(n_Y * n_Y + n_Y + n_Y * n_E)], n_Y, n_E)) * exogenous

    @inline H_t(exogenous, params, t) = sparse(Matrix{Float64}(I, n_Y, n_Y))
    @inline d_t(exogenous, params, t) = sparse(zeros(n_Y))

    @inline R_t(exogenous, params, t) = PDiagMat((params[(n_Y * n_Y + n_Y * n_E + n_Y + 1):(n_Y * n_Y + 2 * n_Y + n_Y * n_E)]) .^
                                                 2)
    @inline Q_t(exogenous, params, t) = PDiagMat((params[(n_Y * n_Y + 2 * n_Y + n_Y * n_E + 1):(n_Y * n_Y + 3 * n_Y + n_Y * n_E)]) .^
                                                 2)

    return GaussianLinearStateSpaceSystem{Float64}(
        A_t, B_t, c_t, H_t, d_t, R_t, Q_t, n_Y, n_Y, dt_model / 1440)
end

function get_nonparametric_black_box_model(; dt_model::Int64 = 1, n_Y = 1, kwargs...)
    function M_t(x, x_scaled, exogenous, u, params, llrs, t)
        n_particules = size(x, 2)
        ind = hcat([x_scaled', repeat(hcat(exogenous)', n_particules)]...)'

        if u... > 0.5
            return x + llrs[1](ind, t)
        else
            return x + llrs[2](ind, t)
        end
    end

    function callback_llrs(llrs, idx, t_idx, X, E, U, μ, σ)
        _, n_particules, n_X = size(X)
    
        knn_data = transpose(hcat([
            StateSpaceIdentification._scale(reshape(X[1:(end - 1), :, :], (:, n_X))', μ, σ)',
            E[idx, :]]...))
        succesors_data = reshape(X[2:end, :, :] - X[1:(end - 1), :, :], (:, n_X))
        ind_aeration_on = (U[1:(end - 1), :] .> 0.5)
    
        knn_data_aeration_on = knn_data[:, repeat(ind_aeration_on[:, 1], n_particules)]
        knn_data_aeration_off = knn_data[:, repeat(.!ind_aeration_on[:, 1], n_particules)]
    
        succesors_data_aeration_on = reshape(
            succesors_data[repeat(ind_aeration_on[:, 1], n_particules), :], (:, n_X))
        succesors_data_aeration_off = reshape(
            succesors_data[repeat(.!ind_aeration_on[:, 1], n_particules), :], (:, n_X))
    
        llrs[1].analog_times_in_days = t_idx[repeat(ind_aeration_on[:, 1], n_particules)[idx]]
        llrs[1].analog_inputs = knn_data_aeration_on
        llrs[1].analog_outputs = succesors_data_aeration_on'
        llrs[1].neighbor_tree = KDTree(knn_data_aeration_on)
    
        llrs[2].analog_times_in_days = t_idx[repeat(.!ind_aeration_on[:, 1], n_particules)[idx]]
        llrs[2].analog_inputs = knn_data_aeration_off
        llrs[2].analog_outputs = succesors_data_aeration_off'
        llrs[2].neighbor_tree = KDTree(knn_data_aeration_off)
    end

    ϵ = 10e-6
    @inline H_t(x, exogenous, params, t) = x
    @inline R_t(exogenous, params, t) = PDiagMat(params[1:(n_Y)] .^ 2) + ϵ .* I(n_Y)
    @inline Q_t(exogenous, params, t) = PDiagMat(params[(n_Y + 1):(2 * n_Y)] .^ 2) +
                                        ϵ .* I(n_Y)

    # Define the system
    return GaussianNonParametricStateSpaceSystem{Float64}(
        M_t, H_t, R_t, Q_t, n_Y, n_Y, dt_model / (1440),
        kwargs[:llrs], μ = kwargs[:μ], σ = kwargs[:σ]),
        callback_llrs
end