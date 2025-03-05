using ASPSimulator
using Random
using Dates
using Distributions
using JLD2
using Match
using StateSpaceIdentification
using PyCall
using Loess

abstract type AbstractDataset end

mutable struct S1Dataset <: AbstractDataset
    name::Any
    seed::Any
    init_state::Any
    init_t::Any
    obs_var::Any
    dt_obs::Any
    σ_ϵ::Any

    T_training::Any
    T_testing::Any

    t::Any
    x::Any
    y::Any
    E::Any
    U::Any
    dt_signals::Any

    function S1Dataset(; seed = 1234)

        # Parameter of the S1 Dataset
        name = "S1"
        T_training = 5.0
        T_testing = 14.0
        T_steady_state = 20.0
        obs_var = [:nh4]
        dt_obs = 5
        σ_ϵ = 0.2

        # Generate data from the ASM1 Simulation model
        init_state, t, x, y, U, E = generate_ASM1_dataset(
            obs_var, T_steady_state, T_training, T_testing, dt_obs,
            σ_ϵ, seed; variable_inlet_concentration = false)

        new(name, seed, init_state, T_steady_state, obs_var,
            dt_obs, σ_ϵ, T_training, T_testing, t, x, y, E, U, 1)
    end
end

mutable struct S2Dataset <: AbstractDataset
    name::Any
    seed::Any
    init_state::Any
    init_t::Any
    obs_var::Any
    dt_obs::Any
    σ_ϵ::Any

    T_training::Any
    T_testing::Any

    t::Any
    x::Any
    y::Any
    E::Any
    U::Any
    dt_signals::Any

    function S2Dataset(; seed = 1234)

        # Parameter of the S1 Dataset
        name = "S2"
        T_training = 5.0
        T_testing = 14.0
        T_steady_state = 20.0
        obs_var = [:nh4, :o2]
        dt_obs = 5
        σ_ϵ = 0.2

        # Generate data from the ASM1 Simulation model
        init_state, t, x, y, U, E = generate_ASM1_dataset(
            obs_var, T_steady_state, T_training, T_testing, dt_obs,
            σ_ϵ, seed; variable_inlet_concentration = false)

        new(name, seed, init_state, T_steady_state, obs_var,
            dt_obs, σ_ϵ, T_training, T_testing, t, x, y, E, U, 1)
    end
end

mutable struct AcigneDataset <: AbstractDataset
    name::Any
    seed::Any
    start_date::Any
    init_state::Any
    init_t::Any
    obs_var::Any
    dt_obs::Any

    T_training::Any
    T_testing::Any

    t::Any
    x::Any
    y::Any
    E::Any
    U::Any
    dt_signals::Any

    function AcigneDataset(; start_date = DateTime("12-04-2023", dateformat"d-m-y"),
            seed = 5, T_training = 7.0)

        # Parameter of the S1 Dataset
        name = "AcigneDataset"
        control_id = [369996]
        observation_ids = [1653]
        exogenous_ids = [257259] #[370028] #257259

        T_training = T_training
        T_testing = 14.0
        obs_var = [:nh4]
        dt_obs = 5

        # Generate data from the ASM1 Simulation model
        X, Y, E, U, t_index = generate_purecontrol_dataset(
            start_date, T_training, T_testing, dt_obs,
            control_id, observation_ids, exogenous_ids)

        init_state = Y[1, :]
        t = getfield.((t_index .- t_index[1]), :value) / (1_000 * 60 * 60 * 24)
        # E = hcat([E, cos.(t*(2*pi)), sin.(t*(2*pi))]...)

        new(name, seed, start_date, init_state, 0.0, obs_var,
            dt_obs, T_training, T_testing, t, X', Y, E, U, 1)
    end
end

function generate_ASM1_dataset(
        obs_var, T_steady_state, T_training, T_testing, dt_obs, σ_ϵ, seed; kwargs...)

    # Fixed parameters
    system = :asm1

    # Define system
    ASPSimulator.set_default_system(system)
    nb_var = ASPSimulator.get_number_variables(system)
    index_obs_var = ASPSimulator.get_indexes_from_symbols(obs_var, system)
    nb_obs_var = size(index_obs_var, 1)
    index_u = ASPSimulator.get_control_index(system)

    # Set up environment
    sim = ASPSimulator.ODECore(system; kwargs...)

    # Let the system evolve during T_steady_state days in order to be stable and get new X_init
    res_steady = ASPSimulator.multi_step!(
        sim, ASPSimulator.redox_control(system), Day(T_steady_state))
    x_init = sim.current_state

    # Let the system evolve during T_training + T_testing days to get training and testing data
    sol_asm1 = ASPSimulator.multi_step!(
        sim, ASPSimulator.redox_control(system), Hour((T_training + T_testing) * 24))

    # Postprocess results
    t = vcat([[T_steady_state], timestamp(sol_asm1)]...)[1:(end - 1)]
    x = hcat([x_init, values(sol_asm1)']...)[:, 1:(end - 1)]
    Random.seed!(seed)
    y = max.(
        (x[index_obs_var, :] + rand(Normal(0, σ_ϵ), (nb_obs_var, size(x, 2))) +
         vcat([reshape([(i - 1) % Int(dt_obs) == 0 ? 0 : NaN for i in 1:size(x, 2)],
                   (1, size(x, 2))) for i in 1:nb_obs_var]...))',
        0)
    U = reshape(x[index_u, :], (:, 1))

    # Get exogenous signals
    Q_in = sim.parameters.Q_in
    X_in = sim.parameters.X_in
    Q_in_t = [Q_in(T_steady_state + t / (24 * 60))
              for t in 0:Int((T_training + T_testing) * 1440 - 1)]
    if isa(X_in, Vector{Float64})
        X_in_t = [X_in for t in 0:Int((T_training + T_testing) * 1440 - 1)]
    else
        X_in_t = [X_in(T_steady_state + t / (24 * 60))
                  for t in 0:Int((T_training + T_testing) * 1440 - 1)]
    end
    E = hcat([Q_in_t, hcat(X_in_t...)']...)

    return x_init, t, x, y, U, E
end

function adapt_signals_to_model!(d::AbstractDataset, symbol_model::Symbol, dt_model)
    if d.dt_obs % dt_model != 0
        @error "dt_obs has to be a multiple of dt_model."
    end

    if symbol_model == :linear_black && typeof(d) == AcigneDataset
        d.E = hcat(d.E, ones(size(d.E, 1), 1))
    end

    d.dt_signals = dt_model
    d.y = d.y[1:dt_model:end, 1:end]
    d.U = mean(reshape(d.U, dt_model, :), dims = 1)'
    d.E = d.E[1:dt_model:end, 1:end]

    if typeof(d) ∈ [S1Dataset, S2Dataset]
        if symbol_model ∈ [:grey, :linear_grey]
            d.E = d.E[:, [1, 11]]
        elseif symbol_model ∈ [:white]
            d.E = hcat([
                d.E[:, 1:1], sum(d.E[:, [3, 5]], dims = 2), d.E[:, [9, 10, 11, 12]]]...)
        elseif symbol_model ∈ [:nonparametric_black, :linear_black]
            ASPSimulator.set_default_system(:asm1)
            d.E = d.E[:,
                vec(vcat([[1], ASPSimulator.get_indexes_from_symbols(d.obs_var) .+ 1]...))]
            if symbol_model == :nonparametric_black
                # Scale data : TODO maybe adapt when the inlet concentration is not constant.
                d.E[:, 1:1] = StateSpaceIdentification.standard_scaler(d.E[:, 1:1])
                d.E[:, 2:end] = StateSpaceIdentification.standard_scaler(
                    d.E[:, 2:end], with_std = false)
            end
        end
    elseif typeof(d) ∈ [AcigneDataset] &&
           symbol_model ∈ [:linear_black, :nonparametric_black]
        if symbol_model == :nonparametric_black
            d.E = StateSpaceIdentification.standard_scaler(d.E)
        end
    else
        @error "No method defined to adapat dataset to model."
    end
end

function train_test_split(d::AbstractDataset)
    x_train = d.x[:, 1:Int(d.T_training * 1440)]
    x_test = d.x[:, Int(d.T_training * 1440 + 1):end]

    y_train = d.y[1:Int(d.T_training * 1440 / d.dt_signals), :]
    y_test = d.y[Int(d.T_training * 1440 / d.dt_signals + 1):end, :]

    U_train = d.U[1:Int(d.T_training * 1440 / d.dt_signals), :]
    U_test = d.U[Int(d.T_training * 1440 / d.dt_signals + 1):end, :]

    E_train = d.E[1:Int(d.T_training * 1440 / d.dt_signals), :]
    E_test = d.E[Int(d.T_training * 1440 / d.dt_signals + 1):end, :]

    return (x_train, y_train, U_train, E_train), (x_test, y_test, U_test, E_test)
end

function get_dataset(name_dataset::String; kwargs...)
    @match name_dataset begin
        "S1" => @match_return S1Dataset(; kwargs...)
        "S2" => @match_return S2Dataset(; kwargs...)
        "AcigneDataset" => @match_return AcigneDataset(; kwargs...)
        _ => @error "Non available dataset."
    end
end

function save(dataset::AbstractDataset; overwrite = false)
    prefix = dataset.name
    seed = dataset.seed

    d_name = savename((@dict seed), "jld2")
    mkpath(datadir("dataset", prefix))

    path_dataset = datadir("dataset", prefix, d_name)
    if ispath(path_dataset)
        if overwrite
            @warn "Dataset with path=$path_dataset already exist. As specified, overwrite current dataset."
            jldsave(path_dataset; dataset)
        else
            @warn "Dataset with path=$path_dataset already exist. already exist. Canceling the saving of the dataset."
        end
    else
        jldsave(path_dataset; dataset)
    end
end

function load_dataset(d_type::Type{<:AbstractDataset}; kwargs...)

    # Convert kwargs to specific name
    f_name = savename(kwargs) * ".jld2"

    # Set prefix according to type
    prefix = ""
    if d_type == S1Dataset
        prefix = "S1"
    elseif d_type == S2Dataset
        prefix = "S2"
    elseif d_type == AcigneDataset
        prefix = "AcigneDataset"
    end

    # Load data
    @info "Load the $prefix dataset with $(savename(kwargs))"
    filename = datadir("dataset", prefix, f_name)
    try
        f = jldopen(filename, "r")
        data = f["dataset"]
        close(f)
        return data
    catch e
        if isa(e, SystemError)
            @error "File $filename doesn't exist."
        end
    end
end

function load_dataset(d_name::String; kwargs...)

    # Convert kwargs to specific name
    f_name = savename(kwargs) * ".jld2"

    # Load data
    @info "Load the $d_name dataset with $(savename(kwargs))"
    filename = datadir("dataset", d_name, f_name)
    try
        f = jldopen(filename, "r")
        data = f["dataset"]
        close(f)
        return data
    catch e
        if isa(e, SystemError)
            @error "File $filename doesn't exist."
        end
    end
end

function adapt_x_test_model(name::Symbol, x_test, observed_var, type_dataset)
    idx_observed = convert(
        Array{Int64}, vcat(ASPSimulator.get_indexes_from_symbols.(observed_var)...))
    if type_dataset ∈ [S1Dataset, S2Dataset]
        @match name begin
            :white => @match_return vcat([
                sum(x_test[[2, 4], :], dims = 1), x_test[[8, 9, 10, 11], :]]...)
            :grey => @match_return x_test[10:10, :]
            :linear_black => @match_return x_test[idx_observed, :]
            :nonparametric_black => @match_return x_test[idx_observed, :]
            :linear_grey => @match_return x_test[10:10, :]
            _ => throw(ArgumentError("Non available symbol. Model has to be in $available_model ."))
        end
    elseif type_dataset ∈ [AcigneDataset]
        return x_test
    else
        @error "Not implemented"
    end
end