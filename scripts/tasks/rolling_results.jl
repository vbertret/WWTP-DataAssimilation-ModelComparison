using DrWatson
@quickactivate "ASP_model_comparison"
using DataFrames, Statistics
using StateSpaceIdentification
using Plots
using StatsPlots
using LaTeXStrings
using Measures
using StateSpaceIdentification
using NearestNeighbors

function Base.convert(LLR, input)
    LLR(input.index_analogs, input.analogs, input.successors, input.tree, input.ignored_nodes, input.k, input.lag_x, input.kernel)
end
# JLD2.load(datadir("rolling_evaluation", "nonparametric_black", "name_dataset=S1_seed=1.jld2"))

include(srcdir("dataset.jl"))
include(srcdir("model.jl"))


# Comparison for S1
df = collect_results(datadir("rolling_evaluation"), subfolders=true)
df = df[in.(df.name_dataset, Ref(["AcigneDataset"])),:]
df = df[in.(df.name_dataset, Ref(["S1"])),:]
df = df[.∈(df.model, Ref([:linear_grey])),:]
df = df[.∉(df.model, Ref([:linear_grey])),:]

# Comparison between S1, S2
df = collect_results(datadir("rolling_evaluation"), subfolders=true)
df = df[.∈(df.model, Ref([:nonparametric_black])),:]

df = collect_results(datadir("rolling_evaluation_obs"), subfolders=true)

# Comparison on AcigneDataset
df = collect_results(datadir("rolling_evaluation_acigne", "50epoch_20min"), subfolders=true)
# df = df[in.(df.name_dataset, Ref(["AcigneDataset"])),:]

# df1 = collect_results(datadir("rolling_evaluation_test", "grey"), subfolders=true)
# df1 = df1[in.(df1.name_dataset, Ref(["S1"])),:]

# df2 = collect_results(datadir("rolling_evaluation_test", "linear_black"), subfolders=true)
# df2 = df2[in.(df2.name_dataset, Ref(["S1"])),:]

# df3 = collect_results(datadir("rolling_evaluation_test", "linear_grey"), subfolders=true)
# df3 = df3[in.(df3.name_dataset, Ref(["S1"])),:]

# df4 = collect_results(datadir("rolling_evaluation_test", "nonparametric_black2"), subfolders=true)
# df4 = df4[in.(df4.name_dataset, Ref(["S1"])),:]
# df4.name_dataset = df4.name_dataset .* "_15"
# select!(df4, Not([:llrs, :μ, :σ]))

# df5 = collect_results(datadir("rolling_evaluation_test", "nonparametric_black_50"), subfolders=true)
# df5 = df5[in.(df5.name_dataset, Ref(["S1"])),:]
# df5.name_dataset = df5.name_dataset .* "_50"
# select!(df5, Not([:llrs, :μ, :σ]))

# df6 = collect_results(datadir("rolling_evaluation_test", "white"), subfolders=true)
# df6 = df6[in.(df6.name_dataset, Ref(["S1"])),:]

# df7 = collect_results(datadir("rolling_evaluation_test", "white_15"), subfolders=true)
# df7 = df7[in.(df7.name_dataset, Ref(["S1"])),:]
# df7.name_dataset = df7.name_dataset .* "_15"

# df1 = collect_results(datadir("test_CPF"), subfolders=true)
# df1.name_dataset = df1.name_dataset .* "_CPF_5"
# df3 = collect_results(datadir("test_CPF_15"), subfolders=true)
# df3.name_dataset = df3.name_dataset .* "_CPF_15"
# df4 = collect_results(datadir("test_CPF_30"), subfolders=true)
# df4.name_dataset = df4.name_dataset .* "_CPF_30"
# df5 = collect_results(datadir("test_CPF_100"), subfolders=true)
# df5.name_dataset = df5.name_dataset .* "_CPF_100"
# df6 = collect_results(datadir("test_CPF_10_100"), subfolders=true)
# df6.name_dataset = df6.name_dataset .* "_CPF_10_100"
# df7 = collect_results(datadir("test_150_iter"), subfolders=true)
# df7.name_dataset = df7.name_dataset .* "_CPF_150_iter"
# df2 = collect_results(datadir("rolling_evaluation"), subfolders=true)
# df2 = df2[in.(df2.name_dataset, Ref(["S1"])),:]
# df2 = df2[in.(df2.model, Ref([:linear_black])),:]
# df2.name_dataset = df2.name_dataset .* "_KF"
# select!(df2, Not([:llrs, :μ, :σ]))
# select!(df1, Not([:llrs, :μ, :σ]))
# res = res[in.(res.name_dataset, Ref(["S1"])),:]
# res = res[in.(res.model, Ref([:linear_grey, :linear_black])),:]
# res = res[in.(res.seed, Ref([1, 2])),:]
#test = res[.!ismissing.(res.rolling_ic),:]

# df = vcat(df1, df2, df3, df4, df5, df6, df7)

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


function get_prediction(tasks_parameters; H=288*2, lag=1, name_directory=nothing, training=true, model=nothing, n_particles=1_000)

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

    if name_directory == datadir("training_acigne", "last_one")
        E_test .= E_test./100
        E_train .= E_train./100
    end

    
    # Get training_results
    sname = savename((@dict name_dataset seed))
    res = collect_results(name_directory; rinclude = [Regex(".*($sname)[.|_].*")])

    if isnothing(model)
        # Build model
        if symbol_model == :nonparametric_black
            # Fix compatibility problems with NearestNeighbors
            res.llrs[1][1].tree = KDTree(res.llrs[1][1].analogs)
            res.llrs[1][2].tree = KDTree(res.llrs[1][2].analogs)

            model, _ = get_model(symbol_model, dataset.init_t, dataset.init_state, typeof(dataset); dt_model=dt_model, opt_params_linear_black_box_model = res.opt_parameters[1], llrs = res.llrs[1], μ=res.μ[1], σ=res.σ[1], observed_var=dataset.obs_var)
        else
            model = get_model(symbol_model, dataset.init_t, dataset.init_state, typeof(dataset); dt_model=dt_model, observed_var=dataset.obs_var)
        end
        
        # Set optimal parameters
        model.parameters = res.opt_parameters[1]
        # if name_dataset == "AcigneDataset"
        #     model.system.llrs[1].lag_time = 60*2/1440
        #     model.system.llrs[2].lag_time = 60*2/1440
        # end

        if training==false
            return model
        end
        # model.parameters[2] = 0.01
        # model.system.llrs[1].k = 500
        # model.system.llrs[2].k = 500

        y_train_nan = similar(y_train)
        y_train_nan .= NaN
        @info "Update model with training set."
        filter_output_train = StateSpaceIdentification.update!(model, y_train, E_train, U_train, positive=true, n_particles=300) 
    end

    new_model, _ = StateSpaceIdentification.update(model, y_test[1:lag, :], E_test[1:lag, :], U_test[1:lag, :], positive=true, n_particles=300) 

    # model.parameters[1] = 0.0005 

    y_test_nan = similar(y_test)
    y_test_nan .= NaN
    # if symbol_model == :linear_black
    #     n_particles = 10_000
    # else
    #     n_particles = n_particles
    # end
    # filter_output_test = StateSpaceIdentification.filter(new_model, y_test_nan[(lag+1):end, :], E_test[(lag+1):end, :], U_test[(lag+1):end, :], filter=ParticleFilter(new_model, positive=true, n_particles=n_particles)) 
    filter_output_test = StateSpaceIdentification.filter(new_model, y_test_nan[(lag+1):end, :], E_test[(lag+1):end, :], U_test[(lag+1):end, :], n_particles=n_particles, positive=true) 
    
    # filter_output_test = StateSpaceIdentification.filter(model, y_test_nan, E_test, U_test, filter=ParticleFilter(model, positive=true, n_particles=300)) 


    output_dict = Dict([i => j[1] for (i,j) in zip(names(res), eachcol(res))])
    output_dict["filter_output_test"] = filter_output_test
    if symbol_model ∈ [:linear_black, :linear_grey]
        # output_dict["filtered_state_train"] = filter_output_train.filtered_state
        output_dict["filtered_state_test"] = filter_output_test.filtered_state
    else
        # output_dict["filtered_state_train"] = filter_output_train.filtered_particles_swarm
        output_dict["filtered_state_test"] = filter_output_test.filtered_particles_swarm
    end
    output_dict["y_train"] = y_train
    output_dict["y_test"] = y_test

    return output_dict, output_dict["filtered_state_test"], x_test, U_test, E_test, model, lag

end

#####################################################################
############################ RMSE ###################################
#####################################################################
n_c = [Symbol("H$i") for i in 1:24]

sub_df = df
# sub_df = df[in.(df.name_dataset, Ref(["S2"])),:]

transform!(sub_df, [:model, :rolling_rmse, :obs_var] => ByRow(get_nh4_metrics) => :rolling_rmse_nh4)
transform!(sub_df, [:model, :rolling_cp, :obs_var] => ByRow(get_nh4_metrics) => :rolling_cp_nh4)
transform!(sub_df, [:model, :rolling_aw, :obs_var] => ByRow(get_nh4_metrics) => :rolling_aw_nh4)

rmse_df = combine(groupby(sub_df, [:model, :name_dataset]), :rolling_rmse_nh4 => (x-> cat_horizon_op(x, y-> mean(y, dims=1)[1])) => AsTable)
q_low_df = combine(groupby(sub_df, [:model, :name_dataset]), :rolling_rmse_nh4 => (x-> cat_horizon_op(x, y-> custom_quantile(y, 0.025))) => AsTable)
q_high_df = combine(groupby(sub_df, [:model, :name_dataset]), :rolling_rmse_nh4 => (x-> cat_horizon_op(x, y-> custom_quantile(y, 0.975))) => AsTable)

μ_rmse_values = Matrix(rmse_df[!, n_c])'
q_low_rmse_values = Matrix(q_low_df[!, n_c])'
q_high_rmse_values = Matrix(q_high_df[!, n_c])'
rmse_label = hcat(join.(eachrow(rmse_df[!, Not(n_c)]), "_")...)

Plots.backend(:gr)

plot_font = "Computer Modern"
font_size = 18#11

default(fontfamily=plot_font, 
        guidefontsize=font_size,
        tickfontsize = font_size, 
        legendfontsize = font_size - 2,
        linewidth=1.1, 
        framestyle=:box, 
        label=nothing, 
        grid=false
)

rmse_label = ["Grey-box" "Linear-black" "" ""]# "Nonparametric-black" "White-box"]

plot(μ_rmse_values, yaxis=:log, 
                    ribbon =(μ_rmse_values - q_low_rmse_values, q_high_rmse_values - μ_rmse_values), 
                    label = rmse_label, 
                    fillalpha=.2, 
                    linestyle = [:solid :dash :dot :dashdotdot]
)
fig1 = plot!(xlabel="Prediction horizon", ylabel="log10(RMSE)", legend_columns=2, legend=(0.28, 0.78))#,legend=:outerbottom)#, legend_columns=3)
safesave(savefig(plot!(size=(600, 400)), plotsdir("paper", "rmse_S1.pdf")))
# safesave(savefig(plotsdir("CPF_vs_KF", "rmse_global.html")))
# safesave(savefig(plotsdir("CPF_vs_KF", "rmse_global.png")))
# safesave(savefig(plotsdir("rmse_more_particles.png")))

plot(μ_rmse_values
)
fig1 = plot!(xlabel="Prediction horizon", ylabel="RMSE")


###################################################################
############################ AW ###################################
###################################################################

aw_df = combine(groupby(sub_df, [:model, :name_dataset]), :rolling_aw_nh4 => (x-> cat_horizon_op(x, y-> mean(y, dims=1)[1])) => AsTable)
q_low_aw_df = combine(groupby(sub_df, [:model, :name_dataset]), :rolling_aw_nh4 => (x-> cat_horizon_op(x, y-> custom_quantile(y, 0.025))) => AsTable)
q_high_aw_df = combine(groupby(sub_df, [:model, :name_dataset]), :rolling_aw_nh4 => (x-> cat_horizon_op(x, y-> custom_quantile(y, 0.975))) => AsTable)

μ_aw_values = Matrix(aw_df[!, n_c])'
q_low_aw_values = Matrix(q_low_aw_df[!, n_c])'
q_high_aw_values = Matrix(q_high_aw_df[!, n_c])'
aw_label = hcat(join.(eachrow(aw_df[!, Not(n_c)]), "_")...)

aw_label = ["" "" "Nonparametric-black" ""]

plot(μ_aw_values, ribbon =(μ_aw_values - q_low_aw_values, q_high_aw_values - μ_aw_values), 
        label = aw_label, 
        fillalpha=.2,
        linestyle = [:solid :dash :dot :dashdotdot]
)
fig2 = plot!(xlabel="Prediction horizon", ylabel="Average Width", legend_columns=2, legend=(0.3, 0.48))#,legend=:outerbottom)#, legend_columns=3)
safesave(savefig(plot!(), plotsdir("paper", "aw_S1.pdf")))
# safesave(savefig(plotsdir("CPF_vs_KF", "aw_global.png")))
# safesave(savefig(plotsdir("aw_more_particles.png")))


#####################################################################
############################ CP ###################################
#####################################################################

cp_df = combine(groupby(sub_df, [:model, :name_dataset]), :rolling_cp_nh4 => (x-> cat_horizon_op(x, y-> mean(y, dims=1)[1])) => AsTable)
q_low_cp_df = combine(groupby(sub_df, [:model, :name_dataset]), :rolling_cp_nh4 => (x-> cat_horizon_op(x, y-> custom_quantile(y, 0.025))) => AsTable)
q_high_cp_df = combine(groupby(sub_df, [:model, :name_dataset]), :rolling_cp_nh4 => (x-> cat_horizon_op(x, y-> custom_quantile(y, 0.975))) => AsTable)

μ_cp_values = Matrix(cp_df[!, n_c])'
q_low_cp_values = Matrix(q_low_cp_df[!, n_c])'
q_high_cp_values = Matrix(q_high_cp_df[!, n_c])'
cp_label = hcat(join.(eachrow(cp_df[!, Not(n_c)]), "_")...)

cp_label = ["" "" "" "White-box"]


plot(μ_cp_values, linestyle = [:solid :dash :dot :dashdotdot], ribbon =(μ_cp_values - q_low_cp_values, q_high_cp_values - μ_cp_values), label = cp_label, fillalpha=.2)
fig3 = plot!(xlabel="Prediction Horizon", ylabel="Coverage Probability", legend_columns=2, legend=(0.3, 0.48))
safesave(savefig(plot!(), plotsdir("paper", "cp_S1.pdf")))
# safesave(savefig(plotsdir("CPF_vs_KF", "cp_global.html")))
# safesave(savefig(plotsdir("cp_more_particles.png")))


plot(fig1, fig2, fig3, layout=(1, 3), size=(1800, 400), subplot_padding=1mm, left_margin=12mm, right_margin=5mm, top_margin=5mm, bottom_margin=15mm)
safesave(savefig(plot!(), plotsdir("paper", "S1.pdf")))

########################################################################
###################### MAKE TABLE PARAMETERS ###########################
########################################################################

filtered_df = sub_df[sub_df.model .== :nonparametric_black, :]
filtered_df.training_time = filtered_df.training_time / 5 # car on a mis 100 itérations mais ca a large converger en 20
sub_df[sub_df.model .== :nonparametric_black, :training_time] = filtered_df.training_time
column1 = combine(groupby(sub_df, :model), :training_time => mean => Symbol("Training time"))

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

sub_df[!, :σ_ϵ] = extract_noise_obs.(sub_df.opt_parameters, sub_df.name_dataset)
sub_df[!, :σ_η] = extract_noise_model.(sub_df.opt_parameters, sub_df.model, sub_df.name_dataset)


# column2 = combine(groupby(df, :model), :σ_ϵ => (x -> abs(mean(x))) => Symbol("Observation Standard Deviation"))
# column3 = combine(groupby(df, :model), :σ_η => (x -> abs.(mean(x))) => Symbol(L"S_{NH}"*" Standard Deviation"))
column2 = combine(groupby(df, :name_dataset), :σ_ϵ => (x -> mean(x)) => Symbol("Observation Standard Deviation"))
column3 = combine(groupby(df, :name_dataset), :σ_η => (x -> mean(x)) => Symbol(L"S_{NH}"*" Standard Deviation"))


using Latexify

table_sol = innerjoin(column1, column2, column3, on=:model)
table_sol = table_sol[[1, 2, 3, 7], :]
table_sol.model = Symbol.(["Grey-box", "Linear-black", "Nonparametric-black", "White-box"])

copy_to_clipboard(true)
l = latexify(table_sol; env = :table, booktabs = true, latex = false, fmt="%.2e", adjustement = :c) |> print

###################################################################
####################### Table metrics S2 ##########################
###################################################################

using Latexify

metrics = DataFrame((name_dataset = ["S1", "S2"], rmse=μ_rmse_values[end, :], aw=μ_aw_values[end, :], cp=μ_cp_values[end, :]))

delete!(column2, [3])
delete!(column3, [3])
table_sol = innerjoin(column2, column3, metrics, on=:name_dataset)
table_sol.name_dataset = Symbol.([L"[Y_{NH}]", L"[Y_{NH}, Y_{O}]"])
rename!(table_sol,:name_dataset => "Observed species")
rename!(table_sol,:rmse => L"\mathcal{L}_{RMSE}(24)")
rename!(table_sol,:aw => L"\mathcal{L}_{AW}(24)")
rename!(table_sol,:cp => L"\mathcal{L}_{CP}(24)")

copy_to_clipboard(true)
l = latexify(table_sol; env = :table, booktabs = true, latex = false, fmt="%.2e", adjustement = :c) |> print



########################################################################
####################### Plots trajectories S1 ##########################
########################################################################
model_table = unique(sub_df.model)
model_label = ["Grey-box" "Linear-black" "Nonparametric-black" "White-box"]

fig_table = []
for (i, model) in enumerate(model_table)
    tasks_parameters = Dict("model" => model, "name_dataset" => "S1", "seed" => 1, "dt_model" => 5)
    output, prediction, x_test, U_test, _, _, lag = get_prediction(tasks_parameters, H=288, lag=1)
    t_time = map(x-> x.t, prediction)
    y_test = output["y_test"]
    if i ==1
        plt = plot(t_time, x_test[:nh4, :][((lag*5)+1):5:end], linestyle=:dashdot, label="True concentration", title=model_label[i])
        scatter!(t_time, y_test[(lag+1):end], label="Observations", markersize=1.0)    
    else
        plt = plot(t_time, x_test[:nh4, :][((lag*5)+1):5:end], linestyle=:dashdot, title=model_label[i])
        scatter!(t_time, y_test[(lag+1):end], markersize=1.0) 
    end
    plot!(t_time, U_test[(lag+1):end])
    if i ==2
        push!(fig_table, plot!(get_nh4_timeseries_from_model(model, prediction, [:nh4]), label=["prediction"]))
    else
        push!(fig_table, plot!(get_nh4_timeseries_from_model(model, prediction, [:nh4])))
    end
    if i > 2
        xlabel!("Time (in days)")
    end
    if i%2 == 1
        ylabel!(L"S_{NH}"*" (mg/L)")
    end
end

plot(fig_table..., layout=(2,2), size=(1200, 800), link=:all, subplot_padding=1mm, left_margin=5mm, bottom_margin=2mm)
safesave(savefig(plot!(), plotsdir("paper", "S1_prediction.pdf")))


########################################################################
####################### Plots trajectories S2 ##########################
########################################################################

model = :nonparametric_black
dataset_label = ["Ammonia", "Ammonia and Oxygen"]
dataset_table = unique(sub_df.name_dataset)
obs_var_table = unique(sub_df.obs_var)
fig_table = []
prediction_table = []
for (i, dataset) in enumerate(dataset_table)
    tasks_parameters = Dict("model" => model, "name_dataset" => dataset, "seed" => 1, "dt_model" => 5)
    output, prediction, x_test, U_test, _, _, lag = get_prediction(tasks_parameters, H=288, lag=1)
    push!(prediction_table, (output, prediction, x_test, U_test, lag))
    t_time = map(x-> x.t, prediction)
    y_test = output["y_test"]
    if i == 1
        plt = plot(t_time, x_test[:nh4, :][((lag*5)+1):5:end], label="True concentration", title=dataset_label[i])
        scatter!(t_time, y_test[(lag+1):end], label="Observations", markersize=1.0)    
        plot!(t_time, U_test[(lag+1):end])
        push!(fig_table, plot!(get_nh4_timeseries_from_model(model, prediction, obs_var_table[i]), label=["prediction"]))
    else
        plt = plot(t_time, x_test[:nh4, :][((lag*5)+1):5:end], title=dataset_label[i])
        scatter!(t_time, y_test[(lag+1):end], markersize=1.0)    
        plot!(t_time, U_test[(lag+1):end])
        push!(fig_table, plot!(get_nh4_timeseries_from_model(model, prediction, obs_var_table[i])))
    end
end

plot(fig_table..., layout=(1,2), size=(1200, 400), link=:all, subplot_padding=1mm, left_margin=5mm, bottom_margin=2mm)


lag = prediction_table[2][end]
x_test = prediction_table[2][3]
U_test = prediction_table[2][4]
t_time = map(x-> x.t, prediction_table[2][2])
y_test = prediction_table[2][1]["y_test"]
fig_1 = plot(t_time, x_test[:o2, :][((lag*5)+1):5:end], linestyle=:dashdot, label="True concentration", legend=:topleft)
scatter!(t_time, y_test[(lag+1):end, 2], label="Observations", markersize=1.0)
plot!(t_time, U_test[(lag+1):end])
plot!(prediction_table[2][2], index=[2])
xlabel!("Time (in days)")
ylabel!(L"S_{O2}"*" (mg/L)")
fig_2 = plot(t_time, x_test[:nh4, :][((lag*5)+1):5:end], linestyle=:dashdot)
scatter!(t_time, y_test[(lag+1):end, 1], markersize=1.0)
plot!(t_time, U_test[(lag+1):end])
plot!(prediction_table[2][2], index=[1], label=["prediction"])
xlabel!("Time (in days)")
ylabel!(L"S_{NH}"*" (mg/L)")
plot(fig_2, fig_1, layout=(1,2), size=(1200, 400), subplot_padding=1mm, left_margin=6mm, bottom_margin=7mm)
safesave(savefig(plot!(), plotsdir("paper", "S2_prediction.pdf")))

######################################################
####################### DEV ##########################
######################################################

# df_nonparam = res[in.(res.model, Ref([:nonparametric_black])),:]

# k_1_tab = []
# k_2_tab = []+
# for seed in df_nonparam.seed
#     tasks_parameters = Dict("model" => :nonparametric_black, "name_dataset" => "S1", "seed" => seed, "dt_model" => 5)
#     model = get_prediction(tasks_parameters)
#     push!(k_1_tab, model.system.llrs[1].k/size(model.system.llrs[1].analogs, 2))
#     push!(k_2_tab, model.system.llrs[2].k/size(model.system.llrs[2].analogs, 2))
# end
# plot(k_1_tab.*100)
# plot(k_2_tab.*100)

t_time = map(x-> x.t, output["filtered_state_train"])
plot(output["filtered_state_train"], label=["Prediction"])
plot!(t_time, x_train[:, 1:5:end]', label="True concentration")

plotlyjs()

########################## Linear black ##########################
begin
    try
        model
    catch UndefVarError 
        model = nothing
    end
    seed = 1
    H = 288
    lag = 1
    model_name = :linear_black
    name_directory = datadir("training_paper_real", "linear_black")
    tasks_parameters = Dict("model" => model_name, "name_dataset" => "AcigneDataset", "seed" => seed, "dt_model" => 5)
    if isnothing(name_directory)
        output, prediction, x_test, U_test, E_test, model, lag = get_prediction(tasks_parameters, H=H, lag=lag, model=model)
    else
        output, prediction, x_test, U_test, E_test, model, lag = get_prediction(tasks_parameters, H=H, lag=lag, model=model, name_directory=name_directory)
    end
    t_time = map(x-> x.t, prediction)
    if tasks_parameters["name_dataset"] == "S1"
        plt = plot(t_time, x_test[10, ((lag*5)+1):5:end], label="True concentration")
    else
        plt = plot(t_time, x_test[((lag*5)+1):5:end], label="True concentration")
    end
    plot!(t_time, U_test[(lag+1):end])
    plot!(prediction, label=["Test"], linestyle = :dash, ic=0.95, color=:green)
    for i in 0.5:0.05:0.95
        plt = plot!(prediction, linestyle = :dash, ic=i, color=:green)
    end
    display(plt)
end

########################## Nonparametric black ##########################
plotlyjs()
# model_nl = nothing
begin
    # fig_table = []
    lag_table = [1, 288*1.25, 288*2.5, 288*5.75]
    for j in 2:2#1:4
        begin
            # try
            #     model_nl
            # catch UndefVarError 
            #     model_nl = nothing
            # end
            # seed = 1
            # H = 288
            # lag = Int(lag_table[j])
            # n_particles = 50_000
            # model_name = :nonparametric_black
            # name_directory = datadir("training_paper_real", "nonparametric_black")
            # tasks_parameters = Dict("model" => model_name, "name_dataset" => "AcigneDataset", "seed" => seed, "dt_model" => 5)
            # if isnothing(name_directory)
            #     output, prediction, x_test, U_test, E_test, model_nl, lag = get_prediction(tasks_parameters, H=H, lag=lag, model=model_nl, n_particles=n_particles)#, name_directory=datadir("training_acigne", "lissage_lag4"))
            # else
            #     output, prediction, x_test, U_test, E_test, model_nl, lag = get_prediction(tasks_parameters, H=H, lag=lag, model=model_nl, name_directory=name_directory, n_particles=n_particles)
            # end

            obs_prediction = deepcopy(prediction)

            # for i in 1:obs_prediction.n_t
            #     obs_prediction[i].particles_state = max.(StateSpaceIdentification.observation(model_nl.system, prediction[i].particles_state, E_test[i,:], model_nl.parameters, prediction[i].t)  + rand(MvNormal(model_nl.system.R_t(E_test[i, :], model_nl.parameters, prediction[i].t)), prediction[i].n_particles), 0.001)
            # end


            t_time = map(x-> x.t, prediction)
            # t_date = d.start_date .+ Dates.Minute.(Int.(round.(t_time.*24*60)))
            if j==1
                plt = scatter(t_time, x_test[((lag*5)+1):5:end], label="Observed concentration", markersize=1.0)
            else
                plt = scatter(t_time, x_test[((lag*5)+1):5:end], markersize=1.0)
            end
            if j==4
                plot!(t_time, U_test[(lag+1):end], label="Surpressor's state")
            else
                plot!(t_time, U_test[(lag+1):end])
            end
            if j==2
                plot!(obs_prediction, label=["prediction"], ic=0.95)#, quantile_tab=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            else
                plot!(obs_prediction)#, quantile_tab=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            end
            for i in 0.5:0.05:0.95
                plt = plot!(obs_prediction, ic=i, color=:green)
            end
            if j>2
                xlabel!("Days since 12 april of 2023")
            end
            if j%2 == 1
                ylabel!(L"S_{NH}"*" (mg/L)")
            end
            # push!(fig_table, plot!())
            fig_table[2] =  plot!()
            # display(plt)
        end
    end

    plot(fig_table..., layout=(2, 2), size=(1200, 800), link=:y, subplot_padding=1mm, left_margin=5mm, bottom_margin=2mm)
end

# fig_table_copy = deepcopy(fig_table)

fig_table = deepcopy(fig_table_copy)

safesave(savefig(plot!(size=(1200, 800)), plotsdir("paper", "square_real.pdf")))






















# safesave(savefig(plot!(size=(1600,800)), plotsdir("acigne", "prob.html")))
# plot!(t_time, E_test[(lag+1):end])

t_time_train = map(x-> x.t, output["filtered_state_train"])
plt = plot(t_time_train, x_train[1:5:end], label="True concentration")
plot!(output["filtered_state_train"], label=["Pred"])

# predictions_tab = []
# x_test = 0.0
# for model_label in aw_label
#     tasks_parameters = Dict("model" => Symbol(model_label), "name_dataset" => "S2", "seed" => 1, "dt_model" => 5)
#     prediction, x_test = get_prediction(tasks_parameters)
#     push!(predictions_tab, prediction)
# end

t_time = map(x-> x.t, predictions_tab[1])
plt = plot(t_time, x_test[10, 1:5:end], label="True concentration")
for i in [3]
    plt = plot!(predictions_tab[i], label=[aw_label[3]], color=l_color[3])
end
display(plt)
savefig(plotsdir("white.png"))


# Make a mean over periods

mean(model.system.llrs[1].successors)
mean(model.system.llrs[2].successors)



using DataFrames
using Statistics
begin
    # Fonction pour assigner chaque temps à un bin
    function assign_bin(temps, bins=0:4:24)
        for (i, bin) in enumerate(bins)
            if temps < bin/24
                return bin
            end
        end
        return bins[end]
    end

    df1 = DataFrame()
    df2 = DataFrame()
    for seed in 1:50
        
        tasks_parameters = Dict("model" => :nonparametric_black, "name_dataset" => "S1", "seed" => seed, "dt_model" => 5)
        model = get_prediction(tasks_parameters, H=288, lag=1, training=false)
        df1_temp = DataFrame(t=model.system.llrs[1].index_analogs.%1, μ=model.system.llrs[1].successors[:, 1])
        df2_temp = DataFrame(t=model.system.llrs[2].index_analogs.%1, μ=model.system.llrs[2].successors[:, 1])
        df1 = vcat(df1, df1_temp)
        df2 = vcat(df2, df2_temp)

    end
    tasks_parameters = Dict("model" => :nonparametric_black, "name_dataset" => "AcigneDataset", "seed" => 1, "dt_model" => 5)
    model = get_prediction(tasks_parameters, H=288, lag=1, name_directory=datadir("training_paper_real", "nonparametric_black"), training=false)
    df3 = DataFrame(t=model.system.llrs[1].index_analogs.%1, μ=model.system.llrs[1].successors[:, 1])
    df4 = DataFrame(t=model.system.llrs[2].index_analogs.%1, μ=model.system.llrs[2].successors[:, 1])

    df1[!, :bin] = assign_bin.(df1.t)
    result1 = combine(groupby(df1, :bin), :μ => mean => :moyenne_aeration_simu)

    df2[!, :bin] = assign_bin.(df2.t)
    result2 = combine(groupby(df2, :bin), :μ => mean => :moyenne_nonaeration_simu)

    df3[!, :bin] = assign_bin.(df3.t)
    result3 = combine(groupby(df3, :bin), :μ => mean => :moyenne_aeration_real)

    df4[!, :bin] = assign_bin.(df4.t)
    result4 = combine(groupby(df4, :bin), :μ => mean => :moyenne_nonaeration_real)

    result = outerjoin(result1, result2, result3, result4, on=:bin)

    bin_labels = ["[0-4h]", "[4h-8h]", "[8h-12h]", "[12h-16h]", "[16h-20h]", "[20h-24h]"]
    # bin_labels = result.bin
    moyenne_simu = result.moyenne_nonaeration_simu./maximum(abs.(result.moyenne_nonaeration_simu))
    moyenne_real = result.moyenne_nonaeration_real./maximum(abs.(result.moyenne_nonaeration_real))
    moyenne = cat([moyenne_simu, moyenne_real]..., dims=2)
    using StatsPlots
    # Tracer les barres
    groupedbar(bin_labels, moyenne, bar_position = :dodge, bar_width=0.7, label=["Fixed inlet (Simulation)" "Variable inlet (Real installation)"], alpha=0.7)
    xlabel!("Hours")
    ylabel!("Scaled mean derivative \n(non aeration)")
end
safesave(savefig(plot!(), plotsdir("paper", "explanation_real.pdf")))


######
tasks_parameters = Dict("model" => :nonparametric_black, "name_dataset" => "AcigneDataset", "seed" => 1, "dt_model" => 5)
model = get_prediction(tasks_parameters, H=288, lag=1, name_directory=datadir("rolling_evaluation_acigne", "6h-knn"), training=false)
df_aeration = DataFrame(t=model.system.llrs[1].index_analogs.%1, successors=model.system.llrs[1].successors[:, 1], analogs_snh=model.system.llrs[1].analogs[1, :], analogs_qin=model.system.llrs[1].analogs[2, :])
df_nonaeration = DataFrame(t=model.system.llrs[2].index_analogs.%1, successors=model.system.llrs[2].successors[:, 1], analogs_snh=model.system.llrs[2].analogs[1, :], analogs_qin=model.system.llrs[2].analogs[2, :])

plotlyjs()
df_plot = df_aeration
df_plot = df_plot[8/24 .<= df_plot.t .< 9/24, :]
mean(df_plot.successors)
hover_text = ["Derivative : $(row.successors) <br> Time : $(row.t) <br> S_{NH}(t) : $(row.analogs_snh) <br> Q_{in}(t) : $(row.analogs_qin)" for row in eachrow(df_plot)]
scatter(df_plot.analogs_snh, df_plot.analogs_qin, hover = hover_text, zcolor = df_plot.successors, lab="")
xlabel!(L"S_{NH}(t)")
ylabel!(L"Q_{in}(t)")
title!(L"S_{NH}(t+1) - S_{NH}(t)")
safesave(savefig(plot!(size=(1600,800)), plotsdir("acigne", "explore_knn.html")))


@benchmark if t != 0 && lag_x != 0
    for i in 1:nb_point_tree
        if (t-lag_x <= llr.index_analogs[i] <= t+lag_x) || ((lag_time != 0) && ((llr.index_analogs[i]%1 <= t%1 - lag_time && llr.index_analogs[i]%1 >= max(t +lag_time, 1)%1) || (llr.index_analogs[i]%1 >= t%1 + lag_time && llr.index_analogs[i]%1 <= 1 + t - lag_time)))
            push!(llr.ignored_nodes, i)
        end
    end
end



llr.ignored_nodes = Set{Int64}()
begin
    if t != 0 && lag_x != 0
        indexes = llr.index_analogs
        ignored_nodes = llr.ignored_nodes
        t_mod = t % 1
        lag_time_mod = lag_time % 1
        t_minus_lag_x = t - lag_x
        t_plus_lag_x = t + lag_x
        t_minus_lag_time = t_mod - lag_time_mod
        t_plus_lag_time = t_mod + lag_time_mod
        max_t_plus_lag_time = max(t_plus_lag_time, 1) % 1
        t_minus_lag_time_mod = t_minus_lag_time % 1
        t_plus_lag_time_mod = t_plus_lag_time % 1

        for i in eachindex(indexes)
            index = indexes[i]
            index_mod = index % 1
            if (t_minus_lag_x <= index <= t_plus_lag_x) ||
               ((lag_time != 0) &&
                ((index_mod <= t_minus_lag_time_mod && index_mod >= max_t_plus_lag_time) ||
                 (index_mod >= t_plus_lag_time_mod && index_mod <= 1 + t_minus_lag_time_mod)))
                push!(ignored_nodes, i)
            end
        end
    end
end

plotlyjs()
llr = model.system.llrs[1]
nb_point_tree = size(llr.tree.data)[1]
t=0.1
lag_x=0.05
lag_time = 0.2

@timed begin
    # llr.ignored_nodes = Set{Int64}()
    if t != 0 && lag_x != 0
        indexes = llr.index_analogs
        n = length(indexes)
        ignored_nodes = BitVector(undef, n)  # Pré-allocations d'un vecteur booléen
        t_mod = t % 1

        # Condition 1: Vérifier les index dans l'intervalle [t-lag_x, t+lag_x]
        in_range = (t - lag_x .<= indexes) .& (indexes .<= t + lag_x)

        # Condition 2: Vérifier les index modulo dans l'intervalle des temps
        mod_indexes = indexes .% 1
        out_of_mod_range = ((mod_indexes .<= t_mod - lag_time) .& (mod_indexes .>= max(t +lag_time, 1) % 1)) .| 
                            ((mod_indexes .>= t_mod + lag_time) .& (mod_indexes .<= 1 + t - lag_time))

        # Combiner les conditions
        ignored_nodes .= in_range .| (lag_time != 0 && out_of_mod_range)

        # Collecter les indices ignorés
        llr.ignored_nodes = Set(findall(ignored_nodes))
    end
end
temp = copy(llr.ignored_nodes)

@timed begin
    llr.ignored_nodes = Set{Int64}()
    if t != 0 && lag_x != 0
        indexes = llr.index_analogs
        ignored_nodes = llr.ignored_nodes
        for i in eachindex(indexes)
            index = indexes[i]
            if (t-lag_x <= index <= t+lag_x) || ((lag_time != 0) && ((index%1 <= t%1 - lag_time && index%1 >= max(t +lag_time, 1)%1) || (index%1 >= t%1 + lag_time && index%1 <= 1 + t - lag_time)))
                push!(ignored_nodes, i)
            end
        end
    end
end
llr.ignored_nodes

println("t=$(t) $(issetequal(llr.ignored_nodes, temp))")

llr.ignored_nodes = Set{Int64}()
if t != 0 && lag_x != 0
    for i in 1:nb_point_tree
        if (t-lag_x <= llr.index_analogs[i] <= t+lag_x)
            push!(llr.ignored_nodes, i)
        elseif (lag_time != 0) && (((t%1 <= llr.index_analogs[i]%1 - lag_time) && (t%1 >= (llr.index_analogs[i] + lag_time)%1)) || ((t%1 >= llr.index_analogs[i]%1 + lag_time) && (t%1 <= (1 + llr.index_analogs[i]%1 - lag_time))))
            push!(llr.ignored_nodes, i)
        end
    end
end
llr.ignored_nodes

t_ignored_nodes = zeros(nb_point_tree)
t_ignored_nodes[collect(llr.ignored_nodes)] .= 1
scatter(llr.index_analogs, t_ignored_nodes)


ignored_nodes = Set([])

nb_point_tree = size(llr.tree.data)[1]
llr = model.system.llrs[1]


sma
movmean(x_train)

@benchmark t_ignored_nodes = (t-lag_x .<= llr.index_analogs .<= t+lag_x) .|| ((lag_time != 0) .&& ((((llr.index_analogs .+ lag_time).%1 .<= t%1 .<= llr.index_analogs.%1 .- lag_time)) .|| ((llr.index_analogs.%1 .+ lag_time .<= t.%1 .<= (1 .+ llr.index_analogs.%1 .- lag_time)))))

using MarketTechnicals

tasks_parameters = Dict("model" => :linear_black, "name_dataset" => "AcigneDataset", "seed" => 1, "dt_model" => 5)

# Get arguments tasks_parameters
symbol_model = tasks_parameters["model"]
name_dataset = tasks_parameters["name_dataset"]
seed = tasks_parameters["seed"]
dt_model = tasks_parameters["dt_model"]

# Load and Split dataset
dataset = load_dataset(name_dataset, seed=seed)
adapt_signals_to_model!(dataset, symbol_model, dt_model)
(x_train, y_train, U_train, E_train), (x_test, y_test, U_test, E_test) = train_test_split(dataset)

plot(dataset.t[1:size(x_train, 2)], x_train')
plot!(dataset.t[1:5:size(x_train, 2)], U_train, markersize=0.5)
scatter!(dataset.t[1:5:size(x_train, 2)], y_train, markersize=0.5)