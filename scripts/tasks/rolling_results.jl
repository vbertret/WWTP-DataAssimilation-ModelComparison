using DrWatson
@quickactivate "ASP_model_comparison"

include(srcdir("utils.jl"))

# Fix Compatibility problems
import Base: convert
function Base.convert(::Type{StateSpaceIdentification.LocalLinearRegressor}, input)
    StateSpaceIdentification.LocalLinearRegressor(input.index_analogs, input.analogs, input.successors, KDTree(input.analogs), n_neighbors=input.k, min_lag_in_days=input.lag_x, kernel_type=input.kernel)
end

Plots.backend(:gr)
plot_font = "Computer Modern"
font_size = 18
default(fontfamily=plot_font, 
        guidefontsize=font_size,
        tickfontsize = font_size, 
        legendfontsize = font_size - 2,
        linewidth=1.1, 
        framestyle=:box, 
        label=nothing, 
        grid=false
)


# Comparison for S1
df = collect_results(datadir("rolling_evaluation"), subfolders=true)

transform!(df, [:model, :rolling_rmse, :obs_var] => ByRow(get_nh4_metrics) => :rolling_rmse_nh4)
transform!(df, [:model, :rolling_cp, :obs_var] => ByRow(get_nh4_metrics) => :rolling_cp_nh4)
transform!(df, [:model, :rolling_aw, :obs_var] => ByRow(get_nh4_metrics) => :rolling_aw_nh4)

s1_df = deepcopy(df)
s1_df = s1_df[in.(s1_df.name_dataset, Ref(["S1"])),:]

s1_s2_np_df = deepcopy(df)
s1_s2_np_df = s1_s2_np_df[.∈(s1_s2_np_df.model, Ref([:nonparametric_black])),:]
s1_s2_np_df = s1_s2_np_df[in.(s1_s2_np_df.name_dataset, Ref(["S1", "S2"])),:]

for i in 1:size(s1_s2_np_df)[1]
    if  ~(typeof(s1_s2_np_df.llrs[i][1]) <: LocalLinearRegressor)
        convert_new_interface(s1_s2_np_df.llrs[i])
    end
end

##########################################################################################################################################################################
################################################################################ FIGURE 3 ################################################################################
##########################################################################################################################################################################

#####################################################################
############################ RMSE - S1 ##############################
#####################################################################

aggregation_results = calculate_grouped_aggregations(s1_df, :rolling_rmse_nh4)

p31 = plot(
    aggregation_results.mean_values,
    ribbon=(aggregation_results.mean_values - aggregation_results.low_quantile_values, 
            aggregation_results.high_quantile_values - aggregation_results.mean_values),
    label=["Grey-box" "Linear-black" "" ""],
    fillalpha=0.2,
    yaxis=:log,
    linestyle = [:solid :dash :dot :dashdotdot],
    xlabel="Prediction horizon", ylabel="log10(RMSE)", legend=(0.28, 0.78), legend_columns=2
)
safesave(savefig(plot!(size=(600, 400)), plotsdir("paper", "figure3-1.pdf")))

###################################################################
############################ AW - S1 ##############################
###################################################################

aggregation_results = calculate_grouped_aggregations(s1_df, :rolling_aw_nh4)

p32 = plot(
    aggregation_results.mean_values,
    ribbon=(aggregation_results.mean_values - aggregation_results.low_quantile_values, 
            aggregation_results.high_quantile_values - aggregation_results.mean_values),
    label=["" "" "Nonparametric-black" ""],
    fillalpha=0.2,
    linestyle = [:solid :dash :dot :dashdotdot],
    xlabel="Prediction horizon", ylabel="Average Width", legend=(0.3, 0.48)
)
safesave(savefig(plot!(size=(600, 400)), plotsdir("paper", "figure3-2.pdf")))

#####################################################################
############################ CP - S1 ################################
#####################################################################

aggregation_results = calculate_grouped_aggregations(s1_df, :rolling_cp_nh4)

p33 = plot(
    aggregation_results.mean_values,
    ribbon=(aggregation_results.mean_values - aggregation_results.low_quantile_values, 
            aggregation_results.high_quantile_values - aggregation_results.mean_values),
    label=["" "" "" "White-box"],
    fillalpha=0.2,
    linestyle = [:solid :dash :dot :dashdotdot],
    xlabel="Prediction horizon", ylabel="Coverage Probability", legend=(0.3, 0.48)
)
safesave(savefig(plot!(size=(600, 400)), plotsdir("paper", "figure3-3.pdf")))


plot(p31, p32, p33, layout=(1, 3), size=(1800, 400), subplot_padding=1mm, left_margin=12mm, right_margin=5mm, top_margin=5mm, bottom_margin=15mm)
safesave(savefig(plot!(), plotsdir("paper", "figure3.pdf")))

##########################################################################################################################################################################
################################################################################ TABLE 2 ################################################################################
##########################################################################################################################################################################

c_tt = combine(groupby(s1_df, :model), :training_time => mean => Symbol("Training time"))

s1_df[!, :σ_ϵ] = extract_noise_obs.(s1_df.opt_parameters, s1_df.name_dataset)
c_σ_ϵ = combine(groupby(s1_df, :model), :σ_ϵ => (x -> abs.(mean(x))) => Symbol("Observation Standard Deviation"))

s1_df[!, :σ_η] = extract_noise_model.(s1_df.opt_parameters, s1_df.model, s1_df.name_dataset)
c_σ_η = combine(groupby(s1_df, :model), :σ_η => (x -> abs.(mean(x))) => Symbol(L"S_{NH}"*" Standard Deviation"))

table_2 = innerjoin(c_tt, c_σ_ϵ, c_σ_η, on=:model)
table_2 = table_2[[1, 2, 3, 7], :]
table_2.model = Symbol.(["Grey-box", "Linear-black", "Nonparametric-black", "White-box"])
copy_to_clipboard(true)
l = latexify(table_2; env = :table, booktabs = true, latex = false, fmt="%.2e", adjustement = :c) |> print

##########################################################################################################################################################################
################################################################################ FIGURE 4 ################################################################################
##########################################################################################################################################################################

model_table = unique(s1_df.model)
model_label = ["Grey-box" "Linear-black" "Nonparametric-black" "White-box"]

fig_table = []
for (i, model) in enumerate(model_table)
    tasks_parameters = Dict("model" => model, "name_dataset" => "S1", "seed" => 1, "dt_model" => 5)
    output, prediction, x_test, U_test, _, _, lag = get_prediction(tasks_parameters, H=288, lag=1)
    t_time = map(x-> x.t, prediction)
    y_test = output["y_test"]
    if i ==1
        plt = plot(t_time, x_test[ASPSimulator.get_indexes_from_symbols(:nh4), :][((lag*5)+1):5:end], linestyle=:dashdot, label="True concentration", title=model_label[i])
        scatter!(t_time, y_test[(lag+1):end], label="Observations", markersize=1.0)    
    else
        plt = plot(t_time, x_test[ASPSimulator.get_indexes_from_symbols(:nh4), :][((lag*5)+1):5:end], linestyle=:dashdot, title=model_label[i])
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
safesave(savefig(plot!(), plotsdir("paper", "figure4.pdf")))

##########################################################################################################################################################################
################################################################################ TABLE 3 ################################################################################
##########################################################################################################################################################################

s1_s2_np_df[!, :σ_ϵ] = extract_noise_obs.(s1_s2_np_df.opt_parameters, s1_s2_np_df.name_dataset)
c_σ_ϵ = combine(groupby(s1_s2_np_df, :name_dataset), :σ_ϵ => (x -> abs.(mean(x))) => Symbol("Observation Standard Deviation"))
delete!(c_σ_ϵ, [3])

s1_s2_np_df[!, :σ_η] = extract_noise_model.(s1_s2_np_df.opt_parameters, s1_s2_np_df.model, s1_s2_np_df.name_dataset)
c_σ_η = combine(groupby(s1_s2_np_df, :name_dataset), :σ_η => (x -> abs.(mean(x))) => Symbol(L"S_{NH}"*" Standard Deviation"))
delete!(c_σ_η, [3])

rmse_s1_s2 = mean(calculate_grouped_aggregations(s1_s2_np_df, :rolling_rmse_nh4).mean_values, dims=1)[1,:]
aw_s1_s2 = mean(calculate_grouped_aggregations(s1_s2_np_df, :rolling_aw_nh4).mean_values, dims=1)[1,:]
cp_s1_s2 = mean(calculate_grouped_aggregations(s1_s2_np_df, :rolling_cp_nh4).mean_values, dims=1)[1,:]
c_metrics = DataFrame((name_dataset = ["S1", "S2"], rmse=rmse_s1_s2, aw=aw_s1_s2, cp=cp_s1_s2))


table_3 = innerjoin(c_σ_ϵ, c_σ_η, c_metrics, on=:name_dataset)
table_3.name_dataset = Symbol.([L"[Y_{NH}]", L"[Y_{NH}, Y_{O}]"])
rename!(table_3,:name_dataset => "Observed species")
rename!(table_3,:rmse => L"\mathcal{L}_{RMSE}(1H, 24H)")
rename!(table_3,:aw => L"\mathcal{L}_{AW}(1H, 24H)")
rename!(table_3,:cp => L"\mathcal{L}_{CP}(1H, 24H)")

copy_to_clipboard(true)
l = latexify(table_3; env = :table, booktabs = true, latex = false, fmt="%.2e", adjustement = :c) |> print


##########################################################################################################################################################################
################################################################################ FIGURE 5 ################################################################################
##########################################################################################################################################################################

model = :nonparametric_black
dataset_label = ["Ammonia", "Ammonia and Oxygen"]
dataset_table = unique(s1_s2_np_df.name_dataset)
obs_var_table = unique(s1_s2_np_df.obs_var)
fig_table = []
prediction_table = []
for (i, dataset) in enumerate(dataset_table)
    tasks_parameters = Dict("model" => model, "name_dataset" => dataset, "seed" => 1, "dt_model" => 5)
    output, prediction, x_test, U_test, _, _, lag = get_prediction(tasks_parameters, H=288, lag=1)
    push!(prediction_table, (output, prediction, x_test, U_test, lag))
    t_time = map(x-> x.t, prediction)
    y_test = output["y_test"]
    if i == 1
        plt = plot(t_time, x_test[ASPSimulator.get_indexes_from_symbols(:nh4), :][((lag*5)+1):5:end], label="True concentration", title=dataset_label[i])
        scatter!(t_time, y_test[(lag+1):end], label="Observations", markersize=1.0)    
        plot!(t_time, U_test[(lag+1):end])
        push!(fig_table, plot!(get_nh4_timeseries_from_model(model, prediction, obs_var_table[i]), label=["prediction"]))
    else
        plt = plot(t_time, x_test[ASPSimulator.get_indexes_from_symbols(:nh4), :][((lag*5)+1):5:end], title=dataset_label[i])
        scatter!(t_time, y_test[(lag+1):end], markersize=1.0)    
        plot!(t_time, U_test[(lag+1):end])
        push!(fig_table, plot!(get_nh4_timeseries_from_model(model, prediction, obs_var_table[i])))
    end
end

# plot(fig_table..., layout=(1,2), size=(1200, 400), link=:all, subplot_padding=1mm, left_margin=5mm, bottom_margin=2mm)

lag = prediction_table[2][end]
x_test = prediction_table[2][3]
U_test = prediction_table[2][4]
t_time = map(x-> x.t, prediction_table[2][2])
y_test = prediction_table[2][1]["y_test"]
fig_51 = plot(t_time, x_test[ASPSimulator.get_indexes_from_symbols(:o2), :][((lag*5)+1):5:end], linestyle=:dashdot, label="True concentration", legend=:topleft)
scatter!(t_time, y_test[(lag+1):end, 2], label="Observations", markersize=1.0)
plot!(t_time, U_test[(lag+1):end])
plot!(prediction_table[2][2], index=[2])
xlabel!("Time (in days)")
ylabel!(L"S_{O2}"*" (mg/L)")
fig_52 = plot(t_time, x_test[ASPSimulator.get_indexes_from_symbols(:nh4), :][((lag*5)+1):5:end], linestyle=:dashdot)
scatter!(t_time, y_test[(lag+1):end, 1], markersize=1.0)
plot!(t_time, U_test[(lag+1):end])
plot!(prediction_table[2][2], index=[1], label=["prediction"])
xlabel!("Time (in days)")
ylabel!(L"S_{NH}"*" (mg/L)")
plot(fig_51, fig_52, layout=(1,2), size=(1200, 400), subplot_padding=1mm, left_margin=6mm, bottom_margin=7mm)
safesave(savefig(plot!(), plotsdir("paper", "figure5.pdf")))

##########################################################################################################################################################################
################################################################################ FIGURE 6 ################################################################################
##########################################################################################################################################################################

begin
    fig_table = []
    lag_table = [1, 288*1.25, 288*2.5, 288*5.75]
    for j in 1:4
        begin
            try
                model_nl
            catch UndefVarError 
                model_nl = nothing
            end
            seed = 1
            H = 288
            lag = Int(lag_table[j])
            n_particles = 50_000
            model_name = :nonparametric_black
            tasks_parameters = Dict("model" => model_name, "name_dataset" => "AcigneDataset", "seed" => seed, "dt_model" => 5)
            name_directory = nothing
            if isnothing(name_directory)
                output, prediction, x_test, U_test, E_test, model_nl, lag = get_prediction(tasks_parameters, H=H, lag=lag, model=model_nl, n_particles=n_particles)
            else
                output, prediction, x_test, U_test, E_test, model_nl, lag = get_prediction(tasks_parameters, H=H, lag=lag, model=model_nl, name_directory=name_directory, n_particles=n_particles)
            end

            obs_prediction = deepcopy(prediction)

            for i in 1:length(obs_prediction)
                obs_prediction[i].particles_state = max.(StateSpaceIdentification.observation(model_nl.system, prediction[i].particles_state, E_test[i,:], model_nl.parameters, prediction[i].t)  + rand(MvNormal(model_nl.system.Q_t(E_test[i, :], model_nl.parameters, prediction[i].t)), size(prediction[1],2)), 0.001)
            end

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
                plot!(obs_prediction, label=["prediction"], ic=0.95)
            else
                plot!(obs_prediction)
            end
            for i in 0.5:0.05:0.95
                plt = plot!(prediction, ic=i, color=:green)
            end
            if j>2
                xlabel!("Days since 12 april of 2023")
            end
            if j%2 == 1
                ylabel!(L"S_{NH}"*" (mg/L)")
            end
            push!(fig_table, plot!())
        end
    end

    plot(fig_table..., layout=(2, 2), size=(1200, 800), link=:y, subplot_padding=1mm, left_margin=5mm, bottom_margin=2mm)
end

safesave(savefig(plot!(size=(1200, 800)), plotsdir("paper", "figure6.pdf")))


##########################################################################################################################################################################
################################################################################ FIGURE 7 ################################################################################
##########################################################################################################################################################################

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
        model = get_prediction(tasks_parameters, training=false)
        df1_temp = DataFrame(t=model.system.llrs[1].analog_times_in_days.%1, μ=model.system.llrs[1].analog_outputs[1, :])
        df2_temp = DataFrame(t=model.system.llrs[2].analog_times_in_days.%1, μ=model.system.llrs[2].analog_outputs[1, :])
        df1 = vcat(df1, df1_temp)
        df2 = vcat(df2, df2_temp)

    end
    tasks_parameters = Dict("model" => :nonparametric_black, "name_dataset" => "AcigneDataset", "seed" => 1, "dt_model" => 5)
    model = get_prediction(tasks_parameters, H=288, lag=1, training=false)
    df3 = DataFrame(t=model.system.llrs[1].analog_times_in_days.%1, μ=model.system.llrs[1].analog_outputs[1, :])
    df4 = DataFrame(t=model.system.llrs[2].analog_times_in_days.%1, μ=model.system.llrs[2].analog_outputs[1, :])

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
    groupedbar(bin_labels, moyenne, bar_position = :dodge, size=(1000, 600), bar_width=0.7, label=["Fixed inlet (Simulation)" "Variable inlet (Real installation)"], alpha=0.7, bottom_margin=10mm, left_margin=5mm)
    xlabel!("Hours")
    ylabel!("Scaled mean derivative \n(non aeration)")
end
safesave(savefig(plot!(), plotsdir("paper", "figure7.pdf")))