using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using BenchmarkTools

include("modules.jl")

using .Samplers
import .GiererMeinhardt
import .GMStabilityAnalysis

using PyCall
pickle = pyimport("pickle")

function prior()
    sampler = filtered_sampler(wide_prior_sampler,
        GMStabilityAnalysis.is_homogeneous_state_unstable)
    parameters = sampler()
    return Dict("parameters" => parameters)
end

function initial_condition_sampler(parameters, domain_size)
    homogeneous_fix_point = GMStabilityAnalysis.homogeneous_state(parameters)
    initial_conditions = reshape(homogeneous_fix_point, (1, 1, 2)) .* (
        1 .+ 0.01 .* rand(domain_size, domain_size, length(homogeneous_fix_point))
    )
    return Dict("initial_conditions" => initial_conditions)
end

function simulate(batch_size; domain_size = 16, tspan = (0.0, 0.1))
    # create batch of prior and ic samples
    parameter_batch = Matrix{Float64}(undef, 5, batch_size)
    initial_condition_batch = Array{Float64}(undef, domain_size, domain_size, 2, batch_size)

    for i in 1:batch_size
        parameters = prior()["parameters"]
        parameter_batch[:, i] = parameters
        initial_condition_batch[:, :, :, i] = initial_condition_sampler(
            parameters, domain_size)["initial_conditions"]
    end

    # pass batches to solve as all PDEs as a parallel ensemble
    final_state_batch = solve_final_states_ensemble(
        parameter_batch, initial_condition_batch, tspan, GiererMeinhardt.initialize_problem)

    return Dict(
        "parameters" => permutedims(parameter_batch, (2, 1)),
        "initial_conditions" => permutedims(initial_condition_batch, (4, 3, 2, 1)),
        "final_states" => permutedims(final_state_batch, (4, 3, 2, 1))
    )
end

domain_size = 64
tspan = (0, 1500)
num_simulations = 4

path = "data/GM-001"
name = "GM-001-019-test-$(domain_size)"
full_save_path = "$path" * "/$name.pkl"

if !ispath(path)
    throw("Directory for output $path does not exist.")
end
if isfile(full_save_path)
    throw("Selected output path:$full_save_path already exists!")
else
    println("Will save to $full_save_path.")
end

# @btime test_sims = simulate(10; domain_size=64, tspan=(0, 1000))
# test_sims = simulate(2; domain_size=domain_size, tspan=(0, 1))
# println("Simulating $(num_simulations) patterns for randomly sampled parameters and initial conditions...")
@time sims = simulate(num_simulations; domain_size = domain_size, tspan = tspan)

###########
# patterns = sims["final_state_batch"][:,:,1,:]
# using Plots
# for i in 1:num_simulations
#     plot_container = heatmap(patterns[:,:,i], size=(300,300), dpi=400, right_margin = 1Plots.mm, aspect_ratio=:equal, c=:seaborn_icefire_gradient, framestyle = :none)
#     savefig(plot_container, "figures/final_state_batch_$(lpad(i,3,"0")).png")
# end
###########

file = open(full_save_path, "w")
pickle.dump(sims, file)
close(file)
