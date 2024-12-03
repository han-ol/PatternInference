#######################################

module GiererMeinhardt
export kernel!, initialize_problem

using LinearAlgebra
using DifferentialEquations
using BenchmarkTools

function create_MxMy_periodic_bc_laplacian(domain_size)
    Mx = Matrix(
        Tridiagonal([1.0 for i in 1:(domain_size - 1)], [-2.0 for i in 1:domain_size],
        [1.0 for i in 1:(domain_size - 1)])
    )
    Mx[end, 1] = 1.0
    Mx[1, end] = 1.0

    My = copy(transpose(Mx))

    return Mx, My
end

function laplacian!(du, u, D, Mx, My)
    mul!(du, Mx, u, D, 1)
    mul!(du, u, My, D, 1)
end

function kernel!(du, u, p, t)
    parameters, Mx, My = p
    a, b, c, δ, s = parameters

    du .= 0
    u1 = @view u[:, :, 1]
    du1 = @view du[:, :, 1]

    u2 = @view u[:, :, 2]
    du2 = @view du[:, :, 2]

    # add diffusion gradient
    laplacian!(du1, u1, s * 1, Mx, My)
    laplacian!(du2, u2, s * δ, Mx, My)

    # add reaction gradient
    @. du1 += a - b * u1 + u1^2 / (u2 * (1 + c * u1^2))
    @. du2 += u1^2 - u2
end

function benchmark_kernel(parameters, initial_conditions)
    domain_size = size(initial_conditions, 1)
    Mx, My = create_MxMy_periodic_bc_laplacian(domain_size)
    p = (parameters, Mx, My)
    u = copy(initial_conditions)
    du = copy(initial_conditions)
    du .= 0
    @btime kernel!($du, $u, $p, 0)
end

function initialize_problem(parameters, initial_conditions, tspan)
    # println("GiererMeinhardt.kernel! benchmark:")
    # benchmark_kernel(parameters, initial_conditions)  # -> 0 allocations !
    domain_size = size(initial_conditions, 1)
    Mx, My = create_MxMy_periodic_bc_laplacian(domain_size)
    p = (parameters, Mx, My)
    prob = ODEProblem(kernel!, initial_conditions, tspan, p)
end

end  # end of module GiererMeinhardt ##
#######################################

#######################################

module Samplers

export wide_prior_sampler, uniform_initial_condition_sampler,
       repeated_uniform_initial_condition_sampler, filtered_sampler, solve_final_state,
       solve_final_states_ensemble
using Random
using Distributions
using ProgressBars
using DifferentialEquations
using Sundials

function wide_prior_sampler(; rng = Xoshiro())
    marginal_distributions = [
        Uniform(0.01, 0.7),
        Uniform(0.4, 2),
        Uniform(0.02, 7),
        Uniform(20, 200),
        Dirac(0.5)
    ]
    parameters = Vector{Float64}(undef, length(marginal_distributions))

    for i in 1:length(marginal_distributions)
        parameters[i] = rand(rng, marginal_distributions[i])
    end
    return parameters
end

function filtered_sampler(sampler, filter_function; rng = Xoshiro())

    # Define the function that does the filtered sampling
    function sample_filtered()
        candidate_parameters = sampler(; rng = rng)
        while !filter_function(candidate_parameters)
            candidate_parameters = sampler(; rng = rng)
        end

        return candidate_parameters
    end

    return sample_filtered
end

function solve_final_state(parameters, initial_conditions, tspan, prob;
        abstol = 1e-6, reltol = 1e-6, length = 100)
    saveat = (tspan[2] - tspan[1]) / length
    sol = solve(prob, CVODE_BDF(linear_solver = :GMRES), tspan = tspan,
        abstol = abstol, reltol = reltol, saveat = saveat, progress = true)

    return sol
end

function solve_final_states_ensemble(parameter_batch, initial_condition_batch, tspan,
        initialize_problem; abstol = 1e-6, reltol = 1e-6, length = 100)
    prob = initialize_problem(
        parameter_batch[:, 1],
        initial_condition_batch[:, :, :, 1],
        tspan
    )

    # jumps to the next ensemble member
    function prob_func(prob, i, repeat)
        prob.u0 .= initial_condition_batch[:, :, :, i]
        for j in 1:size(parameter_batch, 1)
            prob.p[1][j] = parameter_batch[j, i]
        end
        prob
    end

    # instructs to only retain the last frame of each evolution
    output_func(sol, i) = (sol.u[end], false)

    ens_prob = EnsembleProblem(prob; output_func = output_func, prob_func = prob_func)

    saveat = (tspan[2] - tspan[1]) / length
    num_trajectories = size(parameter_batch, 2)
    println("Evolving $(num_trajectories) PDEs from t=$(tspan[1]) to t=$(tspan[2]) in $(Threads.nthreads()) parallel threads.")
    ens_sol = solve(ens_prob, CVODE_BDF(linear_solver = :GMRES), EnsembleThreads(),
        tspan = tspan, trajectories = num_trajectories, abstol = abstol,
        reltol = reltol, saveat = saveat, progress = true)

    final_state_batch = copy(initial_condition_batch)
    for i in 1:num_trajectories
        final_state_batch[:, :, :, i] = ens_sol.u[i]
    end

    return final_state_batch
end

end  # end of module Samplers  ########
#######################################

# #######################################

module GMStabilityAnalysis

export homogeneous_state, is_homogeneous_state_unstable

using LinearAlgebra

const third = 1 // 3

function solve_cubic_eq(poly::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
    # Cubic equation solver for complex polynomial (degree=3)
    # http://en.wikipedia.org/wiki/Cubic_function   Lagrange's method
    a1 = 1 / poly[4]
    E1 = -poly[3] * a1
    E2 = poly[2] * a1
    E3 = -poly[1] * a1
    s0 = E1
    E12 = E1 * E1
    A = 2 * E1 * E12 - 9 * E1 * E2 + 27 * E3 # = s1^3 + s2^3
    B = E12 - 3 * E2                 # = s1 s2
    # quadratic equation: z^2 - Az + B^3=0  where roots are equal to s1^3 and s2^3
    Δ = sqrt(A * A - 4 * B * B * B)
    if real(conj(A) * Δ) >= 0 # scalar product to decide the sign yielding bigger magnitude
        s1 = exp(log(0.5 * (A + Δ)) * third)
    else
        s1 = exp(log(0.5 * (A - Δ)) * third)
    end
    if s1 == 0
        s2 = s1
    else
        s2 = B / s1
    end
    zeta1 = complex(-0.5, sqrt(T(3.0)) * 0.5)
    zeta2 = conj(zeta1)
    return third * (s0 + s1 + s2), third * (s0 + s1 * zeta2 + s2 * zeta1),
    third * (s0 + s1 * zeta1 + s2 * zeta2)
end

function homogeneous_state(p)
    a, b, c, δ, s = @view p[:]
    poly::Vector{Complex{Float64}} = [-b * c, a * c, -b, a + 1][end:-1:1]
    roots = solve_cubic_eq(poly)
    filtered_tuple = filter(x -> imag(x) == 0, roots)

    u_hom_1 = real.(filtered_tuple)[1]
    u_hom_2 = u_hom_1^2
    return [u_hom_1, u_hom_2]
end

function jacobian_hom(u, p)
    a, b, c, δ, s = @view p[:]
    u1 = u[1]
    u2 = u[2]

    du1_du1 = -b + (2 * u1 * (1 + c * u1^2) - 2 * c * u1^3) / (u2 * (1 + c * u1^2)^2)
    du1_du2 = -u1^2 / (u2^2 * (1 + c * u1^2))

    du2_du1 = 2 * u1
    du2_du2 = -1

    # Form the Jacobian matrix
    return [du1_du1 du1_du2;
            du2_du1 du2_du2]
end

function jacobian_perturbation(u, p, q)
    a, b, c, δ, s = @view p[:]
    return jacobian_hom(u, p) - abs(q)^2 * [s 0; 0 s*δ]
end

function max_eigval(u_hom, p, qs)
    # vectorized computation of maximum real part of eigenvalue for range of wavenumbers q
    return map(q -> maximum(real(eigvals(jacobian_perturbation(u_hom, p, q)))), qs)
end

function is_homogeneous_state_unstable(p)
    u_hom = homogeneous_state(p)
    qs = 0.0:0.001:4  # wide range and high resolution to scan wavenumbers
    return maximum(max_eigval(u_hom, p, qs)) > 0
end

end  # end of module GMStabilityAnalysis##
# #######################################
