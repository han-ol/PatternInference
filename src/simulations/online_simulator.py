import os

import bayesflow as bf
import numpy as np

from ..experts import histogram, resistance_distance

os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"
from juliacall import Main as jl

jl.seval('import Pkg; Pkg.activate(".")')
jl.seval('include("src/simulations/modules.jl")')

jl.seval("using Distributions")

jl.seval("import .GiererMeinhardt")
jl.seval("import .GMStabilityAnalysis")
jl.seval("using .Samplers")


def prior():
    """
    Sample from prior for a, b, c, delta, s, delta_x as in to [1].

    [1] Schnörr, D., & Schnörr, C. (2021).
        Learning System Parameters from Turing Patterns
        (No. arXiv:2108.08542). arXiv. http://arxiv.org/abs/2108.08542
    """
    parameters = np.array(jl.wide_prior_sampler())
    return dict(parameters=parameters)


def filtered_sampler(sampler, filter_function):
    candidate_sample = sampler()
    while not filter_function(candidate_sample):
        candidate_sample = sampler()
    return candidate_sample


def filtered_prior():
    """Rejects all parameter vectors that do not exhibit diffusion driven instability."""
    parameters = np.array(filtered_sampler(jl.wide_prior_sampler, jl.GMStabilityAnalysis.is_homogeneous_state_unstable))
    return dict(parameters=parameters)


def ic_sampler(parameters, domain_size=64):
    """Initializes the field with random perturbation around the fix point of the perfectly mixed system"""
    homogeneous_fix_point = np.array(jl.GMStabilityAnalysis.homogeneous_state(parameters))

    initial_conditions = homogeneous_fix_point[None, None, :] * (
        1 + 0.01 * np.random.rand(domain_size, domain_size, len(homogeneous_fix_point))
    )
    return dict(initial_conditions=initial_conditions)


def solve_final_state(parameters, initial_conditions, tspan=(0, 1000)):
    prob = jl.GiererMeinhardt.initialize_problem(jl.Array(parameters), jl.Array(initial_conditions), tspan)
    sol = jl.solve_final_state(
        jl.Array(parameters),
        jl.Array(initial_conditions),
        tspan,
        prob,
    )
    final_state = sol.u[-1][:, :, 0]
    return dict(final_states=final_state)


def get_pattern_simulator():
    return bf.make_simulator([filtered_prior, ic_sampler, solve_final_state])


def subsampled_resistance(final_states, eps=0.003, t1=8, t2=4):
    "Computes the full resistance distance matrix of ONE final_state, reshapes and subsamples it"
    subsampled_resistance_distances = resistance_distance.reshaped_subsample(
        resistance_distance.compute_resistance(final_states, eps), t1=t1, t2=t2
    )
    return dict(subsampled_resistance_distances=subsampled_resistance_distances)


def rdhistogram_expert(subsampled_resistance_distances, R_max, B):
    hist, bin_edges = histogram.histogram_from_distances(subsampled_resistance_distances, R_max=R_max, B=B)
    return dict(expert_rdhs=hist)


def make_rdhistogram_expert_func(R_max, B):
    "Wrapper function until bf.make_simulator supports partial objects"
    rdhist_expert_fun = lambda subsampled_resistance_distances: rdhistogram_expert(
        subsampled_resistance_distances, R_max=R_max, B=B
    )
    return rdhist_expert_fun


def get_expert_simulator(R_max, B):
    return bf.make_simulator([subsampled_resistance, make_rdhistogram_expert_func(R_max, B)])
