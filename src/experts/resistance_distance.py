import numpy as np


def get_adj_matrix(pattern, eps=0.003):
    assert pattern.shape[0] == pattern.shape[1]
    graph_res = pattern.shape[0]
    m = graph_res**2

    adj_matrix = np.zeros((m, m), dtype=np.float64)
    u_mean = np.mean(pattern)

    for i in range(m):
        neighbor_js = np.array([i - 1, i + 1, i - graph_res, i + graph_res])
        neighbor_js = neighbor_js % m

        for j in neighbor_js:
            u_vi = pattern[i % graph_res, i // graph_res]
            u_vj = pattern[j % graph_res, j // graph_res]
            vi_high = u_vi > u_mean
            vj_high = u_vj > u_mean
            if not vi_high ^ vj_high:  # both on same side of u_mean
                adj_matrix[i, j] = 1
            else:
                adj_matrix[i, j] = eps

    return adj_matrix


def compute_resistance_np(pattern, eps):
    assert pattern.shape[0] == pattern.shape[1]
    graph_res = pattern.shape[0]
    m = graph_res**2

    adj_matrix = get_adj_matrix(pattern, eps)
    graph_laplacian = np.diag(adj_matrix @ np.ones((m,))) - adj_matrix
    K = np.linalg.inv(np.ones((m, m)) + graph_laplacian)
    Kvivi = np.ones((m, m)) * np.diagonal(K)
    Kvjvj = np.ones((m, m)) * np.diagonal(K)[:, None]
    resistance = Kvivi + Kvjvj - 2 * K

    assert resistance.shape == (m, m)

    return resistance


try:
    import os

    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"

    from juliacall import Main as jl

    jl.seval("using LinearAlgebra")
    jl.seval('include("src/experts/resistance_distance.jl")')
    jl.seval("using .FastResistanceDistance")

    def compute_resistance_jl(pattern, eps):
        return np.array(jl.compute_resistance(pattern, eps))

    compute_resistance = compute_resistance_jl
except Exception as e:
    compute_resistance = compute_resistance_np
    import warnings

    warnings.warn("Falling back to numpy version of compute_resistance, since Julia version failed with:\n" + str(e))


def reshaped_subsample(distances, t1, t2):
    assert distances.shape[0] == distances.shape[1]
    assert distances.ndim == 2, f"ndim is {distances.ndim}"
    graph_res = 64
    assert graph_res**2 == distances.shape[0]
    subsampled = distances.reshape((graph_res, graph_res, graph_res, graph_res), order="F")[::t1, ::t1, ::t2, ::t2]
    return subsampled
