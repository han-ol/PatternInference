#######################################

module FastResistanceDistance

export inv!, compute_resistance

using LinearAlgebra
using LinearAlgebra: BlasReal, BlasFloat
using Statistics

function inv!(A)
    A = convert(Matrix{Float32}, A)
    inv!(A)
end

function inv!(A::Union{Matrix{Int64}, Matrix{Float32}})
    A = LinearAlgebra.inv!(cholesky!(A))
end

function invlapack!(A::Union{Symmetric{<:BlasReal}, Hermitian{<:BlasFloat}})
    _, info = LAPACK.potrf!(A.uplo, A.data)
    (info == 0) || throw(PosDefException(info))
    LAPACK.potri!(A.uplo, A.data)
    return A
end

function get_adj_matrix(pattern, ε = 0.003)
    return get_adj_matrix(convert(Matrix{Float32}, pattern))
end

function get_adj_matrix(pattern::Union{Matrix{Float64}, Matrix{Float32}}, ε = 0.003)
    @assert size(pattern, 1) == size(pattern, 2)
    graph_res = size(pattern, 1)
    m = graph_res^2
    adj_matrix = zeros(Float32, m, m)
    u_mean = mean(pattern)

    for i in 1:m
        neighbor_js = [i - 1, i + 1, i - graph_res, i + graph_res]
        neighbor_js = mod1.(neighbor_js, m)# .+ 1

        for j in neighbor_js
            u_vi = pattern[mod1(i, graph_res), div(i - 1, graph_res) + 1]
            u_vj = pattern[mod1(j, graph_res), div(j - 1, graph_res) + 1]
            vi_high = u_vi > u_mean
            vj_high = u_vj > u_mean
            adj_matrix[i, j] = vi_high == vj_high ? 1.0 : ε
        end
    end

    return adj_matrix
end

function prepare_graph_laplacian_matrix(pattern, ε)
    graph_res = size(pattern, 1)
    m = graph_res^2
    return prepare_graph_laplacian_matrix(pattern, ε, m)
end

function prepare_graph_laplacian_matrix(pattern, ε, m)
    adj_matrix = get_adj_matrix(pattern, ε)
    graph_laplacian = Diagonal(adj_matrix * ones(Float32, m)) - adj_matrix

    return graph_laplacian
end

function compute_resistance(pattern, ε)
    graph_res = size(pattern, 1)
    m = graph_res^2

    K = Symmetric(prepare_graph_laplacian_matrix(pattern, ε, m) .+ 1.0f0)
    invlapack!(K)  # inv!(K) would be a slightly slower alternative
    return diag(K) .+ diag(K)' - 2 * K
end

end  # end of module FastResistanceDistance ####
#######################################
