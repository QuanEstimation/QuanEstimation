abstract type KrausDynamicsData <: AbstractDynamicsData end

mutable struct Kraus_dm <: KrausDynamicsData
    ρ0::AbstractMatrix
    K::AbstractVector
    dK::AbstractVecOrMat
end

mutable struct Kraus_pure <: KrausDynamicsData
    ψ0::AbstractVector
    K::AbstractVector
    dK::AbstractVecOrMat
end

# Constructor for Kraus dynamics
@doc raw"""

    Kraus(ρ0::AbstractMatrix, K::AbstractVector, dK::AbstractVector)

The parameterization of a state is ``\rho=\sum_i K_i\rho_0K_i^{\dagger}`` with ``\rho`` the evolved density matrix and ``K_i`` the Kraus operator.
- `ρ0`: Initial state (density matrix).
- `K`: Kraus operators.
- `dK`: Derivatives of the Kraus operators with respect to the unknown parameters to be estimated. For example, dK[0] is the derivative vector on the first parameter.
"""
Kraus(ρ0::AbstractMatrix, K::AbstractVector, dK::AbstractVector) =
    Kraus{dm}(Kraus_dm(ρ0, K, dK), :noiseless, :free, :dm)
@doc raw"""

    Kraus(ψ0::AbstractMatrix, K::AbstractVector, dK::AbstractVector)

The parameterization of a state is ``\psi\rangle=\sum_i K_i|\psi_0\rangle`` with ``\psi`` the evolved state and ``K_i`` the Kraus operator.
- `ψ0`: Initial state (ket).
- `K`: Kraus operators.
- `dK`: Derivatives of the Kraus operators with respect to the unknown parameters to be estimated. For example, dK[0] is the derivative vector on the first parameter.
"""
Kraus(ψ0::AbstractVector, K::AbstractVector, dK::AbstractVector) =
    Kraus{ket}(Kraus_pure(ψ0, K, dK), :noiseless, :free, :ket)
Kraus(ρ0::AbstractMatrix, K::AbstractVector, dK::AbstractMatrix) =
    Kraus{dm}(Kraus_dm(ρ0, K, [[dK[i,j,:,:][1] for j in 1:size(dK,2)] for i in 1:size(dK,1)]), :noiseless, :free, :dm)
Kraus(ψ0::AbstractVector, K::AbstractVector, dK::AbstractMatrix) =
    Kraus{ket}(Kraus_pure(ψ0, K,[[dK[i,j,:,:][1] for j in 1:size(dK,2)] for i in 1:size(dK,1)]), :noiseless, :free, :ket)
    
para_type(data::KrausDynamicsData) = length(data.dK[1]) == 1 ? :single_para : :multi_para

get_dim(k::Kraus_dm) = size(k.ρ0, 1)
get_dim(k::Kraus_pure) = size(k.ψ0, 1)

get_para(k::KrausDynamicsData) = length(k.dK[1])
