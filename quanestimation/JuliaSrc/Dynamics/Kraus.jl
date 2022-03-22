# dynamics in Kraus rep.
struct Kraus{R} <: AbstractDynamics
    data::AbstractDynamicsData
    noise_type::Symbol
    ctrl_type::Symbol
    state_rep::Symbol
end

mutable struct Kraus_dm <: AbstractDynamicsData
    K::AbstractVector
    dK::AbstractVecOrMat
    ρ0::AbstractMatrix
end

mutable struct Kraus_pure <: AbstractDynamicsData
    K::AbstractVector
    dK::AbstractVecOrMat
    ψ0::AbstractVector
end

# Constructor for Kraus dynamics
Kraus(K::AbstractVector, dK::AbstractVector, ρ0::AbstractMatrix) =
    Kraus{dm}(Kraus_dm(K, dK, ρ0), :noiseless, :free, :dm)
Kraus(K::AbstractVector, dK::AbstractVector, ψ0::AbstractVector) =
    Kraus{ket}(Kraus_pure(K, dK, ψ0), :noiseless, :free, :ket)
Kraus(K::AbstractVector, dK::AbstractMatrix, ρ0::AbstractMatrix) =
    Kraus{dm}(Kraus_dm(K, [[dK[i,j,:,:][1] for j in 1:size(dK,2)] for i in 1:size(dK,1)], ρ0), :noiseless, :free, :dm)
Kraus(K::AbstractVector, dK::AbstractMatrix, ψ0::AbstractVector) =
    Kraus{ket}(Kraus_pure(K,[[dK[i,j,:,:][1] for j in 1:size(dK,2)] for i in 1:size(dK,1)], ψ0), :noiseless, :free, :ket)

function set_state(dynamics::Kraus, state::AbstractVector)
    temp = deepcopy(dynamics)
    temp.data.ψ0 = state
    temp
end
    
function set_state(dynamics::Kraus, state::AbstractMatrix)
    temp = deepcopy(dynamics)
    temp.data.ρ0 = state
    temp
end

#### evolution of pure states under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Kraus{ket})
    (; K, dK, ψ0) = dynamics.data
    ρ0 = ψ0 * ψ0'
    ρ = [K * ρ0 * K' for K in K] |> sum
    dρ = [[dK * ρ0 * K' + K * ρ0 * dK' for (K,dK) in zip(K,dK)] |> sum for dK in dK]

    ρ, dρ
end

#### evolution of density matrix under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Kraus{dm})
    (; K, dK, ρ0) = dynamics.data
    ρ = [K * ρ0 * K' for K in K] |> sum
    dρ = [[dK * ρ0 * K' + K * ρ0 * dK' for (K,dK) in zip(K,dK)] |> sum for dK in dK]

    ρ, dρ
end
