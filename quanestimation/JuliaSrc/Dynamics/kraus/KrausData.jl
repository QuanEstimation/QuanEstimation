abstract type KrausDynamicsData <: AbstractDynamicsData end

mutable struct Kraus_dm <: KrausDynamicsData
    K::AbstractVector
    dK::AbstractVecOrMat
    ρ0::AbstractMatrix
end

mutable struct Kraus_pure <: KrausDynamicsData
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
    
para_type(data::KrausDynamicsData) = length(data.dK[1]) == 1 ? :single_para : :multi_para

get_dim(k::Kraus_dm) = size(k.ρ0, 1)
get_dim(k::Kraus_pure) = size(k.ψ0, 1)

get_para(k::KrausDynamicsData) = length(k.dK[1])
