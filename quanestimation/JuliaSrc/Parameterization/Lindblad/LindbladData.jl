abstract type LindbladDynamicsData <: AbstractDynamicsData end


mutable struct Lindblad_noiseless_free <: LindbladDynamicsData
    H0::AbstractMatrix
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
end

mutable struct Lindblad_noisy_free <: LindbladDynamicsData
    H0::AbstractMatrix
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end


mutable struct Lindblad_noiseless_timedepend <: LindbladDynamicsData
    H0::AbstractVector
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
end

mutable struct Lindblad_noisy_timedepend <: LindbladDynamicsData
    H0::AbstractVector
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end

mutable struct Lindblad_noiseless_controlled <: LindbladDynamicsData
    H0::AbstractVecOrMat
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    Hc::AbstractVector
    ctrl::AbstractVector
end

mutable struct Lindblad_noisy_controlled <: LindbladDynamicsData
    H0::AbstractVecOrMat
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
    Hc::AbstractVector
    ctrl::AbstractVector
end


mutable struct Lindblad_noiseless_free_pure <: LindbladDynamicsData
    H0::AbstractMatrix
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
end

mutable struct Lindblad_noisy_free_pure <: LindbladDynamicsData
    H0::AbstractMatrix
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end

mutable struct Lindblad_noiseless_timedepend_pure <: LindbladDynamicsData
    H0::AbstractVector
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
end

mutable struct Lindblad_noisy_timedepend_pure <: LindbladDynamicsData
    H0::AbstractVector
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end
mutable struct Lindblad_noiseless_controlled_pure <: LindbladDynamicsData
    H0::AbstractVecOrMat
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    Hc::AbstractVector
    ctrl::AbstractVector
end

mutable struct Lindblad_noisy_controlled_pure <: LindbladDynamicsData
    H0::AbstractVecOrMat
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
    Hc::AbstractVector
    ctrl::AbstractVector
end

# Constructor of Lindblad dynamics
Lindblad(
    H0::AbstractMatrix,
    dH::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
) = Lindblad(Lindblad_noiseless_free(H0, dH, ρ0, tspan), :noiseless, :free)

Lindblad(
    H0::AbstractMatrix,
    dH::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(Lindblad_noisy_free(H0, dH, ρ0, tspan, decay_opt, γ), :noisy, :free)

Lindblad(
    H0::AbstractVector,
    dH::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
) = Lindblad(Lindblad_noiseless_timedepend(H0, dH, ρ0, tspan), :noiseless, :timedepend)

Lindblad(
    H0::AbstractVector,
    dH::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(
    Lindblad_noisy_timedepend(H0, dH, ρ0, tspan, decay_opt, γ),
    :noisy,
    :timedepend,
)

Lindblad(
    H0::AbstractVecOrMat,
    dH::AbstractVector,
    Hc::AbstractVector,
    ctrl::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
) = Lindblad(
    Lindblad_noiseless_controlled(H0, dH, ρ0, tspan, Hc, ctrl),
    :noiseless,
    :controlled,
)
Lindblad(
    H0::AbstractVecOrMat,
    dH::AbstractVector,
    Hc::AbstractVector,
    ctrl::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(
    Lindblad_noisy_controlled(H0, dH, ρ0, tspan, decay_opt, γ, Hc, ctrl),
    :noisy,
    :controlled,
)

Lindblad(
    H0::AbstractMatrix,
    dH::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
) = Lindblad(Lindblad_noiseless_free_pure(H0, dH, ψ0, tspan), :noiseless, :free, :ket)

Lindblad(
    H0::AbstractMatrix,
    dH::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(Lindblad_noisy_free_pure(H0, dH, ψ0, tspan, decay_opt, γ), :noisy, :free, :ket)

Lindblad(
    H0::AbstractVector,
    dH::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
) = Lindblad(
    Lindblad_noiseless_timedepend(H0, dH, ψ0, tspan),
    :noiseless,
    :timedepend,
    :ket,
)

Lindblad(
    H0::AbstractVector,
    dH::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(
    Lindblad_noisy_timedepend_pure(H0, dH, ψ0, tspan, decay_opt, γ),
    :noisy,
    :timedepend,
    :ket,
)

Lindblad(
    H0::AbstractVecOrMat,
    dH::AbstractVector,
    Hc::AbstractVector,
    ctrl::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
) = Lindblad(
    Lindblad_noiseless_controlled(H0, dH, ψ0, tspan, Hc, ctrl),
    :noiseless,
    :controlled,
    :ket,
)

Lindblad(
    H0::AbstractVecOrMat,
    dH::AbstractVector,
    Hc::AbstractVector,
    ctrl::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(
    Lindblad_noisy_controlled_pure(H0, dH, ψ0, tspan, decay_opt, γ, Hc, ctrl),
    :noisy,
    :controlled,
    :ket,
)

Lindblad(data::LindbladDynamicsData, N, C, R) =
    para_type(data) |> P -> Lindblad{((N, C, R, P) .|> eval)...}(data, N, C, R, P)
Lindblad(data::LindbladDynamicsData, N, C) = Lindblad(data, N, C, :dm)

para_type(data::LindbladDynamicsData) = length(data.dH) == 1 ? :single_para : :multi_para

get_dim(d::Lindblad_noiseless_free) = size(d.ρ0, 1)
get_dim(d::Lindblad_noisy_free) = size(d.ρ0, 1)
get_dim(d::Lindblad_noiseless_controlled) = size(d.ρ0, 1)
get_dim(d::Lindblad_noisy_controlled) = size(d.ρ0, 1)
get_dim(d::Lindblad_noiseless_timedepend) = size(d.ρ0, 1)
get_dim(d::Lindblad_noisy_timedepend) = size(d.ρ0, 1)
get_dim(d::Lindblad_noiseless_free_pure) = size(d.ψ0, 1)
get_dim(d::Lindblad_noisy_free_pure) = size(d.ψ0, 1)
get_dim(d::Lindblad_noiseless_controlled_pure) = size(d.ψ0, 1)
get_dim(d::Lindblad_noisy_controlled_pure) = size(d.ψ0, 1)
get_dim(d::Lindblad_noiseless_timedepend_pure) = size(d.ψ0, 1)
get_dim(d::Lindblad_noisy_timedepend_pure) = size(d.ψ0, 1)

get_para(d::LindbladDynamicsData) = length(d.dH)