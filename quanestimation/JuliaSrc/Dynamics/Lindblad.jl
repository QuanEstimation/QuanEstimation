
# dynamics in Lindblad form
struct Lindblad{N,C,R,P} <: AbstractDynamics
    data::AbstractDynamicsData
    noise_type::Symbol
    ctrl_type::Symbol
    state_rep::Symbol
    para_type::Symbol
end

Lindblad(data, N, C, R) =
    para_type(data) |> P -> Lindblad{((N, C, R, P) .|> eval)...}(data, N, C, R, P)
Lindblad(data, N, C) = Lindblad(data, N, C, :dm)

mutable struct Lindblad_noiseless_free <: AbstractDynamicsData
    H0::AbstractMatrix
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
end

mutable struct Lindblad_noisy_free <: AbstractDynamicsData
    H0::AbstractMatrix
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end


mutable struct Lindblad_noiseless_timedepend <: AbstractDynamicsData
    H0::AbstractVector
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
end

mutable struct Lindblad_noisy_timedepend <: AbstractDynamicsData
    H0::AbstractVector
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end

mutable struct Lindblad_noiseless_controlled <: AbstractDynamicsData
    H0::AbstractVecOrMat
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    Hc::AbstractVector
    ctrl::AbstractVector
end

mutable struct Lindblad_noisy_controlled <: AbstractDynamicsData
    H0::AbstractVecOrMat
    dH::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
    Hc::AbstractVector
    ctrl::AbstractVector
end


mutable struct Lindblad_noiseless_free_pure <: AbstractDynamicsData
    H0::AbstractMatrix
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
end

mutable struct Lindblad_noisy_free_pure <: AbstractDynamicsData
    H0::AbstractMatrix
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end

mutable struct Lindblad_noiseless_timedepend_pure <: AbstractDynamicsData
    H0::AbstractVector
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
end

mutable struct Lindblad_noisy_timedepend_pure <: AbstractDynamicsData
    H0::AbstractVector
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end
mutable struct Lindblad_noiseless_controlled_pure <: AbstractDynamicsData
    H0::AbstractVecOrMat
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    Hc::AbstractVector
    ctrl::AbstractVector
end

mutable struct Lindblad_noisy_controlled_pure <: AbstractDynamicsData
    H0::AbstractVecOrMat
    dH::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
    Hc::AbstractVector
    ctrl::AbstractVector
end


para_type(data::AbstractDynamicsData) = length(data.dH) == 1 ? :single_para : :multi_para

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
) = Lindblad(Lindblad_noiseless_free(H0, dH, ψ0, tspan), :noiseless, :free, :ket)

Lindblad(
    H0::AbstractMatrix,
    dH::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(Lindblad_noisy_free(H0, dH, ψ0, tspan, decay_opt, γ), :noisy, :free, :ket)

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
    Lindblad_noisy_timedepend(H0, dH, ψ0, tspan, decay_opt, γ),
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
    Lindblad_noisy_controlled(H0, dH, ψ0, tspan, decay_opt, γ, Hc, ctrl),
    :noisy,
    :controlled,
    :ket,
)

function set_ctrl(dynamics::Lindblad, ctrl)
    temp = deepcopy(dynamics)
    temp.data.ctrl = ctrl
    temp
end

function set_state(dynamics::Lindblad, state::AbstractVector)
    temp = deepcopy(dynamics)
    temp.data.ψ0 = state
    temp
end

function set_state(dynamics::Lindblad, state::AbstractMatrix)
    temp = deepcopy(dynamics)
    temp.data.ρ0 = state
    temp
end

# functions for evolve dynamics in Lindblad form
function liouville_commu(H)
    kron(one(H), H) - kron(H |> transpose, one(H))
end

function liouville_dissip(Γ)
    kron(Γ |> conj, Γ) - 0.5 * kron((Γ |> transpose) * (Γ |> conj), Γ |> one) -
    0.5 * kron(Γ |> one, Γ' * Γ)
end

function liouville_commu_py(A::Array{T}) where {T<:Complex}
    dim = size(A)[1]
    result = zeros(T, dim^2, dim^2)
    @inbounds for i = 1:dim
        @inbounds for j = 1:dim
            @inbounds for k = 1:dim
                ni = dim * (i - 1) + j
                nj = dim * (k - 1) + j
                nk = dim * (i - 1) + k

                result[ni, nj] = A[i, k]
                result[ni, nk] = -A[k, j]
                result[ni, ni] = A[i, i] - A[j, j]
            end
        end
    end
    result
end

function liouville_dissip_py(A::Array{T}) where {T<:Complex}
    dim = size(A)[1]
    result = zeros(T, dim^2, dim^2)
    @inbounds for i = 1:dim
        @inbounds for j = 1:dim
            ni = dim * (i - 1) + j
            @inbounds for k = 1:dim
                @inbounds for l = 1:dim
                    nj = dim * (k - 1) + l
                    L_temp = A[i, k] * conj(A[j, l])
                    @inbounds for p = 1:dim
                        L_temp -=
                            0.5 * float(k == i) * A[p, j] * conj(A[p, l]) +
                            0.5 * float(l == j) * A[p, k] * conj(A[p, i])
                    end
                    result[ni, nj] = L_temp
                end
            end
        end
    end
    result[findall(abs.(result) .< 1e-10)] .= 0.0
    result
end

function dissipation(
    Γ::Vector{Matrix{T}},
    γ::Vector{R},
    t::Int = 0,
) where {T<:Complex,R<:Real}
    [γ[i] * liouville_dissip(Γ[i]) for i = 1:length(Γ)] |> sum
end

function dissipation(
    Γ::Vector{Matrix{T}},
    γ::Vector{Vector{R}},
    t::Int = 0,
) where {T<:Complex,R<:Real}
    [γ[i][t] * liouville_dissip(Γ[i]) for i = 1:length(Γ)] |> sum
end

function free_evolution(H0)
    -1.0im * liouville_commu(H0)
end

function liouvillian(
    H::Matrix{T},
    decay_opt::Vector{Matrix{T}},
    γ,
    t = 1,
) where {T<:Complex}
    freepart = liouville_commu(H)
    dissp = norm(γ) + 1 ≈ 1 ? freepart |> zero : dissipation(decay_opt, γ, t)
    -1.0im * freepart + dissp
end

function Htot(H0::Matrix{T}, Hc::Vector{Matrix{T}}, ctrl) where {T<:Complex,R}
    [H0] .+ ([ctrl[i] .* [Hc[i]] for i = 1:length(ctrl)] |> sum)
end

function Htot(
    H0::Matrix{T},
    Hc::Vector{Matrix{T}},
    ctrl::Vector{R},
) where {T<:Complex,R<:Real}
    H0 + ([ctrl[i] * Hc[i] for i = 1:length(ctrl)] |> sum)
end

function Htot(H0::Vector{Matrix{T}}, Hc::Vector{Matrix{T}}, ctrl) where {T<:Complex}
    H0 + ([ctrl[i] .* [Hc[i]] for i = 1:length(ctrl)] |> sum)
end

function expL(H, decay_opt, γ, dt, tj = 1)
    Ld = dt * liouvillian(H, decay_opt, γ, tj)
    exp(Ld)
end

function expL(H, dt)
    freepart = liouville_commu(H)
    Ld = -1.0im * dt * freepart
    exp(Ld)
end

function expm(
    H0::Matrix{T},
    dH::Matrix{T},
    Hc::Vector{Matrix{T}},
    ctrl::Vector{Vector{R}},
    ρ0::Matrix{T},
    tspan,
    decay_opt::Vector{Matrix{T}},
    γ,
) where {T<:Complex,R<:Real}

    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]

    H = Htot(H0, Hc, ctrl)
    dH_L = liouville_commu(dH)

    Δt = tspan[2] - tspan[1]

    ρt_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i = 1:length(tspan)]
    ∂ρt_∂x_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i = 1:length(tspan)]
    ρt_all[1] = ρ0 |> vec
    ∂ρt_∂x_all[1] = ρt_all[1] |> zero

    for t = 2:length(tspan)
        expL = evolve(H[t-1], decay_opt, γ, Δt, t)
        ρt_all[t] = expL * ρt_all[t-1]
        ∂ρt_∂x_all[t] = -im * Δt * dH_L * ρt_all[t] + expL * ∂ρt_∂x_all[t-1]
    end
    ρt_all |> vec2mat, ∂ρt_∂x_all |> vec2mat
end

function expm(
    H0::Matrix{T},
    dH::Vector{Matrix{T}},
    Hc::Vector{Matrix{T}},
    ctrl::Vector{Vector{R}},
    ρ0::Matrix{T},
    tspan,
    decay_opt::Vector{Matrix{T}},
    γ,
) where {T<:Complex,R<:Real}

    para_num = length(dH)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]

    H = Htot(H0, Hc, ctrl)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]

    Δt = tspan[2] - tspan[1]

    ρt_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i = 1:length(tspan)]
    ∂ρt_∂x_all = [
        [Vector{ComplexF64}(undef, (length(H0))^2) for j = 1:para_num] for
        i = 1:length(tspan)
    ]
    ρt_all[1] = ρ0 |> vec
    for pj = 1:para_num
        ∂ρt_∂x_all[1][pj] = ρt_all[1] |> zero
    end

    for t = 2:length(tspan)
        expL = evolve(H[t-1], decay_opt, γ, Δt, t)
        ρt_all[t] = expL * ρt_all[t-1]
        for pj = 1:para_num
            ∂ρt_∂x_all[t][pj] = -im * Δt * dH_L[pj] * ρt_all[t] + expL * ∂ρt_∂x_all[t-1][pj]
        end
    end
    ρt_all |> vec2mat, ∂ρt_∂x_all |> vec2mat
end

expm(dynamics::Lindblad) = expm(dynamics.data...)


function secondorder_derivative(
    H0,
    dH::Vector{Matrix{T}},
    dH_∂x::Vector{Matrix{T}},
    ρ0::Matrix{T},
    decay_opt::Vector{Matrix{T}},
    γ,
    Hc::Vector{Matrix{T}},
    ctrl::Vector{Vector{R}},
    tspan,
) where {T<:Complex,R<:Real}

    para_num = length(dH)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]

    H = Htot(H0, Hc, ctrl)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    dH_L = [liouville_commu(dH_∂x[i]) for i = 1:para_num]

    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    ∂2ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = evolve(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
        ∂2ρt_∂x =
            [
                (-im * Δt * dH_L[i] + Δt * Δt * dH_L[i] * dH_L[i]) * ρt -
                2 * im * Δt * dH_L[i] * ∂ρt_∂x[i] for i = 1:para_num
            ] + [expL] .* ∂2ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat, ∂2ρt_∂x |> vec2mat
end

#### evolution of pure states under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Lindblad{noiseless,free,ket})
    (; H0, dH, psi0, tspan) = dynamics.data

    para_num = length(dH)
    Δt = tspan[2] - tspan[1]
    U = exp(-im * H0 * Δt)
    psi_t = psi0
    ∂psi_∂x = [psi0 |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        psi_t = U * psi_t
        ∂psi_∂x = [-im * Δt * dH[i] * psi_t for i = 1:para_num] + [U] .* ∂psi_∂x
    end
    ρt = psi_t * psi_t'
    ∂ρt_∂x = [(∂psi_∂x[i] * psi_t' + psi_t * ∂psi_∂x[i]') for i = 1:para_num]
    ρt, ∂ρt_∂x
end

#### evolution of pure states under time-dependent Hamiltonian without noise and controls ####
function evolve(dynamics::Lindblad{noiseless,timedepend,ket})
    (; H0, dH, psi0, tspan) = dynamics.data

    para_num = length(dH)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    ρt = (psi0 * psi0') |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H0[t-1], Δt)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Lindblad{noiseless,free,dm})
    (; H0, dH, ρ0, tspan) = dynamics.data

    para_num = length(dH)
    Δt = tspan[2] - tspan[1]
    expL = expL(H0, Δt)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of pure states under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Lindblad{noiseless,free,ket})
    (; H0, dH, psi0, tspan) = dynamics.data

    para_num = length(dH)
    Δt = tspan[2] - tspan[1]
    U = exp(-im * H0 * Δt)
    psi_t = psi0
    ∂psi_∂x = [psi0 |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        psi_t = U * psi_t
        ∂psi_∂x = [-im * Δt * dH[i] * psi_t for i = 1:para_num] + [U] .* ∂psi_∂x
    end
    ρt = psi_t * psi_t'
    ∂ρt_∂x = [(∂psi_∂x[i] * psi_t' + psi_t * ∂psi_∂x[i]') for i = 1:para_num]
    ρt, ∂ρt_∂x
end

#### evolution of pure states under time-dependent Hamiltonian without noise and controls ####
function evolve(dynamics::Lindblad{noiseless,timedepend,ket})
    (; H0, dH, psi0, tspan) = dynamics.data

    para_num = length(dH)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    ρt = (psi0 * psi0') |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H0[t-1], Δt)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Lindblad{noiseless,free,dm})
    (; H0, dH, ρ0, tspan) = dynamics.data

    para_num = length(dH)
    Δt = tspan[2] - tspan[1]
    expL = expL(H0, Δt)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-dependent Hamiltonian without noise and controls ####
function evolve(dynamics::Lindblad{noiseless,timedepend,dm})
    (; H0, dH, ρ0, tspan) = dynamics.data

    para_num = length(dH)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H0[t-1], Δt)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of pure states under time-independent Hamiltonian  
#### with noise but without controls
function evolve(dynamics::Lindblad{noisy,free,ket})
    (; H0, dH, psi0, tspan, decay_opt, γ) = dynamics.data

    para_num = length(dH)
    ρt = (psi0 * psi0') |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    Δt = tspan[2] - tspan[1]
    expL = expL(H0, decay_opt, γ, Δt, 1)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    for t = 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-independent Hamiltonian  
#### with noise but without controls
function evolve(dynamics::Lindblad{noisy,free,dm})
    (; H0, dH, ρ0, tspan, decay_opt, γ) = dynamics.data

    para_num = length(dH)
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    Δt = tspan[2] - tspan[1]
    expL = expL(H0, decay_opt, γ, Δt, 1)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    for t = 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of pure states under time-dependent Hamiltonian  
#### with noise but without controls
function evolve(dynamics::Lindblad{noisy,timedepend,ket})
    (; H0, dH, psi0, tspan, decay_opt, γ) = dynamics.data

    para_num = length(dH)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    ρt = (psi * psi') |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H0[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-dependent Hamiltonian  
#### with noise but without controls
function evolve(dynamics::Lindblad{noisy,timedepend,dm})
    (; H0, dH, ρ0, tspan, decay_opt, γ) = dynamics.data

    para_num = length(dH)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H0[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-independent Hamiltonian 
#### with controls but without noise #### 
function evolve(dynamics::Lindblad{noiseless,controlled,dm})
    (; H0, dH, ρ0, tspan, Hc, ctrl) = dynamics.data

    para_num = length(dH)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]
    H = Htot(H0, Hc, ctrl)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H[t-1], Δt)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-independent Hamiltonian with noise and controls #### 
function evolve(dynamics::Lindblad{noisy,controlled,dm})
    (; H0, dH, ρ0, tspan, decay_opt, γ, Hc, ctrl) = dynamics.data

    para_num = length(dH)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]
    H = Htot(H0, Hc, ctrl)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL_t = expL(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL_t * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL_t] .* ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end
# # 
# function dynamics_grad(H0, dH, ρ0, tspan, decay_opt, γ, Hc, ctrl)
#     para_num = length(dH)
#     ctrl_num = length(Hc)
#     ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
#     ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]
#     H = Htot(H0, Hc, ctrl)
#     dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
#     ρt = ρ0 |> vec
#     ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
#     for t = 2:length(tspan)
#         Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
#         expL = expL(H[t-1], decay_opt, γ, Δt, t)
#         ρt = expL * ρt
#         ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
#     end
#     ρt = exp(vec(H[end])' * zero(ρt)) * ρt
#     ρt |> vec2mat, ∂ρt_∂x |> vec2mat
# end

# function dynamics_grad(H0, dH, ρ0, tspan, decay_opt, γ, Hc, ctrl)
#     para_num = length(dH)
#     ctrl_num = length(Hc)
#     ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
#     ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]
#     H = Htot(H0, Hc, ctrl)
#     dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
#     ρt = ρ0 |> vec
#     ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
#     for t = 2:length(tspan)
#         Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
#         expL = expL(H[t-1], decay_opt, γ, Δt, t)
#         ρt = expL * ρt
#         ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
#     end
#     ρt = exp(vec(H[end])' * zero(ρt)) * ρt
#     ρt |> vec2mat, ∂ρt_∂x |> vec2mat
# end

function propagate(
    dynamics::Lindblad{noisy,controlled,dm,P},
    ρₜ::AbstractMatrix,
    dρₜ::AbstractVector,
    ctrl::Vector{R},
    Δt::Real,
) where {R<:Real,P}
    (; H0, dH, decay_opt, γ, Hc, tspan) = dynamics.data
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    para_num = length(dH)
    H = Htot(H0, Hc, ctrl)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    expL_t = expL(H, decay_opt, γ, Δt)
    ρₜ_next = ρₜ |> vec
    dρₜ_next = [(dρₜ[para] |> vec) for para = 1:para_num]
    for i = 1:ctrl_interval
        ρₜ_next = expL_t * ρₜ_next
        for para = 1:para_num
            dρₜ_next[para] = -im * Δt * dH_L[para] * ρₜ_next + expL_t * dρₜ_next[para]
        end
    end
    ρₜ_next |> vec2mat, dρₜ_next |> vec2mat
end

function propagate(
    dynamics::Lindblad{noiseless,controlled,dm,P},
    ρₜ::AbstractMatrix,
    dρₜ::AbstractVector,
    ctrl::Vector{R},
    Δt::Real,
) where {R<:Real,P}
    (; H0, dH, Hc, tspan) = dynamics.data
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    para_num = length(dH)
    H = Htot(H0, Hc, ctrl)
    dH_L = [liouville_commu(dH[i]) for i = 1:para_num]
    expL_t = expL(H, Δt)
    ρₜ_next = ρₜ |> vec
    dρₜ_next = [(dρₜ[para] |> vec) for para = 1:para_num]
    for i = 1:ctrl_interval
        ρₜ_next = expL_t * ρₜ_next
        for para = 1:para_num
            dρₜ_next[para] = -im * Δt * dH_L[para] * ρₜ_next + expL_t * dρₜ_next[para]
        end
    end
    ρₜ_next |> vec2mat, dρₜ_next |> vec2mat
end

function propagate(ρₜ, dρₜ, dynamics, ctrl, t = 1, ctrl_interval = 1)
    Δt = dynamics.tspan[t+1] - system.tspan[t]
    propagate(dynamics, ρₜ, dρₜ, ctrl, Δt, ctrl_interval)
end

# function propagate!(system)
#     system.ρ, system.∂ρ_∂x = propagate(system.freeHamiltonian, system.dH, system.ρ0,
#                             system.decay_opt, system.γ, system.Hc, 
#                             system.ctrl, system.tspan)
# # end
