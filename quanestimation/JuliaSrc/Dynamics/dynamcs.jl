function liouville_commu(H) 
    kron(one(H), H) - kron(H |> transpose, one(H))
end

function liouville_dissip(Γ)
    kron(Γ |> conj, Γ) - 0.5 * kron((Γ |> transpose) * (Γ |> conj), Γ |> one) - 0.5 * kron(Γ |> one, Γ' * Γ)
end

function liouville_commu_py(A::Array{T}) where {T <: Complex}
    dim = size(A)[1]
    result = zeros(T, dim^2, dim^2)
    @inbounds for i in 1:dim
        @inbounds for j in 1:dim
            @inbounds for k in 1:dim
                ni = dim * (i - 1) + j
                nj = dim * (k - 1) + j
                nk = dim * (i - 1) + k

                result[ni,nj] = A[i,k]
                result[ni,nk] = -A[k,j]
                result[ni,ni] = A[i,i] - A[j,j]
            end
        end
    end
    result
end

function liouville_dissip_py(A::Array{T}) where {T <: Complex}
    dim = size(A)[1]
    result =  zeros(T, dim^2, dim^2)
    @inbounds for i = 1:dim
        @inbounds for j in 1:dim
            ni = dim * (i - 1) + j
            @inbounds for k in 1:dim
                @inbounds for l in 1:dim 
                    nj = dim * (k - 1) + l
                    L_temp = A[i,k] * conj(A[j,l])
                    @inbounds for p in 1:dim
                        L_temp -= 0.5 * float(k == i) * A[p,j] * conj(A[p,l]) + 0.5 * float(l == j) * A[p,k] * conj(A[p,i])
                    end
                    result[ni,nj] = L_temp
                end
            end 
        end
    end
    result[findall(abs.(result) .< 1e-10)] .= 0.
    result
end

function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{R}, t::Int=0) where {T <: Complex,R <: Real}
    [γ[i] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end

function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{Vector{R}}, t::Int=0) where {T <: Complex,R <: Real}
    [γ[i][t] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end

function free_evolution(H0)
    -1.0im * liouville_commu(H0)
end

function liouvillian(H::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, t::Real) where {T <: Complex} 
    freepart = liouville_commu(H)
    dissp = norm(γ) +1 ≈ 1 ? freepart|>zero : dissipation(decay_opt, γ, t)
    -1.0im * freepart + dissp
end

function Htot(H0::Matrix{T}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients) where {T <: Complex, R}
    Htot = [H0] .+ ([control_coefficients[i] .* [control_Hamiltonian[i]] for i in 1:length(control_coefficients)] |> sum )
end

function Htot(H0::Matrix{T}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{R}) where {T <: Complex, R<:Real}
    Htot = H0 + ([control_coefficients[i] * control_Hamiltonian[i] for i in 1:length(control_coefficients)] |> sum )
end

function Htot(H0::Vector{Matrix{T}}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients) where {T <: Complex}
    Htot = H0 + ([control_coefficients[i] .* [control_Hamiltonian[i]] for i in 1:length(control_coefficients)] |> sum )
end

function evolute(H, decay_opt, γ, dt, tj)
    Ld = dt * liouvillian(H, decay_opt, γ, tj)
    exp(Ld)
end

function propagate(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}},
                   γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}
    dim = size(ρ0)[1]
    para_num = length(∂H_∂x)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ρₜ = [Vector{ComplexF64}(undef, dim^2) for i in 1:length(tspan)]
    ∂ₓρₜ = [[Vector{ComplexF64}(undef, dim^2) for i in 1:length(tspan)] for para in 1:para_num]
    Δt = tspan[2] - tspan[1]
    ρₜ[1] = ρ0 |> vec
    for para in 1:para_num
        ∂ₓρₜ[para][1] = ρₜ[1] |> zero
    end
    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρₜ[t] = expL * ρₜ[t-1]
        for para in 1:para_num
            ∂ₓρₜ[para][t] = -im * Δt * ∂H_L[para] * ρₜ[t] + expL * ∂ₓρₜ[para][t - 1]
        end
    end
    ρₜ .|> vec2mat, ∂ₓρₜ .|> vec2mat
end

function propagate(ρₜ::Matrix{T}, ∂ₓρₜ::Vector{Matrix{T}}, H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}},  decay_opt::Vector{Matrix{T}},
                   γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{R}, Δt::Real, t::Int=0, ctrl_interval::Int=1) where {T <: Complex,R <: Real}
    para_num = length(∂H_∂x)
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    expL = evolute(H, decay_opt, γ, Δt, t)
    ρₜ_next = ρₜ |> vec
    ∂ₓρₜ_next = [(∂ₓρₜ[para] |> vec) for para in 1:para_num]
    for i in 1:ctrl_interval
        ρₜ_next = expL * ρₜ_next 
        for para in 1:para_num
            ∂ₓρₜ_next[para] = -im * Δt * ∂H_L[para] * ρₜ_next + expL * ∂ₓρₜ_next[para]
        end
    end
    ρₜ_next |> vec2mat, ∂ₓρₜ_next |> vec2mat
end

function propagate(ρₜ, ∂ₓρₜ, system, ctrl, t=1)
    Δt = system.tspan[2] - system.tspan[1]
    propagate(ρₜ, ∂ₓρₜ, system.freeHamiltonian, system.Hamiltonian_derivative, system.decay_opt, system.γ, 
              system.control_Hamiltonian, ctrl, Δt, t, system.ctrl_interval)
end

function propagate!(system)
    system.ρ, system.∂ρ_∂x = propagate(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0,
                                       system.decay_opt, system.γ, system.control_Hamiltonian, 
                                       system.control_coefficients, system.tspan)
end

function expm(H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}

    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = tspan[2] - tspan[1]

    ρt_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i in 1:length(tspan)]
    ∂ρt_∂x_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i in 1:length(tspan)]
    ρt_all[1] = ρ0 |> vec
    ∂ρt_∂x_all[1] = ρt_all[1] |> zero
    
    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt_all[t] = expL * ρt_all[t-1]
        ∂ρt_∂x_all[t] = -im * Δt * ∂H_L * ρt_all[t] + expL * ∂ρt_∂x_all[t-1]
    end
    ρt_all |> vec2mat, ∂ρt_∂x_all |> vec2mat
end

function expm(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}

    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    
    ρt_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i in 1:length(tspan)]
    ∂ρt_∂x_all = [[Vector{ComplexF64}(undef, (length(H0))^2) for j in 1:para_num] for i in 1:length(tspan)]
    ρt_all[1] = ρ0 |> vec
    for pj in 1:para_num
        ∂ρt_∂x_all[1][pj] = ρt_all[1] |> zero
    end

    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt_all[t] = expL * ρt_all[t-1]
        for pj in 1:para_num
            ∂ρt_∂x_all[t][pj] = -im * Δt * ∂H_L[pj] * ρt_all[t] + expL* ∂ρt_∂x_all[t-1][pj]
        end
    end
    ρt_all |> vec2mat, ∂ρt_∂x_all |> vec2mat
end

######## single parameter ########
function dynamics(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T<:Complex,R<:Real}
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero

    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

function dynamics_TimeIndepend(H0::Matrix{T}, ∂H_∂x::Matrix{T}, psi_initial::Vector{T}, tspan) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    U = exp(-im * H0 * Δt)
    psi_t = psi_initial
    ∂psi_∂x = psi_initial |> zero 
    for t in 2:length(tspan)
        psi_t = U * psi_t
        ∂psi_∂x = -im * Δt * ∂H_∂x * psi_t + U * ∂psi_∂x
    end
    ρt = psi_t*psi_t'
    ∂ρt_∂x = ∂psi_∂x*psi_t'+psi_t*∂psi_∂x'
    ρt, ∂ρt_∂x
end

function dynamics_TimeIndepend(H0::Matrix{T}, ∂H_∂x::Matrix{T}, psi::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    ρ0 = psi*psi'
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero 
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

function dynamics_TimeIndepend(H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero 
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp(vec(H0)' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

function dynamics_noctrl(H0::Vector{Matrix{T}}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan) where {T<:Complex,R<:Real}

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        expL = evolute(H0[t-1], decay_opt, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp( vec(H0[end])' * zero(ρt) ) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

######## multi-parameter ########
function dynamics(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T<:Complex,R<:Real}

    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

function dynamics_TimeIndepend(H0::Vector{Matrix{T}}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan) where {T<:Complex,R<:Real}
    para_num = length(∂H_∂x)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        expL = evolute(H0[t-1], decay_opt, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp( vec(H0[end])' * zero(ρt) ) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

function dynamics_TimeIndepend(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, psi_initial::Vector{T}, tspan) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    U = exp(-im * H0 * Δt)
    psi_t = psi_initial
    ∂psi_∂x = [psi_initial |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        psi_t = U * psi_t
        ∂psi_∂x = [-im * Δt * ∂H_∂x[i] * psi_t for i in 1:para_num] + [U] .* ∂psi_∂x
    end
    ρt = psi_t * psi_t'
    ∂ρt_∂x = [(∂psi_∂x[i] * psi_t' + psi_t * ∂psi_∂x[i]') for i in 1:para_num]
    ρt, ∂ρt_∂x
end

function dynamics_TimeIndepend(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, psi::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    ρ0 = psi*psi'
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

function dynamics_TimeIndepend(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end


function dynamics_secondorder(H0, ∂H_∂x::Matrix{T}, ∂2H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T<:Complex,R<:Real}
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)
    ∂2H_L = liouville_commu(∂2H_∂x)

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero]
    ∂2ρt_∂x = [ρt |> zero]

    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x[1] = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
        ∂2ρt_∂x[1] = (-im*Δt*∂2H_L + Δt*Δt*∂H_L*∂H_L)*ρt - 2*im*Δt*∂H_L*∂ρt_∂x[1] + expL * ∂2ρt_∂x[1]
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat, ∂2ρt_∂x |> vec2mat
end

function secondorder_derivative(H0, ∂H_∂x::Vector{Matrix{T}}, ∂2H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T<:Complex,R<:Real}

    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    ∂2H_L = [liouville_commu(∂2H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    ∂2ρt_∂x = [ρt |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
        ∂2ρt_∂x = [(-im*Δt*∂2H_L[i] + Δt*Δt*∂H_L[i]*∂H_L[i])*ρt - 2*im*Δt*∂H_L[i]*∂ρt_∂x[i] for i in 1:para_num] + [expL] .* ∂2ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat, ∂2ρt_∂x |> vec2mat
end
