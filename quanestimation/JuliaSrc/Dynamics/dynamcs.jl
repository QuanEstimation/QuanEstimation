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

function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{R}, t::Real) where {T <: Complex,R <: Real}
    [γ[i] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end

function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{Vector{R}}, t::Real) where {T <: Complex,R <: Real}
    [γ[i][t] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end

function free_evolution(H0)
    -1.0im * liouville_commu(H0)
end

function liouvillian(H::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ, t::Real) where {T <: Complex} 
    freepart = liouville_commu(H)
    dissp = norm(γ) +1 ≈ 1 ? freepart|>zero : dissipation(Liouville_operator, γ, t)
    -1.0im * freepart + dissp
end

function Htot(H0::Matrix{T}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients) where {T <: Complex}
    Htot = [H0] .+  ([control_coefficients[i] .* [control_Hamiltonian[i]] for i in 1:length(control_coefficients)] |> sum )
end

function evolute(H, Liouville_operator, γ, dt, tj)
    Ld = dt * liouvillian(H, Liouville_operator, γ, tj)
    exp(Ld)
end

function propagate(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}},
                   γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}
    dim = size(H0)[1]
    para_num = length(∂H_∂x)
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ρt = [Vector{ComplexF64}(undef, dim^2)  for i in 1:length(times)]
    ∂ρt_∂x = [[Vector{ComplexF64}(undef, dim^2) for i in 1:length(times)] for para in 1:para_num]
    Δt = times[2] - times[1]
    ρt[1] = ρ_initial |> vec
    for para in  1:para_num
        ∂ρt_∂x[para][1] = ρt[1] |> zero
    end
    for t in 2:length(times)
        expL = evolute(H[t-1], Liouville_operator, γ, Δt, t)
        ρt[t] =  expL * ρt[t-1]
        for para in para_num
            ∂ρt_∂x[para][t] = -im * Δt * liouville_commu(∂H_∂x[para]) * ρt[t] + expL * ∂ρt_∂x[para][t - 1]
        end
    end
    ρt .|> vec2mat, ∂ρt_∂x .|> vec2mat
end

function propagate!(system)
    system.ρ, system.∂ρ_∂x = propagate(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ_initial,
                                       system.Liouville_operator, system.γ, system.control_Hamiltonian, 
                                       system.control_coefficients, system.times )
end

function expm(H0::Matrix{T}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}

    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = (length(times)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = times[2] - times[1]

    ρt_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i in 1:length(times)]
    ∂ρt_∂x_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i in 1:length(times)]
    ρt_all[1] = ρ_initial |> vec
    ∂ρt_∂x_all[1] = ρt_all[1] |> zero
    
    for t in 2:length(times)
        expL = evolute(H[t-1], Liouville_operator, γ, Δt, t)
        ρt_all[t] =  expL * ρt_all[t-1]
        ∂ρt_∂x_all[t] = -im * Δt * ∂H_L * ρt_all[t] + expL * ∂ρt_∂x_all[t-1]
    end
    ρt_all |> vec2mat, ∂ρt_∂x_all |> vec2mat
end

function expm(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}

    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = (length(times)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = times[2] - times[1]
    
    ρt_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i in 1:length(times)]
    ∂ρt_∂x_all = [[Vector{ComplexF64}(undef, (length(H0))^2) for j in 1:para_num] for i in 1:length(times)]
    ρt_all[1] = ρ_initial |> vec
    for pj in 1:para_num
        ∂ρt_∂x_all[1][pj] = ρt_all[1] |> zero
    end

    for t in 2:length(times)
        expL = evolute(H[t-1], Liouville_operator, γ, Δt, t)
        ρt_all[t] =  expL * ρt_all[t-1]
        for pj in 1:para_num
            ∂ρt_∂x_all[t][pj] = -im * Δt * ∂H_L[pj] * ρt_all[t] + expL* ∂ρt_∂x_all[t-1][pj]
        end
    end
    ρt_all |> vec2mat, ∂ρt_∂x_all |> vec2mat
end
