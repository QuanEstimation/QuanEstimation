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

# function expm(H::Vector{Matrix{T}}, ∂H_∂x::Vector{Vector{T}},  ρ_in::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
#     Δt = times[2] - times[1]
#     println(111)
#     para_num = length(∂H_∂x)
#     ρt = evolute(H[1], Liouville_operator, γ, Δt, 1) * (ρ_in |> vec)
#     ∂ρt_∂x = [-im * Δt * ∂H_∂x[i] * ρt for i in 1:para_num]
#     println(ρt)
#     println(∂ρt_∂x)
#     for t in 2:length(times)
#         expL = evolute(H[t], Liouville_operator, γ, Δt, t)
#         ρt = expL * ρt
#         ∂ρt_∂x = [-im * Δt * ∂H_∂x[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
#     end
#     ρt, ∂ρt_∂x
# end

# function evolute_ODE!(grape::Gradient)
#     H(p) = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, p)
#     dt = grape.times[2] - grape.times[1]    
#     tspan = (grape.times[1], grape.times[end])
#     u0 = grape.ρ_initial
#     Γ = grape.Liouville_operator
#     f(u, p, t) = -im * (H(p)[t2Num(tspan[1], dt, t)] * u + u * H(p)[t2Num(tspan[1], dt, t)]) + 
#                  ([grape.γ[i] * (Γ[i] * u * Γ[i]' - (Γ[i]' * Γ[i] * u + u * Γ[i]' * Γ[i] )) for i in 1:length(Γ)] |> sum)
#     prob = ODEProblem(f, u0, tspan, grape.control_coefficients, saveat=dt)
#     sol = solve(prob)
#     sol.u
# end

# function propagate_ODEAD!(grape::Gradient)
#     H(p) = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, p)
#     dt = grape.times[2] - grape.times[1]    
#     tspan = (grape.times[1], grape.times[end])
#     u0 = grape.ρ_initial
#     Γ = grape.Liouville_operator
#     f(u, p, t) = -im * (H(p)[t2Num(tspan[1], dt, t)] * u + u * H(p)[t2Num(tspan[1], dt, t)]) + 
#                  ([grape.γ[i] * (Γ[i] * u * Γ[i]' - (Γ[i]' * Γ[i] * u + u * Γ[i]' * Γ[i] )) for i in 1:length(Γ)] |> sum)
#     p = grape.control_coefficients
#     prob = ODEProblem(f, u0, tspan, p, saveat=dt)
#     u = solve(prob).u
#     du = Zygote.jacobian(solve(remake(prob, u0=u, p), sensealg=QuadratureAdjoint()))
#     u, du
# end

# function propagate_L_ODE!(grape::Gradient)
#     H = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients)
#     Δt = grape.times[2] - grape.times[1]    
#     tspan = (grape.times[1], grape.times[end])
#     u0 = grape.ρ_initial |> vec
#     evo(p, t) = evolute(p[t2Num(tspan[1], Δt,  t)], grape.Liouville_operator, grape.γ, grape.times, t2Num(tspan[1], Δt, t)) 
#     f(u, p, t) = evo(p, t) * u
#     prob = DiscreteProblem(f, u0, tspan, H,dt=Δt)
#     ρt = solve(prob).u 
#     ∂ρt_∂x = Vector{Vector{Vector{eltype(u0)}}}(undef, 1)
#     for para in 1:length(grape.Hamiltonian_derivative)
#         devo(p, t) = -1.0im * Δt * liouville_commu(grape.Hamiltonian_derivative[para]) * evo(p, t) 
#         du0 = devo(H, tspan[1]) * u0
#         g(du, p, t) = evo(p, t) * du + devo(p, t) * ρt[t2Num(tspan[1], Δt,  t)] 
#         dprob = DiscreteProblem(g, du0, tspan, H,dt=Δt) 
#         ∂ρt_∂x[para] = solve(dprob).u
#     end

#     grape.ρ, grape.∂ρ_∂x = ρt |> vec2mat, ∂ρt_∂x |> vec2mat
# end