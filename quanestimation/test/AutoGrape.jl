module AutoGrape 
using LinearAlgebra
using Zygote
using DifferentialEquations
using JLD
using Random
abstract type Control end
mutable struct GrapeControl{T <: Complex,M <: Real} <: Control
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ_initial::Matrix{T}
    times::StepRangeLen{M,Base.TwicePrecision{M},Base.TwicePrecision{M}}
    Liouville_operator::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ϵ::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    GrapeControl(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, ρ_initial::Matrix{T},
                 times::StepRangeLen{M,Base.TwicePrecision{M},Base.TwicePrecision{M}},
                 Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
                 control_coefficients::Vector{Vector{M}}, ϵ=0.1, ρ=Vector{Matrix{T}}(undef, 1), 
                 ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1),∂ρ_∂V=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = 
                 new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ_initial, times, Liouville_operator, γ, control_Hamiltonian,
                          control_coefficients, ϵ, ρ, ∂ρ_∂x) 
end

sigmax() = [.0im 1.;1. 0.]
sigmay() = [0. -1.0im;1.0im 0.]
sigmaz() = [1.0  .0im;0. -1.]
sigmap() = [.0im 1.;0. 0.]
sigmam() = [.0im 0.;1. 0.]
sigmax(i, N) = kron(I(2^(i-1)), sigmax(), I(2^(N-i)))
sigmay(i, N) = kron(I(2^(i-1)), sigmay(), I(2^(N-i)))
sigmaz(i, N) = kron(I(2^(i-1)), sigmaz(), I(2^(N-i)))
sigmap(i, N) = kron(I(2^(i-1)), sigmap(), I(2^(N-i)))
sigmam(i, N) = kron(I(2^(i-1)), sigmam(), I(2^(N-i)))

function vec2mat(x::Vector{T}) where {T <: Number}
    reshape(x, x |> length |> sqrt |> Int, :)  
end

function vec2mat(x)
    vec2mat.(x)
end

function vec2mat(x::Matrix)
    throw(ErrorException("vec2mating a matrix of size $(size(x))"))
end

function Base.repeat(grape::GrapeControl, N)
    [grape for i in 1:N]
end

function Base.repeat(grape::GrapeControl, M, N)
    reshape(repeat(grape, M*N), M, N)
end

function liouville_commu(H) 
    kron(one(H), H) - kron(H |> transpose, one(H))
end

function destroy(M)
    spdiagm(M, M, 1 => map(x -> x |> sqrt, 1:(M - 1)))
end

function liouville_dissip(Γ)
    kron(Γ |> conj, Γ) - 0.5 * kron((Γ |> transpose) * Γ |> conj, Γ |> one) - 0.5 * kron(Γ |> one, Γ' * Γ)
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

function evolute(H::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ, 
                 times::Vector{M}, t::M)::Matrix{T} where {T <: Complex,R <: Real,M <: Real}
    tj = Int(round((t - times[1]) / (times[2] - times[1]))) + 1 
    dt = times[2] - times[1]
    Ld = dt * liouvillian(H, Liouville_operator, γ, tj)
    exp(Ld)
end

function evolute(H::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,
                 times::StepRangeLen{M,Base.TwicePrecision{M},Base.TwicePrecision{M}}, 
                 t::M)::Matrix{T} where {T <: Complex,R <: Real,M <: Real}
    tj = Int(round((t - times[1]) / (times[2] - times[1]))) + 1 
    dt = times[2] - times[1]
    Ld = dt * liouvillian(H, Liouville_operator, γ, tj)
    exp(Ld)
end

function evolute(H, Liouville_operator, γ, times, t)
    tj = Int(round((t - times[1]) / (times[2] - times[1]))) + 1 
    dt = times[2] - times[1]
    Ld = dt * liouvillian(H, Liouville_operator, γ, tj)
    exp(Ld)
end

function filterZeros!(x::Matrix{T}) where {T <: Complex}
    x[abs.(x) .< eps()] .= zero(T)
    x
end

function filterZeros!(x) 
    filterZeros!.(x)
end

function t2Num(t0, dt, t)
    Int(round((t - t0) / dt)) + 1 
end

function evolute_ODE(grape::GrapeControl)
    H(p) = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, p)
    dt = grape.times[2] - grape.times[1]    
    tspan = (grape.times[1], grape.times[end])
    u0 = grape.ρ_initial
    Γ = grape.Liouville_operator
    f(u, p, t) = -im * (H(p)[t2Num(tspan[1], dt, t)] * u + u * H(p)[t2Num(tspan[1], dt, t)]) + 
                 ([grape.γ[i] * (Γ[i] * u * Γ[i]' - (Γ[i]' * Γ[i] * u + u * Γ[i]' * Γ[i] )) for i in 1:length(Γ)] |> sum)
    prob = ODEProblem(f, u0, tspan, grape.control_coefficients, saveat=dt)
    sol = solve(prob)
    sol.u
end

function propagate_ODEAD(grape::GrapeControl)
    H(p) = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, p)
    dt = grape.times[2] - grape.times[1]    
    tspan = (grape.times[1], grape.times[end])
    u0 = grape.ρ_initial
    Γ = grape.Liouville_operator
    f(u, p, t) = -im * (H(p)[t2Num(tspan[1], dt, t)] * u + u * H(p)[t2Num(tspan[1], dt, t)]) + 
                 ([grape.γ[i] * (Γ[i] * u * Γ[i]' - (Γ[i]' * Γ[i] * u + u * Γ[i]' * Γ[i] )) for i in 1:length(Γ)] |> sum)
    p = grape.control_coefficients
    prob = ODEProblem(f, u0, tspan, p, saveat=dt)
    u = solve(prob).u
    du = Zygote.jacobian(solve(remake(prob, u0=u, p), sensealg=QuadratureAdjoint()))
    u, du
end

function propagate_L_ODE!(grape::GrapeControl)
    H = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients)
    Δt = grape.times[2] - grape.times[1]    
    tspan = (grape.times[1], grape.times[end])
    u0 = grape.ρ_initial |> vec
    evo(p, t) = evolute(p[t2Num(tspan[1], Δt,  t)], grape.Liouville_operator, grape.γ, grape.times, t2Num(tspan[1], Δt, t)) 
    f(u, p, t) = evo(p, t) * u
    prob = DiscreteProblem(f, u0, tspan, H,dt=Δt)
    ρt = solve(prob).u 
    ∂ρt_∂x = Vector{Vector{Vector{eltype(u0)}}}(undef, 1)
    for para in 1:length(grape.Hamiltonian_derivative)
        devo(p, t) = -1.0im * Δt * liouville_commu(grape.Hamiltonian_derivative[para]) * evo(p, t) 
        du0 = devo(H, tspan[1]) * u0
        g(du, p, t) = evo(p, t) * du + devo(p, t) * ρt[t2Num(tspan[1], Δt,  t)] 
        dprob = DiscreteProblem(g, du0, tspan, H,dt=Δt) 
        ∂ρt_∂x[para] = solve(dprob).u
    end

    grape.ρ, grape.∂ρ_∂x = ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

function propagate_analitical(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}},
                   γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}
    dim = size(H0)[1]
    tnum = length(times)
    para_num = length(∂H_∂x)
    ctrl_num = length(δH_δV)
    Δt = times[2] - times[1]

    ρt = [Vector{ComplexF64}(undef, dim^2)  for i in 1:tnum]
    ∂ρt_∂x = [[Vector{ComplexF64}(undef, dim^2) for para in 1:para_num] for i in 1:tnum]
    δρt_δV = [[] for ctrl in 1:ctrl_num]
    ∂xδρt_δV = [[[] for ctrl in 1:ctrl_num] for i in 1:para_num]
    ∂H_L = [Vector{ComplexF64}(undef, dim^2)  for i in 1:para_num]
    Hc_L = [Vector{ComplexF64}(undef, dim^2)  for i in 1:ctrl_num]

    ρt[1] = ρ_initial |> vec
    for pi in 1:para_num
        ∂ρt_∂x[1][pi] = zeros(T, dim^2)   #the shape???
        ∂H_L[pi] = liouville_commu(∂H_∂x[pi])
        for ci in 1:ctrl_num
            append!(δρt_δV[ci], [zeros(T, dim^2)])
            append!(∂xδρt_δV[pi][ci], [zeros(T, dim^2)])
        end 
    end

    for cj in 1:ctrl_num
        Hc_L[cj] = liouville_commu(control_Hamiltonian[cj])
    end

    for ti in 2:tnum
        expL = evolute(H[ti], Liouville_operator, γ, times, times[ti])
        ρt[i] =  expL * ρt[ti-1]
        for pk in 1:para_num
            ∂ρt_∂x[ti][pk] = -im*Δt*∂H_L[pk]*ρt[ti] + expL * ∂ρt_∂x[ti-1][pk]
            for ck in 1:ctrl_num
                for tk in 1:ti
                    δρt_δV_first = popfirsr!(δρt_δV[ck])
                    ∂xδρt_δV_first = popfirsr!(∂xδρt_δV[pk][ck])
                    δρt_δV_tp = expL*δρt_δV_first
                    ∂xδρt_δV_tp = -im*Δt*∂H_L[pk]*expL*∂xδρt_δV_first+expL*∂xδρt_δV_first
                    append!(δρt_δV[ck], [δρt_δV_tp])
                    append!(∂xδρt_δV[pk][ck], [∂xδρt_δV_tp])
                end
                δρt_δV_last = -im*Δt*Hc_L[ck]*ρt[ti]
                ∂xδρt_δV_last = -im*Δt*Hc_L[ck]*∂ρt_∂x[ti-1][pk]
                append!(δρt_δV[ck], [δρt_δV_last])
                append!(∂xδρt_δV[pk][ck], [∂xδρt_δV_last])
            end
        end
    end

    ρt_T = ρt[end] |> vec2mat
    ∂ρt_T = [∂ρt_∂x[end][para] |> vec2mat for para in 1:para_num]
    SLD = SLD(ρt_T, ∂ρt_T)
    F_T = QFIM(ρt_T, ∂ρt_T)

    if para_num == 1
        δF = [[0.0 for i in 1:tnum] for ctrl in 1:ctrl_num]
        for tm in 1:tnum
            for cm in 1:ctrl_num
                ∂ρt_T_δV = ∂ρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = ∂xδρt_T_δV*SLD[1]
                anti_commu = 2*SLD[pm]*SLD[1]
                term2 = ∂ρt_T_δV*anti_commu
                δF[cm][tm] = δF[cm][tm]-((2*term1-term2)|>real)
            end
        end
        
    elseif para_num == 2
        F_det = F_T[0][0]*F_T[1][1]-F_T[0][1]*F_T[1][0]
        δF = [[0.0 for i in 1:tnum] for ctrl in 1:ctrl_num]
        for tm in 1:tnum
            for cm in 1:ctrl_num
                for pm in 1:para_num
                    ∂ρt_T_δV = ∂ρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = ∂xδρt_T_δV*SLD[pm]
                    anti_commu = 2*SLD[pm]*SLD[pm]
                    term2 = ∂ρt_T_δV*anti_commu
                    δF[cm][tm] = δF[cm][tm]-((2*term1-term2)|>real)
                end
            δF[cm][tm] = δF[cm][tm]/F_det
            end
        end

    else
        δF = [[0.0 for i in 1:tnum] for ctrl in 1:ctrl_num]
        for tm in 1:tnum
            for cm in 1:ctrl_num
                for pm in 1:para_num
                    ∂ρt_T_δV = ∂ρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = ∂xδρt_T_δV*SLD[pm]
                    anti_commu = 2*SLD[pm]*SLD[pm]
                    term2 = ∂ρt_T_δV*anti_commu
                    δF[cm][tm] = δF[cm][tm]-((2*term1-term2)|>real) /(F_T[pm][pm]*F_T[pm][pm])
                end
            end
        end
    end
    ρt |> vec2mat |> filterZeros!, ∂ρt_∂x |> vec2mat |> filterZeros!, δF
end

function propagate(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}},
                   γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}
    dim = size(H0)[1]
    para_num = length(∂H_∂x)
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ρt = [Vector{ComplexF64}(undef, dim^2)  for i in 1:length(times)]
    ∂ρt_∂x = [[Vector{ComplexF64}(undef, dim^2) for i in 1:length(times)] for para in 1:para_num]
    Δt = times[2] - times[1]
    ρt[1] = evolute(H[1], Liouville_operator, γ, times, times[1]) * (ρ_initial |> vec)
    for para in  1:para_num
        ∂ρt_∂x[para][1] = -im * Δt * liouville_commu(∂H_∂x[para]) * ρt[1]
    end
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, times, times[t])
        ρt[t] =  expL * ρt[t - 1]
        for para in para_num
            ∂ρt_∂x[para][t] = -im * Δt * liouville_commu(∂H_∂x[para]) * ρt[t] + expL * ∂ρt_∂x[para][t - 1]
        end
    end
    ρt .|> vec2mat, ∂ρt_∂x .|> vec2mat
end

function propagate!(grape::GrapeControl)
    grape.ρ, grape.∂ρ_∂x = propagate(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial,
                                                grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, 
                                                grape.control_coefficients, grape.times )
end

function propagate_analitical!(grape::GrapeControl)
    grape.ρ, grape.∂ρ_∂x, δF = propagate_analitical(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial,
                                                grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, 
                                                grape.control_coefficients, grape.times )
    δF
end

function basis(dim, si, ::T)::Array{T} where {T <: Complex}
    result = zeros(T, dim)
    result[si] = 1.0
    result
end

function Adam(gt, t, para, m_t, v_t, alpha=0.01, beta1=0.90, beta2=0.99, epsilon=1e-8)
    t = t+1
    m_t = beta1*m_t + (1-beta1)*gt
    v_t = beta2*v_t + (1-beta2)*(gt*gt)
    m_cap = m_t/(1-(beta1^t))
    v_cap = v_t/(1-(beta2^t))
    para = para+(alpha*m_cap)/(sqrt(v_cap)+epsilon)
    return para, m_t, v_t
end

function Adam!(grape, δ, mt=0.0, vt=0.0)
    for ctrl in 1:length(δ)
        for ti in 1:length(grape.times)
            grape.control_coefficients[ctrl][ti], mt, vt = Adam(δ[ctrl][ti], ti, grape.control_coefficients[ctrl][ti], mt, vt)
        end
    end
end

function CFI(ρ, dρ, M)
    m_num = length(M)
    p = zero(ComplexF64)
    dp = zero(ComplexF64)
    F = 0.
    for i in 1:m_num
        mp = M[i]
        p += tr(ρ * mp)
        dp = tr(dρ * mp)
        cadd = 0.
        if p != 0
            cadd = (dp^2) / p
        end
        F += cadd
    end 
    real(F)
end

function CFI(M::Vector{Matrix{T}}, H::Vector{Matrix{T}}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    ρt = evolute(H[1], Liouville_operator, γ, times, times[1]) * (ρ_initial |> vec)
    ∂ρt_∂x = -im * Δt * liouville_commu(∂H_∂x) * ρt
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, times, times[t])
        ρt=  expL * ρt
        ∂ρt_∂x= -im * Δt * liouville_commu(∂H_∂x) * ρt + expL * ∂ρt_∂x
    end
    CFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
end

function CFIM(ρ, dρ, M)
    m_num = length(M)
    cfim = [tr.(kron(dρ', dρ).*M[i]) / tr(ρ*M[i])  for i in 1:m_num] |> sum
end

function CFIM(M::Vector{Matrix{T}}, H::Vector{Matrix{T}}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    para_num = length(∂H_∂x)
    ρt = evolute(H[1], Liouville_operator, γ, times, times[1]) * ρ_initial[:]
    ∂ρt_∂x = [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num]
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, times, times[t])
        ρt=  expL * ρt
        ∂ρt_∂x= [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    CFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
end

function CFI(M, grape::GrapeControl)
    CFI(M,Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times)
end

function CFIM(M, grape::GrapeControl)
    CFIM(M,Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients), grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times)
end

function gradient_CFI!(grape::GrapeControl{T}, M) where {T <: Complex}
    δI = gradient(x->CFI(M, Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients)[1].|>real
    grape.control_coefficients += grape.ϵ*δI
end

function gradient_CFIM!(grape::GrapeControl{T}, M) where {T <: Complex}
    δI = gradient(x->CFIM(M, Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients).|>real
    grape.control_coefficients += grape.ϵ*δI
end

function gradient_CFI_ADAM!(grape::GrapeControl{T}, M) where {T <: Complex}
    δI = gradient(x->CFI(M, Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients)[1].|>real
    Adam!(grape, δI)
end

function gradient_CFIM_ADAM!(grape::GrapeControl{T}, M, mt, vt) where {T <: Complex}
    δI = gradient(x->CFIM(M, Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients).|>real
    Adam!(grape, δI)
end

function SLD(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T <: Complex}
    2 * pinv(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ)) * vec(∂ρ_∂x) |> vec2mat
end

function SLD(ρ::Vector{T},∂ρ_∂x::Vector{T}) where {T <: Complex}
    SLD(ρ |> vec2mat, ∂ρ_∂x |> vec2mat)
end

function SLD(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T <: Complex}
    (x->SLD(ρ, x)).(∂ρ_∂x)
end

function SLD_qr(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T <: Complex}
    2 * (qr(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), Val(true)) \ vec(∂ρ_∂x)) |> vec2mat
end

function SLD_eig(ρ::Array{T}, dρ::Array{T})::Array{T} where {T <: Complex}
    dim = size(ρ)[1]
    if typeof(dρ) == Array{T,2}
        purity = tr(ρ * ρ)
        SLD_res = zeros(T, dim, dim)
        if abs(1 - purity) < 1e-8
            SLD_res = 2 * dρ
        else
            val, vec_mat = eigen(ρ)
            for fi in 1:dim
                for fj in 1:dim
                    coeff = 2 / (val[fi] + val[fj])
                    SLD_res[fi, fj] = coeff * (vec_mat[:,fi]' * (dρ * vec_mat[:, fj]))
                end
            end
            SLD_res[findall(SLD_res == Inf)] .= 0.
            SLD_res = vec_mat * (SLD_res * vec_mat')
        end
    else
        # multi-parameter scenario
        purity = tr(ρ * ρ)
        if abs(1 - purity) < 1e-8
            SLD_res = [2 * dρ[i] for i in 1:length(dρ)]
        else
            # SLD_res = [zeros(T,dim,dim) for i in 1:length(dρ)]
            dim = ndims(ρ)
            val, vec_mat = eigens(ρ)
            for para_i in 1:length(dρ)
                SLD_tp = zeros(T, dim, dim)
                for fi in 1:dim
                    for fj in 1:dim
                        coeff = 2. / (val[fi] + val[fj])
                        SLD_tp[fi][fj] = coeff * (vec[fi]' * (dρ[para_i] * vec[fj]))
                    end
                end
                SLD_tp[findall(SLD_rp == Inf)] .= 0.
                SLD_res[para_i] = vec_mat * (SLD_tp * vec_mat')
            end
        end
    end
    SLD_res
end

function RLD(ρ::Matrix{T}, dρ::Matrix{T}) where {T <: Complex}
    dρ * pinv(ρ)
end

function QFI_RLD(ρ, dρ)
    RLD_tp = RLD(ρ, dρ)
    F = tr(ρ * RLD_tp * RLD_tp')
    F |> real
end

function QFI(ρ, dρ)
    SLD_tp = SLD(ρ, dρ)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end

function QFIM(ρ, dρ)
    SLD_tp = SLD(ρ, dρ)
    [0.5*ρ] .* (kron(SLD_tp, SLD_tp') + kron(SLD_tp', SLD_tp)).|> tr .|>real 
end

function QFI(H::Vector{Matrix{T}}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    ρt = evolute(H[1], Liouville_operator, γ, times, times[1]) * (ρ_initial |> vec)
    ∂ρt_∂x = -im * Δt * liouville_commu(∂H_∂x) * ρt
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, times, times[t])
        ρt=  expL * ρt
        ∂ρt_∂x= -im * Δt * liouville_commu(∂H_∂x) * ρt + expL * ∂ρt_∂x
    end
    QFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFIM(H::Vector{Matrix{T}}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    para_num = length(∂H_∂x)
    ρt = evolute(H[1], Liouville_operator, γ, times, times[1]) * ρ_initial[:]
    ∂ρt_∂x = [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num]
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, times, times[t])
        ρt=  expL * ρt
        ∂ρt_∂x= [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    QFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFI(grape::GrapeControl)
    QFI(Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times)
end

function QFIM(grape::GrapeControl)
    QFIM(Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients), grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times)
end

function gradient_QFI!(grape::GrapeControl{T}) where {T <: Complex}
    δF = gradient(x->QFI(Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients)[1].|>real
    grape.control_coefficients += grape.ϵ*δF
end

function gradient_QFIM!(grape::GrapeControl{T}) where {T <: Complex}
    δF = gradient(x->1/(QFIM(Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times) |> pinv |> tr |>real), grape.control_coefficients).|>real |>sum
    grape.control_coefficients += grape.ϵ*δF
end

function gradient_QFI_ADAM!(grape::GrapeControl{T}) where {T <: Complex}
    δF = gradient(x->QFI(Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients)[1].|>real
    Adam!(grape, δF)
end

function gradient_QFIM_ADAM!(grape::GrapeControl{T}) where {T <: Complex}
    δF = gradient(x->1/(QFIM(Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times) |> pinv |> tr |>real), grape.control_coefficients).|>real |>sum
    Adam!(grape, δF)
end

function gradient_QFI_analitical_ADAM!(grape::GrapeControl{T}) where {T <: Complex}
    δF = propagate_analitical!(grape)
    Adam!(grape, δF)
end

function gradient_QFIM_ODE(grape::GrapeControl)
    H = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients)
    Δt = grape.times[2] - grape.times[1]
    t_num = length(grape.times)
    para_num = length(grape.Hamiltonian_derivative)    
    ctrl_num = length(grape.control_Hamiltonian)
    tspan(j) = (grape.times[1], grape.times[j])
    tspan() = (grape.times[1], grape.times[end])
    u0 = grape.ρ_initial |> vec
    evo(p, t) = evolute(p[t2Num(tspan()[1], Δt,  t)], grape.Liouville_operator, grape.γ, grape.times, t2Num(tspan()[1], Δt, t)) 
    f(u, p, t) = evo(p, t) * u
    prob = DiscreteProblem(f, u0, tspan(), H,dt=Δt)
    ρt = solve(prob).u 
    ∂ρt_∂x = Vector{Vector{Vector{eltype(u0)}}}(undef, 1)
    for para in 1:para_num
        devo(p, t) = -1.0im * Δt * liouville_commu(grape.Hamiltonian_derivative[para]) * evo(p, t) 
        du0 = devo(H, tspan()[1]) * u0
        g(du, p, t) = evo(p, t) * du + devo(p, t) * ρt[t2Num(tspan()[1], Δt,  t)] 
        dprob = DiscreteProblem(g, du0, tspan(), H,dt=Δt) 
        ∂ρt_∂x[para] = solve(dprob).u
    end
    δρt_δV = Matrix{Vector{Vector{eltype(u0)}}}(undef,ctrl_num,length(grape.times))
    for ctrl in 1:ctrl_num
        for j in 1:t_num
            devo(p, t) = -1.0im * Δt * liouville_commu(grape.control_Hamiltonian[ctrl]) * evo(p, t) 
            du0 = devo(H, tspan()[1]) * u0
            g(du, p, t) = evo(p, t) * du + devo(p, t) * ρt[t2Num(tspan()[1], Δt,  t)] 
            dprob = DiscreteProblem(g, du0, tspan(j), H,dt=Δt) 
            δρt_δV[ctrl,j] = solve(dprob).u
        end
    end
    ∂xδρt_δV = Array{Vector{eltype(u0)}, 3}(undef,para_num, ctrl_num,length(grape.times))
    for para in 1:para_num
        for ctrl in 1:ctrl_num
            dxevo = -1.0im * Δt * liouville_commu(grape.Hamiltonian_derivative[para]) 
            dkevo = -1.0im * Δt * liouville_commu(grape.control_Hamiltonian[ctrl])
            for j in 1:t_num
                g(du, p, t) = dxevo * dkevo  * evo(p, t) * ρt[t2Num(tspan()[1], Δt,  t)] +
                              dxevo * evo(p, t) * δρt_δV[ctrl, j][t2Num(tspan()[1], Δt,  t)] +
                              dkevo * evo(p, t) * ∂ρt_∂x[para][t2Num(tspan()[1], Δt,  t)] + 
                              evo(p, t) * du
                du0 = dxevo * dkevo  * evo(H,tspan()[1]) * ρt[t2Num(tspan()[1], Δt, tspan()[1])]
                dprob = DiscreteProblem(g, du0, tspan(j), H, dt=Δt)
                ∂xδρt_δV[para, ctrl, j] = solve(dprob).u[end]
            end
        end
    end
    δF = grape.control_coefficients .|> zero
    for para in 1:para_num
        SLD_tp = SLD(ρt[end], ∂ρt_∂x[para][end])
        for ctrl in 1:ctrl_num
            for j in 1:t_num   
                δF[ctrl][j] -= 2 * tr((∂xδρt_δV[para,ctrl,j]|> vec2mat) * SLD_tp) - 
                                   tr((δρt_δV[ctrl, j][end] |> vec2mat) * SLD_tp^2) |> real
            end
        end
    end
    δF
end

function gradient_QFIM_ODE!(grape::GrapeControl{T}) where {T <: Complex}
    grape.control_coefficients += grape.ϵ * gradient_QFIM_ODE(grape)
end

function Run(grape)
    println("AutoGrape strategies")
    println("quantum parameter estimation")
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter estimation scenario")
        qfi_ini = QFI(grape)
        qfi_list = [qfi_ini]
        println("initial QFI is $(qfi_ini)")
        gradient_QFI!(grape)
        while true
            qfi_now = QFI(grape)
            gradient_QFI!(grape)
            if  0 < (qfi_now - qfi_ini) < 1e-4
                println("\n Iteration over, data saved.")
                println("Final QFI is ", qfi_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "qfi", qfi_list)
                break
            else
                qfi_ini = qfi_now
                append!(qfi_list,qfi_now)
                print("current QFI is ", qfi_now, " ($(qfi_list|>length) epochs)    \r")
            end
        end
    else
        println("multiple parameters estimation scenario")
        f_ini =1/(grape |> QFIM |> inv |> tr)
        f_list = [f_ini]
        println("initial 1/tr(F^-1) is $(f_ini)")
        gradient_QFIM!(grape)
        while true
            f_now = 1/(grape |> QFIM |> inv |> tr)
            gradient_QFIM!(grape)
            if  0< f_now - f_ini < 1e-4
                println("\n Iteration over, data saved.")
                println("Final 1/tr(F^-1) is ", f_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "f", f_list)
                break
            else
                f_ini = f_now
                append!(f_list,f_now)
                print("current 1/tr(F^-1) is ", f_now, " ($(f_list|>length) epochs)    \r")
            end
        end
    end
end

function Run(M, grape::GrapeControl{T}) where {T<: Complex}
    println("classical parameter estimation")
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter estimation scenario")
        cfi_ini = CFI(M, grape)
        cfi_list = [cfi_ini]
        println("initial CFI is $(cfi_ini)")
        gradient_CFI!(M, grape)
        while true
            cfi_now = CFI(M, grape)
            gradient_CFI!(M, grape)
            if  0 < (cfi_now - cfi_ini) < 1e-4
                println("\n Iteration over, data saved.")
                println("Final CFI is ", cfi_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "cfi", cfi_list)
                break
            else
                cfi_ini = cfi_now
                append!(cfi_list,cfi_now)
                print("current CFI is ", cfi_now, " ($(cfi_list|>length) epochs)    \r")
            end
        end
    else
        println("multiple parameters estimation scenario")
        f_ini =1/(grape |> CFIM |> inv |> tr)
        f_list = [f_ini]
        println("initial 1/tr(F^-1) is $(f_ini)")
        gradient_CFIM!(M, grape)
        while true
            f_now = 1/(grape |> CFIM |> inv |> tr)
            gradient_CFIM!(M, grape)
            if  0< f_now - f_ini < 1e-4
                println("\n Iteration over, data saved.")
                println("Final 1/tr(I^-1) is ", f_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "f", f_list)
                break
            else
                f_ini = f_now
                append!(f_list,f_now)
                print("current 1/tr(I^-1) is ", f_now, " ($(f_list|>length) epochs)    \r")
            end
        end
    end
end

function RunADAM(grape)
    println("AutoGrape strategies")
    println("quantum parameter estimation")
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter estimation scenario")
        qfi_ini = QFI(grape)
        qfi_list = [qfi_ini]
        println("initial QFI is $(qfi_ini)")
        gradient_QFI_ADAM!(grape)
        while true
            qfi_now = QFI(grape)
            gradient_QFI_ADAM!(grape)
            if  0 < (qfi_now - qfi_ini) < 1e-4
                println("\n Iteration over, data saved.")
                println("Final QFI is ", qfi_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "qfi", qfi_list)
                break
            else
                qfi_ini = qfi_now
                append!(qfi_list,qfi_now)
                print("current QFI is ", qfi_now, " ($(qfi_list|>length) epochs)    \r")
            end
        end
    else
        println("multiple parameters estimation scenario")
        f_ini =1/(grape |> QFIM |> inv |> tr)
        f_list = [f_ini]
        println("initial 1/tr(F^-1) is $(f_ini)")
        gradient_QFIM_ADAM!(grape)
        while true
            f_now = 1/(grape |> QFIM |> inv |> tr)
            gradient_QFIM_ADAM!(grape)
            if  0< f_now - f_ini < 1e-4
                println("\n Iteration over, data saved.")
                println("Final 1/tr(F^-1) is ", f_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "f", f_list)
                break
            else
                f_ini = f_now
                append!(f_list,f_now)
                print("current 1/tr(F^-1) is ", f_now, " ($(f_list|>length) epochs)    \r")
            end
        end
    end
end

function RunAnaliticalADAM(grape)
    println("Analitical strategies")
    println("quantum parameter estimation")
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter estimation scenario")
        qfi_ini = QFI(grape)
        qfi_list = [qfi_ini]
        println("initial QFI is $(qfi_ini)")
        gradient_QFI_analitical_ADAM!(grape)
        while true
            qfi_now = QFI(grape)
            gradient_QFI_analitical_ADAM!(grape)
            if  0 < (qfi_now - qfi_ini) < 1e-4
                println("\n Iteration over, data saved.")
                println("Final QFI is ", qfi_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "qfi", qfi_list)
                break
            else
                qfi_ini = qfi_now
                append!(qfi_list,qfi_now)
                print("current QFI is ", qfi_now, " ($(qfi_list|>length) epochs)    \r")
            end
        end
    end
end

function RunADAM(M, grape::GrapeControl{T}) where {T<: Complex}
    println("classical parameter estimation")
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter estimation scenario")
        cfi_ini = CFI(M, grape)
        cfi_list = [cfi_ini]
        println("initial CFI is $(cfi_ini)")
        gradient_CFI_ADAM!(M, grape)
        while true
            cfi_now = CFI(M, grape)
            gradient_CFI_ADAM!(M, grape)
            if  0 < (cfi_now - cfi_ini) < 1e-4
                println("\n Iteration over, data saved.")
                println("Final CFI is ", cfi_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "cfi", cfi_list)
                break
            else
                cfi_ini = cfi_now
                append!(cfi_list,cfi_now)
                print("current CFI is ", cfi_now, " ($(cfi_list|>length) epochs)    \r")
            end
        end
    else
        println("multiple parameters estimation scenario")
        f_ini =1/(grape |> CFIM |> inv |> tr)
        f_list = [f_ini]
        println("initial 1/tr(F^-1) is $(f_ini)")
        gradient_CFIM!(M, grape)
        while true
            f_now = 1/(grape |> CFIM |> inv |> tr)
            gradient_CFIM!(M, grape)
            if  0< f_now - f_ini < 1e-4
                println("\n Iteration over, data saved.")
                println("Final 1/tr(I^-1) is ", f_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "f", f_list)
                break
            else
                f_ini = f_now
                append!(f_list,f_now)
                print("current 1/tr(I^-1) is ", f_now, " ($(f_list|>length) epochs)    \r")
            end
        end
    end
end

function RunMixed( grape::GrapeControl{T}, particle_num= 1, c0=0.5, c1=0.5, c2=0.5, prec_rough = 0.1, sd=1234, episode=400) where {T<: Complex}
    println("Combined strategies")
    println("searching initial controls with particle swarm optimization ")
    tnum = length(grape.times)
    ctrl_num = length(grape.control_Hamiltonian)
    particles, velocity = repeat(grape, particle_num), [[10*ones(tnum) for i in 1:ctrl_num] for j in 1:particle_num]
    pbest = [[zeros(tnum) for i in 1:ctrl_num] for j in 1:particle_num]
    gbest = [zeros(tnum) for i in 1:ctrl_num]
    velocity_best = [zeros(Float64, tnum) for i in 1:ctrl_num]
    p_fit = zeros(particle_num)
    qfi_ini = QFI(grape)
    println("initial QFI is $(qfi_ini)")
    for ei in 1:episode
        fit_pre = 0.0
        fit = 0.0
        for pi in 1:particle_num
            propagate!(particles[pi])
            f_now = QFI(particles[pi])
            
            if f_now > p_fit[pi]
                p_fit[pi] = f_now
                for di in 1:ctrl_num
                    for ni in 1:tnum
                        pbest[pi][di][ni] = particles[pi].control_coefficients[di][ni]
                    end
                end
            end
        end
        for pj in 1:particle_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                for dj in 1:ctrl_num
                    for nj in 1:tnum
                        gbest[dj][nj] =  particles[pj].control_coefficients[dj][nj]
                        velocity_best[dj][nj] = velocity[pj][dj][nj]
                    end
                end
            end
        end
        Random.seed!(sd)
        for pk in 1:particle_num
            for dk in 1:ctrl_num
                for ck in 1:tnum
                    velocity[pk][dk][ck]  = c0*velocity[pk][dk][ck] + c1*rand()*(pbest[pk][dk][ck] - particles[pk].control_coefficients[dk][ck]) 
                                          + c2*rand()*(gbest[dk][ck] - particles[pk].control_coefficients[dk][ck])  
                    particles[pk].control_coefficients[dk][ck] = particles[pk].control_coefficients[dk][ck] + velocity[pk][dk][ck]
                end
            end
        end
        fit_pre = fit
        if abs(fit-fit_pre) < prec_rough
            grape.control_coefficients = gbest
            println("PSO strategy finished, switching to GRAPE")
            Run(grape)
            return nothing
        end
        print("current QFI is $fit ($ei epochs)    \r")
    end
end

function RunODE(grape::GrapeControl{T}) where {T<: Complex}
    println("quantum parameter estimation")
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter estimation scenario")
        qfi_ini = QFI(grape)
        qfi_list = [qfi_ini]
        println("initial QFI is $(qfi_ini)")
        gradient_QFIM_ODE!(grape)
        while true
            qfi_now = QFI(grape)
            gradient_QFIM_ODE!(grape)
            if  0 < (qfi_now - qfi_ini) < 1e-4
                println("\n Iteration over, data saved.")
                println("Final QFI is ", qfi_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "qfi", qfi_list)
                break
            else
                qfi_ini = qfi_now
                append!(qfi_list,qfi_now)
                print("current QFI is ", qfi_now, " ($(qfi_list|>length) epochs)    \r")
            end
        end
    else
        println("multiple parameters estimation scenario")
        f_ini =1/(grape |> QFIM |> inv |> tr)
        f_list = [f_ini]
        println("initial 1/tr(F^-1) is $(f_ini)")
        gradient_QFIM!(grape)
        while true
            f_now = 1/(grape |> QFIM |> inv |> tr)
            gradient_QFIM!(grape)
            if  0< f_now - f_ini < 1e-4
                println("\n Iteration over, data saved.")
                println("Final 1/tr(F^-1) is ", f_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "f", f_list)
                break
            else
                f_ini = f_now
                append!(f_list,f_now)
                print("current 1/tr(F^-1) is ", f_now, " ($(f_list|>length) epochs)    \r")
            end
        end
    end
end 

function RunPSO(grape::GrapeControl{T}, particle_num= 1,  c0=0.5, c1=0.5, c2=0.5,seed=1234, episode=400) where {T<: Complex}
    println("PSO strategies")
    println("searching optimal controls with particle swarm optimization ")
    tnum = length(grape.times)
    ctrl_num = length(grape.control_Hamiltonian)
    particles, velocity = repeat(grape, particle_num), [[10*ones(tnum) for i in 1:ctrl_num] for j in 1:particle_num]
    pbest = [[zeros(tnum) for i in 1:ctrl_len] for j in 1:particle_num]
    gbest = [zeros(tnum) for i in 1:ctrl_len]
    velocity_best = [zeros(Float64, tnum) for i in 1:ctrl_num]
    p_fit = zeros(particle_num)
    qfi_ini = QFI(grape)
    println("initial QFI is $(qfi_ini)")
    for ei in 1:episode
        fit_pre = 0.0
        fit = 0.0
        for pi in 1:particle_num
            propagate!(particle[pi])
            f_now = QFI(ρt[end], ∂ρt_∂x[1][end])
            
            if f_now > p_fit[pi]
                p_fit[pi] = f_now
                for di in 1:ctrl_num
                    for ni in 1:tnum
                        pbest[pi][di][ni] = particles[pi].control_coefficients[di][ni]
                    end
                end
            end
        end
        for pj in 1:particle_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                for dj in 1:ctrl_num
                    for nj in 1:cnum
                        gbest[dj][nj] =  particles[pk].control_coefficients[dj][nj]
                        velocity_best[dj][nj] = velocity[pj][dj][nj]
                    end
                end
            end
        end
        Random.seed!(seed)
        for pk in 1:particle_num
            for dk in 1:ctrl_num
                for ck in 1:tnum
                    velocity[pk][dk][ck]  = c0*velocity[pk][dk][ck] + c1*rand()*(pbest[pk][dk][ck] - particles[pk].control_coefficients[dk][ck]) 
                                          + c2*rand()*(gbest[dk][ck] - particles[pk].control_coefficients)  
                    ctrl_span[pk][dk][ck] = particles[pk].control_coefficients[dk][ck] + velocity[pk][dk][ck]
                end
            end
        end
        fit_pre = fit
        if abs(fit-fit_pre) < 1e-4
            grape.control_coefficients = gbest
            println("Final QFI is ", fit)
            return nothing
        end
        print("current QFI is $fit ($ei epochs)    \r")
    end
end

export sigmax, sigmay, sigmaz, sigmam
export GrapeControl, evolute, propagate!, QFI, CFI, gradient_CFI!,gradient_QFIM!, Run, RunODE, RunMixed, RunADAM
end
