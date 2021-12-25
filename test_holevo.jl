using LinearAlgebra
using JLD
using Random
using SharedArrays
using Base.Threads
using SparseArrays
using Convex
using SCS
using NPZ
using DelimitedFiles
using StatsBase

function vec2mat(x::Vector{T}) where {T <: Number}
    reshape(x, x |> length |> sqrt |> Int, :)  
end

function vec2mat(x)
    vec2mat.(x)
end


function evolute(H, decay_opt, γ, dt, tj)
    Ld = dt * liouvillian(H, decay_opt, γ, tj)
    exp(Ld)
end

function liouvillian(H::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, t::Real) where {T <: Complex} 
    freepart = liouville_commu(H)
    dissp = norm(γ) +1 ≈ 1 ? freepart|>zero : dissipation(decay_opt, γ, t)
    -1.0im * freepart + dissp
end

function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{R}, t::Int=0) where {T <: Complex,R <: Real}
    [γ[i] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end

function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{Vector{R}}, t::Int=0) where {T <: Complex,R <: Real}
    [γ[i][t] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end

function liouville_commu(H) 
    kron(one(H), H) - kron(H |> transpose, one(H))
end

function liouville_dissip(Γ)
    kron(Γ |> conj, Γ) - 0.5 * kron((Γ |> transpose) * (Γ |> conj), Γ |> one) - 0.5 * kron(Γ |> one, Γ' * Γ)
end

function decomposition(A)
    C = bunchkaufman(A; check=false)
    R = sqrt(Array(C.D))*C.U'C.P
    return R
end

function Holevo_bound(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}, C::Matrix{Float64}) where {T<:Complex}
    dim = size(ρ)[1]
    num = dim*dim
    para_num = length(∂ρ_∂x)
    suN = suN_generator(dim)/sqrt(2)
    Lambda = [Matrix{ComplexF64}(I,dim,dim)/sqrt(2)]
    append!(Lambda, [suN[i] for i in 1:length(suN)])
    vec_∂ρ = [[0.0 for i in 1:num] for j in 1:para_num]
    for pa in 1:para_num
        for ra in 2:num
            vec_∂ρ[pa][ra] = (∂ρ_∂x[pa]*Lambda[ra]) |> tr |> real
        end
    end
    S = zeros(ComplexF64, num, num)
    for a in 1:num
        for b in 1:num
            S[a, b] = (Lambda[a]*Lambda[b]*ρ) |> tr
        end
    end
    R = decomposition(round.(digits=1, S))

    #=========optimization variables===========#
    V = Variable(para_num, para_num)
    X = Variable(num, para_num)
    #============add constraints===============#
    constraints = [[V X'*R'; R*X Matrix{Float64}(I,num,num)] ⪰ 0 ]
    for i in 1:para_num
        for j in 1:para_num
            if i == j
                constraints += [X[:,i]'*vec_∂ρ[j] == 1]
            else
                constraints += [X[:,i]'*vec_∂ρ[j] == 0]
            end
        end
    end
    problem = minimize(tr(C*V), constraints)
    solve!(problem, SCS.Optimizer(verbose=false))
    return evaluate(tr(C*V))
end


function suN_generatorU(n, k)
    tmp1, tmp2 = ceil((1+sqrt(1+8k))/2), ceil((-1+sqrt(1+8k))/2) 
    i = k - tmp2*(tmp2-1)/2 |> Int
    j =  tmp1 |> Int
    return sparse([i, j], [j,i], [1, 1], n, n)
end

function suN_generatorV(n, k)
    tmp1, tmp2 = ceil((1+sqrt(1+8k))/2), ceil((-1+sqrt(1+8k))/2) 
    i = k - tmp2*(tmp2-1)/2 |> Int
    j =  tmp1 |> Int 
    return sparse([i, j], [j,i], [-im, im], n, n)
end

function suN_generatorW(n, k)
    diagw = spzeros(n)
    diagw[1:k] .=1
    diagw[k+1] = -k
    return spdiagm(n,n,diagw)
end

function suN_generator(n)
    result = Vector{SparseMatrixCSC{ComplexF64, Int64}}(undef, n^2-1)
    idx = 2
    itr = 1

    for i in 1:n-1
       idx_t = idx
       while idx_t > 0
            result[itr] = iseven(idx_t) ? suN_generatorU(n, (i*(i-1)+idx-idx_t+2)/2) : suN_generatorV(n, (i*(i-1)+idx-idx_t+1)/2)
            itr += 1
            idx_t -= 1
       end
       result[itr] = sqrt(2/(i+i*i))*suN_generatorW(n, i)
       itr += 1
       idx += 2
    end
    return result
end

sigmax() = [.0im 1.;1. 0.]
sigmay() = [0. -1.0im;1.0im 0.]
sigmaz() = [1.0  .0im;0. -1.]

# example 1 
# rho = 0.5*(Matrix{ComplexF64}(I,2,2)+sigmax())
# drho1 = -0.5*sigmaz()
# drho2 = 0.5*sigmay()
# drho = [drho1, drho2]
# C = Matrix{Float64}(I,2,2)
# F = Holevo_bound(rho, drho, C)

# example 2
# cr = 0.5
# rspan = LinRange(0.01,0.99,100)
# Holevo_value = zeros(100)
# f_ana = zeros(100)
# for ri in 1:length(rspan)
#     rho = 0.5*(Matrix{ComplexF64}(I,2,2) + rspan[ri]*sigmaz())
#     drho1 = -0.5*sigmaz()
#     drho2 = 0.5*rspan[ri]*sigmay()
#     drho = [drho1, drho2]
#     C = [cr 0.0; 0.0 rspan[ri]*rspan[ri]]
#     Holevo_value[ri] = Holevo_bound(rho, drho, C)
#     f_ana[ri] = cr*(1-rspan[ri]*rspan[ri])+1
# end

# println(Holevo_value-f_ana)

function run()
    # psi0 = [1.0+0.0im; 0.0+0.0im; 0.0+0.0im; 1.0+0.0im]/sqrt(2)
    # rho0 = psi0*psi0'
    # # psi0 = 0.5*[1.0+0.0im; -1.0+0.0im; -1.0+0.0im; 1.0+0.0im] 
    # # rho0 = psi0*psi0'

    # #Hamiltonian
    # omega1, omega2, g = 1.0, 1.0, 0.1
    ide = [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im]
    sx = [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]
    sy = [0.0+0.0im 0.0-1.0im; 0.0+1.0im 0.0+0.0im]
    sz = [1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im]
    # H0 = omega1*kron(sz,ide)+omega2*kron(ide,sz)+g*kron(sx,sx)  
    # ∂H_∂x = [kron(ide,sz), kron(sx,sx)] 
    # decay_opt = [kron(sz,ide), kron(ide, sz)]
    # γ = [0.05, 0.05]
    # C = [1.0 0.0; 0.0 1.0]

    # tspan = LinRange(0.0,10.0,1000)
    # Δt = tspan[2] - tspan[1]
    # para_num = length(∂H_∂x)
    # expL = evolute(H0, decay_opt, γ, Δt, 1) 
    # ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    # ρt = rho0 |> vec
    # ∂ρt_∂x = [ρt |> zero for i in 1:para_num]

    # f_value = zeros(length(tspan))
    # for t in 2:length(tspan)
    #     ρt = expL * ρt
    #     ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    #     f_value[t] = Holevo_bound(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, C)
    # end
    # f_value

    # example 1
    # rho = 0.5*(ide + sx)
    # drho = Matrix{ComplexF64}[]
    # push!(drho, -0.5*sz, 0.5*sy)
    # C = [1.0 0.0; 0.0 1.0]
    # f = Holevo_bound(rho, drho, C)
    # print(f) 

    # example 2
    cr = 0.5
    rspan = LinRange(0.01,0.99,100)
    f_all = zeros(length(rspan))
    f_analy = zeros(length(rspan))
    for ri in 1:length(rspan)
        rho = 0.5*(ide+rspan[ri]*sz)
        drho = Matrix{ComplexF64}[]
        push!(drho, 0.5*sz, 0.5*rspan[ri]*sx)
        C = [cr 0.0; 0.0 rspan[ri]*rspan[ri]]
        f_all[ri] = Holevo_bound(rho, drho, C)
        f_analy[ri] = cr*(1-rspan[ri]*rspan[ri])+1
    end
    # f_all.append(f)
    # f_analy.append(cr*(1-r*r)+1)
    println([f_all[i]-f_analy[i] for i in 1:length(rspan)])
end

f_value = run()
# open("value_jl.csv","w") do f
#     writedlm(f, f_value)
# end
