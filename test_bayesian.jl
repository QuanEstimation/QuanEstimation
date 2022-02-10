using LinearAlgebra
using BoundaryValueDiffEq
using Trapz
using Interpolations

include("quanestimation/JuliaSrc/Common/common.jl")
include("quanestimation/JuliaSrc/Dynamics/dynamics.jl")
include("quanestimation/JuliaSrc/AsymptoticBound/CramerRao.jl")
include("quanestimation/JuliaSrc/BayesianBound/BayesianCramerRao.jl")
include("quanestimation/JuliaSrc/BayesianBound/ZivZakai.jl")

function secondorder(H0, ∂H_∂x::Vector{Matrix{T}}, ∂2H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, tspan) where {T<:Complex,R<:Real}

    para_num = length(∂H_∂x)
    
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    ∂2H_L = [liouville_commu(∂2H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    ∂2ρt_∂x = [ρt |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        expL = evolute(H0, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
        ∂2ρt_∂x = [(-im*Δt*∂2H_L[i] + Δt*Δt*∂H_L[i]*∂H_L[i])*ρt - 2*im*Δt*∂H_L[i]*∂ρt_∂x[i] for i in 1:para_num] + [expL] .* ∂2ρt_∂x
    end
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat, ∂2ρt_∂x |> vec2mat
end


#initial state
rho0 = 0.5*[1.0 1.0+0.0im; 1.0 1.0]
#Hamiltonian
B = pi/2.0
sx = [0.0 1.0; 1.0 0.0im]
sy = [0.0 -1.0im; 1.0im 0.0]
sz = [1.0 0.0im; 0.0 -1.0]

function H0_res(x)
    return 0.5*B*(sx*cos(x)+sz*sin(x))
end
function dH_res(x)
    return 0.5*B*(-sx*sin(x)+sz*cos(x))
end
function d2H_res(x)
    return 0.5*B*(-sx*cos(x)-sz*sin(x))
end

#measurement
M1 = 0.5*[1.0 1.0; 1.0 1.0+0.0im]
M2 = 0.5*[1.0 -1.0; -1.0 1.0+0.0im]
Measurement = [M1, M2]

tspan = range(0.0, stop=1.0, length=1000)

dim = 2
para_num = 1

#### flat distribution ####
xspan = range(0.0, stop=pi/2.0, length=100)
x1, x2 = xspan[1], xspan[end]
p = (1.0/(x2-x1))*ones(length(xspan))
dp = zeros(length(xspan)) 

rho_all = [zeros(ComplexF64, dim, dim) for i in 1:length(xspan)]
drho_all = [[zeros(ComplexF64, dim, dim) for j in 1:para_num] for i in 1:length(xspan)]
d2rho_all = [[zeros(ComplexF64, dim, dim) for j in 1:para_num] for i in 1:length(xspan)]
for i in 1:length(xspan)
    H0 = H0_res(xspan[i])
    dH = dH_res(xspan[i])
    d2H = d2H_res(xspan[i])
    rho, drho, d2rho = secondorder(H0, [dH], [d2H], rho0, tspan)
    rho_all[i] = rho
    for j in 1:para_num
        drho_all[i][j] = drho[j]
        d2rho_all[i][j] = d2rho[j]
    end
end

f1 = BQCRB(rho_all, drho_all, p, [x1, x2], accuracy=1e-8)
f2 = TWC(rho_all, drho_all, p, [dp], [x1, x2], accuracy=1e-8)
f3 = OBB(rho_all, drho_all, d2rho_all, p, [dp], [x1, x2], accuracy=1e-8)
f4 = QZZB(rho_all, p, [x1,x2])
println(f1, f2, f3, f4)

