using Random
using StatsBase
include("../src/QuanEstimation.jl")

# initial state
rho0 = 0.5*[1.0 1.0+0.0im; 1.0 1.0]
# free Hamiltonian
sx = [0.0 1.0; 1.0 0.0im]
sy = [0.0 -1.0im; 1.0im 0.0]
sz = [1.0 0.0im; 0.0 -1.0]
function H0_func(x)
    return 0.5*x[2]*(sx*cos(x[1])+sz*sin(x[1]))
end
function dH_func(x)
    return [0.5*x[2]*(-sx*sin(x[1])+sz*cos(x[1])), 0.5*(sx*cos(x[1])+sz*sin(x[1]))]
end
# measurement
M1 = 0.5*[1.0+0.0im  1.0; 1.0  1.0]
M2 = 0.5*[1.0+0.0im -1.0; -1.0  1.0]
M = [M1, M2]
# dynamics
tspan = range(0.0, stop=1.0, length=1000)
# dissipation
decay_opt = [zeros(ComplexF64,size(rho0)[1],size(rho0)[1])] 
gamma = [0.0]
# prior distribution
x = [range(0.0, stop=pi/2.0, length=100), range(pi/2-0.1, stop=pi/2+0.1, length=10)].|>Vector
p = (1.0/(x[1][end]-x[1][1]))*(1.0/(x[2][end]-x[2][1]))*ones((length(x[1]), length(x[2])))

rho = Matrix{Matrix{ComplexF64}}(undef, length.(x)...)
for i = 1:length(x[1])
    for j = 1:length(x[2])  
        x_tp = [x[1][i], x[2][j]]
        H0_tp = H0_func(x_tp)
        dH_tp = dH_func(x_tp)
        rho_tp, drho_tp = QuanEstimation.expm(H0_tp, dH_tp, [zeros(ComplexF64,size(rho0)[1],size(rho0)[1])], [zeros(length(tspan)-1)],rho0, tspan,decay_opt, gamma)
        rho[i,j] = rho_tp[end]
    end
end
# Bayesian estimation
Random.seed!(1234)
y = [0 for i in 1:500]
res_rand = sample(1:length(y), 125, replace=false)
for i in 1:length(res_rand)
    y[res_rand[i]] = 1
end
pout, xout = QuanEstimation.Bayes(x, p, rho, y, M=M, savefile=false)
# adaptive
p = pout
H, dH = QuanEstimation.AdaptiveInput(x, H0_func, dH_func; channel="dynamics")
QuanEstimation.adaptive(x, p, rho0, tspan, H, dH, M=M, max_episode=10)
