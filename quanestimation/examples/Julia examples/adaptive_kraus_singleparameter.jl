using Random
using StatsBase
include("../src/QuanEstimation.jl")

# initial state
rho0 = 0.5*[1.0 1.0+0.0im; 1.0 1.0]
# free Hamiltonian
function K_func(x)
    K1 = [1.0 0.0; 0.0 sqrt(1 - x[1])]
    K2 = [0.0 sqrt(x[1]); 0.0 0.0]
    return [K1, K2]
end
function dK_func(x)
    dK1 = [1.0 0.0; 0.0 -0.5 / sqrt(1 - x[1])]
    dK2 = [0.0 0.5 / sqrt(x[1]); 0.0 0.0]
    return [[dK1],[dK2]]
end
# measurement 
M1 = 0.5*[1.0+0.0im  1.0; 1.0  1.0]
M2 = 0.5*[1.0+0.0im -1.0; -1.0  1.0]
M = [M1, M2]
#### prior distribution ####
x = [range(0.1, stop=0.9, length=100)].|>Vector
p = (1.0/(x[1][end]-x[1][1]))*ones(length(x[1]))

rho = Vector{Matrix{ComplexF64}}(undef, length(x[1]))
for i = 1:length(x[1]) 
    K_tp = K_func([x[1][i]])
    rho[i] = [K * rho0 * K' for K in K_tp] |> sum
end

# Bayesian estimation
Random.seed!(1234)
y = [0 for i in 1:500]
res_rand = sample(1:length(y), 125, replace=false)
for i in 1:length(res_rand)
    y[res_rand[i]] = 1
end
pout, xout = QuanEstimation.Bayes(x, p, rho, y, M=M, savefile=false)

p = pout
K, dK = QuanEstimation.AdaptiveInput(x, K_func, dK_func; channel="kraus")
QuanEstimation.adaptive(x, p, rho0, K, dK; M=M, max_episode=10)
