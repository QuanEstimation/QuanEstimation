using Random
using StatsBase
include("../src/QuanEstimation.jl")

# initial state
rho0 = 0.5*[1.0 1.0+0.0im; 1.0 1.0]
psi0 = [1, 0]
psi1 = [0, 1]
function K_func(x)
    n, p = x[1], x[2]
    K0 = sqrt(1-n)*(psi0*psi0'+sqrt(1-p)*psi1*psi1')
    K1 = sqrt(p-p*n)*psi0*psi1'
    K2 = sqrt(n)*(sqrt(1-p)*psi0*psi0'+psi1*psi1')
    K3 = sqrt(p*n)*psi1*psi0'
    return [K0, K1, K2, K3]
end
function dK_func(x)
    n, p = x[1], x[2]
    dK0_n = -0.5*(psi0*psi0'+sqrt(1-p)*psi1*psi1')/sqrt(1-n)
    dK1_n = -0.5*p*psi0*psi1'/sqrt(p-p*n)
    dK2_n = 0.5*(sqrt(1-p)*psi0*psi0'+psi1*psi1')/sqrt(n)
    dK3_n = 0.5*p*psi1*psi0'/sqrt(p*n)
    dK0_p = -0.5*sqrt(1-n)*psi1*psi1'/sqrt(1-p)
    dK1_p = 0.5*(1-n)*psi0*psi1'/sqrt(p-p*n)
    dK2_p = -0.5*sqrt(n)*psi0*psi0'/sqrt(1-p)
    dK3_p = -0.5*sqrt(n)*psi0*psi0'/sqrt(1-p)
    dK3_p = 0.5*n*psi1*psi0'/sqrt(p*n)
    return [[dK0_n, dK0_p], [dK1_n, dK1_p], [dK2_n, dK2_p], [dK3_n, dK3_p]]
end
# measurement
M1 = 0.5*[1.0+0.0im  1.0; 1.0  1.0]
M2 = 0.5*[1.0+0.0im -1.0; -1.0  1.0]
M = [M1, M2]
# prior distribution
x = [range(0.1, stop=0.9, length=100), range(0.1, stop=0.9, length=10)].|>Vector
p = (1.0/(x[1][end]-x[1][1]))*(1.0/(x[2][end]-x[2][1]))*ones((length(x[1]), length(x[2])))

rho = Matrix{Matrix{ComplexF64}}(undef, length.(x)...)
for i = 1:length(x[1])
    for j = 1:length(x[2])  
        x_tp = [x[1][i], x[2][j]]
        K_tp = K_func(x_tp)
        rho[i,j]  = [K * rho0 * K' for K in K_tp] |> sum
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
K, dK = QuanEstimation.AdaptiveInput(x, K_func, dK_func; channel="kraus")
QuanEstimation.adaptive(x, p, rho0, K, dK, M=M, max_episode=10)
